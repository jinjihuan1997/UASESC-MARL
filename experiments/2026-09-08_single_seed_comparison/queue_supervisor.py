"""Supervise frozen jobs, isolate worker failures, and adopt healthy live jobs."""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time


def stamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def process_info(pid):
    path = Path("/proc") / str(pid)
    try:
        raw = (path / "stat").read_text()
        fields = raw[raw.rfind(")") + 2:].split()
        return dict(pid=int(pid), state=fields[0], ppid=int(fields[1]),
                    pgid=int(fields[2]), start_ticks=int(fields[19]),
                    wait_status=int(fields[49]), uid=path.stat().st_uid)
    except (OSError, ValueError, IndexError):
        return None


def identity_matches(identity):
    current = process_info(identity["pid"])
    return (current is not None and current["uid"] == identity["uid"]
            and current["start_ticks"] == identity["start_ticks"])


def group_members(pgid):
    result = []
    for path in Path("/proc").iterdir():
        if path.name.isdigit():
            info = process_info(int(path.name))
            if info and info["pgid"] == pgid and info["uid"] == os.getuid():
                result.append(info)
    return result


def live(identity):
    current = process_info(identity["pid"])
    return (current is not None and current["uid"] == identity["uid"]
            and current["start_ticks"] == identity["start_ticks"]
            and current["state"] != "Z")


def stop_group(identity, grace=5.0):
    """Signal only a verified, independent training process group."""
    if identity["pgid"] != identity["pid"] or identity["pgid"] == os.getpgrp():
        raise RuntimeError("Refusing to signal a non-isolated process group")
    members = group_members(identity["pgid"])
    if not identity_matches(identity):
        # An exited leader can leave verified children in its original group.
        members = [p for p in members if p["start_ticks"] >= identity["start_ticks"]]
        if not members:
            return
        if any(p["pid"] == identity["pid"] for p in members):
            raise RuntimeError("Process-group leader PID was reused")
    if not members:
        return
    try:
        os.killpg(identity["pgid"], signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + grace
    while time.monotonic() < deadline:
        if not any(identity_matches(p) and live(p) for p in members):
            return
        time.sleep(.05)
    if any(identity_matches(p) and live(p) for p in members):
        try:
            os.killpg(identity["pgid"], signal.SIGKILL)
        except ProcessLookupError:
            pass


def health(job, progress_path, stale_seconds=180.0, startup_seconds=120.0):
    """Inspect progress AND all workers; a live parent is insufficient."""
    if not live(job["identity"]):
        return "training_process_exited"
    path = Path(progress_path)
    age = time.time() - job["started_wall"]
    status = read(path) if path.exists() else {}
    if status.get("attempt") == job["attempt"] and status.get("state") == "complete":
        # Workers can briefly be zombies while a successful job joins them.
        return "finalization_timeout" if time.time() - path.stat().st_mtime > stale_seconds else None
    members = group_members(job["identity"]["pgid"])
    dead = [p for p in members if p["pid"] != job["identity"]["pid"] and p["state"] == "Z"]
    if dead:
        return "dead_worker:" + json.dumps(dead, sort_keys=True)
    if not path.exists():
        return "initialization_timeout" if age > startup_seconds else None
    if status.get("attempt") != job["attempt"]:
        return "initialization_timeout" if age > startup_seconds else None
    if status.get("state") == "failed":
        return "training_reported_failure:" + status.get("error", "unknown")
    if time.time() - path.stat().st_mtime > stale_seconds:
        return "no_progress_timeout"
    return None


def replace_supervisor(identity, expected_children):
    """Retire only the old coordinator, retaining independent worker groups."""
    if not live(identity) or identity["uid"] != os.getuid():
        raise RuntimeError("Old supervisor identity is not live or owned")
    for child in expected_children:
        now = process_info(child["pid"])
        if (now is None or not identity_matches(child) or now["ppid"] != identity["pid"]
                or child["pgid"] != child["pid"] or child["pgid"] == identity["pgid"]):
            raise RuntimeError("Child identity/group does not match the coordinator")
    os.kill(identity["pid"], signal.SIGSTOP)
    try:
        if not identity_matches(identity):
            raise RuntimeError("Coordinator identity changed")
        # SIGTERM/KeyboardInterrupt would run the old kill-all cleanup.
        # Its independent children retain their own log descriptors and state.
        os.kill(identity["pid"], signal.SIGKILL)
    except BaseException:
        if identity_matches(identity):
            os.kill(identity["pid"], signal.SIGCONT)
        raise


class Supervisor:
    def __init__(self, out, jobs=4, stale_seconds=180, max_attempts=2, train_only=False):
        self.out = Path(out).resolve()
        sys.path.insert(0, str(self.out / "frozen"))
        import protocol
        import run as frozen_run
        if Path(protocol.__file__).resolve() != self.out / "frozen/protocol.py":
            raise RuntimeError("Supervisor must use the run's frozen protocol")
        self.protocol, self.checks = protocol, frozen_run.complete_checks
        self.manifest = protocol.verify_run(self.out)
        if not read(self.out / "preflight.json").get("passed"):
            raise RuntimeError("Prepare the run and pass its frozen tests before training")
        self.capacity, self.stale_seconds, self.max_attempts = jobs, stale_seconds, max_attempts
        self.train_only = train_only
        self.active, self.completed, self.failed, self.pending = {}, [], {}, []
        self.record = self.out / "execution" / ("recovery_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
        self.record.mkdir(parents=True, exist_ok=False)
        for name in ("queue_supervisor.py", "spawn_train.py"):
            shutil.copyfile(Path(__file__).resolve().parent / name, self.record / name)
        write(self.record / "manifest.json", dict(created_utc=stamp(),
            supervisor_pid=os.getpid(), parallel_jobs=jobs, worker_start_method="spawn",
            stale_seconds=stale_seconds, max_attempts=max_attempts,
            frozen_scientific_inputs_unchanged=True,
            sources={name: hashlib.sha256((self.record/name).read_bytes()).hexdigest()
                     for name in ("queue_supervisor.py", "spawn_train.py")}))

    def event(self, kind, **extra):
        entry = dict(utc=stamp(), event=kind, **extra)
        with (self.record / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(json.dumps(entry, ensure_ascii=False), flush=True)

    def job_path(self, method):
        return self.out / "jobs" / method / "status.json"

    def prepare(self, snapshot):
        for method in self.manifest["methods"]:
            path = self.job_path(method)
            status = read(path) if path.exists() else {}
            pid = snapshot.get("active_pids", {}).get(method)
            if pid and (identity := process_info(pid)) and identity["state"] != "Z":
                if status.get("pid") != pid or status.get("method") != method:
                    raise RuntimeError(f"Live task identity mismatch: {method}")
                self.active[method] = dict(identity=identity, attempt=status["attempt"],
                    started_wall=time.time(), proc=None, log=None, adopted=True)
                self.event("adopted", method=method, pid=pid, steps=status.get("completed_steps"))
            elif status.get("state") == "complete":
                self.protocol.verify_checkpoint(status)
                self.completed.append(method)
            else:
                self.pending.append(method)

    def launch(self, method):
        self.protocol.verify_run(self.out)
        attempts = [int(p.name.rsplit("_", 1)[1])
                    for p in (self.out / "jobs" / method).glob("attempt_*")]
        attempts.extend(int(p.stem.rsplit("_", 1)[1])
                        for p in (self.out / "logs").glob(f"{method}_[0-9]*.log"))
        attempt = max(attempts, default=0) + 1
        if attempt > self.max_attempts:
            self.failed[method] = "attempt_limit_reached"
            return
        log_path = self.out / "logs" / f"{method}_{attempt:02d}.log"
        log = log_path.open("x")
        env = dict(os.environ, PYTHONHASHSEED="0", PYTHONFAULTHANDLER="1",
            OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
            TF_CPP_MIN_LOG_LEVEL="3", CUBLAS_WORKSPACE_CONFIG=":4096:8")
        command = [sys.executable, "-u", str(self.record / "spawn_train.py"),
                   "--output", str(self.out), "--method", method, "--attempt", str(attempt)]
        try:
            proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                cwd=self.out / "frozen", env=env, start_new_session=True)
        except BaseException:
            log.close()
            raise
        identity = process_info(proc.pid)
        if identity is None:
            log.close()
            raise RuntimeError("New process disappeared before registration")
        self.active[method] = dict(identity=identity, attempt=attempt,
            started_wall=time.time(), proc=proc, log=log, adopted=False)
        self.event("started", method=method, pid=proc.pid, attempt=attempt,
                   start_method="spawn", log=str(log_path))

    def release(self, method, error=None):
        job = self.active.pop(method)
        path = self.job_path(method)
        status = read(path) if path.exists() else dict(method=method, attempt=job["attempt"])
        if error:
            evidence = dict(error=error, status_before=status,
                process_group=group_members(job["identity"]["pgid"]), updated_utc=stamp())
            write(self.record / f"{method}_attempt_{job['attempt']:02d}_failure.json", evidence)
            if not job["adopted"] and error == "no_progress_timeout" and live(job["identity"]):
                os.kill(job["identity"]["pid"], signal.SIGUSR1)
                time.sleep(.2)
            stop_group(job["identity"])
            if status.get("attempt") != job["attempt"]:
                status = dict(method=method, attempt=job["attempt"], completed_steps=0)
            status.update(state="failed", error=error, updated_utc=stamp())
            write(path, status)
            attempt_dir = path.parent / f"attempt_{job['attempt']:02d}"
            if attempt_dir.exists():
                write(attempt_dir / "supervisor_failure.json", evidence)
            if job["attempt"] < self.max_attempts:
                self.pending.append(method)
            else:
                self.failed[method] = error
            self.event("isolated_failure", method=method, error=error,
                       retry_queued=job["attempt"] < self.max_attempts)
        else:
            self.protocol.verify_checkpoint(status)
            self.completed.append(method)
            self.event("completed", method=method, steps=status.get("completed_steps"))
        if job["proc"] is not None:
            job["proc"].wait(timeout=10)
        if job["log"] is not None:
            job["log"].close()

    def report(self, state="training"):
        progress = {}
        for method in self.manifest["methods"]:
            path = self.job_path(method)
            status = read(path) if path.exists() else {}
            current = status.get("state", "queued")
            if method in self.pending:
                current = "retry_queued" if status.get("state") == "failed" else "queued"
            progress[method] = dict(state=current,
                completed_steps=status.get("completed_steps", 0),
                target_steps=self.manifest["steps_per_method"])
        write(self.out / "status.json", dict(state=state, active_methods=list(self.active),
            active_pids={m:j["identity"]["pid"] for m,j in self.active.items()},
            completed_jobs=len(self.completed), total_jobs=len(self.manifest["methods"]),
            parallel_jobs=self.capacity, supervisor_pid=os.getpid(), progress=progress,
            failed_jobs=self.failed, pending_methods=self.pending,
            worker_health_checked=True, recovery_record=str(self.record), updated_utc=stamp()))

    def run(self):
        while self.active or self.pending:
            for method, job in list(self.active.items()):
                if job["proc"] is not None:
                    job["proc"].poll()  # Reap exited children before /proc checks.
                path = self.job_path(method)
                status = read(path) if path.exists() else {}
                if not live(job["identity"]) and status.get("state") == "complete":
                    self.release(method)
                    continue
                reason = health(job, path, self.stale_seconds)
                if reason:
                    self.release(method, reason)
            while self.pending and len(self.active) < self.capacity:
                self.launch(self.pending.pop(0))
            self.report()
            time.sleep(2)
        verification = self.checks(self.out)
        if self.failed:
            self.report("attention_required")
            return
        if verification["verified_completed_jobs"] != len(self.manifest["methods"]):
            raise RuntimeError("Completion count does not match the frozen method list")
        if self.train_only:
            write(self.out / "verification.json", self.checks(self.out))
            self.report("training_complete")
            return
        if not (self.out / "evaluation/status.json").exists():
            self.report("evaluating")
            attempt = len(list((self.out / "evaluation").glob("attempt_*"))) + 1
            log_path = self.out / "logs" / f"evaluation_{attempt:02d}.log"
            with log_path.open("x") as log:
                proc = subprocess.Popen([sys.executable, "-u", str(self.out / "frozen/evaluate.py"),
                    "--output", str(self.out), "--attempt", str(attempt)],
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                while proc.poll() is None:
                    self.report("evaluating")
                    time.sleep(5)
                if proc.returncode:
                    raise RuntimeError(f"Evaluation failed; see {log_path}")
        verification = self.checks(self.out)
        write(self.out / "verification.json", verification)
        self.report("complete")
        self.event("comparison_complete", evaluation=verification.get("evaluation"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--takeover-pid", type=int)
    parser.add_argument("--stale-seconds", type=float, default=180)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--train-only", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.jobs <= 7 or args.stale_seconds < 30 or args.max_attempts < 1:
        parser.error("Invalid capacity, timeout, or retry limit")
    out = args.output.resolve()
    with (out / ".recovery.lock").open("a+") as recovery_lock:
        fcntl.flock(recovery_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        supervisor = Supervisor(out, args.jobs, args.stale_seconds, args.max_attempts, args.train_only)
        snapshot = read(out / "status.json")
        write(supervisor.record / "status_before.json", snapshot)
        with (out / ".lock").open("a+") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                pid = args.takeover_pid
                if pid is None or snapshot.get("supervisor_pid") != pid:
                    raise RuntimeError("Run is already managed; an exact takeover PID is required")
                identity = process_info(pid)
                command = (Path("/proc") / str(pid) / "cmdline").read_bytes().decode(errors="replace")
                if "run.py" not in command or out.name not in command:
                    raise RuntimeError("PID does not identify this frozen run's old coordinator")
                children = []
                for method, child_pid in snapshot.get("active_pids", {}).items():
                    state = read(supervisor.job_path(method))
                    if state.get("pid") != child_pid or state.get("method") != method:
                        raise RuntimeError("Child status identity mismatch")
                    child = process_info(child_pid)
                    if child:
                        children.append(child)
                write(supervisor.record / "takeover_identities.json", dict(parent=identity, children=children))
                replace_supervisor(identity, children)
                deadline = time.monotonic() + 10
                while True:
                    try:
                        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        break
                    except BlockingIOError:
                        if time.monotonic() > deadline:
                            raise RuntimeError("Old lock did not release; children left intact")
                        time.sleep(.1)
                supervisor.event("coordinator_replaced", old_pid=pid,
                    retained_child_pids=[p["pid"] for p in children])
            supervisor.prepare(snapshot)
            try:
                supervisor.run()
            except BaseException as exc:
                # A coordinator failure must never destroy healthy experiments.
                supervisor.event("supervision_failed_children_preserved", error=repr(exc))
                supervisor.report("supervision_failed")
                raise


if __name__ == "__main__":
    main()
