#!/usr/bin/env python3
"""One command: freeze -> tests -> seven training jobs -> eleven-policy comparison."""
from pathlib import Path
import argparse
import datetime
import fcntl
import os
import shutil
import signal
import subprocess
import sys
import time
from protocol import (VERSION, TABLE_SHA, METHODS, RULES, SCENARIOS, configurations,
                      read, sha, stamp, verify_checkpoint, verify_run, write)


def start_child(command, log_path):
    env = dict(os.environ, PYTHONHASHSEED="0", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", TF_CPP_MIN_LOG_LEVEL="3", CUBLAS_WORKSPACE_CONFIG=":4096:8")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("x")
    try:
        proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env,
                                cwd=VERSION, start_new_session=True)
    except BaseException:
        log.close()
        raise
    return dict(proc=proc, log=log, log_path=log_path)


def stop_child(job):
    proc = job["proc"]
    if proc.poll() is None:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
    job["log"].close()


def finish_child(job):
    code = job["proc"].wait()
    job["log"].close()
    if code:
        log_path = job["log_path"]
        tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-35:])
        raise RuntimeError(f"Subprocess exited {code}. Log: {log_path}\n{tail}")


def child(command, log_path):
    """Wait for a managed process group; preserve logs and stop only its children."""
    job = start_child(command, log_path)
    try:
        finish_child(job)
    except BaseException:
        stop_child(job)
        raise


def run_training_queue(out, manifest, parallel_jobs):
    pending, completed, active = [], [], {}
    for method in manifest["methods"]:
        path = out/"jobs"/method/"status.json"
        if path.exists() and read(path)["state"] == "complete":
            verify_checkpoint(read(path))
            completed.append(method)
            print(f"Verified, skipping {method}", flush=True)
        else:
            pending.append(method)
    launch = dict(started_utc=stamp(), supervisor_pid=os.getpid(), parallel_jobs=parallel_jobs,
                  device=manifest["device"], methods=manifest["methods"])
    index = len(list((out/"execution").glob("launch_*.json")))+1
    write(out/"execution"/f"launch_{index:02d}.json", launch)
    last_report = 0.0
    try:
        while pending or active:
            while pending and len(active) < parallel_jobs:
                verify_run(out)
                method = pending.pop(0)
                attempt = len(list((out/"jobs"/method).glob("attempt_*")))+1
                path = out/"logs"/f"{method}_{attempt:02d}.log"
                active[method] = start_child([sys.executable, "-u", str(out/"frozen/train.py"),
                    "--output", str(out), "--method", method, "--attempt", str(attempt)], path)
                print(f"Started {method}; PID {active[method]['proc'].pid}; {len(active)}/{parallel_jobs} slots; log: {path}", flush=True)
                last_report = 0.0
            for method in list(active):
                if active[method]["proc"].poll() is not None:
                    finish_child(active[method])
                    verify_checkpoint(read(out/"jobs"/method/"status.json"))
                    del active[method]
                    completed.append(method)
                    print(f"Completed {method}; {len(completed)}/{len(manifest['methods'])} methods", flush=True)
                    last_report = 0.0
            if time.monotonic()-last_report >= 5:
                progress = {}
                for method in manifest["methods"]:
                    path = out/"jobs"/method/"status.json"
                    state = read(path) if path.exists() else {}
                    progress[method] = dict(state=state.get("state", "queued"),
                        completed_steps=state.get("completed_steps", 0), target_steps=manifest["steps_per_method"])
                write(out/"status.json", dict(state="training", active_methods=list(active),
                    active_pids={m:j["proc"].pid for m,j in active.items()},
                    completed_jobs=len(completed), total_jobs=len(manifest["methods"]),
                    parallel_jobs=parallel_jobs, supervisor_pid=os.getpid(), progress=progress,
                    updated_utc=stamp()))
                last_report = time.monotonic()
            if active:
                time.sleep(.5)
    except BaseException:
        for job in active.values():
            stop_child(job)
        raise


def freeze(args, out):
    if sha(VERSION/"inputs/profile.npz") != TABLE_SHA:
        raise RuntimeError("The final average table changed")
    # Validate budgets before creating an output directory.
    configs = configurations(args.seed, args.steps, args.threads, args.device)
    if args.methods:
        configs = {m:configs[m] for m in args.methods}
    out.mkdir(parents=True, exist_ok=False)
    frozen = out / "frozen"
    frozen.mkdir()
    for name in ("runtime", "config", "inputs", "tests"):
        shutil.copytree(VERSION/name, frozen/name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name in ("run.py", "train.py", "evaluate.py", "protocol.py", "README.md", "source_provenance.json"):
        shutil.copyfile(VERSION/name, frozen/name)
    for name, config in configs.items():
        config["env_args"].update(semantic_registry_path=str(frozen/"inputs/mode_registry.json"),
                                   semantic_profile_path=str(frozen/"inputs/profile.npz"))
        config["algo_args"]["logger"]["log_dir"] = str(out/"jobs"/name)
        write(out/"configs"/f"{name}.json", config)
    import importlib.metadata
    versions = {}
    for name in ("numpy", "torch", "gymnasium", "gym", "scipy", "tensorboard", "tensorboardX", "PyYAML"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    inputs = [p for folder in (frozen, out/"configs") for p in folder.rglob("*") if p.is_file()]
    manifest = dict(version="single_seed_comparison_v4_resilient_workers", created_utc=stamp(), seed=args.seed,
                    steps_per_method=args.steps, total_training_steps=len(configs)*args.steps,
                    rollout_threads=args.threads, rollout_length=400,
                    updates_per_method=args.steps//(400*args.threads),
                    methods=list(configs), rules=RULES, parallel_jobs=args.jobs or 4,
                    evaluation_seeds=list(range(20261201, 20261201+args.eval_episodes)),
                    evaluation_role="development, not the reserved final confirmation set",
                    scenarios=SCENARIOS, device=args.device, smoke=args.smoke,
                    engineering_changes=["batched equivalent simplex log-density and entropy",
                        "registered GPU ValueNorm statistics", "optional CPU inference and GPU optimization",
                        "independent methods run in concurrent subprocesses"],
                    table_sha256=TABLE_SHA, mode_interface="all 16 SCI/rate rows",
                    source_version_dir=str(VERSION), python=sys.executable,
                    python_version=sys.version, dependencies=versions,
                    checkpoint_selection="final fixed-budget checkpoint; no evaluation-based selection",
                    resume_policy="Skip verified completed jobs; restart interrupted jobs from scratch in a new attempt, preserving old artifacts",
                    ablation_boundary="Hidden instruction removes explicit actor task ID and targets; critic, quality masks and task-derived pending loads still carry task information",
                    common_reward="base_reward - A_penalty[g]*max(0,(max_A-A_limit[g])/A_limit[g]) + 0.1*mean_DS(max(A_pre-A_post,0))/A_limit[g]",
                    input_hashes={str(p.relative_to(out)): sha(p) for p in sorted(inputs)})
    write(out/"manifest.json", manifest)
    write(out/"status.json", dict(state="prepared", updated_utc=stamp()))
    return manifest


def complete_checks(out, deep=False):
    manifest = verify_run(out)
    complete = []
    for method in manifest["methods"]:
        path = out/"jobs"/method/"status.json"
        if path.exists() and read(path)["state"] == "complete":
            verify_checkpoint(read(path))
            complete.append(method)
    pairs = [("IC_HAPPO", "HAPPO_hidden_instruction"), ("IC_MAPPO", "MAPPO_hidden_instruction")]
    for a, b in pairs:
        if a in complete and b in complete:
            x, y = read(out/"jobs"/a/"status.json"), read(out/"jobs"/b/"status.json")
            if (x["initial_actor_hashes"] != y["initial_actor_hashes"] or
                x["initial_critic_hash"] != y["initial_critic_hash"]):
                raise AssertionError(f"Unpaired initial parameters: {a}, {b}")
    report = dict(verified_inputs=True, verified_completed_jobs=len(complete),
                  paired_initial_parameters_checked=all(a in complete and b in complete for a, b in pairs))
    eval_status = out/"evaluation/status.json"
    if eval_status.exists():
        status = read(eval_status)
        directory = Path(status["evaluation_dir"])
        for name, expected in status["summary_hashes"].items():
            if sha(directory/name) != expected:
                raise RuntimeError(f"Evaluation summary changed: {name}")
        if deep:
            for row in read(directory/"episode_summary.json"):
                if sha(row["trace_file"]) != row["trace_sha256"]:
                    raise RuntimeError(f"Evaluation trace changed: {row['trace_file']}")
        report.update(evaluation=status)
    return report


def execute(args):
    if args.output is None:
        if args.action in ("verify", "evaluate") or args.resume:
            raise ValueError("--output is required for an existing run")
        tag = "smoke" if args.smoke else "train"
        args.output = VERSION/"runs"/f"{tag}_seed{args.seed}_{datetime.datetime.now():%Y%m%d_%H%M%S}"
    out = args.output.expanduser().resolve()
    if args.action == "verify":
        print(complete_checks(out, deep=True))
        return
    if out.exists():
        if not args.resume and args.action != "evaluate":
            raise FileExistsError(f"Preserving existing run {out}; use --resume or a new --output")
        manifest = verify_run(out)
    else:
        if args.resume or args.action == "evaluate":
            raise FileNotFoundError(out)
        manifest = freeze(args, out)
    parallel_jobs = args.jobs or manifest.get("parallel_jobs", 1)
    if manifest["methods"] != [m[0] for m in METHODS] and not args.train_only and args.action != "prepare":
        raise ValueError("A performance-probe subset requires --train-only; full comparison requires all seven methods")
    print(f"Run: {out}\nSeed {manifest['seed']}; {len(manifest['methods'])} jobs × {manifest['steps_per_method']:,} steps; device={manifest['device']}; concurrent jobs={parallel_jobs}.", flush=True)
    print("Resumed runs keep frozen training parameters; --jobs only changes execution concurrency.", flush=True)
    with (out/".lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            verify_run(out)
            preflight = out/"preflight.json"
            if not preflight.exists() or not read(preflight)["passed"]:
                attempt = len(list((out/"logs").glob("preflight_*.log")))+1
                child([sys.executable, "-m", "unittest", "discover", "-s", str(out/"frozen/tests"), "-v"],
                      out/"logs"/f"preflight_{attempt:02d}.log")
                write(preflight, dict(passed=True, updated_utc=stamp()))
                print("Environment and protocol tests passed.", flush=True)
            if args.action == "prepare":
                print(f"Prepared and tested. Start with: {sys.executable} {out/'frozen/run.py'} run --output {out} --resume", flush=True)
                return
            if args.action == "run":
                run_training_queue(out, manifest, parallel_jobs)
                complete_checks(out)
            if args.train_only:
                write(out/"verification.json", complete_checks(out))
                write(out/"status.json", dict(state="training_complete", completed_jobs=len(manifest["methods"]),
                    note="Training-only run; evaluation not requested", updated_utc=stamp()))
                print("All requested training jobs completed and verified.", flush=True)
                return
            for method in manifest["methods"]:
                verify_checkpoint(read(out/"jobs"/method/"status.json"))
            if not (out/"evaluation/status.json").exists():
                attempt = len(list((out/"evaluation").glob("attempt_*")))+1
                write(out/"status.json", dict(state="evaluating", updated_utc=stamp()))
                print(f"Evaluating 11 methods in 13 paired scenarios; progress: {out/'evaluation'/f'attempt_{attempt:02d}'/'progress.json'}", flush=True)
                child([sys.executable, "-u", str(out/"frozen/evaluate.py"), "--output", str(out),
                       "--attempt", str(attempt)], out/"logs"/f"evaluation_{attempt:02d}.log")
            verification = complete_checks(out)
            write(out/"verification.json", verification)
            comparison = Path(verification["evaluation"]["evaluation_dir"])/"comparison.md"
            write(out/"status.json", dict(state="complete", completed_jobs=7,
                  comparison=str(comparison), updated_utc=stamp()))
            print(f"Complete: {comparison}", flush=True)
        except BaseException as exc:
            write(out/"status.json", dict(state="failed", error=repr(exc), updated_utc=stamp()))
            raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", nargs="?", default="run", choices=("run", "prepare", "evaluate", "verify"))
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--steps", type=int, default=10_000_000, help="Environment steps per learned method")
    p.add_argument("--threads", type=int, default=10)
    p.add_argument("--eval-episodes", type=int, default=20, help="Paired 600-slot episodes per scenario and method")
    p.add_argument("--device", choices=("cpu", "cuda", "hybrid"), default="hybrid")
    p.add_argument("--jobs", type=int, help="Concurrent independent methods; default 4 for a new run")
    p.add_argument("--train-only", action="store_true", help="Train and verify weights; skip evaluation")
    p.add_argument("--methods", nargs="+", choices=[m[0] for m in METHODS], help="Performance probe subset; requires --train-only")
    p.add_argument("--output", type=Path)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--smoke", action="store_true", help="All methods; 8,000 steps/job and one evaluation seed")
    args = p.parse_args()
    if args.seed < 0 or args.eval_episodes < 1:
        p.error("seed must be nonnegative; eval-episodes must be positive")
    if args.jobs is not None and not 1 <= args.jobs <= len(METHODS):
        p.error("jobs must be between 1 and 7")
    if args.methods and (len(set(args.methods)) != len(args.methods) or not args.train_only):
        p.error("--methods must be unique and requires --train-only")
    if args.smoke:
        args.steps, args.eval_episodes = 8000, 1
    execute(args)


if __name__ == "__main__":
    main()
