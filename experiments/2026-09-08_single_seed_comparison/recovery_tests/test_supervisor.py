"""Fault-injection tests, independent of the immutable scientific preflight."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from types import SimpleNamespace

SOURCE = Path(__file__).resolve().parents[1] / "queue_supervisor.py"
spec = importlib.util.spec_from_file_location("queue_recovery", SOURCE)
q = importlib.util.module_from_spec(spec)
spec.loader.exec_module(q)


class SupervisorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.children = []

    def tearDown(self):
        for proc, identity in reversed(self.children):
            q.stop_group(identity, grace=.1)
            proc.wait(timeout=5)
        self.temp.cleanup()

    def process(self, code, *args):
        proc = subprocess.Popen([sys.executable, "-c", code, *map(str, args)],
                                start_new_session=True)
        identity = q.process_info(proc.pid)
        self.children.append((proc, identity))
        return proc, identity

    def wait_for(self, condition):
        deadline = time.monotonic() + 5
        while not condition():
            if time.monotonic() > deadline:
                self.fail("Timed out waiting for test process")
            time.sleep(.02)

    def test_dead_worker_detected_and_isolated_from_healthy_job(self):
        failed, identity = self.process("import os,time; p=os.fork(); os._exit(23) if p==0 else time.sleep(60)")
        healthy, sibling = self.process("import time; time.sleep(60)")
        self.wait_for(lambda: any(p["state"] == "Z" for p in q.group_members(failed.pid)))
        job = dict(identity=identity, attempt=1, started_wall=time.time())
        reason = q.health(job, self.root / "missing.json")
        self.assertTrue(reason.startswith("dead_worker:"), reason)
        self.assertIsNone(failed.poll())  # A parent-only poll misses this failure.
        q.stop_group(identity, grace=.1)
        self.assertTrue(q.live(sibling))
        failed.wait(timeout=5)

    def test_progress_timeout_and_successful_cleanup(self):
        proc, identity = self.process("import time; time.sleep(60)")
        path = self.root / "status.json"
        job = dict(identity=identity, attempt=1, started_wall=time.time())
        q.write(path, dict(attempt=1, state="training"))
        os.utime(path, (time.time()-200, time.time()-200))
        self.assertEqual(q.health(job, path), "no_progress_timeout")
        q.write(path, dict(attempt=1, state="complete"))
        self.assertIsNone(q.health(job, path))
        wrong_identity = dict(identity, start_ticks=identity["start_ticks"]+1)
        self.assertFalse(q.live(wrong_identity))

    def test_takeover_retains_independent_child(self):
        pid_path = self.root / "child.pid"
        code = ("import subprocess,sys,time; from pathlib import Path; "
                "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'],start_new_session=True); "
                "Path(sys.argv[1]).write_text(str(p.pid)); time.sleep(60)")
        old, identity = self.process(code, pid_path)
        self.wait_for(pid_path.exists)
        child = q.process_info(int(pid_path.read_text()))
        try:
            q.replace_supervisor(identity, [child])
            old.wait(timeout=5)
            self.assertTrue(q.live(child))
            self.assertEqual(q.process_info(child["pid"])["start_ticks"], child["start_ticks"])
        finally:
            q.stop_group(child, grace=.1)

    def test_failed_method_does_not_block_queued_method(self):
        for folder in ("frozen", "logs", "record"):
            (self.root / folder).mkdir()
        # Real isolated processes, including a parent stuck with a dead worker.
        (self.root / "record/spawn_train.py").write_text('''
import argparse, json, os, time
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--output'); p.add_argument('--method'); p.add_argument('--attempt',type=int)
a=p.parse_args(); out=Path(a.output); work=out/'jobs'/a.method/f'attempt_{a.attempt:02d}'
work.mkdir(parents=True)
def status(state):
    path=work.parent/'status.json'; temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(dict(method=a.method,attempt=a.attempt,state=state,pid=os.getpid(),completed_steps=16)))
    temp.replace(path)
status('training')
if a.method=='broken':
    child=os.fork()
    if child==0: os._exit(23)
    time.sleep(60)
else:
    time.sleep(1)
    status('complete')
''')
        supervisor = q.Supervisor.__new__(q.Supervisor)
        supervisor.out, supervisor.record = self.root, self.root / "record"
        supervisor.manifest = dict(methods=["broken", "healthy", "queued"], steps_per_method=16)
        supervisor.protocol = SimpleNamespace(verify_run=lambda out: None,
            verify_checkpoint=lambda status: self.assertEqual(status["state"], "complete"))
        supervisor.checks = lambda out: dict(verified_completed_jobs=len(supervisor.completed))
        supervisor.capacity, supervisor.stale_seconds, supervisor.max_attempts = 2, 180, 1
        supervisor.train_only = True
        supervisor.active, supervisor.completed, supervisor.failed = {}, [], {}
        supervisor.pending = supervisor.manifest["methods"].copy()
        try:
            supervisor.run()
            self.assertEqual(set(supervisor.completed), {"healthy", "queued"})
            self.assertEqual(set(supervisor.failed), {"broken"})
            self.assertEqual(q.read(self.root / "status.json")["state"], "attention_required")
            events = [json.loads(line) for line in (supervisor.record / "events.jsonl").read_text().splitlines()]
            self.assertTrue(any(e["event"] == "started" and e["method"] == "queued" for e in events))
        finally:
            for job in supervisor.active.values():
                q.stop_group(job["identity"], grace=.1)
                job["proc"].wait(timeout=5)
                job["log"].close()


if __name__ == "__main__":
    unittest.main()
