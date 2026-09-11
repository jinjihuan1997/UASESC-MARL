"""Verify a simulated thermal event pauses only the launched test process tree."""
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import queue_supervisor as q
import stability_check as stability


class StabilityGuardTests(unittest.TestCase):
    def test_thermal_event_preserves_launched_process_tree(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            code = ("import subprocess,sys,time; from pathlib import Path; "
                    "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); "
                    "Path(sys.argv[1]).write_text(str(p.pid)); time.sleep(60)")
            spawned = []
            original = subprocess.Popen
            def launch(*args, **kwargs):
                proc = original(*args, **kwargs)
                spawned.append(proc)
                deadline = time.monotonic()+5
                while not (root/'child.pid').exists():
                    if time.monotonic() > deadline:
                        raise TimeoutError('Test child did not start')
                    time.sleep(.01)
                return proc
            options = SimpleNamespace(max_cpu_c=85, hot_seconds=5, max_seconds=30)
            try:
                with patch.object(stability.subprocess, 'Popen', side_effect=launch), \
                     patch.object(stability, 'cpu_temperature', side_effect=[45, 96]):
                    with self.assertRaisesRegex(RuntimeError, 'Thermal guard triggered'):
                        stability.monitor([sys.executable, '-c', code, str(root/'child.pid')],
                                          root, 'test', options)
                journal = q.read(root/'test_pause.json')
                expected = {spawned[0].pid, int((root/'child.pid').read_text())}
                self.assertEqual({p['pid'] for p in journal['processes']}, expected)
                self.assertTrue(all(q.process_info(p['pid'])['state'] == 'T'
                                    for p in journal['processes']))
                self.assertEqual(q.read(root/'status.json')['state'], 'stopped_for_review')
            finally:
                for proc in spawned:
                    identity = q.process_info(proc.pid)
                    if identity:
                        q.stop_group(identity, grace=.1)
                    proc.wait(timeout=5)


if __name__ == '__main__':
    unittest.main()
