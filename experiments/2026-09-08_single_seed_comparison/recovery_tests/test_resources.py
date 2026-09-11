"""Resource control must preserve healthy processes and bound new admission."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
from types import SimpleNamespace

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import queue_supervisor as q
import resource_control as control
import optimized_queue as optimized


class ResourceTests(unittest.TestCase):
    def test_takeover_validates_run_and_preserves_previously_adopted_job(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            script = root/'queue_supervisor.py'
            script.write_text('import time; time.sleep(60)')
            manager = subprocess.Popen([sys.executable,str(script),'--output',str(root)],start_new_session=True)
            job = subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'],start_new_session=True)
            manager_id, job_id = q.process_info(manager.pid), q.process_info(job.pid)
            try:
                path = root/'job.json'
                q.write(path,dict(method='test',pid=job.pid))
                snapshot = dict(supervisor_pid=manager.pid,active_pids={'test':job.pid})
                supervisor = SimpleNamespace(out=root/'wrong', record=root, job_path=lambda method:path,
                                             event=lambda *args,**kwargs:None)
                with self.assertRaisesRegex(RuntimeError,'another run'):
                    optimized.retire_coordinator(supervisor,snapshot,manager.pid)
                self.assertTrue(q.live(manager_id))
                supervisor.out = root
                optimized.retire_coordinator(supervisor,snapshot,manager.pid)
                manager.wait(timeout=5)
                self.assertTrue(q.live(job_id))
            finally:
                q.stop_group(manager_id,grace=.1)
                q.stop_group(job_id,grace=.1)
                manager.wait(timeout=5)
                job.wait(timeout=5)

    def test_pause_tree_resume_and_affinity_preserve_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            code = ("import subprocess,sys,time; from pathlib import Path; "
                "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); "
                "Path(sys.argv[1]).write_text(str(p.pid)); time.sleep(60)")
            proc = subprocess.Popen([sys.executable,'-c',code,str(root/'pid')],start_new_session=True)
            identity = q.process_info(proc.pid)
            try:
                deadline = time.monotonic()+5
                while not (root/'pid').exists():
                    self.assertLess(time.monotonic(),deadline)
                    time.sleep(.02)
                child = int((root/'pid').read_text())
                selected = {min(os.sched_getaffinity(0))}
                optimized.set_group_affinity(identity,selected)
                self.assertEqual(os.sched_getaffinity(child),selected)
                journal = root/'pause.json'
                record = control.pause([proc.pid],journal)
                self.assertEqual({p['pid'] for p in record['processes']},{proc.pid,child})
                self.assertTrue(all(q.process_info(p['pid'])['state']=='T' for p in record['processes']))
                control.resume(journal,defer_pids=[proc.pid])
                time.sleep(.05)
                self.assertEqual(q.process_info(proc.pid)['state'],'T')
                self.assertNotEqual(q.process_info(child)['state'],'T')
                control.resume(journal)
                time.sleep(.05)
                self.assertTrue(q.live(identity))
                self.assertNotEqual(q.process_info(proc.pid)['state'],'T')
            finally:
                q.stop_group(identity,grace=.1)
                proc.wait(timeout=5)

    def test_headroom_blocks_new_jobs_and_gpu_probe_failure_is_nonfatal(self):
        supervisor = optimized.OptimizedSupervisor.__new__(optimized.OptimizedSupervisor)
        supervisor.last_admission = None
        supervisor.last_admission_time = 0
        with patch.object(optimized,'resources',return_value=dict(memory_available_mib=6000,gpu_memory_free_mib=8000)):
            self.assertFalse(supervisor.can_launch())
        supervisor.last_admission = None
        with patch.object(optimized,'resources',return_value=dict(memory_available_mib=10000,gpu_memory_free_mib=1700)):
            self.assertFalse(supervisor.can_launch())
        supervisor.last_admission = None
        with patch.object(optimized,'resources',return_value=dict(memory_available_mib=10000,gpu_memory_free_mib=8000)):
            self.assertTrue(supervisor.can_launch())
        with patch.object(optimized.subprocess,'check_output',side_effect=subprocess.TimeoutExpired('nvidia-smi',5)):
            sample = optimized.resources()
        self.assertIsNone(sample['gpu_memory_free_mib'])
        self.assertIn('gpu_query_error',sample)


if __name__=='__main__':
    unittest.main()
