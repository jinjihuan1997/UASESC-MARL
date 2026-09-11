"""Thermal control must keep IPC clocks and healthy jobs running."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import guarded_queue as guarded


class GuardedQueueTests(unittest.TestCase):
    def test_adoption_refuses_overcapacity_or_paused_jobs(self):
        queue = guarded.GuardedSupervisor.__new__(guarded.GuardedSupervisor)
        queue.capacity = 4
        with patch.object(guarded.optimized.OptimizedSupervisor, 'prepare') as parent, \
             patch.object(guarded.resource_control, 'pause') as pause:
            for snapshot in ({'active_pids': {str(i): i for i in range(5)}},
                             {'suspended_jobs': {'method': {'pid': 1}}}):
                with self.assertRaisesRegex(RuntimeError, 'queued jobs must be unstarted'):
                    queue.prepare(snapshot)
            parent.assert_not_called()
            pause.assert_not_called()
            queue.prepare({'active_pids': {'healthy': 2}})
            parent.assert_called_once()

    def test_cooling_reduces_cpu_without_stopping_workers_or_hiding_failures(self):
        with tempfile.TemporaryDirectory() as folder:
            queue = guarded.GuardedSupervisor.__new__(guarded.GuardedSupervisor)
            queue.record = Path(folder)
            queue.cpus = list(range(12))
            queue.fallback_cpus = [0, 2, 8, 10]
            queue.capacity = 6
            queue.active = {'example': {'identity': {'pid': 12345}}}
            queue.derated = False
            queue.hot_since = 0
            queue.cool_since = None
            queue.cooling = None
            queue.stale_seconds = 180
            events = []
            queue.event = lambda event, **extra: events.append((event, extra))
            with patch.object(guarded.q, 'live', return_value=True), \
                 patch.object(guarded.os, 'sched_setaffinity') as own_affinity, \
                 patch.object(guarded.optimized, 'set_group_affinity') as groups, \
                 patch.object(guarded.resource_control, 'pause') as pause, \
                 patch.object(guarded.resource_control, 'resume') as resume:
                with patch.object(guarded, 'cpu_temperature', return_value=86), \
                     patch.object(guarded.time, 'monotonic', return_value=10):
                    self.assertTrue(queue.cycle_ready())
                    self.assertEqual(queue.capacity, 2)
                    self.assertEqual(queue.cpus, [0, 2, 8, 10])
                with patch.object(guarded, 'cpu_temperature', return_value=96), \
                     patch.object(guarded.time, 'monotonic', return_value=12):
                    self.assertTrue(queue.cycle_ready())
                    self.assertEqual(queue.cpus, [8])
                    self.assertEqual(queue.capacity, 1)
                with patch.object(guarded, 'cpu_temperature', return_value=65), \
                     patch.object(guarded.time, 'monotonic', side_effect=[100, 116]):
                    self.assertTrue(queue.cycle_ready())
                    self.assertEqual(queue.cpus, [8])
                    self.assertTrue(queue.cycle_ready())
                    self.assertEqual(queue.cpus, [0, 2, 8, 10])
                    self.assertEqual(queue.capacity, 2)
                    self.assertIsNone(queue.cooling)
                pause.assert_not_called()
                resume.assert_not_called()
                self.assertEqual(set(queue.active), {'example'})
                self.assertEqual(own_affinity.call_count, 3)
                self.assertEqual(groups.call_count, 3)
            for reason in ('no_progress_timeout', 'dead_worker'):
                with patch.object(guarded.q, 'health', return_value=reason):
                    self.assertEqual(queue.job_health({}, 'unused'), reason)
            self.assertEqual([event for event, _ in events],
                             ['thermal_derated', 'thermal_cpu_cooling', 'thermal_resources_restored'])


if __name__ == '__main__':
    unittest.main()
