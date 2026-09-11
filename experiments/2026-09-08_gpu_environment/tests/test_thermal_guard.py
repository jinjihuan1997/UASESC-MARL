"""Thermal hysteresis and a real child-process pause/cool/resume cycle."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from thermal_guard import Guard, ThermalPolicy, identity, read, same_live, write


FAKE_SUPERVISOR = '''
import json, os, signal, sys, time
from pathlib import Path
run=Path(sys.argv[sys.argv.index('--output')+1])
path=run/'status.json'
old=json.loads(path.read_text()) if path.exists() else {}
counter=old.get('counter',0) if '--resume' in sys.argv else 123
def save(state):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(dict(state=state,supervisor_pid=os.getpid(),counter=counter,resumed='--resume' in sys.argv)))
    temporary.replace(path)
def stop(signum,frame):
    save('paused')
    raise SystemExit(0)
signal.signal(signal.SIGTERM,stop)
save('training')
while True:time.sleep(.01)
'''


class ThermalGuardTests(unittest.TestCase):
    def fixture(self, parent):
        run = Path(parent)/'run'
        (run/'source').mkdir(parents=True)
        write(run/'manifest.json', dict(jobs=[], total_training_steps=100,
              resource_lock=str(run/'resource.lock'), resources={}))
        return run

    def test_hot_spikes_and_emergency(self):
        p = ThermalPolicy()
        self.assertIsNone(p.stop_reason(85, 0))
        self.assertIsNone(p.stop_reason(84, 4))
        self.assertIsNone(p.stop_reason(85, 5))
        self.assertIsNone(p.stop_reason(86, 9.9))
        self.assertEqual(p.stop_reason(86, 10), 'cpu_sustained_high_temperature')
        self.assertEqual(ThermalPolicy().stop_reason(95, 0), 'cpu_emergency_temperature')

    def test_cooling_requires_minimum_rest_and_continuous_cool(self):
        p = ThermalPolicy()
        p.begin_cooling(0)
        self.assertFalse(p.can_resume(70, 0))
        self.assertFalse(p.can_resume(70, 179))
        self.assertFalse(p.can_resume(75, 180))
        self.assertFalse(p.can_resume(74, 181))
        self.assertFalse(p.can_resume(74, 240))
        self.assertTrue(p.can_resume(74, 241))

    def test_missing_sensor_cannot_restart(self):
        p = ThermalPolicy()
        self.assertEqual(p.stop_reason(None, 0), 'temperature_sensor_unavailable')
        p.begin_cooling(0)
        self.assertFalse(p.can_resume(65, 0))
        self.assertFalse(p.can_resume(None, 181))
        self.assertFalse(p.can_resume(float('nan'), 182))
        self.assertFalse(p.can_resume(65, 183))
        self.assertTrue(p.can_resume(65, 243))

    def test_pid_reuse_never_signals_new_process(self):
        old = dict(pid=1234, uid=1000, start_ticks=10)
        with patch('thermal_guard.identity', return_value=dict(pid=1234, uid=1000, state='S', start_ticks=11)):
            self.assertFalse(same_live(old))
            with tempfile.TemporaryDirectory() as tmp:
                guard = Guard(self.fixture(tmp))
                try:
                    with patch('thermal_guard.os.kill') as kill:
                        guard.signal_term(old)
                        kill.assert_not_called()
                finally:
                    guard.close()

    def test_single_guard_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = self.fixture(tmp)
            guard = Guard(run)
            try:
                with self.assertRaises(BlockingIOError):
                    Guard(run)
            finally:
                guard.close()

    def test_manual_pause_and_unrelated_failure_do_not_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = self.fixture(tmp)
            guard = Guard(run)
            try:
                write(run/'status.json', dict(state='paused'))
                self.assertEqual(guard.tick(60, 0), 'manual_pause_preserved')
                write(run/'status.json', dict(state='attention_required', error='FloatingPointError'))
                with self.assertRaisesRegex(RuntimeError, 'non-thermal'):
                    guard.tick(60, 181)
            finally:
                guard.close()

    def test_checkpoint_failure_latch_overrides_old_thermal_exit(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = self.fixture(tmp)
            guard = Guard(run)
            try:
                write(run/'status.json', dict(state='attention_required', error='95C emergency'))
                guard.set_control(mode='attention_required', error='Checkpoint hash mismatch')
                with self.assertRaisesRegex(RuntimeError, 'latched'):
                    guard.tick(60, 0)
                self.assertIsNone(guard.policy.paused_since)
            finally:
                guard.close()

    def test_real_process_thermal_pause_and_automatic_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = self.fixture(tmp)
            script = run/'source/three_seed_train.py'
            script.write_text(FAKE_SUPERVISOR)
            old = subprocess.Popen([sys.executable, str(script), '--output', str(run)])
            guard = None
            try:
                deadline = time.monotonic()+5
                while read(run/'status.json').get('state') != 'training' and time.monotonic() < deadline:
                    time.sleep(.02)
                self.assertEqual(read(run/'status.json')['counter'], 123)
                write(run/'current_execution.json', identity(old.pid))
                guard = Guard(run, ThermalPolicy(minimum_pause=10, cool_seconds=3))
                self.assertEqual(guard.tick(86, 0), 'monitoring')
                self.assertEqual(guard.tick(86, 5), 'saving_and_stopping')
                old.wait(timeout=5)
                self.assertEqual(read(run/'status.json')['state'], 'paused')
                self.assertEqual(guard.tick(60, 6), 'cooling')
                self.assertEqual(guard.tick(80, 10), 'cooling')
                self.assertEqual(guard.tick(60, 11), 'cooling')
                self.assertEqual(guard.tick(60, 15), 'cooling')
                with patch.object(guard, 'verify_saved_state', return_value=dict(completed_training_steps=123)), \
                     patch('thermal_guard.cpu_temperature', return_value=60):
                    self.assertEqual(guard.tick(60, 16), 'restarting')
                deadline = time.monotonic()+5
                while not read(run/'status.json').get('resumed') and time.monotonic() < deadline:
                    time.sleep(.02)
                current = read(run/'status.json')
                self.assertTrue(current['resumed'])
                self.assertEqual(current['counter'], 123)
                self.assertNotEqual(current['supervisor_pid'], old.pid)
                self.assertEqual(guard.tick(60, 17), 'monitoring')
                self.assertEqual(guard.control['mode'], 'watching')
                events = [json.loads(x)['event'] for x in (run/'thermal_guard/events.jsonl').read_text().splitlines()]
                for name in ['thermal_pause_requested', 'sigterm_sent', 'cooldown_started',
                             'thermal_resume_started', 'thermal_resume_confirmed']:
                    self.assertIn(name, events)
            finally:
                if old.poll() is None:
                    old.terminate()
                    old.wait(timeout=5)
                if guard:
                    for child in guard.children:
                        if child.poll() is None:
                            child.terminate()
                        child.wait(timeout=5)
                    guard.close()


if __name__ == '__main__':
    unittest.main()
