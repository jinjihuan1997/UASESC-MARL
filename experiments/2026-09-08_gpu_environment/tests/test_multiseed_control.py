"""Integrity, non-destructive startup and thermal hysteresis checks."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aggregate_seeds import stats
from formal_train import train
from multiseed_protocol import prepare, read, verify_run
from three_seed_train import ThermalControl, recorded_process_alive


class MultiSeedControl(unittest.TestCase):
    def test_freezes_21_jobs_and_detects_changed_input(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)/'run'
            result = prepare(out, seeds=[19, 317, 863], smoke=True)
            self.assertEqual(result['seeds'], [19, 317, 863])
            self.assertEqual(result['seed_selection'], 'provided explicitly')
            self.assertEqual(result['total_training_jobs'], 21)
            self.assertEqual(result['total_training_steps'], 168000)
            self.assertEqual(sum(j['device'] == 'cuda:0' for j in result['jobs']), 3)
            self.assertEqual(result['total_evaluation_jobs'], 25)
            self.assertEqual(result['total_evaluation_slots'], 30000)
            self.assertEqual(verify_run(out), result)
            path = out/result['jobs'][0]['config']
            config = read(path)
            config['algo_args']['seed']['seed'] = 42
            path.write_text(json.dumps(config))
            with self.assertRaises(RuntimeError):
                verify_run(out)

    def test_duplicate_seed_and_overlapping_environment_streams_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            for seeds in ([1, 1, 2], [1, 2, 1001]):
                with self.assertRaises(ValueError):
                    prepare(Path(temporary)/'run', seeds=seeds, smoke=True)

    def test_existing_job_cannot_be_overwritten_by_fresh_start(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'status.json'
            original = '{"state": "complete", "evidence": "preserve"}'
            path.write_text(original)
            with self.assertRaises(FileExistsError):
                train('unused', temporary, 'cpu', 'unused')
            self.assertEqual(path.read_text(), original)

    def test_thermal_hysteresis_preserves_gpu_control_core(self):
        thermal = ThermalControl()
        self.assertFalse(thermal.tick(84, 0))
        self.assertFalse(thermal.tick(86, 1))
        self.assertTrue(thermal.tick(86, 7))
        self.assertEqual(thermal.level, 1)
        self.assertEqual(thermal.cores(13, 'cpu'), [16, 17, 19, 11])
        self.assertEqual(thermal.cores(18, 'cuda:0'), [18])
        self.assertTrue(thermal.tick(86, 18))
        self.assertEqual(thermal.level, 2)
        thermal.tick(74, 20)
        self.assertTrue(thermal.tick(74, 81))
        self.assertEqual(thermal.level, 1)
        self.assertTrue(thermal.tick(74, 142))
        self.assertEqual(thermal.level, 0)
        with self.assertRaises(RuntimeError):
            thermal.tick(95, 150)

    def test_sample_sd_is_over_training_seed_means(self):
        result = stats([1., 2., 3.])
        self.assertEqual(result, dict(mean=2., sample_std=1., n=3))

    def test_orphan_detection_checks_process_identity_not_pid_alone(self):
        with tempfile.TemporaryDirectory() as temporary:
            Path(temporary, 'process.json').write_text(json.dumps(dict(pid=123, uid=1000, start_ticks=10)))
            with patch('three_seed_train.process_info', return_value=dict(pid=123, uid=1000, start_ticks=10, state='S')):
                self.assertTrue(recorded_process_alive(temporary))
            with patch('three_seed_train.process_info', return_value=dict(pid=123, uid=1000, start_ticks=20, state='S')):
                self.assertFalse(recorded_process_alive(temporary))


if __name__ == '__main__':
    unittest.main()
