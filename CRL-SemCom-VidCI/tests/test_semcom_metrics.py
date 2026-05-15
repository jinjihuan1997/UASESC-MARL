import math
import unittest

from experiment_scripts.comm.metrics import compute_batch_comm_metrics, compute_ls_side

import torch


class SemComMetricTests(unittest.TestCase):
    def test_default_rate_level_one_case(self):
        rate_mask = torch.zeros(1, 48, 32, 32, dtype=torch.float32)
        rate_mask[:, :12] = 1.0
        snr_db = 10.0

        metrics = compute_batch_comm_metrics(
            rate_mask,
            input_hw=(256, 256),
            rate_levels=4,
            snr_db=snr_db,
        )

        expected_kept_real_symbols = 12 * 32 * 32
        expected_ls_main = expected_kept_real_symbols / 2.0
        expected_ls_side = (256 * 256 / 32.0) / math.log2(1.0 + 10.0)
        expected_ls_total = expected_ls_main + expected_ls_side

        self.assertEqual(float(metrics['kept_real_symbols_per_clip'][0]), expected_kept_real_symbols)
        self.assertEqual(float(metrics['ls_main_per_clip'][0]), expected_ls_main)
        self.assertAlmostEqual(float(metrics['ls_side_per_clip'][0]), expected_ls_side, places=5)
        self.assertLess(abs(float(metrics['ls_total_per_clip'][0]) - expected_ls_total), 1e-3)
        self.assertEqual(float(metrics['avg_rate_level']), 1.0)
        self.assertEqual(float(metrics['avg_kept_real_symbols']), expected_kept_real_symbols)

    def test_ls_side_formula(self):
        self.assertAlmostEqual(compute_ls_side(256, 256, snr_db=10.0), (256 * 256 / 32.0) / math.log2(11.0), places=10)


if __name__ == '__main__':
    unittest.main()
