import math

import torch


def compute_ls_side(height, width, snr_db, side_info_bits=None):
    """Return the side-information cost using a Shannon-style term.

    Formula:
        l_s_side = l_b / log2(1 + snr)

    where:
      - l_b is the side-information bit budget. By default we use
        l_b = H * W / 32 from the previous implementation.
      - snr is the linear SNR, converted from the configured snr_db.
    """
    if side_info_bits is None:
        side_info_bits = float(height) * float(width) / 32.0
    if side_info_bits <= 0:
        raise ValueError('side_info_bits must be positive')

    snr_linear = 10 ** (float(snr_db) / 10.0)
    capacity = math.log2(1.0 + snr_linear)
    if capacity <= 0:
        raise ValueError('log2(1 + snr) must be positive')

    return float(side_info_bits) / capacity


def compute_batch_comm_metrics(
    rate_mask,
    input_hw,
    rate_levels,
    snr_db,
    side_info_bits=None,
):
    """Compute communication metrics from the retained latent-entry mask.

    The source-of-truth quantity is the number of retained real-valued latent
    entries n_z^(i):
        n_z^(i) = sum(rate_mask_i)

    The paper's main-message communication cost uses complex modulated symbols:
        l_s_main^(i) = n_z^(i) / 2

    The total per-clip communication cost is:
        l_s^(i) = l_s_main^(i) + l_s_side

    The second term now uses:
        l_s_side = l_b / log2(1 + snr)

    with l_b = H * W / 32 by default and snr converted from snr_db.

    This function returns both per-clip tensors and batch averages so the code
    can report the paper metric while still keeping the old real-entry proxy.
    """
    if rate_mask.ndim != 4:
        raise ValueError('rate_mask must have shape [B, C, H_latent, W_latent]')
    if rate_levels <= 0:
        raise ValueError('rate_levels must be positive')

    latent_channels = int(rate_mask.shape[1])
    if latent_channels % rate_levels != 0:
        raise ValueError('latent_channels must be divisible by rate_levels')

    height, width = int(input_hw[0]), int(input_hw[1])
    batch_size, _, latent_height, latent_width = rate_mask.shape
    symbols_per_level = latent_channels // rate_levels

    kept_real_symbols_per_clip = rate_mask.sum(dim=(1, 2, 3)).detach()
    rate_level_denominator_per_clip = torch.full_like(
        kept_real_symbols_per_clip,
        float(latent_height * latent_width * symbols_per_level),
    )
    avg_rate_level_per_clip = kept_real_symbols_per_clip / rate_level_denominator_per_clip

    ls_main_per_clip = kept_real_symbols_per_clip / 2.0
    ls_side_value = compute_ls_side(
        height,
        width,
        snr_db=snr_db,
        side_info_bits=side_info_bits,
    )
    ls_side_per_clip = torch.full_like(ls_main_per_clip, float(ls_side_value))
    ls_total_per_clip = ls_main_per_clip + ls_side_per_clip

    return {
        'batch_size': batch_size,
        'latent_channels': latent_channels,
        'latent_height': latent_height,
        'latent_width': latent_width,
        'symbols_per_level': symbols_per_level,
        'kept_real_symbols_per_clip': kept_real_symbols_per_clip,
        'rate_level_denominator_per_clip': rate_level_denominator_per_clip,
        'avg_rate_level_per_clip': avg_rate_level_per_clip,
        'side_info_bits_per_clip': torch.full_like(ls_main_per_clip, float(side_info_bits if side_info_bits is not None else (height * width / 32.0))),
        'snr_db': float(snr_db),
        'snr_linear': float(10 ** (float(snr_db) / 10.0)),
        'ls_main_per_clip': ls_main_per_clip,
        'ls_side_per_clip': ls_side_per_clip,
        'ls_total_per_clip': ls_total_per_clip,
        'avg_rate_level': avg_rate_level_per_clip.mean(),
        'avg_kept_real_symbols': kept_real_symbols_per_clip.mean(),
        'avg_ls_main': ls_main_per_clip.mean(),
        'avg_ls_side': ls_side_per_clip.mean(),
        'avg_ls_total': ls_total_per_clip.mean(),
    }


class CommunicationMetricAccumulator:
    """Accumulate exact dataset-level communication metrics over clips."""

    def __init__(self):
        self.total_clips = 0
        self.total_kept_real_symbols = 0.0
        self.total_rate_level_denominator = 0.0
        self.total_ls_main = 0.0
        self.total_ls_side = 0.0
        self.total_ls_total = 0.0

    def update(self, comm_info):
        kept_real_symbols_per_clip = comm_info['kept_real_symbols_per_clip']
        rate_level_denominator_per_clip = comm_info['rate_level_denominator_per_clip']
        ls_main_per_clip = comm_info['ls_main_per_clip']
        ls_side_per_clip = comm_info['ls_side_per_clip']
        ls_total_per_clip = comm_info['ls_total_per_clip']

        self.total_clips += int(kept_real_symbols_per_clip.numel())
        self.total_kept_real_symbols += float(kept_real_symbols_per_clip.sum().item())
        self.total_rate_level_denominator += float(rate_level_denominator_per_clip.sum().item())
        self.total_ls_main += float(ls_main_per_clip.sum().item())
        self.total_ls_side += float(ls_side_per_clip.sum().item())
        self.total_ls_total += float(ls_total_per_clip.sum().item())

    def summary(self):
        if self.total_clips == 0:
            raise ValueError('Cannot summarize communication metrics with zero clips')

        avg_kept_real_symbols = self.total_kept_real_symbols / self.total_clips
        avg_rate_level = self.total_kept_real_symbols / self.total_rate_level_denominator
        bar_ls_main = self.total_ls_main / self.total_clips
        avg_ls_side = self.total_ls_side / self.total_clips
        bar_ls_total = self.total_ls_total / self.total_clips
        return {
            'num_clips': self.total_clips,
            'avg_kept_real_symbols': avg_kept_real_symbols,
            'avg_rate_level': avg_rate_level,
            'bar_ls_main': bar_ls_main,
            'avg_ls_side': avg_ls_side,
            'bar_ls_total': bar_ls_total,
            'bar_ls_paper': bar_ls_total,
        }


def validate_default_rate_level_one_case(snr_db=10.0, side_info_bits=None):
    """Return expected values for the default 48-channel, 4-level, 32x32 case."""
    rate_mask = torch.zeros(1, 48, 32, 32, dtype=torch.float32)
    rate_mask[:, :12] = 1.0
    metrics = compute_batch_comm_metrics(
        rate_mask,
        input_hw=(256, 256),
        rate_levels=4,
        snr_db=snr_db,
        side_info_bits=side_info_bits,
    )

    expected_kept_real_symbols = 12 * 32 * 32
    expected_ls_main = expected_kept_real_symbols / 2.0
    expected_ls_side = compute_ls_side(256, 256, snr_db=snr_db, side_info_bits=side_info_bits)
    expected_ls_total = expected_ls_main + expected_ls_side

    if float(metrics['kept_real_symbols_per_clip'][0]) != expected_kept_real_symbols:
        raise AssertionError('rate level 1 should keep exactly 12288 real latent entries')
    if float(metrics['ls_main_per_clip'][0]) != expected_ls_main:
        raise AssertionError('rate level 1 should map to 6144 main complex symbols')
    if float(metrics['ls_total_per_clip'][0]) != expected_ls_total:
        raise AssertionError('paper l_s should equal l_s_main + l_s_side')

    return {
        'expected_kept_real_symbols_per_clip': expected_kept_real_symbols,
        'expected_ls_main_per_clip': expected_ls_main,
        'expected_ls_side_per_clip': expected_ls_side,
        'expected_ls_total_per_clip': expected_ls_total,
    }
