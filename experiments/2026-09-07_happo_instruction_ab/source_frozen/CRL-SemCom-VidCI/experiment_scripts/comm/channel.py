import torch


def build_rate_mask(rate_map, max_symbols=48, levels=4):
    """Build the retained-entry mask for the latent tensor.

    The resulting binary mask is the source of truth for communication cost:
    `rate_mask.sum()` counts retained real-valued latent entries n_z, not the
    paper's final l_s metric.
    """
    symbols_per_level = max_symbols // levels
    channel_idx = torch.arange(max_symbols, device=rate_map.device).view(1, max_symbols, 1, 1)
    keep = rate_map.long().clamp_(1, levels) * symbols_per_level
    return (channel_idx < keep).to(rate_map.dtype)


def power_normalize(symbols, eps=1e-6, rate_mask=None):
    """Unit mean power on transmitted real coordinates, per sample.

    Paired real coordinates represent sqrt(2)-scaled I/Q components, so
    (I + jQ)/sqrt(2) has unit average complex-symbol power. Padding consumes
    neither energy nor channel uses. Use the explicit mask, not symbols != 0.
    """
    mask = torch.ones_like(symbols) if rate_mask is None else rate_mask.to(symbols)
    active_count = mask.sum(dim=(1, 2, 3), keepdim=True).clamp_min(1.0)
    masked = symbols * mask
    power = (masked.square().sum(dim=(1, 2, 3), keepdim=True) / active_count).clamp_min(eps)
    return masked / power.sqrt()


def apply_channel(symbols, snr_db=None, training=True, rate_mask=None):
    if snr_db is None:
        return symbols

    normalized = power_normalize(symbols, rate_mask=rate_mask)
    snr = 10 ** (snr_db / 10.0)
    noise_std = (1.0 / snr) ** 0.5
    noise = torch.randn_like(normalized) * noise_std
    if rate_mask is not None:
        noise = noise * rate_mask.to(noise)
    return normalized + noise
