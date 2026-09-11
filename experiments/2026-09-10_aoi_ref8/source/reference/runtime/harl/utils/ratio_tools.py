"""Utilities for numerically stable importance-ratio aggregation."""

import torch


def aggregate_action_ratio(log_ratio, action_aggregation: str, clip: float = 20.0):
    """Aggregate per-dimension log-ratios into a stable scalar ratio."""
    log_ratio = torch.nan_to_num(log_ratio, nan=0.0, posinf=clip, neginf=-clip)
    if action_aggregation == "prod":
        agg_log_ratio = torch.sum(log_ratio, dim=-1, keepdim=True)
        agg_log_ratio = torch.clamp(agg_log_ratio, -clip, clip)
        return torch.exp(agg_log_ratio)

    log_ratio = torch.clamp(log_ratio, -clip, clip)
    return getattr(torch, action_aggregation)(
        torch.exp(log_ratio),
        dim=-1,
        keepdim=True,
    )
