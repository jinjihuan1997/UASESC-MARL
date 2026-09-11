"""One full objective for scalar NumPy and batched Torch execution.

Physical delivery gains and usages are inputs; an independent evaluator must
recompute them and the complete formula without calling this function.
"""
import numpy as np
import torch


def reward_components(p, quality_gain_sum, channel_uses, before, after, gid):
    if torch.is_tensor(after):
        weights = torch.as_tensor(p.reward_weights_by_instruction, dtype=after.dtype, device=after.device)[gid]
        limits = torch.as_tensor(p.A_limit_by_instruction, dtype=after.dtype, device=after.device)[gid]
        penalties = torch.as_tensor(p.constraint_penalty_A_by_instruction, dtype=after.dtype, device=after.device)[gid]
        maximum = after.amax(-1)
    else:
        after, before = np.asarray(after), np.asarray(before)
        weights = np.asarray(p.reward_weights_by_instruction)[gid]
        limits = np.asarray(p.A_limit_by_instruction)[gid]
        penalties = np.asarray(p.constraint_penalty_A_by_instruction)[gid]
        maximum = after.max(-1)
    q = quality_gain_sum / p.n_ds
    load = channel_uses / p.reward_load_ref
    mean = (after / p.aoi_reward_ref).mean(-1)
    worst = maximum / p.aoi_reward_ref
    tail = ((after - p.aoi_tail_threshold) / p.aoi_reward_ref).clip(0, None).mean(-1)
    age = p.aoi_mean_weight*mean + p.aoi_max_weight*worst + p.aoi_tail_weight*tail
    violation = ((maximum-limits)/limits).clip(0, None)
    quality_credit = weights[..., 0]*q
    age_mean_cost = weights[..., 1]*p.aoi_mean_weight*mean
    age_max_cost = weights[..., 1]*p.aoi_max_weight*worst
    age_tail_cost = weights[..., 1]*p.aoi_tail_weight*tail
    resource_cost = weights[..., 2]*p.reward_load_scale*load
    service_violation_cost = penalties*violation
    reception_credit = p.eta_recv_aoi_bonus*(before-after).clip(0, None).mean(-1)/limits
    base = quality_credit - weights[..., 1]*age - resource_cost
    auxiliary = -service_violation_cost + reception_credit
    objective = base+auxiliary
    return dict(quality_term=q, load_term=load, aoi_term=age,
                age_mean_term=mean, age_max_term=worst, age_tail_term=tail,
                quality_credit=quality_credit, age_mean_cost=age_mean_cost,
                age_max_cost=age_max_cost, age_tail_cost=age_tail_cost,
                resource_cost=resource_cost, service_violation_cost=service_violation_cost,
                reception_credit=reception_credit, max_aoi_violation=violation,
                base_reward=base, auxiliary=auxiliary, objective_reward=objective,
                common_reward=objective, reward=objective, display_score=100*objective)
