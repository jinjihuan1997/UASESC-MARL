"""Decentralized deployment: each decision takes ONLY that actor's observation.

No environment, critic, global state, future tape, or other UAV observation is
accepted. The SUT score predictor is fitted offline from calibration data.
"""
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class PublicSpec:
    uavs: int = 3
    ds_per_uav: int = 10
    modes: int = 16
    age_ref: float = 8.0
    observation_load_ref: float = 100000.0
    quality_ref: float = 33.0
    quality_min: float = 21.0
    reward_load_ref: float = 60000.0
    age_max: float = 600.0
    age_tail: float = 4.0
    weights: tuple = ((.5, .4, .1), (.2, .7, .1), (.7, .2, .1))


def sut_candidates(sut_obs):
    """[equal, backlog, urgency], using only the existing SUT summary fields."""
    obs = sut_obs.to(torch.float64)
    count = (obs[:, 10:13] * 10).round().clamp_min(0)
    urgency = (obs[:, 13:16] * 80).round().clamp_min(0)
    scores = torch.stack([torch.ones_like(count), count, urgency], 1)
    shares = torch.where(scores.sum(-1, keepdim=True) > 0,
                         scores / scores.sum(-1, keepdim=True).clamp_min(1e-12),
                         torch.full_like(scores, 1 / 3))
    return (2 * shares - 1).to(torch.float32)


def score_features(sut_obs):
    """Known candidate ID is an action argument, not an additional observation."""
    count = len(sut_obs)
    return torch.cat([sut_obs[:, None, :27].expand(-1, 3, -1),
                      torch.eye(3, dtype=sut_obs.dtype, device=sut_obs.device)[None].expand(count, -1, -1)], -1)


class TreeScorePredictor:
    """Portable numeric-only HGB trees, verified against sklearn before use."""
    def __init__(self, path):
        with np.load(path, allow_pickle=False) as z:
            self.feature = torch.as_tensor(z['feature'].copy(), dtype=torch.int64)
            self.threshold = torch.as_tensor(z['threshold'].copy(), dtype=torch.float64)
            self.left = torch.as_tensor(z['left'].copy(), dtype=torch.int64)
            self.right = torch.as_tensor(z['right'].copy(), dtype=torch.int64)
            self.leaf = torch.as_tensor(z['leaf'].copy(), dtype=torch.bool)
            self.value = torch.as_tensor(z['value'].copy(), dtype=torch.float64)
            self.baseline = float(z['baseline'])
            self.depth = int(z['depth'])
        self.tree_index = torch.arange(len(self.feature))[None]

    def predict(self, x):
        assert x.shape[-1] == 30 and torch.isfinite(x).all()
        shape = x.shape[:-1]
        x = x.reshape(-1, 30).to(torch.float64)
        node = torch.zeros((len(x), len(self.feature)), dtype=torch.int64)
        rows = torch.arange(len(x))[:, None]
        for _ in range(self.depth):
            trees = self.tree_index
            feature = self.feature[trees, node]
            left = x[rows, feature] <= self.threshold[trees, node]
            next_node = torch.where(left, self.left[trees, node], self.right[trees, node])
            node = torch.where(self.leaf[trees, node], node, next_node)
        assert self.leaf[self.tree_index, node].all()
        return (self.value[self.tree_index, node].sum(-1) + self.baseline).reshape(shape)


class SUTGreedy:
    def __init__(self, predictor=None, static_family_by_instruction=None):
        self.predictor = predictor
        self.static = static_family_by_instruction
        assert (predictor is None) != (static_family_by_instruction is None)

    @torch.no_grad()
    def act(self, sut_obs):
        actions = sut_candidates(sut_obs)
        if self.predictor is not None:
            scores = self.predictor.predict(score_features(sut_obs))
            chosen = scores.argmax(-1)
        else:
            gid = sut_obs[:, 23:26].argmax(-1)
            chosen = torch.as_tensor(self.static, dtype=torch.int64)[gid]
        return actions[torch.arange(len(sut_obs)), chosen], chosen


class UAVGreedy:
    """Local one-slot utility; replace unobserved global max by local max/U.

Mean/tail AoI, quality, and usage terms retain their exact additive global
normalizers. True environment scoring is NEVER changed to this local proxy.
"""
    def __init__(self, candidates, spec=PublicSpec()):
        self.candidates = tuple(candidates)
        self.spec = spec
        self.weights = torch.as_tensor(spec.weights, dtype=torch.float64)

    @torch.no_grad()
    def act(self, local_obs, local_mask):
        p = self.spec
        obs = local_obs.to(torch.float64)
        E, K = len(obs), p.ds_per_uav
        budget = obs[:, 0] * p.observation_load_ref
        q = obs[:, 1:11] > .5
        cache_age = (obs[:, 11:21] * p.age_ref).round().clamp_min(0)
        aoi = (obs[:, 21:31] * p.age_ref).round().clamp_min(0)
        load = obs[:, 31:47] * p.observation_load_ref
        quality = obs[:, 47:63] * p.quality_ref
        gid = obs[:, 67:70].argmax(-1)
        req = obs[:, 74] * p.quality_ref
        # Actor masks are authoritative. The all-ones no-feasible fallback is
        # disambiguated from the actor-visible load/quality values only.
        plausible = (quality >= req[:, None] - 1e-5) & (load <= budget[:, None] + 2e-3)
        has_mode = q.any(-1) & plausible.any(-1)
        valid = local_mask[:, :16] > .5
        preferred = torch.as_tensor(self.candidates)[None].expand(E, -1)
        fallback = valid.long().argmax(-1)[:, None]
        effective = torch.where(valid.gather(-1, preferred), preferred, fallback)
        cost = load.gather(-1, effective)
        psnr = quality.gather(-1, effective)
        ranks = torch.argsort(torch.where(q, aoi, -torch.inf), dim=-1, descending=True, stable=True)
        ranked_q = q.gather(-1, ranks)
        ranked_age = cache_age.gather(-1, ranks)
        ranked_aoi = aoi.gather(-1, ranks)
        remaining = budget[:, None].expand_as(cost).clone()
        takes = []
        for j in range(K):
            take = has_mode[:, None] & ranked_q[:, j, None] & (cost <= remaining + 1e-9)
            remaining -= torch.where(take, cost, 0.)
            takes.append(take)
        served = torch.stack(takes, -1)
        count = served.sum(-1)
        after = torch.where(served, ranked_age[:, None] + 1, ranked_aoi[:, None] + 1).clamp_max(p.age_max)
        total_ds = p.uavs * K
        q_gain = ((psnr - p.quality_min) / (p.quality_ref - p.quality_min)).clamp_min(0) * count / total_ds
        age_cost = (.4 * after.sum(-1) / total_ds + .3 * after.max(-1).values / p.uavs +
                    .3 * (after - p.age_tail).clamp_min(0).sum(-1) / total_ds) / p.age_ref
        weights = self.weights[gid]
        score = weights[:, :1] * q_gain - weights[:, 1:2] * age_cost - weights[:, 2:3] * .2 * cost * count / p.reward_load_ref
        chosen = score.argmax(-1)
        # Return a preferred mode, preserving the environment's common fallback.
        return F.one_hot(preferred[torch.arange(E), chosen], p.modes).to(torch.float32)
