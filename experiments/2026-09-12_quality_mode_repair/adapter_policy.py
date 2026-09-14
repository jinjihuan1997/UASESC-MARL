"""Frozen original UAV + independently trainable zero-initialized mode residual."""
from repair_support import torch
from harl.models.base.distributions import FixedCategorical
from torch import nn

class ResidualPolicy(nn.Module):
    instruction_slice = slice(67, 70)
    quality_id = 2
    def __init__(self, base, obs_dim, arm, device):
        super().__init__()
        assert obs_dim == 76 and arm in ('residual_all', 'residual_quality')
        self.base_policy = base.requires_grad_(False).eval()
        self.arm = arm
        self.adapter = nn.Sequential(nn.Linear(obs_dim, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 16)).to(device)
        for layer in (self.adapter[0], self.adapter[2]):
            nn.init.orthogonal_(layer.weight, gain=nn.init.calculate_gain('relu')); nn.init.zeros_(layer.bias)
        nn.init.zeros_(self.adapter[4].weight); nn.init.zeros_(self.adapter[4].bias)
        self.tpdv = dict(dtype=torch.float32, device=torch.device(device))
        self.train(False)

    def train(self, mode=True):
        super().train(mode)
        self.base_policy.eval()
        return self

    def requires_grad_(self, requires_grad=True):
        self.adapter.requires_grad_(requires_grad)
        self.base_policy.requires_grad_(False)
        return self

    def gate(self, obs):
        gid = obs[..., self.instruction_slice].argmax(-1)
        return (gid == self.quality_id) if self.arm == 'residual_quality' else torch.ones_like(gid, dtype=torch.bool)

    def raw_logits(self, obs):
        obs = torch.as_tensor(obs, **self.tpdv)
        with torch.no_grad():
            features = self.base_policy.base(obs)
            base = self.base_policy.act.action_out.categorical_heads[0].get_logits(features).squeeze(1)
        delta = self.adapter(obs)
        return base, delta, base + self.gate(obs).to(delta.dtype).unsqueeze(-1) * delta

    def distribution(self, obs, available_actions):
        _, _, logits = self.raw_logits(obs)
        if available_actions is not None:
            mask = torch.as_tensor(available_actions, device=logits.device)[..., :16] > .5
            mask = torch.where(mask.any(-1, keepdim=True), mask, torch.ones_like(mask))
            logits = logits.masked_fill(~mask, -1e10)
        return FixedCategorical(logits=logits.unsqueeze(1))

    @staticmethod
    def encoded_logp(dist, index):
        return dist.log_probs(index).reshape(-1, 1).expand(-1, 16) / 16

    def forward(self, obs, rnn_states, masks, available_actions=None, deterministic=False):
        dist = self.distribution(obs, available_actions)
        index = dist.mode() if deterministic else dist.sample()
        action = 2 * torch.nn.functional.one_hot(index.reshape(-1), 16).to(torch.float32) - 1
        return action, self.encoded_logp(dist, index), rnn_states

    def evaluate_actions(self, obs, rnn_states, action, masks, available_actions=None, active_masks=None):
        dist = self.distribution(obs, available_actions)
        action = torch.as_tensor(action, **self.tpdv)
        index = action.argmax(-1).reshape(-1, 1, 1)
        entropy = dist.entropy().sum(-1)
        if active_masks is not None:
            active_masks = torch.as_tensor(active_masks, **self.tpdv).reshape(-1)
            entropy = (entropy * active_masks).sum() / active_masks.sum()
        else: entropy = entropy.mean()
        return self.encoded_logp(dist, index), entropy, dist
