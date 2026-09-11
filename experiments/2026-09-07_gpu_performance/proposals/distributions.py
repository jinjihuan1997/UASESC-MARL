"""Modify standard PyTorch distributions so they to make compatible with this codebase."""
import torch
import torch.nn as nn
from harl.utils.models_tools import init, get_init_method


class FixedCategorical(torch.distributions.Categorical):
    """Modify standard PyTorch Categorical."""

    def sample(self):
        return super().sample().unsqueeze(-1)

    def log_probs(self, actions):
        return (
            super()
            .log_prob(actions.squeeze(-1))
            .unsqueeze(-1)
        )

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)


class FixedNormal(torch.distributions.Normal):
    """Modify standard PyTorch Normal."""

    def log_probs(self, actions):
        return super().log_prob(actions)

    def entropy(self):
        return super().entropy().sum(-1)

    def mode(self):
        return self.mean


class FixedSimplex:
    """Masked simplex-valued distribution with exact Dirichlet log-probabilities.

    Each batch row is interpreted over its active simplex coordinates only. Inactive
    coordinates are clamped to zero and excluded from the Dirichlet density.
    """

    def __init__(self, logits, temperature=0.67, mask=None, eps=1e-6, concentration_floor=1e-3):
        self.eps = float(eps)
        # `temperature` is kept in the signature for backward compatibility with
        # existing action-spec construction, but the density is now parameterized
        # directly as a Dirichlet over the feasible simplex coordinates.
        self.temperature = torch.as_tensor(float(temperature), dtype=logits.dtype, device=logits.device)
        self.concentration_floor = float(concentration_floor)
        if mask is None:
            mask = torch.ones_like(logits, dtype=torch.bool)
        else:
            mask = mask.to(dtype=torch.bool, device=logits.device)

        valid = mask.any(dim=-1, keepdim=True)
        safe_mask = mask.clone()
        if safe_mask.shape[-1] > 0:
            safe_mask[~valid.expand_as(safe_mask)] = False
            safe_mask[~valid.squeeze(-1), 0] = True
        self.mask = mask
        self.safe_mask = safe_mask
        self.logits = logits
        concentration = torch.nn.functional.softplus(logits) + self.concentration_floor
        self.concentration = torch.where(
            safe_mask,
            concentration,
            torch.zeros_like(concentration),
        )

    def _normalize(self, actions):
        probs = torch.clamp(actions, min=0.0, max=1.0)
        probs = torch.where(self.safe_mask, probs, torch.zeros_like(probs))
        active = self.safe_mask.any(dim=-1, keepdim=True)
        if probs.shape[-1] > 0:
            fallback = torch.zeros_like(probs)
            fallback[..., 0] = 1.0
            probs = torch.where(active, probs, fallback)
        probs = torch.clamp(probs, min=self.eps)
        probs = probs * self.safe_mask.to(dtype=probs.dtype)
        denom = probs.sum(dim=-1, keepdim=True).clamp(min=self.eps)
        probs = probs / denom
        return probs

    def _active_index_list(self):
        return [torch.nonzero(row, as_tuple=False).squeeze(-1) for row in self.safe_mask]

    def _dirichlet_row(self, row_idx, active_idx):
        return torch.distributions.Dirichlet(self.concentration[row_idx, active_idx])

    def sample(self):
        probs = torch.zeros_like(self.concentration)
        active_list = self._active_index_list()
        if probs.shape[-1] > 0:
            fallback = torch.zeros_like(probs)
            fallback[..., 0] = 1.0
            probs = fallback
        for row_idx, active_idx in enumerate(active_list):
            if active_idx.numel() == 0:
                continue
            if active_idx.numel() == 1:
                probs[row_idx].zero_()
                probs[row_idx, active_idx[0]] = 1.0
                continue
            probs[row_idx].zero_()
            probs[row_idx, active_idx] = self._dirichlet_row(row_idx, active_idx).sample()
        return probs

    def _masked_stats(self):
        alpha = torch.where(self.safe_mask, self.concentration, torch.ones_like(self.concentration))
        count = self.safe_mask.sum(dim=-1)
        total = self.concentration.sum(dim=-1)
        log_beta = torch.where(self.safe_mask, torch.lgamma(alpha), torch.zeros_like(alpha)).sum(dim=-1) - torch.lgamma(total)
        return alpha, count, total, log_beta

    def log_probs(self, actions):
        probabilities = self._normalize(actions)
        alpha, count, _, log_beta = self._masked_stats()
        safe_probabilities = torch.where(self.safe_mask, probabilities, torch.ones_like(probabilities))
        terms = torch.where(self.safe_mask, (alpha - 1.) * safe_probabilities.log(), torch.zeros_like(alpha))
        result = terms.sum(dim=-1) - log_beta
        return torch.where(count > 1, result, torch.zeros_like(result)).unsqueeze(-1)

    def entropy(self):
        alpha, count, total, log_beta = self._masked_stats()
        terms = torch.where(self.safe_mask, (alpha - 1.) * torch.digamma(alpha), torch.zeros_like(alpha))
        result = log_beta + (total - count) * torch.digamma(total) - terms.sum(dim=-1)
        return torch.where(count > 1, result, torch.zeros_like(result))

    def entropy_proxy(self):
        return self.entropy()

    def kl_divergence(self, other):
        out = torch.zeros((self.concentration.shape[0],), dtype=self.concentration.dtype, device=self.concentration.device)
        for row_idx, active_idx in enumerate(self._active_index_list()):
            other_active_idx = torch.nonzero(other.safe_mask[row_idx], as_tuple=False).squeeze(-1)
            if active_idx.numel() <= 1 and other_active_idx.numel() <= 1:
                continue
            if active_idx.numel() != other_active_idx.numel() or (
                active_idx.numel() > 0 and not torch.equal(active_idx, other_active_idx)
            ):
                raise ValueError("Simplex KL requires identical active-coordinate masks.")
            out[row_idx] = torch.distributions.kl_divergence(
                self._dirichlet_row(row_idx, active_idx),
                other._dirichlet_row(row_idx, active_idx),
            )
        return out

    def mode(self):
        # Use the Dirichlet mean as the deterministic action; unlike the strict
        # mode, it is always defined and remains in the simplex interior.
        probs = torch.zeros_like(self.concentration)
        if probs.shape[-1] > 0:
            probs[..., 0] = 1.0
        active_mass = self.concentration.sum(dim=-1, keepdim=True).clamp(min=self.eps)
        valid = self.safe_mask.any(dim=-1, keepdim=True)
        mean = torch.where(self.safe_mask, self.concentration / active_mass, torch.zeros_like(self.concentration))
        return torch.where(valid, mean, probs)


class Categorical(nn.Module):
    """A linear layer followed by a Categorical distribution."""

    def __init__(
        self, num_inputs, num_outputs, initialization_method="orthogonal_", gain=0.01
    ):
        super(Categorical, self).__init__()
        init_method = get_init_method(initialization_method)

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain)

        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x, available_actions=None):
        x = self.linear(x)
        if available_actions is not None:
            x[available_actions == 0] = -1e10
        return FixedCategorical(logits=x)


class DiagGaussian(nn.Module):
    """A linear layer followed by a Diagonal Gaussian distribution."""

    def __init__(
        self,
        num_inputs,
        num_outputs,
        initialization_method="orthogonal_",
        gain=0.01,
        args=None,
    ):
        super(DiagGaussian, self).__init__()

        init_method = get_init_method(initialization_method)

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain)

        if args is not None:
            self.std_x_coef = args["std_x_coef"]
            self.std_y_coef = args["std_y_coef"]
        else:
            self.std_x_coef = 1.0
            self.std_y_coef = 0.5
        self.fc_mean = init_(nn.Linear(num_inputs, num_outputs))
        log_std = torch.ones(num_outputs) * self.std_x_coef
        self.log_std = torch.nn.Parameter(log_std)

    def forward(self, x, available_actions=None):
        action_mean = self.fc_mean(x)
        action_std = torch.sigmoid(self.log_std / self.std_x_coef) * self.std_y_coef
        return FixedNormal(action_mean, action_std)


class Simplex(nn.Module):
    """A linear layer followed by a masked simplex-valued Dirichlet distribution."""

    def __init__(
        self,
        num_inputs,
        num_outputs,
        initialization_method="orthogonal_",
        gain=0.01,
        args=None,
    ):
        super(Simplex, self).__init__()
        init_method = get_init_method(initialization_method)

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain)

        self.fc_logits = init_(nn.Linear(num_inputs, num_outputs))
        if args is not None:
            self.temperature = float(args.get("simplex_temperature", 0.67))
            self.concentration_floor = float(args.get("simplex_concentration_floor", 1e-3))
        else:
            self.temperature = 0.67
            self.concentration_floor = 1e-3

    def forward(self, x, available_actions=None):
        logits = self.fc_logits(x)
        if available_actions is None:
            mask = None
        else:
            mask = available_actions > 0.5
        return FixedSimplex(
            logits,
            temperature=self.temperature,
            mask=mask,
            concentration_floor=self.concentration_floor,
        )
