"""Frozen pre-optimization density; test oracle only, never imported by training."""
import torch

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

    def log_probs(self, actions):
        probs = self._normalize(actions)
        out = torch.zeros((probs.shape[0], 1), dtype=probs.dtype, device=probs.device)
        for row_idx, active_idx in enumerate(self._active_index_list()):
            if active_idx.numel() <= 1:
                continue
            out[row_idx, 0] = self._dirichlet_row(row_idx, active_idx).log_prob(
                probs[row_idx, active_idx]
            )
        return out

    def entropy(self):
        out = torch.zeros((self.concentration.shape[0],), dtype=self.concentration.dtype, device=self.concentration.device)
        for row_idx, active_idx in enumerate(self._active_index_list()):
            if active_idx.numel() <= 1:
                continue
            out[row_idx] = self._dirichlet_row(row_idx, active_idx).entropy()
        return out

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

