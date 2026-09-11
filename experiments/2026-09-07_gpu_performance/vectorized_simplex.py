"""Equivalent masked Dirichlet log-density/entropy without per-row loops.

Sampling, normalization, mask fallback and deterministic actions are inherited
unchanged. Install only in the isolated performance probe process.
"""
import torch
from harl.models.base.distributions import FixedSimplex


class VectorizedFixedSimplex(FixedSimplex):
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


def install():
    import harl.models.base.distributions as distributions
    distributions.FixedSimplex = VectorizedFixedSimplex
