from pathlib import Path
import sys
sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[2] / "HARL/HARL")]
import numpy as np
import pytest
import torch
from harl.models.base.distributions import FixedSimplex
from harl.common.valuenorm import ValueNorm
from vectorized_simplex import VectorizedFixedSimplex


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_log_density_entropy_and_gradients(device, dtype):
    torch.manual_seed(173)
    raw = torch.randn(96, 5, device=device, dtype=dtype) * 3
    mask = torch.rand(96, 5, device=device) > .4
    mask[:8] = False
    mask[8:16] = False; mask[8:16, 3] = True
    mask[16:32] = True
    actions = torch.randn_like(raw) * 2
    actions[32:40] = 0
    old_logits = raw.clone().requires_grad_()
    new_logits = raw.clone().requires_grad_()
    old_actions = actions.clone().requires_grad_()
    new_actions = actions.clone().requires_grad_()
    old = FixedSimplex(old_logits, mask=mask)
    new = VectorizedFixedSimplex(new_logits, mask=mask)
    old_logp, new_logp = old.log_probs(old_actions), new.log_probs(new_actions)
    old_entropy, new_entropy = old.entropy(), new.entropy()
    torch.testing.assert_close(new_logp, old_logp, rtol=1e-4, atol=5e-5)
    torch.testing.assert_close(new_entropy, old_entropy, rtol=1e-4, atol=5e-5)
    weights = torch.linspace(.25, 1.5, 96, device=device, dtype=dtype)
    old_loss = (old_logp[:, 0] * weights + .013 * old_entropy).sum()
    new_loss = (new_logp[:, 0] * weights + .013 * new_entropy).sum()
    old_grad = torch.autograd.grad(old_loss, (old_logits, old_actions))
    new_grad = torch.autograd.grad(new_loss, (new_logits, new_actions))
    for actual, expected in zip(new_grad, old_grad):
        torch.testing.assert_close(actual, expected, rtol=2e-4, atol=1e-4)
        assert torch.isfinite(actual).all()
    assert torch.count_nonzero(new_grad[0][~new.safe_mask]) == 0
    assert torch.count_nonzero(new_logp[:16]) == 0
    assert torch.count_nonzero(new_entropy[:16]) == 0


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_sampling_and_rng_unchanged(device):
    raw = torch.linspace(-2, 2, 60, device=device).reshape(20, 3)
    mask = torch.ones_like(raw, dtype=torch.bool)
    mask[:3] = False
    mask[3:7, 1:] = False
    mask[7:11, 1] = False
    old = FixedSimplex(raw, mask=mask)
    new = VectorizedFixedSimplex(raw, mask=mask)
    torch.manual_seed(912)
    expected = old.sample()
    cpu_rng = torch.get_rng_state()
    gpu_rng = torch.cuda.get_rng_state()
    torch.manual_seed(912)
    actual = new.sample()
    assert torch.equal(actual, expected)
    assert torch.equal(torch.get_rng_state(), cpu_rng)
    assert torch.equal(torch.cuda.get_rng_state(), gpu_rng)
    assert torch.equal(old.mode(), new.mode())
    torch.testing.assert_close(actual.sum(-1), torch.ones(20, device=device))
    assert torch.count_nonzero(actual[~new.safe_mask]) == 0


def test_gpu_normalizer_registered_state_round_trip(tmp_path):
    # Constructing on CPU then moving the registered module preserves fields.
    fixed = ValueNorm(1, device=torch.device("cpu")).to("cuda")
    fixed.tpdv = dict(dtype=torch.float32, device=torch.device("cuda"))
    fixed.update(np.arange(31, dtype=np.float32).reshape(-1, 1))
    assert set(fixed.state_dict()) == {"running_mean", "running_mean_sq", "debiasing_term"}
    destination = tmp_path / "value_normalizer.pt"
    torch.save(fixed.state_dict(), destination)
    restored = ValueNorm(1)
    restored.load_state_dict(torch.load(destination, map_location="cpu", weights_only=True), strict=True)
    sample = np.linspace(-2, 2, 11, dtype=np.float32).reshape(-1, 1)
    np.testing.assert_allclose(restored.denormalize(sample), fixed.denormalize(sample), rtol=1e-6, atol=1e-6)
