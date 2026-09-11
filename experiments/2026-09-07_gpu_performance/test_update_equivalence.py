from pathlib import Path
import copy
import importlib.util
import json
import sys
sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[2] / "HARL/HARL"), str(Path(__file__).resolve().parents[2] / "HARL/HARL/examples")]
import evaluate_instruction_constraints
import numpy as np
import pytest
import torch
import harl.models.base.distributions as distributions
from harl.algorithms.actors.happo import HAPPO
from harl.common.buffers.on_policy_critic_buffer_ep import OnPolicyCriticBufferEP
from harl.common.valuenorm import ValueNorm
from harl.envs.uav_escs.SC.uav_escs_env_sc import SCUAVEnv
from vectorized_simplex import VectorizedFixedSimplex


def setup_config():
    root = Path(__file__).resolve().parents[2]
    return json.loads((root / "experiments/2026-09-07_happo_instruction_ab/configs/B_explicit_instruction_seed1.json").read_text())


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_full_happo_update_fixed_minibatch(device, monkeypatch):
    torch.set_num_threads(1)
    torch.manual_seed(618)
    rng = np.random.default_rng(718)
    config = setup_config()
    env = SCUAVEnv(config["env_args"])
    algo = config["algo_args"]
    original = HAPPO({**algo["model"], **algo["algo"]}, env.observation_space[0], env.action_space[0], torch.device(device))
    optimized = copy.deepcopy(original)
    args = original.args
    batch = 64
    obs = rng.normal(0, .2, (batch, env.observation_space[0].shape[0])).astype(np.float32)
    recurrent = np.zeros((batch, args["recurrent_n"], args["hidden_sizes"][-1]), dtype=np.float32)
    masks = np.ones((batch, 1), dtype=np.float32)
    available = np.ones((batch, env.action_space[0].shape[0]), dtype=np.float32)
    with torch.no_grad():
        actions, old_logp, _ = original.get_actions(obs, recurrent, masks, available)
    actions = actions.detach().cpu().numpy()
    old_logp = old_logp.detach().cpu().numpy()
    active = np.ones((batch, 1), dtype=np.float32)
    active[:7] = 0
    advantages = rng.normal(0, 1, (batch, 1)).astype(np.float32)
    factor = rng.uniform(.5, 1.5, (batch, 1)).astype(np.float32)
    sample = (obs, recurrent, actions, masks, active, old_logp, advantages, available, factor)
    initial = {k: v.detach().clone() for k, v in original.actor.state_dict().items()}
    old_metrics = original.update(sample)
    monkeypatch.setattr(distributions, "FixedSimplex", VectorizedFixedSimplex)
    new_metrics = optimized.update(sample)
    for old, new in zip(old_metrics, new_metrics):
        torch.testing.assert_close(torch.as_tensor(new), torch.as_tensor(old), rtol=1e-4, atol=2e-5)
    for old, new in zip(original.actor.parameters(), optimized.actor.parameters()):
        torch.testing.assert_close(new.grad, old.grad, rtol=2e-4, atol=2e-5)
        torch.testing.assert_close(new, old, rtol=2e-4, atol=2e-5)
    assert any(not torch.equal(initial[k], v) for k, v in optimized.actor.state_dict().items())
    # The Adam state matters to a real optimizer step, not just its loss value.
    for old, new in zip(original.actor_optimizer.state.values(), optimized.actor_optimizer.state.values()):
        for key in old:
            torch.testing.assert_close(new[key], old[key], rtol=2e-4, atol=2e-5)
    env.close()


def test_hybrid_gae_matches_gpu_with_truncation_and_termination():
    config = setup_config()
    env = SCUAVEnv(config["env_args"])
    algo = config["algo_args"]
    gpu_buffer = OnPolicyCriticBufferEP({**algo["train"], **algo["model"], **algo["algo"]}, env.share_observation_space[0])
    rng = np.random.default_rng(618)
    gpu_buffer.rewards[:] = rng.normal(0, .2, gpu_buffer.rewards.shape)
    gpu_buffer.value_preds[:] = rng.normal(0, 1, gpu_buffer.value_preds.shape)
    gpu_buffer.masks[50, 0] = 0
    gpu_buffer.masks[220, 3] = 0
    gpu_buffer.bad_masks[220, 3] = 0
    cpu_buffer = copy.deepcopy(gpu_buffer)
    norm_gpu = ValueNorm(1).to("cuda")
    norm_gpu.tpdv = dict(dtype=torch.float32, device=torch.device("cuda"))
    norm_gpu.update(rng.normal(7, 3, (100, 1)).astype(np.float32))
    norm_cpu = ValueNorm(1)
    norm_cpu.load_state_dict(norm_gpu.state_dict(), strict=True)
    next_value = rng.normal(size=(10, 1)).astype(np.float32)
    gpu_buffer.compute_returns(next_value, norm_gpu)
    cpu_buffer.compute_returns(next_value, norm_cpu)
    np.testing.assert_allclose(cpu_buffer.returns, gpu_buffer.returns, rtol=2e-5, atol=2e-5)
    assert np.isfinite(cpu_buffer.returns).all()
    env.close()


def test_proposed_direct_gpu_constructor_saves_statistics(tmp_path):
    path = Path(__file__).resolve().parent / "proposals/valuenorm.py"
    spec = importlib.util.spec_from_file_location("proposed_value_norm", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fixed = module.ValueNorm(1, device=torch.device("cuda"))
    data = np.arange(31, dtype=np.float32).reshape(-1, 1)
    fixed.update(data)
    assert set(fixed.state_dict()) == {"running_mean", "running_mean_sq", "debiasing_term"}
    saved = tmp_path / "normalizer.pt"
    torch.save(fixed.state_dict(), saved)
    restored = module.ValueNorm(1)
    restored.load_state_dict(torch.load(saved, map_location="cpu", weights_only=True), strict=True)
    np.testing.assert_allclose(restored.denormalize(data), fixed.denormalize(data), rtol=1e-6, atol=1e-6)
