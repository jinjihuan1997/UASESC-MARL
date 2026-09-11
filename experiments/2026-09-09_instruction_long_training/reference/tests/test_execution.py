"""Fixed-input equivalence checks for the execution optimization."""
from pathlib import Path
import copy
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from protocol import activate_runtime, configurations
activate_runtime()
import numpy as np
import torch
from reference_simplex import FixedSimplex as ReferenceSimplex
import harl.models.base.distributions as distributions
from harl.common.valuenorm import ValueNorm
from harl.algorithms.actors import ALGO_REGISTRY
from harl.common.buffers.on_policy_critic_buffer_ep import OnPolicyCriticBufferEP
from harl.envs.uav_escs.SC.uav_escs_env_sc import SCUAVEnv

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


class ExecutionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_batch_density_entropy_and_gradients(self):
        for device in DEVICES:
            for dtype in (torch.float32, torch.float64):
                with self.subTest(device=device, dtype=dtype):
                    torch.manual_seed(173)
                    raw = torch.randn(96, 5, device=device, dtype=dtype)*3
                    mask = torch.rand(96, 5, device=device) > .4
                    mask[:8] = False
                    mask[8:16] = False
                    mask[8:16, 3] = True
                    mask[16:32] = True
                    actions = torch.randn_like(raw)*2
                    actions[32:40] = 0
                    x, y = raw.clone().requires_grad_(), raw.clone().requires_grad_()
                    a, b = actions.clone().requires_grad_(), actions.clone().requires_grad_()
                    original, optimized = ReferenceSimplex(x, mask=mask), distributions.FixedSimplex(y, mask=mask)
                    lp, lq = original.log_probs(a), optimized.log_probs(b)
                    hp, hq = original.entropy(), optimized.entropy()
                    torch.testing.assert_close(lq, lp, rtol=1e-4, atol=5e-5)
                    torch.testing.assert_close(hq, hp, rtol=1e-4, atol=5e-5)
                    weights = torch.linspace(.25, 1.5, 96, device=device, dtype=dtype)
                    old_grad = torch.autograd.grad((lp[:, 0]*weights+.013*hp).sum(), (x, a))
                    new_grad = torch.autograd.grad((lq[:, 0]*weights+.013*hq).sum(), (y, b))
                    for actual, expected in zip(new_grad, old_grad):
                        torch.testing.assert_close(actual, expected, rtol=2e-4, atol=1e-4)
                        self.assertTrue(torch.isfinite(actual).all())
                    self.assertEqual(int(torch.count_nonzero(new_grad[0][~optimized.safe_mask])), 0)
                    self.assertEqual(int(torch.count_nonzero(lq[:16])), 0)
                    self.assertEqual(int(torch.count_nonzero(hq[:16])), 0)

    def test_sampling_and_rng_unchanged(self):
        for device in DEVICES:
            with self.subTest(device=device):
                raw = torch.linspace(-2, 2, 60, device=device).reshape(20, 3)
                mask = torch.ones_like(raw, dtype=torch.bool)
                mask[:3] = False
                mask[3:7, 1:] = False
                mask[7:11, 1] = False
                original, optimized = ReferenceSimplex(raw, mask=mask), distributions.FixedSimplex(raw, mask=mask)
                torch.manual_seed(912)
                expected = original.sample()
                cpu_rng = torch.get_rng_state()
                gpu_rng = torch.cuda.get_rng_state() if device == "cuda" else None
                torch.manual_seed(912)
                actual = optimized.sample()
                self.assertTrue(torch.equal(actual, expected))
                self.assertTrue(torch.equal(torch.get_rng_state(), cpu_rng))
                if gpu_rng is not None:
                    self.assertTrue(torch.equal(torch.cuda.get_rng_state(), gpu_rng))
                self.assertTrue(torch.equal(original.mode(), optimized.mode()))

    def test_happo_and_mappo_full_minibatch_update(self):
        for device in DEVICES:
            for algorithm in ("happo", "mappo"):
                with self.subTest(device=device, algorithm=algorithm):
                    torch.manual_seed(618)
                    rng = np.random.default_rng(718)
                    config = configurations(steps=8000)["IC_HAPPO"]
                    env = SCUAVEnv(config["env_args"])
                    algo = config["algo_args"]
                    original = ALGO_REGISTRY[algorithm]({**algo["model"], **algo["algo"]},
                        env.observation_space[0], env.action_space[0], torch.device(device))
                    optimized = copy.deepcopy(original)
                    batch = 64
                    obs = rng.normal(0, .2, (batch, env.obs_dim_common)).astype(np.float32)
                    recurrent = np.zeros((batch, 1, 256), dtype=np.float32)
                    masks = np.ones((batch, 1), dtype=np.float32)
                    available = np.ones((batch, 3), dtype=np.float32)
                    with torch.no_grad():
                        actions, old_logp, _ = original.get_actions(obs, recurrent, masks, available)
                    active = np.ones((batch, 1), dtype=np.float32)
                    active[:7] = 0
                    advantages = rng.normal(0, 1, (batch, 1)).astype(np.float32)
                    sample = (obs, recurrent, actions.cpu().numpy(), masks, active,
                              old_logp.cpu().numpy(), advantages, available)
                    if algorithm == "happo":
                        sample += (rng.uniform(.5, 1.5, (batch, 1)).astype(np.float32),)
                    with patch.object(distributions, "FixedSimplex", ReferenceSimplex):
                        old_metrics = original.update(sample)
                    new_metrics = optimized.update(sample)
                    for x, y in zip(old_metrics, new_metrics):
                        torch.testing.assert_close(torch.as_tensor(y), torch.as_tensor(x), rtol=1e-4, atol=2e-5)
                    for x, y in zip(original.actor.parameters(), optimized.actor.parameters()):
                        torch.testing.assert_close(y.grad, x.grad, rtol=2e-4, atol=2e-5)
                        torch.testing.assert_close(y, x, rtol=2e-4, atol=2e-5)
                    for x, y in zip(original.actor_optimizer.state.values(), optimized.actor_optimizer.state.values()):
                        for key in x:
                            torch.testing.assert_close(y[key], x[key], rtol=2e-4, atol=2e-5)
                    env.close()

    def test_normalizer_checkpoint_contains_all_statistics(self):
        for device in DEVICES:
            with self.subTest(device=device), tempfile.TemporaryDirectory() as tmp:
                norm = ValueNorm(1, device=torch.device(device))
                data = np.arange(31, dtype=np.float32).reshape(-1, 1)
                norm.update(data)
                self.assertEqual(set(norm.state_dict()), {"running_mean", "running_mean_sq", "debiasing_term"})
                path = Path(tmp)/"value_normalizer.pt"
                torch.save(norm.state_dict(), path)
                restored = ValueNorm(1)
                restored.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
                np.testing.assert_allclose(restored.denormalize(data), norm.denormalize(data), rtol=1e-6, atol=1e-6)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA host execution required")
    def test_hybrid_gae_matches_gpu_at_terminal_and_truncated_steps(self):
        config = configurations(steps=8000)["IC_HAPPO"]
        env, algo = SCUAVEnv(config["env_args"]), config["algo_args"]
        gpu_buffer = OnPolicyCriticBufferEP({**algo["train"], **algo["model"], **algo["algo"]}, env.share_observation_space[0])
        rng = np.random.default_rng(618)
        gpu_buffer.rewards[:] = rng.normal(0, .2, gpu_buffer.rewards.shape)
        gpu_buffer.value_preds[:] = rng.normal(0, 1, gpu_buffer.value_preds.shape)
        gpu_buffer.masks[50, 0] = 0
        gpu_buffer.masks[220, 3] = 0
        gpu_buffer.bad_masks[220, 3] = 0
        cpu_buffer = copy.deepcopy(gpu_buffer)
        norm_gpu = ValueNorm(1, device=torch.device("cuda"))
        norm_gpu.update(rng.normal(7, 3, (100, 1)).astype(np.float32))
        norm_cpu = ValueNorm(1)
        norm_cpu.load_state_dict(norm_gpu.state_dict(), strict=True)
        next_value = rng.normal(size=(10, 1)).astype(np.float32)
        gpu_buffer.compute_returns(next_value, norm_gpu)
        cpu_buffer.compute_returns(next_value, norm_cpu)
        np.testing.assert_allclose(cpu_buffer.returns, gpu_buffer.returns, rtol=2e-5, atol=2e-5)
        env.close()


if __name__ == "__main__":
    unittest.main()
