"""CPU rollout and GAE, GPU optimization; synchronize after every PPO update."""
import copy
import numpy as np
import torch
from harl.common.valuenorm import ValueNorm


def cpu_inference_copy(agent, network_name):
    result = copy.deepcopy(agent)
    result.device = torch.device("cpu")
    result.tpdv = dict(dtype=torch.float32, device=result.device)
    network = getattr(result, network_name)
    network.to(result.device)
    for module in network.modules():
        if hasattr(module, "tpdv"):
            module.tpdv = result.tpdv
    result.prep_rollout()
    return result


class HybridInferenceMixin:
    def configure_hybrid(self):
        self.hybrid = self.algo_args["train"].get("hybrid_inference", False)
        if not self.hybrid:
            return
        if self.device.type != "cuda" or self.value_normalizer is None:
            raise ValueError("Hybrid execution requires CUDA and ValueNorm")
        self.cpu_actors = [cpu_inference_copy(a, "actor") for a in self.actor]
        self.cpu_critic = cpu_inference_copy(self.critic, "critic")
        self.normalizer_cpu = ValueNorm(1, device=torch.device("cpu"))
        self.sync_inference_copies()

    @torch.no_grad()
    def sync_inference_copies(self):
        for src, dst in zip(self.actor, self.cpu_actors):
            dst.actor.load_state_dict(src.actor.state_dict(), strict=True)
        self.cpu_critic.critic.load_state_dict(self.critic.critic.state_dict(), strict=True)
        self.normalizer_cpu.load_state_dict(self.value_normalizer.state_dict(), strict=True)

    def collect(self, step):
        if not self.hybrid:
            return super().collect(step)
        actors, critic = self.actor, self.critic
        try:
            self.actor, self.critic = self.cpu_actors, self.cpu_critic
            return super().collect(step)
        finally:
            self.actor, self.critic = actors, critic

    def compute(self):
        if not self.hybrid:
            return super().compute()
        critic, normalizer = self.critic, self.value_normalizer
        try:
            self.critic, self.value_normalizer = self.cpu_critic, self.normalizer_cpu
            return super().compute()
        finally:
            self.critic, self.value_normalizer = critic, normalizer

    def train(self):
        result = super().train()
        if self.hybrid:
            self.sync_inference_copies()
        return result

    @torch.no_grad()
    def validate_hybrid(self):
        if not self.hybrid:
            return None
        max_error = 0.0
        for i, (gpu, cpu) in enumerate(zip(self.actor, self.cpu_actors)):
            for name, value in gpu.actor.state_dict().items():
                torch.testing.assert_close(value.cpu(), cpu.actor.state_dict()[name], rtol=0, atol=0)
            b = self.actor_buffer[i]
            args = (b.obs[0], b.rnn_states[0], b.masks[0], b.available_actions[0])
            gpu.prep_rollout()
            cpu.prep_rollout()
            ga, gl, _ = gpu.get_actions(*args, deterministic=True)
            ca, cl, _ = cpu.get_actions(*args, deterministic=True)
            torch.testing.assert_close(ga.cpu(), ca, rtol=1e-4, atol=2e-5)
            torch.testing.assert_close(gl.cpu(), cl, rtol=1e-4, atol=2e-5)
            # Compare density of the same sampled action, not device RNG streams.
            actions, _, _ = cpu.get_actions(*args, deterministic=False)
            values = (args[0], args[1], actions.numpy(), args[2], args[3], b.active_masks[0])
            gp, ge, _ = gpu.evaluate_actions(*values)
            cp, ce, _ = cpu.evaluate_actions(*values)
            torch.testing.assert_close(gp.cpu(), cp, rtol=1e-4, atol=2e-5)
            torch.testing.assert_close(ge.cpu(), ce, rtol=1e-4, atol=2e-5)
            max_error = max(max_error, float((gp.cpu()-cp).abs().max()))
        for name, value in self.critic.critic.state_dict().items():
            torch.testing.assert_close(value.cpu(), self.cpu_critic.critic.state_dict()[name], rtol=0, atol=0)
        self.critic.prep_rollout()
        self.cpu_critic.prep_rollout()
        b = self.critic_buffer
        args = (b.share_obs[0], b.rnn_states_critic[0], b.masks[0])
        gv, _ = self.critic.get_values(*args)
        cv, _ = self.cpu_critic.get_values(*args)
        torch.testing.assert_close(gv.cpu(), cv, rtol=1e-4, atol=2e-5)
        values = np.linspace(-2, 2, 21, dtype=np.float32).reshape(-1, 1)
        np.testing.assert_allclose(self.value_normalizer.denormalize(values),
                                   self.normalizer_cpu.denormalize(values), rtol=1e-5, atol=1e-5)
        return dict(synchronized_parameters_exact=True, deterministic_actions_close=True,
                    same_action_log_probability_max_abs_error=max_error,
                    critic_values_close=True, normalizer_denormalization_close=True)
