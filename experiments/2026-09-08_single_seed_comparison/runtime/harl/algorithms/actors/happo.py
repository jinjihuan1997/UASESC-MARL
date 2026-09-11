"""HAPPO algorithm."""
import numpy as np
import torch
import torch.nn as nn
from harl.utils.envs_tools import check
from harl.utils.models_tools import get_grad_norm
from harl.utils.ratio_tools import aggregate_action_ratio
from harl.algorithms.actors.on_policy_base import OnPolicyBase


def _to_float(x, default=0.0):
    try:
        if hasattr(x, "detach"):
            x = x.detach()
        if hasattr(x, "cpu"):
            x = x.cpu()
        if hasattr(x, "item"):
            return float(x.item())
        return float(x)
    except Exception:
        return float(default)


class HAPPO(OnPolicyBase):
    def __init__(self, args, obs_space, act_space, device=torch.device("cpu")):
        """Initialize HAPPO algorithm.
        Args:
            args: (dict) arguments.
            obs_space: (gym.spaces or list) observation space.
            act_space: (gym.spaces) action space.
            device: (torch.device) device to use for tensor operations.
        """
        super(HAPPO, self).__init__(args, obs_space, act_space, device)

        self.clip_param = args["clip_param"]
        self.ppo_epoch = args["ppo_epoch"]
        self.actor_num_mini_batch = args["actor_num_mini_batch"]
        self.entropy_coef = args["entropy_coef"]
        self.use_max_grad_norm = args["use_max_grad_norm"]
        self.max_grad_norm = args["max_grad_norm"]
        self.use_instruction_adv_norm = bool(args.get("use_instruction_adv_norm", False))

    def update(self, sample):
        """Update actor network.
        Args:
            sample: (Tuple) contains data batch with which to SC networks.
        Returns:
            policy_loss: (torch.Tensor) actor(policy) loss value.
            dist_entropy: (torch.Tensor) action entropies.
            actor_grad_norm: (torch.Tensor) gradient norm from actor SC.
            imp_weights: (torch.Tensor) importance sampling weights.
        """
        (
            obs_batch,
            rnn_states_batch,
            actions_batch,
            masks_batch,
            active_masks_batch,
            old_action_log_probs_batch,
            adv_targ,
            available_actions_batch,
            factor_batch,
        ) = sample

        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)
        factor_batch = check(factor_batch).to(**self.tpdv)
        adv_targ = torch.nan_to_num(adv_targ, nan=0.0, posinf=0.0, neginf=0.0)
        active_masks_batch = torch.nan_to_num(
            active_masks_batch, nan=0.0, posinf=0.0, neginf=0.0
        )
        factor_batch = torch.nan_to_num(factor_batch, nan=0.0, posinf=0.0, neginf=0.0)

        # Reshape to do evaluations for all steps in a single forward pass
        action_log_probs, dist_entropy, _ = self.evaluate_actions(
            obs_batch,
            rnn_states_batch,
            actions_batch,
            masks_batch,
            available_actions_batch,
            active_masks_batch,
        )

        # actor SC
        log_ratio = action_log_probs - old_action_log_probs_batch
        imp_weights = aggregate_action_ratio(log_ratio, self.action_aggregation, clip=20.0)
        imp_weights = torch.nan_to_num(imp_weights, nan=1.0, posinf=1e6, neginf=0.0)
        imp_weights = torch.clamp(imp_weights, min=0.0, max=1e3)
        surr1 = imp_weights * adv_targ
        surr2 = (
            torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param)
            * adv_targ
        )

        if self.use_policy_active_masks:
            active_mask_sum = active_masks_batch.sum().clamp(min=1e-6)
            policy_action_loss = (
                -torch.sum(factor_batch * torch.min(surr1, surr2), dim=-1, keepdim=True)
                * active_masks_batch
            ).sum() / active_mask_sum
        else:
            policy_action_loss = -torch.sum(
                factor_batch * torch.min(surr1, surr2), dim=-1, keepdim=True
            ).mean()

        policy_loss = policy_action_loss
        total_loss = policy_loss - dist_entropy * self.entropy_coef

        # Skip invalid update to avoid corrupting actor parameters.
        if not torch.isfinite(total_loss):
            imp_weights = torch.nan_to_num(
                imp_weights, nan=1.0, posinf=1e3, neginf=0.0
            ).detach()
            return policy_loss.detach(), dist_entropy.detach(), 0.0, imp_weights

        self.actor_optimizer.zero_grad()
        total_loss.backward()  # add entropy term

        if self.use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(
                self.actor.parameters(), self.max_grad_norm
            )
        else:
            actor_grad_norm = get_grad_norm(self.actor.parameters())
        actor_grad_norm = _to_float(actor_grad_norm, 0.0)
        if not np.isfinite(actor_grad_norm):
            self.actor_optimizer.zero_grad(set_to_none=True)
            imp_weights = torch.nan_to_num(
                imp_weights, nan=1.0, posinf=1e3, neginf=0.0
            ).detach()
            return policy_loss.detach(), dist_entropy.detach(), 0.0, imp_weights
        self.actor_optimizer.step()

        return policy_loss, dist_entropy, actor_grad_norm, imp_weights

    def train(self, actor_buffer, advantages, state_type):
        """Perform a training SC using minibatch GD.
        Args:
            actor_buffer: (OnPolicyActorBuffer) buffer containing training data related to actor.
            advantages: (np.ndarray) advantages.
            state_type: (str) type of state.
        Returns:
            train_info: (dict) contains information regarding training SC (e.g. loss, grad norms, etc).
        """
        train_info = {}
        train_info["policy_loss"] = 0
        train_info["dist_entropy"] = 0
        train_info["actor_grad_norm"] = 0
        train_info["ratio"] = 0

        if np.all(actor_buffer.active_masks[:-1] == 0.0):
            return train_info

        if state_type == "EP" and not self.use_instruction_adv_norm:
            advantages_copy = advantages.copy()
            advantages_copy[actor_buffer.active_masks[:-1] == 0.0] = np.nan
            mean_advantages = np.nanmean(advantages_copy)
            std_advantages = np.nanstd(advantages_copy)
            advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)

        for _ in range(self.ppo_epoch):
            if self.use_recurrent_policy:
                data_generator = actor_buffer.recurrent_generator_actor(
                    advantages, self.actor_num_mini_batch, self.data_chunk_length
                )
            elif self.use_naive_recurrent_policy:
                data_generator = actor_buffer.naive_recurrent_generator_actor(
                    advantages, self.actor_num_mini_batch
                )
            else:
                data_generator = actor_buffer.feed_forward_generator_actor(
                    advantages, self.actor_num_mini_batch
                )

            for sample in data_generator:
                policy_loss, dist_entropy, actor_grad_norm, imp_weights = self.update(
                    sample
                )

                train_info["policy_loss"] += policy_loss.item()
                train_info["dist_entropy"] += dist_entropy.item()
                train_info["actor_grad_norm"] += _to_float(actor_grad_norm, 0.0)
                train_info["ratio"] += _to_float(imp_weights.mean(), 1.0)

        num_updates = self.ppo_epoch * self.actor_num_mini_batch

        for k in train_info.keys():
            train_info[k] /= num_updates

        return train_info
