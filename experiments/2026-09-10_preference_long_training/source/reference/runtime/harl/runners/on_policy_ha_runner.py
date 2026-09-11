"""Runner for on-policy HARL algorithms."""
import numpy as np
import torch
from harl.utils.trans_tools import _t2n
from harl.utils.ratio_tools import aggregate_action_ratio
from harl.runners.on_policy_base_runner import OnPolicyBaseRunner


class OnPolicyHARunner(OnPolicyBaseRunner):
    """Runner for on-policy HA algorithms."""

    def _instruction_ids_from_share_obs(self):
        n_modes = int(self.algo_args["algo"].get("n_instruction_modes", 0))
        if n_modes <= 0:
            return None
        share_obs = self.critic_buffer.share_obs[:-1]
        if share_obs.shape[-1] < n_modes:
            return None
        feature_dim = int(self.algo_args["algo"].get("instruction_feature_dim", n_modes))
        position = str(self.algo_args["algo"].get("instruction_feature_position", "head")).lower()
        if position == "tail" and feature_dim >= n_modes and share_obs.shape[-1] >= feature_dim:
            instruction_obs = share_obs[..., -feature_dim : -feature_dim + n_modes]
        else:
            instruction_obs = share_obs[..., :n_modes]
        return np.argmax(instruction_obs, axis=-1)

    def _instruction_active_masks(self, advantages):
        if self.state_type == "FP":
            active_masks_collector = [
                self.actor_buffer[i].active_masks for i in range(self.num_agents)
            ]
            return np.stack(active_masks_collector, axis=2)[:-1]
        return np.minimum.reduce(
            [self.actor_buffer[i].active_masks[:-1] for i in range(self.num_agents)]
        )

    def _normalize_advantages(self, advantages):
        if not self.algo_args["algo"].get("use_instruction_adv_norm", False):
            return advantages, {}
        n_modes = int(self.algo_args["algo"].get("n_instruction_modes", 0))
        ids = self._instruction_ids_from_share_obs()
        if ids is None or n_modes <= 0:
            return advantages, {}
        min_count = int(self.algo_args["algo"].get("instruction_adv_norm_min_count", 8))
        active_masks = self._instruction_active_masks(advantages)
        ids = ids[..., None]
        if advantages.shape != active_masks.shape:
            ids = np.broadcast_to(ids, advantages.shape)
            if active_masks.shape != advantages.shape:
                active_masks = np.broadcast_to(active_masks, advantages.shape)
        valid = active_masks > 0.0
        global_values = advantages[valid]
        if global_values.size == 0:
            return advantages, {}
        global_mean = float(np.mean(global_values))
        global_std = float(np.std(global_values))
        normalized = advantages.copy()
        stats = {
            "adv_global_mean_before": global_mean,
            "adv_global_std_before": global_std,
        }
        for idx in range(n_modes):
            mask = (ids == idx) & valid
            values = advantages[mask]
            if values.size >= min_count:
                mean = float(np.mean(values))
                std = float(np.std(values))
                fallback = 0.0
            else:
                mean = global_mean
                std = global_std
                fallback = 1.0
            normalized[ids == idx] = (advantages[ids == idx] - mean) / (std + 1e-5)
            after_values = normalized[mask]
            stats[f"adv_instruction_{idx}_count"] = int(values.size)
            stats[f"adv_instruction_{idx}_fallback_global"] = fallback
            stats[f"adv_instruction_{idx}_mean_before"] = float(np.mean(values)) if values.size else 0.0
            stats[f"adv_instruction_{idx}_std_before"] = float(np.std(values)) if values.size else 0.0
            stats[f"adv_instruction_{idx}_mean_after"] = float(np.mean(after_values)) if after_values.size else 0.0
            stats[f"adv_instruction_{idx}_std_after"] = float(np.std(after_values)) if after_values.size else 0.0
        return normalized, stats

    def train(self):
        """Train the model."""
        actor_train_infos = []

        # factor is used for considering updates made by previous agents
        factor = np.ones(
            (
                self.algo_args["train"]["episode_length"],
                self.algo_args["train"]["n_rollout_threads"],
                1,
            ),
            dtype=np.float32,
        )

        # compute advantages
        if self.value_normalizer is not None:
            advantages = self.critic_buffer.returns[
                :-1
            ] - self.value_normalizer.denormalize(self.critic_buffer.value_preds[:-1])
        else:
            advantages = (
                self.critic_buffer.returns[:-1] - self.critic_buffer.value_preds[:-1]
            )

        instruction_adv_stats = {}
        if self.algo_args["algo"].get("use_instruction_adv_norm", False):
            advantages, instruction_adv_stats = self._normalize_advantages(advantages)

        # normalize advantages for FP
        if self.state_type == "FP" and not self.algo_args["algo"].get("use_instruction_adv_norm", False):
            active_masks_collector = [
                self.actor_buffer[i].active_masks for i in range(self.num_agents)
            ]
            active_masks_array = np.stack(active_masks_collector, axis=2)
            advantages_copy = advantages.copy()
            advantages_copy[active_masks_array[:-1] == 0.0] = np.nan
            mean_advantages = np.nanmean(advantages_copy)
            std_advantages = np.nanstd(advantages_copy)
            advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)

        if self.fixed_order:
            agent_order = list(range(self.num_agents))
        else:
            agent_order = list(torch.randperm(self.num_agents).numpy())
        for agent_id in agent_order:
            self.actor_buffer[agent_id].update_factor(
                factor
            )  # current actor save factor

            # the following reshaping combines the first two dimensions (i.e. episode_length and n_rollout_threads) to form a batch
            available_actions = (
                None
                if self.actor_buffer[agent_id].available_actions is None
                else self.actor_buffer[agent_id]
                .available_actions[:-1]
                .reshape(-1, *self.actor_buffer[agent_id].available_actions.shape[2:])
            )

            # compute action log probs for the actor before SC.
            old_actions_logprob, _, _ = self.actor[agent_id].evaluate_actions(
                self.actor_buffer[agent_id]
                .obs[:-1]
                .reshape(-1, *self.actor_buffer[agent_id].obs.shape[2:]),
                self.actor_buffer[agent_id]
                .rnn_states[0:1]
                .reshape(-1, *self.actor_buffer[agent_id].rnn_states.shape[2:]),
                self.actor_buffer[agent_id].actions.reshape(
                    -1, *self.actor_buffer[agent_id].actions.shape[2:]
                ),
                self.actor_buffer[agent_id]
                .masks[:-1]
                .reshape(-1, *self.actor_buffer[agent_id].masks.shape[2:]),
                available_actions,
                self.actor_buffer[agent_id]
                .active_masks[:-1]
                .reshape(-1, *self.actor_buffer[agent_id].active_masks.shape[2:]),
            )

            # SC actor
            if self.state_type == "EP":
                actor_train_info = self.actor[agent_id].train(
                    self.actor_buffer[agent_id], advantages.copy(), "EP"
                )
            elif self.state_type == "FP":
                actor_train_info = self.actor[agent_id].train(
                    self.actor_buffer[agent_id], advantages[:, :, agent_id].copy(), "FP"
                )

            # compute action log probs for updated agent
            new_actions_logprob, _, _ = self.actor[agent_id].evaluate_actions(
                self.actor_buffer[agent_id]
                .obs[:-1]
                .reshape(-1, *self.actor_buffer[agent_id].obs.shape[2:]),
                self.actor_buffer[agent_id]
                .rnn_states[0:1]
                .reshape(-1, *self.actor_buffer[agent_id].rnn_states.shape[2:]),
                self.actor_buffer[agent_id].actions.reshape(
                    -1, *self.actor_buffer[agent_id].actions.shape[2:]
                ),
                self.actor_buffer[agent_id]
                .masks[:-1]
                .reshape(-1, *self.actor_buffer[agent_id].masks.shape[2:]),
                available_actions,
                self.actor_buffer[agent_id]
                .active_masks[:-1]
                .reshape(-1, *self.actor_buffer[agent_id].active_masks.shape[2:]),
            )

            # SC factor for next agent
            ratio = aggregate_action_ratio(
                new_actions_logprob - old_actions_logprob,
                self.action_aggregation,
                clip=20.0,
            )
            ratio = torch.nan_to_num(ratio, nan=1.0, posinf=1e6, neginf=0.0)
            ratio = torch.clamp(ratio, min=0.0, max=1e3)
            factor = factor * _t2n(
                ratio.reshape(
                    self.algo_args["train"]["episode_length"],
                    self.algo_args["train"]["n_rollout_threads"],
                    1,
                )
            )
            factor = np.nan_to_num(factor, nan=1.0, posinf=1e6, neginf=0.0)
            factor = np.clip(factor, 0.0, 1e3)
            actor_train_infos.append(actor_train_info)

        # SC critic
        critic_train_info = self.critic.train(self.critic_buffer, self.value_normalizer)
        critic_train_info.update(instruction_adv_stats)

        return actor_train_infos, critic_train_info
