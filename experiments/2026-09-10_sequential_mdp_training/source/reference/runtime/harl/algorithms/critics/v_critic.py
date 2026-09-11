"""V Critic."""
import torch
import torch.nn as nn
from harl.utils.models_tools import (
    get_grad_norm,
    huber_loss,
    mse_loss,
    update_linear_schedule,
)
from harl.utils.envs_tools import check
from harl.models.value_function_models.v_net import VNet


class VCritic:
    """V Critic.
    Critic that learns a V-function.
    """

    def __init__(self, args, cent_obs_space, device=torch.device("cpu")):
        self.args = args
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)

        self.clip_param = args["clip_param"]
        self.critic_epoch = args["critic_epoch"]
        self.critic_num_mini_batch = args["critic_num_mini_batch"]
        self.data_chunk_length = args["data_chunk_length"]
        self.value_loss_coef = args["value_loss_coef"]
        self.max_grad_norm = args["max_grad_norm"]
        self.huber_delta = args["huber_delta"]

        self.use_recurrent_policy = args["use_recurrent_policy"]
        self.use_naive_recurrent_policy = args["use_naive_recurrent_policy"]
        self.use_max_grad_norm = args["use_max_grad_norm"]
        self.use_clipped_value_loss = args["use_clipped_value_loss"]
        self.use_huber_loss = args["use_huber_loss"]
        self.use_policy_active_masks = args["use_policy_active_masks"]
        self.instruction_conditioned_critic = bool(args.get("instruction_conditioned_critic", False))
        self.instruction_critic_type = str(args.get("instruction_critic_type", "single")).lower()
        self.n_instruction_modes = int(args.get("n_instruction_modes", 0))
        self.instruction_feature_dim = int(args.get("instruction_feature_dim", self.n_instruction_modes))
        self.instruction_feature_position = str(args.get("instruction_feature_position", "head")).lower()

        self.critic_lr = args["critic_lr"]
        self.opti_eps = args["opti_eps"]
        self.weight_decay = args["weight_decay"]

        self.share_obs_space = cent_obs_space

        self.critic = VNet(args, self.share_obs_space, self.device)

        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=self.critic_lr,
            eps=self.opti_eps,
            weight_decay=self.weight_decay,
        )

    def lr_decay(self, episode, episodes):
        """Decay the actor and critic learning rates.
        Args:
            episode: (int) current training episode.
            episodes: (int) total number of training episodes.
        """
        update_linear_schedule(self.critic_optimizer, episode, episodes, self.critic_lr)

    def get_values(self, cent_obs, rnn_states_critic, masks):
        """Get value function predictions.
        Args:
            cent_obs: (np.ndarray) centralized input to the critic.
            rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
            masks: (np.ndarray) denotes points at which RNN states should be reset.
        Returns:
            values: (torch.Tensor) value function predictions.
            rnn_states_critic: (torch.Tensor) updated critic network RNN states.
        """
        values, rnn_states_critic = self.critic(cent_obs, rnn_states_critic, masks)
        return values, rnn_states_critic

    def cal_value_loss(
        self, values, value_preds_batch, return_batch, value_normalizer=None
    ):
        """Calculate value function loss.
        Args:
            values: (torch.Tensor) value function predictions.
            value_preds_batch: (torch.Tensor) "old" value  predictions from data batch (used for value clip loss)
            return_batch: (torch.Tensor) reward to go returns.
            value_normalizer: (ValueNorm) normalize the rewards, denormalize critic outputs.
        Returns:
            value_loss: (torch.Tensor) value function loss.
        """
        value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(
            -self.clip_param, self.clip_param
        )
        if value_normalizer is not None:
            value_normalizer.update(return_batch)
            error_clipped = (
                value_normalizer.normalize(return_batch) - value_pred_clipped
            )
            error_original = value_normalizer.normalize(return_batch) - values
        else:
            error_clipped = return_batch - value_pred_clipped
            error_original = return_batch - values

        if self.use_huber_loss:
            value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
            value_loss_original = huber_loss(error_original, self.huber_delta)
        else:
            value_loss_clipped = mse_loss(error_clipped)
            value_loss_original = mse_loss(error_original)

        if self.use_clipped_value_loss:
            value_loss = torch.max(value_loss_original, value_loss_clipped)
        else:
            value_loss = value_loss_original

        value_loss = value_loss.mean()

        return value_loss

    def update(self, sample, value_normalizer=None):
        """Update critic network.
        Args:
            sample: (Tuple) contains data batch with which to SC networks.
            value_normalizer: (ValueNorm) normalize the rewards, denormalize critic outputs.
        Returns:
            value_loss: (torch.Tensor) value function loss.
            critic_grad_norm: (torch.Tensor) gradient norm from critic SC.
        """
        (
            share_obs_batch,
            rnn_states_critic_batch,
            value_preds_batch,
            return_batch,
            masks_batch,
        ) = sample

        value_preds_batch = check(value_preds_batch).to(**self.tpdv)
        return_batch = check(return_batch).to(**self.tpdv)

        values, _ = self.get_values(
            share_obs_batch, rnn_states_critic_batch, masks_batch
        )

        value_loss = self.cal_value_loss(
            values, value_preds_batch, return_batch, value_normalizer=value_normalizer
        )
        per_instruction_value_loss = self._per_instruction_value_loss(
            share_obs_batch,
            values,
            value_preds_batch,
            return_batch,
            value_normalizer,
        )

        self.critic_optimizer.zero_grad()

        (value_loss * self.value_loss_coef).backward()

        if self.use_max_grad_norm:
            critic_grad_norm = nn.utils.clip_grad_norm_(
                self.critic.parameters(), self.max_grad_norm
            )
        else:
            critic_grad_norm = get_grad_norm(self.critic.parameters())

        self.critic_optimizer.step()

        return value_loss, critic_grad_norm, per_instruction_value_loss

    def _per_instruction_value_loss(
        self,
        share_obs_batch,
        values,
        value_preds_batch,
        return_batch,
        value_normalizer=None,
    ):
        if not (
            self.instruction_conditioned_critic
            and self.n_instruction_modes > 0
            and share_obs_batch is not None
        ):
            return {}
        share_obs = check(share_obs_batch).to(**self.tpdv)
        if share_obs.shape[-1] < self.n_instruction_modes:
            return {}
        if (
            self.instruction_feature_position == "tail"
            and self.instruction_feature_dim >= self.n_instruction_modes
            and share_obs.shape[-1] >= self.instruction_feature_dim
        ):
            instruction_obs = share_obs[
                ...,
                -self.instruction_feature_dim : -self.instruction_feature_dim
                + self.n_instruction_modes,
            ]
        else:
            instruction_obs = share_obs[..., : self.n_instruction_modes]
        instruction_ids = torch.argmax(instruction_obs, dim=-1).long()
        if value_normalizer is not None:
            target = value_normalizer.normalize(return_batch)
        else:
            target = return_batch
        error = target - values
        if self.use_huber_loss:
            per_sample_loss = torch.where(
                torch.abs(error) <= self.huber_delta,
                0.5 * error.pow(2),
                self.huber_delta * (torch.abs(error) - 0.5 * self.huber_delta),
            )
        else:
            per_sample_loss = 0.5 * error.pow(2)
        out = {}
        flat_ids = instruction_ids.reshape(-1)
        flat_loss = per_sample_loss.reshape(-1)
        for idx in range(self.n_instruction_modes):
            mask = flat_ids == idx
            if torch.any(mask):
                out[f"value_loss_instruction_{idx}"] = float(flat_loss[mask].mean().detach().cpu().item())
                out[f"value_count_instruction_{idx}"] = int(mask.sum().detach().cpu().item())
        return out

    def train(self, critic_buffer, value_normalizer=None):
        """Perform a training SC using minibatch GD.
        Args:
            critic_buffer: (OnPolicyCriticBufferEP or OnPolicyCriticBufferFP) buffer containing training data related to critic.
            value_normalizer: (ValueNorm) normalize the rewards, denormalize critic outputs.
        Returns:
            train_info: (dict) contains information regarding training SC (e.g. loss, grad norms, etc).
        """

        train_info = {}

        train_info["value_loss"] = 0
        train_info["critic_grad_norm"] = 0

        for _ in range(self.critic_epoch):
            if self.use_recurrent_policy:
                data_generator = critic_buffer.recurrent_generator_critic(
                    self.critic_num_mini_batch, self.data_chunk_length
                )
            elif self.use_naive_recurrent_policy:
                data_generator = critic_buffer.naive_recurrent_generator_critic(
                    self.critic_num_mini_batch
                )
            else:
                data_generator = critic_buffer.feed_forward_generator_critic(
                    self.critic_num_mini_batch
                )

            for sample in data_generator:
                value_loss, critic_grad_norm, per_instruction_value_loss = self.update(
                    sample, value_normalizer=value_normalizer
                )

                train_info["value_loss"] += value_loss.item()
                train_info["critic_grad_norm"] += critic_grad_norm
                for key, value in per_instruction_value_loss.items():
                    train_info.setdefault(key, 0)
                    train_info[key] += value

        num_updates = self.critic_epoch * self.critic_num_mini_batch

        for k, _ in train_info.items():
            train_info[k] /= num_updates

        return train_info

    def prep_training(self):
        """Prepare for training."""
        self.critic.train()

    def prep_rollout(self):
        """Prepare for rollout."""
        self.critic.eval()
