import torch
import torch.nn as nn
from harl.models.base.cnn import CNNBase
from harl.models.base.mlp import MLPBase
from harl.models.base.rnn import RNNLayer
from harl.utils.envs_tools import check, get_shape_from_obs_space
from harl.utils.models_tools import init, get_init_method


class VNet(nn.Module):
    """V Network. Outputs value function predictions given global states."""

    def __init__(self, args, cent_obs_space, device=torch.device("cpu")):
        """Initialize VNet model.
        Args:
            args: (dict) arguments containing relevant model information.
            cent_obs_space: (gym.Space) centralized observation space.
            device: (torch.device) specifies the device to run on (cpu/gpu).
        """
        super(VNet, self).__init__()
        self.hidden_sizes = args["hidden_sizes"]
        self.initialization_method = args["initialization_method"]
        self.use_naive_recurrent_policy = args["use_naive_recurrent_policy"]
        self.use_recurrent_policy = args["use_recurrent_policy"]
        self.recurrent_n = args["recurrent_n"]
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.instruction_conditioned_critic = bool(args.get("instruction_conditioned_critic", False))
        self.instruction_critic_type = str(args.get("instruction_critic_type", "single")).lower()
        self.n_instruction_modes = int(args.get("n_instruction_modes", 0))
        self.instruction_feature_dim = int(args.get("instruction_feature_dim", 0))
        self.instruction_feature_position = str(args.get("instruction_feature_position", "head")).lower()
        self.use_multi_head_instruction_critic = (
            self.instruction_conditioned_critic
            and self.instruction_critic_type == "multi_head"
            and self.n_instruction_modes > 0
            and self.instruction_feature_dim > 0
        )
        init_method = get_init_method(self.initialization_method)

        cent_obs_shape = get_shape_from_obs_space(cent_obs_space)
        if (
            self.use_multi_head_instruction_critic
            and len(cent_obs_shape) == 1
            and cent_obs_shape[0] > self.instruction_feature_dim
        ):
            cent_obs_shape = (cent_obs_shape[0] - self.instruction_feature_dim,)
            self.strip_instruction_features = True
        else:
            self.strip_instruction_features = False
        base = CNNBase if len(cent_obs_shape) == 3 else MLPBase
        self.base = base(args, cent_obs_shape)

        if self.use_naive_recurrent_policy or self.use_recurrent_policy:
            self.rnn = RNNLayer(
                self.hidden_sizes[-1],
                self.hidden_sizes[-1],
                self.recurrent_n,
                self.initialization_method,
            )

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        if self.use_multi_head_instruction_critic:
            self.v_heads = nn.ModuleList(
                [init_(nn.Linear(self.hidden_sizes[-1], 1)) for _ in range(self.n_instruction_modes)]
            )
        else:
            self.v_out = init_(nn.Linear(self.hidden_sizes[-1], 1))

        self.to(device)

    def forward(self, cent_obs, rnn_states, masks):
        """Compute actions from the given inputs.
        Args:
            cent_obs: (np.ndarray / torch.Tensor) observation inputs into network.
            rnn_states: (np.ndarray / torch.Tensor) if RNN network, hidden states for RNN.
            masks: (np.ndarray / torch.Tensor) mask tensor denoting if RNN states should be reinitialized to zeros.
        Returns:
            values: (torch.Tensor) value function predictions.
            rnn_states: (torch.Tensor) updated RNN hidden states.
        """
        cent_obs = check(cent_obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        if self.use_multi_head_instruction_critic:
            if self.instruction_feature_position == "tail":
                instruction_obs = cent_obs[
                    ...,
                    -self.instruction_feature_dim : -self.instruction_feature_dim
                    + self.n_instruction_modes,
                ]
            else:
                instruction_obs = cent_obs[..., : self.n_instruction_modes]
            instruction_ids = torch.argmax(instruction_obs, dim=-1).long()
            instruction_ids = torch.clamp(instruction_ids, 0, self.n_instruction_modes - 1)
            if self.strip_instruction_features and self.instruction_feature_position == "tail":
                base_obs = cent_obs[..., : -self.instruction_feature_dim]
            elif self.strip_instruction_features:
                base_obs = cent_obs[..., self.instruction_feature_dim :]
            else:
                base_obs = cent_obs
        else:
            instruction_ids = None
            base_obs = cent_obs

        critic_features = self.base(base_obs)
        if self.use_naive_recurrent_policy or self.use_recurrent_policy:
            critic_features, rnn_states = self.rnn(critic_features, rnn_states, masks)
        if self.use_multi_head_instruction_critic:
            values_by_head = torch.cat([head(critic_features) for head in self.v_heads], dim=-1)
            values = values_by_head.gather(-1, instruction_ids.unsqueeze(-1))
        else:
            values = self.v_out(critic_features)

        return values, rnn_states
