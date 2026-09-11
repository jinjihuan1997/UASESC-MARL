import torch
import torch.nn as nn

from harl.models.base.distributions import (
    Categorical,
    DiagGaussian,
    FixedCategorical,
    Simplex,
)
from harl.utils.models_tools import get_init_method, init


class GroupedCategorical(nn.Module):
    """One linear head that outputs `num_groups` categorical distributions."""

    def __init__(self, inputs_dim, num_groups, group_size, initialization_method, gain):
        super().__init__()
        self.num_groups = int(num_groups)
        self.group_size = int(group_size)
        init_method = get_init_method(initialization_method)

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain)

        self.linear = init_(nn.Linear(inputs_dim, self.num_groups * self.group_size))

    def get_logits(self, x):
        return self.linear(x).view(-1, self.num_groups, self.group_size)

    def forward(self, x, available_actions=None):
        logits = self.get_logits(x)
        if available_actions is not None:
            mask = available_actions > 0.5
            valid = mask.sum(dim=-1, keepdim=True) > 0
            mask = torch.where(valid, mask, torch.ones_like(mask, dtype=torch.bool))
            logits = logits.masked_fill(~mask, -1e10)
        return FixedCategorical(logits=logits)


class HybridActionDistribution:
    """Container for hybrid categorical/gaussian action components."""

    def __init__(self, components, action_dim):
        self.components = list(components)
        self.action_dim = int(action_dim)

    @staticmethod
    def _parts(comp):
        if "parts" in comp:
            return list(comp["parts"])
        return [{"start": int(comp["start"]), "stop": int(comp["stop"])}]

    @classmethod
    def _same_layout(cls, src, tgt):
        src_parts = cls._parts(src)
        tgt_parts = cls._parts(tgt)
        if len(src_parts) != len(tgt_parts):
            return False
        if src["kind"] != tgt["kind"]:
            return False
        for p_src, p_tgt in zip(src_parts, tgt_parts):
            if int(p_src["start"]) != int(p_tgt["start"]) or int(p_src["stop"]) != int(
                p_tgt["stop"]
            ):
                return False
        return True

    @staticmethod
    def _scatter_block(out, parts, block):
        offset = 0
        for part in parts:
            start, stop = int(part["start"]), int(part["stop"])
            width = stop - start
            out[:, start:stop] = block[:, offset : offset + width]
            offset += width
        return out

    def kl_divergence(self, other):
        if len(self.components) != len(other.components):
            raise ValueError("Hybrid action component count mismatch.")
        batch_size = int(self.components[0]["batch_size"]) if self.components else 0
        device = self.components[0]["device"] if self.components else torch.device("cpu")
        dtype = self.components[0]["dtype"] if self.components else torch.float32
        out = torch.zeros((batch_size, self.action_dim), dtype=dtype, device=device)
        for src, tgt in zip(self.components, other.components):
            if not self._same_layout(src, tgt):
                raise ValueError("Hybrid action component layout mismatch.")
            parts = self._parts(src)
            total_dim = int(sum(int(part["stop"]) - int(part["start"]) for part in parts))
            if src["kind"] == "categorical":
                kl_vals = torch.distributions.kl_divergence(src["dist"], tgt["dist"])  # (B, G)
                group_size = int(src["group_size"])
                kl_block = (
                    kl_vals.unsqueeze(-1)
                    .expand(-1, -1, group_size)
                    .reshape(batch_size, total_dim)
                    / float(max(group_size, 1))
                )
                self._scatter_block(out, parts, kl_block)
            elif src["kind"] == "simplex":
                kl_vals = src["dist"].kl_divergence(tgt["dist"]).unsqueeze(-1)
                mask = src.get("mask", None)
                if mask is None:
                    mask = tgt.get("mask", None)
                elif tgt.get("mask", None) is not None:
                    mask = mask * tgt["mask"].to(dtype=mask.dtype, device=mask.device)
                if mask is None:
                    active = torch.ones(
                        (batch_size, total_dim), dtype=dtype, device=device
                    )
                else:
                    active = mask.to(dtype=dtype, device=device)
                active_count = active.sum(dim=-1, keepdim=True).clamp(min=1.0)
                kl_block = active * (kl_vals.to(dtype=dtype) / active_count)
                self._scatter_block(out, parts, kl_block)
            else:
                # Reuse the closed-form diagonal Gaussian KL already used by TRPO.
                var_ratio = (src["dist"].scale.to(torch.float64) / tgt["dist"].scale.to(torch.float64)).pow(2)
                t1 = (
                    (src["dist"].loc.to(torch.float64) - tgt["dist"].loc.to(torch.float64))
                    / tgt["dist"].scale.to(torch.float64)
                ).pow(2)
                kl_vals = 0.5 * (var_ratio + t1 - 1.0 - var_ratio.log())
                mask = src.get("mask", None)
                if mask is None:
                    mask = tgt.get("mask", None)
                elif tgt.get("mask", None) is not None:
                    mask = mask * tgt["mask"].to(dtype=mask.dtype, device=mask.device)
                if mask is not None:
                    kl_vals = kl_vals * mask.to(dtype=kl_vals.dtype, device=kl_vals.device)
                self._scatter_block(out, parts, kl_vals.to(dtype=dtype))
        return out


class HybridBoxACTLayer(nn.Module):
    """Hybrid action head for UAV-ESCS SC: categorical sub-groups + Gaussian slices."""

    def __init__(self, spec, inputs_dim, initialization_method, gain, args=None):
        super().__init__()
        self.spec = spec
        self.action_dim = int(spec["action_dim"])
        self.categorical_specs = list(spec.get("categorical_groups", []))
        self.continuous_specs = self._normalize_continuous_specs(spec)

        self.categorical_heads = nn.ModuleList(
            [
                GroupedCategorical(
                    inputs_dim,
                    item["num_groups"],
                    item["group_size"],
                    initialization_method,
                    gain,
                )
                for item in self.categorical_specs
            ]
        )
        self.continuous_heads = nn.ModuleList(
            [
                (
                    Simplex(
                        inputs_dim,
                        int(item["dim"]),
                        initialization_method,
                        gain,
                        args,
                    )
                    if str(item.get("kind", "gaussian")).lower() == "simplex"
                    else DiagGaussian(
                        inputs_dim,
                        int(item["dim"]),
                        initialization_method,
                        gain,
                        args,
                    )
                )
                for item in self.continuous_specs
            ]
        )

    @staticmethod
    def _normalize_continuous_specs(spec):
        groups = list(spec.get("continuous_groups", []))
        if not groups:
            groups = list(spec.get("continuous_slices", []))
        normalized = []
        for item in groups:
            parts = item.get("parts", None)
            if not parts:
                parts = [
                    {
                        "name": str(item.get("name", "")),
                        "start": int(item["start"]),
                        "stop": int(item["stop"]),
                    }
                ]
            parts = [
                {
                    "name": str(part.get("name", item.get("name", ""))),
                    "start": int(part["start"]),
                    "stop": int(part["stop"]),
                }
                for part in parts
            ]
            normalized.append(
                {
                    "name": str(item.get("name", parts[0]["name"])),
                    "kind": str(item.get("kind", "gaussian")).lower(),
                    "parts": parts,
                    "dim": int(
                        sum(int(part["stop"]) - int(part["start"]) for part in parts)
                    ),
                }
            )
        return normalized

    def _empty_outputs(self, x):
        batch = x.shape[0]
        actions = torch.zeros((batch, self.action_dim), dtype=x.dtype, device=x.device)
        action_log_probs = torch.zeros_like(actions)
        return actions, action_log_probs

    def _prepare_available_actions(self, available_actions, x):
        if available_actions is None:
            return None
        mask = available_actions.to(device=x.device, dtype=x.dtype)
        if mask.dim() == 1:
            mask = mask.unsqueeze(0)
        if mask.shape[1] < self.action_dim:
            pad = torch.ones(
                (mask.shape[0], self.action_dim - mask.shape[1]),
                dtype=mask.dtype,
                device=mask.device,
            )
            mask = torch.cat([mask, pad], dim=-1)
        elif mask.shape[1] > self.action_dim:
            mask = mask[:, : self.action_dim]
        return mask

    @staticmethod
    def _slice_mask(available_actions, start, stop, shape, device):
        if available_actions is None:
            return None
        mask = available_actions[:, start:stop]
        return (mask > 0.5).reshape(shape).to(device=device)

    @staticmethod
    def _parts_total_dim(parts):
        return int(sum(int(part["stop"]) - int(part["start"]) for part in parts))

    def _read_group_block(self, tensor, spec):
        blocks = []
        for part in spec["parts"]:
            start, stop = int(part["start"]), int(part["stop"])
            blocks.append(tensor[:, start:stop])
        return torch.cat(blocks, dim=-1)

    def _write_group_block(self, target, spec, block):
        offset = 0
        for part in spec["parts"]:
            start, stop = int(part["start"]), int(part["stop"])
            width = stop - start
            target[:, start:stop] = block[:, offset : offset + width]
            offset += width

    def _group_mask(self, spec, slice_masks, available_actions, batch_size, device):
        part_masks = []
        saw_mask = False
        for part in spec["parts"]:
            start, stop = int(part["start"]), int(part["stop"])
            width = stop - start
            mask = slice_masks.get(part["name"], None)
            if mask is None:
                mask = self._slice_mask(
                    available_actions, start, stop, (batch_size, width), device
                )
            if mask is not None:
                saw_mask = True
                part_masks.append(mask.to(dtype=torch.bool, device=device))
            else:
                part_masks.append(torch.ones((batch_size, width), dtype=torch.bool, device=device))
        if not saw_mask:
            return None
        return torch.cat(part_masks, dim=-1)

    @staticmethod
    def _spread_event_logp(event_logp, mask, total_dim, dtype, device):
        event_logp = event_logp.reshape(-1, 1).to(dtype=dtype, device=device)
        if mask is None:
            return event_logp.expand(-1, total_dim) / float(max(total_dim, 1))
        active = mask.to(dtype=dtype, device=device)
        active_count = active.sum(dim=-1, keepdim=True).clamp(min=1.0)
        return active * (event_logp / active_count)

    def _decode_categorical_choices_from_action(self, action):
        choices = {}
        batch = action.shape[0]
        for spec in self.categorical_specs:
            start, stop = int(spec["start"]), int(spec["stop"])
            num_groups, group_size = int(spec["num_groups"]), int(spec["group_size"])
            raw = action[:, start:stop].reshape(batch, num_groups, group_size)
            choices[str(spec.get("name", ""))] = torch.argmax(raw, dim=-1)
        return choices

    def _continuous_masks(self, available_actions, categorical_choices, batch_size, device):
        masks = {}
        for spec in self.continuous_specs:
            for part in spec["parts"]:
                start, stop = int(part["start"]), int(part["stop"])
                masks[str(part.get("name", ""))] = self._slice_mask(
                    available_actions,
                    start,
                    stop,
                    (batch_size, stop - start),
                    device,
                )
        return masks

    def _entropy_reduce(self, entropy_per_sample, active_masks):
        if active_masks is not None:
            mask = active_masks.squeeze(-1)
            active_sum = mask.sum().clamp(min=1e-6)
            return (entropy_per_sample * mask).sum() / active_sum
        return entropy_per_sample.mean()

    def forward(self, x, available_actions=None, deterministic=False):
        available_actions = self._prepare_available_actions(available_actions, x)
        actions, action_log_probs = self._empty_outputs(x)
        components = []
        categorical_choices = {}

        for spec, head in zip(self.categorical_specs, self.categorical_heads):
            start, stop = int(spec["start"]), int(spec["stop"])
            num_groups, group_size = int(spec["num_groups"]), int(spec["group_size"])
            group_mask = None
            if available_actions is not None:
                group_mask = available_actions[:, start:stop].reshape(x.shape[0], num_groups, group_size)
            dist = head(x, group_mask)
            choice = dist.mode() if deterministic else dist.sample()  # (B, G, 1)
            choice = choice.long()
            one_hot = torch.full(
                (x.shape[0], num_groups, group_size),
                -1.0,
                dtype=x.dtype,
                device=x.device,
            )
            one_hot.scatter_(-1, choice, 1.0)
            actions[:, start:stop] = one_hot.reshape(x.shape[0], stop - start)
            categorical_choices[str(spec.get("name", ""))] = choice.squeeze(-1)

            logp = dist.log_probs(choice).squeeze(-1)  # (B, G)
            logp_block = (
                logp.unsqueeze(-1)
                .expand(-1, -1, group_size)
                .reshape(x.shape[0], stop - start)
                / float(max(group_size, 1))
            )
            action_log_probs[:, start:stop] = logp_block
            components.append(
                {
                    "kind": "categorical",
                    "dist": dist,
                    "start": start,
                    "stop": stop,
                    "group_size": group_size,
                    "batch_size": x.shape[0],
                    "device": x.device,
                    "dtype": x.dtype,
                }
            )

        continuous_masks = self._continuous_masks(
            available_actions,
            categorical_choices,
            x.shape[0],
            x.device,
        )
        for spec, head in zip(self.continuous_specs, self.continuous_heads):
            cont_mask = self._group_mask(
                spec, continuous_masks, available_actions, x.shape[0], x.device
            )
            kind = str(spec.get("kind", "gaussian")).lower()
            if kind == "simplex":
                dist = head(
                    x,
                    cont_mask.to(dtype=x.dtype) if cont_mask is not None else None,
                )
                sample_prob = dist.mode() if deterministic else dist.sample()
                raw_group = 2.0 * sample_prob - 1.0
                if cont_mask is not None:
                    raw_group = torch.where(
                        cont_mask, raw_group, -torch.ones_like(raw_group)
                    )
                logp_block = self._spread_event_logp(
                    dist.log_probs(sample_prob),
                    cont_mask,
                    int(spec["dim"]),
                    x.dtype,
                    x.device,
                )
                self._write_group_block(actions, spec, raw_group)
                self._write_group_block(action_log_probs, spec, logp_block)
            else:
                dist = head(x, None)
                sample_raw = dist.mode() if deterministic else dist.sample()
                logp_raw = dist.log_probs(sample_raw)
                if cont_mask is not None:
                    sample_raw = torch.where(
                        cont_mask, sample_raw, torch.zeros_like(sample_raw)
                    )
                    logp_raw = torch.where(
                        cont_mask, logp_raw, torch.zeros_like(logp_raw)
                    )
                self._write_group_block(actions, spec, sample_raw)
                self._write_group_block(action_log_probs, spec, logp_raw)
            components.append(
                {
                    "kind": kind,
                    "dist": dist,
                    "parts": spec["parts"],
                    "mask": cont_mask.to(dtype=x.dtype) if cont_mask is not None else None,
                    "batch_size": x.shape[0],
                    "device": x.device,
                    "dtype": x.dtype,
                }
            )

        return actions, action_log_probs, HybridActionDistribution(components, self.action_dim)

    def get_logits(self, x, available_actions=None):
        available_actions = self._prepare_available_actions(available_actions, x)
        logits = torch.zeros((x.shape[0], self.action_dim), dtype=x.dtype, device=x.device)
        for spec, head in zip(self.categorical_specs, self.categorical_heads):
            start, stop = int(spec["start"]), int(spec["stop"])
            num_groups, group_size = int(spec["num_groups"]), int(spec["group_size"])
            group_mask = None
            if available_actions is not None:
                group_mask = available_actions[:, start:stop].reshape(x.shape[0], num_groups, group_size)
            dist = head(x, group_mask)
            logits[:, start:stop] = dist.logits.reshape(x.shape[0], stop - start)
        for spec, head in zip(self.continuous_specs, self.continuous_heads):
            out_block = (
                head.fc_logits(x)
                if str(spec.get("kind", "gaussian")).lower() == "simplex"
                else head.fc_mean(x)
            )
            self._write_group_block(logits, spec, out_block)
        return logits

    def evaluate_actions(self, x, action, available_actions=None, active_masks=None):
        available_actions = self._prepare_available_actions(available_actions, x)
        action = action.to(dtype=x.dtype)
        action_log_probs = torch.zeros_like(action)
        entropy_per_sample = torch.zeros((x.shape[0],), dtype=x.dtype, device=x.device)
        components = []
        categorical_choices = {}

        for spec, head in zip(self.categorical_specs, self.categorical_heads):
            start, stop = int(spec["start"]), int(spec["stop"])
            num_groups, group_size = int(spec["num_groups"]), int(spec["group_size"])
            group_mask = None
            if available_actions is not None:
                group_mask = available_actions[:, start:stop].reshape(x.shape[0], num_groups, group_size)
            dist = head(x, group_mask)
            raw = action[:, start:stop].reshape(x.shape[0], num_groups, group_size)
            choice = torch.argmax(raw, dim=-1, keepdim=True)
            categorical_choices[str(spec.get("name", ""))] = choice.squeeze(-1)
            logp = dist.log_probs(choice).squeeze(-1)
            logp_block = (
                logp.unsqueeze(-1)
                .expand(-1, -1, group_size)
                .reshape(x.shape[0], stop - start)
                / float(max(group_size, 1))
            )
            action_log_probs[:, start:stop] = logp_block
            entropy_per_sample = entropy_per_sample + dist.entropy().sum(dim=-1)
            components.append(
                {
                    "kind": "categorical",
                    "dist": dist,
                    "start": start,
                    "stop": stop,
                    "group_size": group_size,
                    "batch_size": x.shape[0],
                    "device": x.device,
                    "dtype": x.dtype,
                }
            )

        continuous_masks = self._continuous_masks(
            available_actions,
            categorical_choices,
            x.shape[0],
            x.device,
        )
        for spec, head in zip(self.continuous_specs, self.continuous_heads):
            cont_mask = self._group_mask(
                spec, continuous_masks, available_actions, x.shape[0], x.device
            )
            kind = str(spec.get("kind", "gaussian")).lower()
            act_group = self._read_group_block(action, spec)
            if kind == "simplex":
                dist = head(
                    x,
                    cont_mask.to(dtype=x.dtype) if cont_mask is not None else None,
                )
                prob_group = 0.5 * (act_group + 1.0)
                if cont_mask is not None:
                    prob_group = torch.where(
                        cont_mask, prob_group, torch.zeros_like(prob_group)
                    )
                logp_block = self._spread_event_logp(
                    dist.log_probs(prob_group),
                    cont_mask,
                    int(spec["dim"]),
                    x.dtype,
                    x.device,
                )
                self._write_group_block(action_log_probs, spec, logp_block)
                if cont_mask is not None:
                    entropy_per_sample = entropy_per_sample + dist.entropy() * cont_mask.any(dim=-1).to(dtype=x.dtype)
                else:
                    entropy_per_sample = entropy_per_sample + dist.entropy().to(dtype=x.dtype)
            else:
                dist = head(x, None)
                logp_raw = dist.log_probs(act_group)
                raw_entropy = torch.distributions.Normal.entropy(dist)
                if cont_mask is not None:
                    logp_raw = torch.where(
                        cont_mask, logp_raw, torch.zeros_like(logp_raw)
                    )
                    entropy_per_sample = entropy_per_sample + (
                        raw_entropy * cont_mask.to(dtype=raw_entropy.dtype)
                    ).sum(dim=-1)
                else:
                    entropy_per_sample = entropy_per_sample + raw_entropy.sum(dim=-1)
                self._write_group_block(action_log_probs, spec, logp_raw)
            components.append(
                {
                    "kind": kind,
                    "dist": dist,
                    "parts": spec["parts"],
                    "mask": cont_mask.to(dtype=x.dtype) if cont_mask is not None else None,
                    "batch_size": x.shape[0],
                    "device": x.device,
                    "dtype": x.dtype,
                }
            )

        dist_entropy = self._entropy_reduce(entropy_per_sample, active_masks)
        return action_log_probs, dist_entropy, HybridActionDistribution(components, self.action_dim)


class ACTLayer(nn.Module):
    """MLP Module to compute actions."""

    def __init__(
        self, action_space, inputs_dim, initialization_method, gain, args=None
    ):
        """Initialize ACTLayer.
        Args:
            action_space: (gym.Space) action space.
            inputs_dim: (int) dimension of network input.
            initialization_method: (str) initialization method.
            gain: (float) gain of the output layer of the network.
            args: (dict) arguments relevant to the network.
        """
        super(ACTLayer, self).__init__()
        self.action_type = action_space.__class__.__name__
        self.multidiscrete_action = False
        self.hybrid_box_action = False

        hybrid_spec = getattr(action_space, "hybrid_action_spec", None)
        if hybrid_spec is not None:
            self.hybrid_box_action = True
            self.action_dim = int(hybrid_spec["action_dim"])
            self.action_out = HybridBoxACTLayer(
                hybrid_spec, inputs_dim, initialization_method, gain, args
            )
        elif action_space.__class__.__name__ == "Discrete":
            action_dim = action_space.n
            self.action_out = Categorical(
                inputs_dim, action_dim, initialization_method, gain
            )
        elif action_space.__class__.__name__ == "Box":
            action_dim = action_space.shape[0]
            self.action_dim = int(action_dim)
            self.action_out = DiagGaussian(
                inputs_dim, action_dim, initialization_method, gain, args
            )
        elif action_space.__class__.__name__ == "MultiDiscrete":
            self.multidiscrete_action = True
            action_dims = action_space.nvec
            action_outs = []
            for action_dim in action_dims:
                action_outs.append(
                    Categorical(inputs_dim, action_dim, initialization_method, gain)
                )
            self.action_outs = nn.ModuleList(action_outs)

    def forward(self, x, available_actions=None, deterministic=False):
        """Compute actions and action logprobs from given input."""

        if self.hybrid_box_action:
            actions, action_log_probs, _ = self.action_out(
                x, available_actions, deterministic
            )
            return actions, action_log_probs

        if self.multidiscrete_action:
            actions = []
            action_log_probs = []
            for action_out in self.action_outs:
                action_distribution = action_out(x, available_actions)
                action = (
                    action_distribution.mode()
                    if deterministic
                    else action_distribution.sample()
                )
                action_log_prob = action_distribution.log_probs(action)
                actions.append(action)
                action_log_probs.append(action_log_prob)
            actions = torch.cat(actions, dim=-1)
            action_log_probs = torch.cat(action_log_probs, dim=-1).sum(
                dim=-1, keepdim=True
            )
        else:
            action_distribution = self.action_out(x, available_actions)
            actions = (
                action_distribution.mode()
                if deterministic
                else action_distribution.sample()
            )
            action_log_probs = action_distribution.log_probs(actions)

        return actions, action_log_probs

    def get_logits(self, x, available_actions=None):
        """Get action logits from inputs."""
        if self.hybrid_box_action:
            return self.action_out.get_logits(x, available_actions)
        if self.multidiscrete_action:
            action_logits = []
            for action_out in self.action_outs:
                action_distribution = action_out(x, available_actions)
                action_logits.append(action_distribution.logits)
        else:
            action_distribution = self.action_out(x, available_actions)
            action_logits = action_distribution.logits

        return action_logits

    def evaluate_actions(self, x, action, available_actions=None, active_masks=None):
        """Compute action log probability, distribution entropy, and action distribution."""
        if self.hybrid_box_action:
            return self.action_out.evaluate_actions(
                x, action, available_actions, active_masks
            )

        if self.multidiscrete_action:
            action = torch.transpose(action, 0, 1)
            action_log_probs = []
            dist_entropy = []
            for action_out, act in zip(self.action_outs, action):
                action_distribution = action_out(x)
                action_log_probs.append(
                    action_distribution.log_probs(act.unsqueeze(-1))
                )
                if active_masks is not None:
                    active_sum = active_masks.sum().clamp(min=1e-6)
                    dist_entropy.append(
                        (action_distribution.entropy() * active_masks)
                        / active_sum
                    )
                else:
                    dist_entropy.append(
                        action_distribution.entropy() / action_log_probs[-1].size(0)
                    )
            action_log_probs = torch.cat(action_log_probs, dim=-1).sum(
                dim=-1, keepdim=True
            )
            dist_entropy = (
                torch.cat(dist_entropy, dim=-1).sum(dim=-1, keepdim=True).mean()
            )
            return action_log_probs, dist_entropy, None

        action_distribution = self.action_out(x, available_actions)
        action_log_probs = action_distribution.log_probs(action)
        if active_masks is not None:
            active_sum = active_masks.sum().clamp(min=1e-6)
            dist_entropy = (
                action_distribution.entropy() * active_masks.squeeze(-1)
            ).sum() / active_sum
        else:
            dist_entropy = action_distribution.entropy().mean()

        return action_log_probs, dist_entropy, action_distribution
