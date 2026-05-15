import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True),
        )

    def forward(self, x):
        return x + self.body(x)


class RateAllocationNetwork(nn.Module):
    def __init__(self, in_channels=48, hidden_channels=64, levels=4):
        super().__init__()
        self.levels = levels
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=True),
            nn.GELU(),
            ResidualBlock(hidden_channels),
            ResidualBlock(hidden_channels),
            nn.Conv2d(hidden_channels, levels, kernel_size=1, bias=True),
        )

    def forward(self, symbols, train=True, forced_rate_level=None):
        batch, _, height, width = symbols.shape
        if forced_rate_level is not None:
            rate_map = torch.full(
                (batch, 1, height, width),
                float(forced_rate_level),
                device=symbols.device,
                dtype=symbols.dtype,
            )
            return {'logits': None, 'probs': None, 'rate_map': rate_map, 'log_prob': None}

        logits = self.body(symbols)
        probs = torch.softmax(logits, dim=1)
        dist = torch.distributions.Categorical(probs=probs.permute(0, 2, 3, 1))

        if train:
            actions = dist.sample()
            log_prob = dist.log_prob(actions).unsqueeze(1)
        else:
            actions = probs.argmax(dim=1)
            log_prob = None

        rate_map = actions.unsqueeze(1).to(symbols.dtype) + 1.0
        return {'logits': logits, 'probs': probs, 'rate_map': rate_map, 'log_prob': log_prob}
