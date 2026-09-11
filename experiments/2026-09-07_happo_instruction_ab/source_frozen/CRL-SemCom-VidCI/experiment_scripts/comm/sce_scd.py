import torch
import torch.nn as nn


class ResidualConvBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True),
        )

    def forward(self, x):
        return x + self.body(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=True),
            nn.GELU(),
            ResidualConvBlock(out_channels),
        )

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=True),
            nn.GELU(),
            ResidualConvBlock(out_channels),
        )

    def forward(self, x):
        return self.block(x)


class SemanticChannelEncoder(nn.Module):
    def __init__(self, in_channels=24, latent_channels=48, hidden_channels=64):
        super().__init__()
        self.embedding = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=True),
            nn.GELU(),
            ResidualConvBlock(hidden_channels),
        )
        self.down1 = DownBlock(hidden_channels, hidden_channels)
        self.down2 = DownBlock(hidden_channels, hidden_channels * 2)
        self.down3 = DownBlock(hidden_channels * 2, hidden_channels * 2)
        self.mapping = nn.Sequential(
            ResidualConvBlock(hidden_channels * 2),
            nn.Conv2d(hidden_channels * 2, latent_channels, kernel_size=3, padding=1, bias=True),
        )

    def forward(self, sensor_data):
        x = self.embedding(sensor_data)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        return self.mapping(x)


class SemanticChannelDecoder(nn.Module):
    def __init__(self, out_channels=24, latent_channels=48, hidden_channels=64, levels=4):
        super().__init__()
        self.levels = levels
        self.embedding = nn.Sequential(
            nn.Conv2d(latent_channels + 1, hidden_channels * 2, kernel_size=3, padding=1, bias=True),
            nn.GELU(),
            ResidualConvBlock(hidden_channels * 2),
        )
        self.up1 = UpBlock(hidden_channels * 2, hidden_channels * 2)
        self.up2 = UpBlock(hidden_channels * 2, hidden_channels)
        self.up3 = UpBlock(hidden_channels, hidden_channels)
        self.mapping = nn.Sequential(
            ResidualConvBlock(hidden_channels),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1, bias=True),
        )

    def forward(self, received_symbols, rate_map):
        rate_map = (rate_map / float(self.levels)).to(received_symbols.dtype)
        x = torch.cat([received_symbols, rate_map], dim=1)
        x = self.embedding(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        return self.mapping(x)

