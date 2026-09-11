import torch.nn as nn

from .channel import apply_channel, build_rate_mask
from .metrics import compute_batch_comm_metrics
from .ran import RateAllocationNetwork
from .sce_scd import SemanticChannelDecoder, SemanticChannelEncoder


class SemanticTransmissionSystem(nn.Module):
    def __init__(
        self,
        sensor_channels=24,
        latent_channels=48,
        hidden_channels=64,
        rate_levels=4,
        snr_db=10.0,
        forced_rate_level=None,
        channel_coding_rate=0.5,
        modulation_order=4,
    ):
        super().__init__()
        self.rate_levels = rate_levels
        self.snr_db = snr_db
        self.forced_rate_level = forced_rate_level
        self.channel_coding_rate = channel_coding_rate
        self.modulation_order = modulation_order
        self.encoder = SemanticChannelEncoder(
            in_channels=sensor_channels,
            latent_channels=latent_channels,
            hidden_channels=hidden_channels,
        )
        self.ran = RateAllocationNetwork(
            in_channels=latent_channels,
            hidden_channels=hidden_channels,
            levels=rate_levels,
        )
        self.decoder = SemanticChannelDecoder(
            out_channels=sensor_channels,
            latent_channels=latent_channels,
            hidden_channels=hidden_channels,
            levels=rate_levels,
        )

    def forward(self, sensor_data, train=True, forced_rate_level=None):
        latent_symbols = self.encoder(sensor_data)
        effective_forced_rate = self.forced_rate_level if forced_rate_level is None else forced_rate_level
        ran_out = self.ran(latent_symbols, train=train, forced_rate_level=effective_forced_rate)
        rate_level_map = ran_out['rate_map']
        rate_mask = build_rate_mask(rate_level_map, max_symbols=latent_symbols.shape[1], levels=self.rate_levels)
        masked_symbols = latent_symbols * rate_mask
        received_symbols = apply_channel(
            masked_symbols, snr_db=self.snr_db, training=train, rate_mask=rate_mask
        )
        reconstructed_sensor_data = self.decoder(received_symbols, rate_level_map)
        comm_metrics = compute_batch_comm_metrics(
            rate_mask,
            input_hw=sensor_data.shape[-2:],
            rate_levels=self.rate_levels,
            snr_db=self.snr_db,
        )
        return reconstructed_sensor_data, {
            'latent_symbols': latent_symbols,
            'received_symbols': received_symbols,
            'rate_map': rate_level_map,
            'rate_level_map': rate_level_map,
            'reward_rate_map': rate_level_map,
            'rate_mask': rate_mask,
            'log_prob': ran_out['log_prob'],
            'probs': ran_out['probs'],
            **comm_metrics,
        }
