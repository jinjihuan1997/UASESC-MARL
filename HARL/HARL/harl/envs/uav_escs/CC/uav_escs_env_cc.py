from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from harl.envs.uav_escs.SC.uav_escs_env_sc import SCUAVEnv


class CCUAVEnv(SCUAVEnv):
    """Conventional H.264 + LDPC communication baseline.

    The baseline keeps the same Dec-POMDP, topology, cache, AoI, SUT resource
    allocation, and UAV scheduling logic as SCUAVEnv. The only modeling change
    is the per-mode stream table:

    - mode m selects an H.264 QP and an LDPC code rate.
    - Lambda_sem[n,k,m] is the reliable digital transmission load in complex
      channel uses.
    - Q_hat_rec[n,k,m] is the predicted H.264 reconstruction PSNR.

    This makes uav_escs_cc directly comparable with uav_escs_sc under the same
    HARL training code and reward interface.
    """

    def __init__(self, args):
        args = dict(args)
        args["use_semcom_registry"] = False
        args.setdefault("use_direct_ds_logits", False)
        args.setdefault("scheduling_weight_transform", "softmax")
        args.setdefault("n_semantic_modes", 5)
        super().__init__(args)
        self.communication_model = "h264_ldpc"
        self.codec_name = str(self._arg("cc_codec_name", "h264"))
        self.channel_code_name = str(self._arg("cc_channel_code_name", "ldpc"))
        self.h264_qp_modes = self._as_float_array(
            self._arg("h264_qp_modes", [42.0, 38.0, 34.0, 30.0, 26.0]),
            self.n_semantic_modes,
            [42.0, 38.0, 34.0, 30.0, 26.0],
        )
        self.ldpc_code_rate_modes = np.clip(
            self._as_float_array(
                self._arg("ldpc_code_rate_modes", [0.50, 2.0 / 3.0, 0.75, 5.0 / 6.0, 5.0 / 6.0]),
                self.n_semantic_modes,
                [0.50, 2.0 / 3.0, 0.75, 5.0 / 6.0, 5.0 / 6.0],
            ),
            1.0e-3,
            1.0,
        )
        self.ldpc_required_snr_db = self._as_float_array(
            self._arg("ldpc_required_snr_db", [-1.0, 2.0, 5.0, 8.0, 8.0]),
            self.n_semantic_modes,
            [-1.0, 2.0, 5.0, 8.0, 8.0],
        )
        self.h264_ref_qp = self._finite_float("h264_ref_qp", 42.0)
        self.h264_ref_bpp = self._finite_float("h264_ref_bpp", 0.0025, lower=1.0e-9)
        self.h264_content_bpp_gain = self._finite_float("h264_content_bpp_gain", 0.60, lower=0.0)
        self.h264_ref_psnr_db = self._finite_float("h264_ref_psnr_db", 28.0)
        self.h264_psnr_per_qp_step_db = self._finite_float("h264_psnr_per_qp_step_db", 0.55, lower=0.0)
        self.h264_content_psnr_penalty_db = self._finite_float("h264_content_psnr_penalty_db", 1.20, lower=0.0)
        self.cc_header_bits = self._finite_float("cc_header_bits", 512.0, lower=0.0)
        self.cc_source_h = max(1, int(self._arg("cc_source_h", self.H)))
        self.cc_source_w = max(1, int(self._arg("cc_source_w", self.W)))
        self.cc_source_l = max(1, int(self._arg("cc_source_l", self.raw_video_l)))
        self.cc_frame_pixels = float(self.cc_source_h * self.cc_source_w * self.cc_source_l)

        # Optional offline H.264+LDPC profile generated from REAL codec + channel
        # simulation (scripts/build_cc_h264_ldpc_profile.py). When provided, the
        # per-mode quality/load tables are interpolated from measured PSNR and
        # real channel uses instead of the synthetic formula below. Empty keeps
        # the synthetic fallback.
        self.cc_profile_path = str(self._arg("cc_profile_path", "") or "")
        self.cc_profile: Optional[Dict[str, np.ndarray]] = None
        if self.cc_profile_path:
            self.cc_profile = self._load_cc_profile(self.cc_profile_path)

    @staticmethod
    def _resolve_profile_path(path: str) -> Path:
        out = Path(path).expanduser()
        if out.is_absolute():
            return out
        base_dir = Path(__file__).resolve().parents[1] / "semantic_models"
        return base_dir / out

    def _load_cc_profile(self, path: str) -> Dict[str, np.ndarray]:
        resolved = self._resolve_profile_path(path)
        if not resolved.exists():
            raise ValueError(f"cc_profile_path does not exist: {resolved}")
        raw = np.load(resolved, allow_pickle=True)
        for key in ("snr_grid_db", "q_hat_mean", "bar_ls_main_mean"):
            if key not in raw:
                raise ValueError(f"CC profile {resolved} is missing required array {key!r}")
        snr_grid = np.asarray(raw["snr_grid_db"], dtype=float).reshape(-1)
        q_hat = np.asarray(raw["q_hat_mean"], dtype=float)
        l_z = np.asarray(raw["bar_ls_main_mean"], dtype=float)
        n_z = (
            np.asarray(raw["avg_kept_real_symbols_mean"], dtype=float)
            if "avg_kept_real_symbols_mean" in raw
            else 2.0 * l_z
        )
        if q_hat.shape != (self.n_semantic_modes, snr_grid.size):
            raise ValueError(
                f"CC profile {resolved} q_hat_mean shape {q_hat.shape} does not match "
                f"(n_modes={self.n_semantic_modes}, n_snr={snr_grid.size})"
            )
        if l_z.shape != q_hat.shape or n_z.shape != q_hat.shape:
            raise ValueError(f"CC profile {resolved} load arrays must match q_hat_mean shape {q_hat.shape}")
        order = np.argsort(snr_grid)
        return {
            "snr_grid_db": snr_grid[order],
            "q_hat_mean": q_hat[:, order],
            "l_z_mean": l_z[:, order],
            "n_z_mean": n_z[:, order],
        }

    def _update_semantic_tables_from_profile(self) -> None:
        prof = self.cc_profile
        snr_grid = prof["snr_grid_db"]
        q_profile = prof["q_hat_mean"]
        l_profile = prof["l_z_mean"]
        n_profile = prof["n_z_mean"]
        snr_db = 10.0 * np.log10(np.maximum(self.gamma_bh, 1.0e-12))
        for n in range(self.n_uav):
            for m in range(self.n_semantic_modes):
                q_val = float(np.interp(float(snr_db[n]), snr_grid, q_profile[m]))
                l_val = float(np.interp(float(snr_db[n]), snr_grid, l_profile[m]))
                n_val = float(np.interp(float(snr_db[n]), snr_grid, n_profile[m]))
                self.Q_hat_rec[n, :, m] = q_val
                # Digital scheme: the reliable load is the real channel-use count
                # (modulation + LDPC), independent of instantaneous SNR. No semantic
                # rate-map side channel, so Lambda_sem == L_z.
                self.L_z[n, :, m] = l_val
                self.n_z[n, :, m] = n_val
                self.Lambda_sem[n, :, m] = l_val
        self.M_feas = (self.Q_hat_rec >= self.Q_min) & self.owner_mask[:, :, None]

    def _build_semantic_mode_lookup(self) -> np.ndarray:
        """Map SNR bucket and 4-way mu action to H.264/LDPC rank modes.

        CC has five quality/load rank modes rather than the SC registry's
        4-SNR-bucket x 4-mu grid. Keep the UAV action as four mu logits, but
        expose four adjacent rank choices per bucket so every bucket has a
        non-degenerate feasible mode set.
        """
        if int(self.n_semantic_modes) >= 5:
            return np.asarray(
                [
                    [0, 1, 2, 3],
                    [0, 1, 2, 3],
                    [1, 2, 3, 4],
                    [1, 2, 3, 4],
                ],
                dtype=int,
            )

        lookup = np.zeros((4, 4), dtype=int)
        for bucket_id in range(4):
            for mu_id in range(4):
                lookup[bucket_id, mu_id] = int(min(mu_id, max(self.n_semantic_modes - 1, 0)))
        return lookup

    def _update_semantic_tables(self):
        if self.cc_profile is not None:
            self._update_semantic_tables_from_profile()
            return

        psi_mean = np.mean(self.psi, axis=-1)
        snr_db = 10.0 * np.log10(np.maximum(self.gamma_bh, 1.0e-12))

        for n in range(self.n_uav):
            channel_eff = max(float(np.log2(1.0 + self.gamma_bh[n])), 1.0e-12)
            for m in range(self.n_semantic_modes):
                qp = float(self.h264_qp_modes[m])
                code_rate = float(self.ldpc_code_rate_modes[m])

                qp_bpp_scale = 2.0 ** ((self.h264_ref_qp - qp) / 6.0)
                bpp = self.h264_ref_bpp * qp_bpp_scale * (1.0 + self.h264_content_bpp_gain * psi_mean[n])
                source_bits = self.cc_frame_pixels * bpp + self.cc_header_bits
                coded_bits = source_bits / max(code_rate, 1.0e-12)
                self.Lambda_sem[n, :, m] = coded_bits / channel_eff
                self.n_z[n, :, m] = coded_bits
                self.L_z[n, :, m] = self.Lambda_sem[n, :, m]

                ldpc_margin = float(snr_db[n] - self.ldpc_required_snr_db[m])
                ldpc_penalty = max(-ldpc_margin, 0.0) * 2.0
                self.Q_hat_rec[n, :, m] = (
                    self.h264_ref_psnr_db
                    + self.h264_psnr_per_qp_step_db * (self.h264_ref_qp - qp)
                    - self.h264_content_psnr_penalty_db * psi_mean[n]
                    - ldpc_penalty
                )

        self.M_feas = (self.Q_hat_rec >= self.Q_min) & self.owner_mask[:, :, None]

    def _build_info(self, slot_t: int, reward: float) -> Dict[str, Any]:
        info = super()._build_info(slot_t, reward)
        info.update(
            {
                "communication_model": self.communication_model,
                "codec_name": self.codec_name,
                "channel_code_name": self.channel_code_name,
                "h264_qp_modes": self.h264_qp_modes.copy(),
                "ldpc_code_rate_modes": self.ldpc_code_rate_modes.copy(),
                "ldpc_required_snr_db": self.ldpc_required_snr_db.copy(),
                "cc_header_bits": float(self.cc_header_bits),
                "cc_source_shape_hwl": np.asarray(
                    [self.cc_source_h, self.cc_source_w, self.cc_source_l],
                    dtype=int,
                ),
                "cc_frame_pixels": float(self.cc_frame_pixels),
            }
        )
        return info
