from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


ALL_FIXED_SNR_MU_GRID_SET = "ran_adaptive_snr0_15_mu_grid_16modes"
FIXED_SNR_MU_SWEEP_SETS = (
    "ran_adaptive_snr0_mu_sweep_4modes",
    "ran_adaptive_snr5_mu_sweep_4modes",
    "ran_adaptive_snr10_mu_sweep_4modes",
    "ran_adaptive_snr15_mu_sweep_4modes",
)


class SemanticModeLibrary:
    """Loads CRL-SemCom-VidCI checkpoints as HARL semantic-mode metadata.

    The registry is intentionally metadata-first. HARL training needs fast online
    estimates of semantic load and predicted reconstruction quality; running
    SCE/RAN/SCD video reconstruction inside every RL step is reserved for
    offline evaluation.
    """

    def __init__(self, name: str, modes: List[Dict[str, Any]], profile_path: Optional[str] = None):
        self.name = str(name)
        self.modes = list(modes)
        self.num_modes = int(len(self.modes))
        self.enabled = self.num_modes > 0
        self.ids = [str(item.get("id", f"mode_{i}")) for i, item in enumerate(self.modes)]
        self.checkpoints = [str(item.get("checkpoint", "")) for item in self.modes]
        self.families = [str(item.get("family", "")) for item in self.modes]
        self.trained_snr_db = np.asarray(
            [self._float_or(item.get("trained_snr_db"), 10.0) for item in self.modes],
            dtype=float,
        )
        self.mu_comm = np.asarray(
            [self._float_or(item.get("mu_comm"), 5.0e-4) for item in self.modes],
            dtype=float,
        )
        self.latent_channels = np.asarray(
            [max(1, int(self._float_or(item.get("comm_latent_channels"), 48))) for item in self.modes],
            dtype=float,
        )
        self.rate_levels = np.asarray(
            [max(1, int(self._float_or(item.get("comm_rate_levels"), 4))) for item in self.modes],
            dtype=float,
        )

        self.rate_level_proxy = self._derive_rate_level_proxy()
        self.quality_bias_db = self._derive_quality_bias_db()
        self.content_sensitivity = self._derive_content_sensitivity()
        self.profile_path = ""
        self.profile: Dict[str, Any] = {}
        if profile_path:
            self.profile_path = str(self._resolve_path(profile_path))
            self.profile = self._load_offline_profile(self.profile_path)

    @classmethod
    def from_config(
        cls,
        *,
        enabled: bool,
        set_name: str,
        registry_path: Optional[str] = None,
        profile_path: Optional[str] = None,
    ) -> "SemanticModeLibrary":
        if not enabled or not profile_path or not registry_path:
            raise ValueError("This version requires an explicit frozen table and row registry")

        base_dir = Path(__file__).resolve().parent
        path = Path(registry_path).expanduser() if registry_path else base_dir / "semcom_model_sets.json"
        if not path.is_absolute():
            path = base_dir / path
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ValueError(f"Failed to load semantic model registry {path}: {exc}") from exc

        selected = str(set_name or data.get("recommended_default", "")).strip()
        if not selected:
            selected = str(data.get("recommended_default", "")).strip()
        sets = data.get("sets", {})
        if selected not in sets:
            available = ", ".join(sorted([*sets.keys(), ALL_FIXED_SNR_MU_GRID_SET]))
            raise ValueError(f"Unknown semantic_model_set={selected!r}. Available: {available}")
        return cls(selected, list(sets[selected].get("modes", [])), profile_path=profile_path)

    @staticmethod
    def _resolve_path(path: str) -> Path:
        base_dir = Path(__file__).resolve().parent
        out = Path(path).expanduser()
        if not out.is_absolute():
            out = base_dir / out
        return out

    @staticmethod
    def _float_or(value: Any, default: float) -> float:
        try:
            out = float(value)
        except Exception:
            out = float(default)
        return out if np.isfinite(out) else float(default)

    @staticmethod
    def _safe_norm(values: np.ndarray, fallback: float = 0.5) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        if values.size == 0:
            return values
        lo = float(np.min(values))
        hi = float(np.max(values))
        if hi - lo <= 1.0e-12:
            return np.full_like(values, float(fallback), dtype=float)
        return (values - lo) / (hi - lo)

    def _derive_rate_level_proxy(self) -> np.ndarray:
        """Approximate average RAN rate level from training metadata.

        Mode load should represent the RAN quality/load operating point, not a
        free advantage for checkpoints trained at higher SNR. Lower `mu_comm`
        spends more semantic symbols for quality; higher `mu_comm` is more
        bandwidth-frugal. The trained SNR affects quality matching in phi_Q.
        """
        if self.num_modes == 0:
            return np.asarray([], dtype=float)
        mu_norm = self._safe_norm(np.log10(np.maximum(self.mu_comm, 1.0e-12)))
        rate = 1.0 + 3.0 * np.power(np.clip(1.0 - mu_norm, 0.0, 1.0), 0.75)
        return np.clip(rate, 1.0, 4.0)

    def _derive_quality_bias_db(self) -> np.ndarray:
        if self.num_modes == 0:
            return np.asarray([], dtype=float)
        snr_norm = self._safe_norm(self.trained_snr_db)
        mu_norm = self._safe_norm(np.log10(np.maximum(self.mu_comm, 1.0e-12)))
        snr_ceiling = 0.4 * snr_norm
        mu_quality_gain = 2.2 * np.power(np.clip(1.0 - mu_norm, 0.0, 1.0), 0.85)
        return snr_ceiling + mu_quality_gain

    def _derive_content_sensitivity(self) -> np.ndarray:
        if self.num_modes == 0:
            return np.asarray([], dtype=float)
        mu_norm = self._safe_norm(np.log10(np.maximum(self.mu_comm, 1.0e-12)))
        return np.clip(0.25 + 0.45 * mu_norm, 0.2, 0.8)

    def _load_offline_profile(self, profile_path: str) -> Dict[str, Any]:
        path = self._resolve_path(profile_path)
        if not path.exists():
            raise ValueError(f"semantic_profile_path does not exist: {path}")
        try:
            raw = np.load(path, allow_pickle=True)
        except Exception as exc:
            raise ValueError(f"Failed to load semantic profile {path}: {exc}") from exc

        def require_array(name: str, *, ndim: Optional[int] = None) -> np.ndarray:
            if name not in raw:
                raise ValueError(f"Semantic profile {path} is missing required array {name!r}")
            arr = np.asarray(raw[name])
            if ndim is not None and arr.ndim != ndim:
                raise ValueError(f"Semantic profile {path} array {name!r} must be {ndim}D, got shape {arr.shape}")
            return arr

        snr_grid_db = np.asarray(require_array("snr_grid_db", ndim=1), dtype=float)
        q_hat_mean = np.asarray(require_array("q_hat_mean", ndim=2), dtype=float)
        if snr_grid_db.size < 1:
            raise ValueError(f"Semantic profile {path} has an empty snr_grid_db")
        if q_hat_mean.shape != (self.num_modes, snr_grid_db.size):
            raise ValueError(
                f"Semantic profile {path} q_hat_mean shape {q_hat_mean.shape} does not match "
                f"(num_modes={self.num_modes}, num_snr={snr_grid_db.size})"
            )
        if not np.all(np.isfinite(snr_grid_db)) or not np.all(np.isfinite(q_hat_mean)):
            raise ValueError(f"Semantic profile {path} contains non-finite SNR or quality values")
        if np.unique(snr_grid_db).size != snr_grid_db.size:
            raise ValueError(f"Semantic profile {path} has duplicate SNR grid points")
        if "mode_ids" in raw and list(np.asarray(raw["mode_ids"]).astype(str)) != self.ids:
            raise ValueError(f"Semantic profile {path} mode_ids do not match registry order")

        if "bar_ls_main_mean" in raw:
            l_z_mean = np.asarray(raw["bar_ls_main_mean"], dtype=float)
        elif "L_z_mean" in raw:
            l_z_mean = np.asarray(raw["L_z_mean"], dtype=float)
        else:
            raise ValueError(f"Semantic profile {path} is missing bar_ls_main_mean or L_z_mean")
        if l_z_mean.shape != q_hat_mean.shape:
            raise ValueError(f"Semantic profile {path} L_z shape {l_z_mean.shape} must match q_hat_mean")
        if not np.all(np.isfinite(l_z_mean)) or np.any(l_z_mean < 0.0):
            raise ValueError(f"Semantic profile {path} contains invalid L_z values")

        if "avg_kept_real_symbols_mean" in raw:
            n_z_mean = np.asarray(raw["avg_kept_real_symbols_mean"], dtype=float)
            if n_z_mean.shape != q_hat_mean.shape:
                raise ValueError(f"Semantic profile {path} n_z shape {n_z_mean.shape} must match q_hat_mean")
            if not np.all(np.isfinite(n_z_mean)) or np.any(n_z_mean < 0.0):
                raise ValueError(f"Semantic profile {path} contains invalid n_z values")
        else:
            n_z_mean = 2.0 * l_z_mean

        order = np.argsort(snr_grid_db)
        profile = {
            "path": str(path),
            "snr_grid_db": snr_grid_db[order],
            "q_hat_mean": q_hat_mean[:, order],
            "l_z_mean": l_z_mean[:, order],
            "n_z_mean": n_z_mean[:, order],
            "quality_metric": self._scalar_string(raw, "quality_metric", "psnr"),
            "dataset_name": self._scalar_string(raw, "dataset_name", ""),
            "semantic_model_set": self._scalar_string(raw, "semantic_model_set", ""),
        }
        if profile["semantic_model_set"] and profile["semantic_model_set"] != self.name:
            raise ValueError(
                f"Semantic profile {path} was built for semantic_model_set={profile['semantic_model_set']!r}, "
                f"but config selected {self.name!r}"
            )
        return profile

    @staticmethod
    def _scalar_string(raw: Any, name: str, default: str) -> str:
        if name not in raw:
            return str(default)
        value = raw[name]
        try:
            if np.asarray(value).shape == ():
                return str(np.asarray(value).item())
        except Exception:
            pass
        return str(value)

    def _predict_from_offline_profile(
        self,
        *,
        gamma_bh: np.ndarray,
        psi_mean: np.ndarray,
        side_info_bits: float,
    ) -> Dict[str, np.ndarray]:
        n_uav, n_ds = psi_mean.shape
        n_modes = self.num_modes
        n_z = np.zeros((n_uav, n_ds, n_modes), dtype=float)
        l_z = np.zeros_like(n_z)
        lambda_sem = np.zeros_like(n_z)
        q_hat = np.zeros_like(n_z)

        gamma = np.maximum(np.asarray(gamma_bh, dtype=float), 1.0e-12)
        snr_db = 10.0 * np.log10(gamma)
        snr_grid_db = np.asarray(self.profile["snr_grid_db"], dtype=float)
        q_profile = np.asarray(self.profile["q_hat_mean"], dtype=float)
        l_profile = np.asarray(self.profile["l_z_mean"], dtype=float)
        n_profile = np.asarray(self.profile["n_z_mean"], dtype=float)
        for n in range(n_uav):
            side_uses = float(side_info_bits) / np.log2(1.0 + float(gamma[n]))
            for m in range(n_modes):
                q_val = float(np.interp(float(snr_db[n]), snr_grid_db, q_profile[m]))
                l_val = float(np.interp(float(snr_db[n]), snr_grid_db, l_profile[m]))
                n_val = float(np.interp(float(snr_db[n]), snr_grid_db, n_profile[m]))
                n_z[n, :, m] = n_val
                l_z[n, :, m] = l_val
                lambda_sem[n, :, m] = l_val + side_uses
                q_hat[n, :, m] = q_val
        return {
            "n_z": n_z,
            "L_z": l_z,
            "Lambda_sem": lambda_sem,
            "Q_hat_rec": q_hat,
        }

    def predict_mode_tables(
        self,
        *,
        gamma_bh: np.ndarray,
        psi_mean: np.ndarray,
        h_s: int,
        w_s: int,
        side_info_bits: float,
    ) -> Dict[str, np.ndarray]:
        n_uav, n_ds = psi_mean.shape
        n_modes = self.num_modes
        if self.profile:
            return self._predict_from_offline_profile(
                gamma_bh=gamma_bh,
                psi_mean=psi_mean,
                side_info_bits=side_info_bits,
            )

        n_z = np.zeros((n_uav, n_ds, n_modes), dtype=float)
        l_z = np.zeros_like(n_z)
        lambda_sem = np.zeros_like(n_z)
        q_hat = np.zeros_like(n_z)

        snr_db = 10.0 * np.log10(np.maximum(np.asarray(gamma_bh, dtype=float), 1.0e-12))
        for n in range(n_uav):
            for m in range(n_modes):
                rate_level = float(self.rate_level_proxy[m])
                latent_scale = float(self.latent_channels[m]) / 48.0
                n_z_val = 12.0 * rate_level * float(h_s * w_s) * latent_scale
                n_z[n, :, m] = n_z_val
                l_z[n, :, m] = np.ceil(n_z_val / 2.0)
                side_uses = float(side_info_bits) / np.log2(1.0 + max(float(gamma_bh[n]), 1.0e-12))
                lambda_sem[n, :, m] = l_z[n, :, m] + side_uses

                mismatch = abs(float(snr_db[n]) - float(self.trained_snr_db[m]))
                # Metadata-based phi_Q surrogate. It rewards matching the online
                # channel to the checkpoint's training SNR and penalizes larger
                # content complexity.
                q_hat[n, :, m] = (
                    27.0
                    + 0.65 * float(snr_db[n])
                    - 0.32 * mismatch
                    + float(self.quality_bias_db[m])
                    - float(self.content_sensitivity[m]) * psi_mean[n]
                )
        return {
            "n_z": n_z,
            "L_z": l_z,
            "Lambda_sem": lambda_sem,
            "Q_hat_rec": q_hat,
        }

    def diagnostic(self) -> Dict[str, Any]:
        return {
            "semantic_model_set": self.name,
            "semantic_mode_ids": list(self.ids),
            "semantic_mode_checkpoints": list(self.checkpoints),
            "semantic_mode_family": list(self.families),
            "semantic_mode_trained_snr_db": None,
            "semantic_mode_mu_comm": None,
            "semantic_mode_names": [item["name"] for item in self.modes],
            "semantic_profile_enabled": bool(self.profile),
            "semantic_profile_path": str(self.profile.get("path", "")),
            "semantic_profile_snr_grid_db": np.asarray(self.profile.get("snr_grid_db", []), dtype=float),
            "semantic_profile_quality_metric": str(self.profile.get("quality_metric", "")),
            "semantic_profile_dataset_name": str(self.profile.get("dataset_name", "")),
        }
