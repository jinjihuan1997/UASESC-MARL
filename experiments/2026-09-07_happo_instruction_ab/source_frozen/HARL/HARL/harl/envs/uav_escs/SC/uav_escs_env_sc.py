from __future__ import annotations

import ast
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
from gymnasium.spaces import Box

from harl.envs.uav_escs.semantic_models.semantic_registry import SemanticModeLibrary


class SCUAVEnv:
    """
    Fixed multi-objective semantic video acquisition Dec-POMDP.

    Agent 0 is the SUT and allocates transparent satellite uplink fractions.
    Agents 1..N are UAVs and each select one common semantic mode plus a
    cached-stream scheduling vector for their permanently associated DS set.
    """

    INVALID_TAU = -1

    def __init__(self, args):
        self.args = args
        self.seed()

        self.env_name = str(self._arg("env_name", "semantic_video_acquisition_fixed_multiobj"))
        self.n_sut = 1
        self.n_uav = max(1, int(self._arg("n_uav", 3)))
        self.n_ds = max(1, int(self._arg("n_ds", 30)))
        if self.n_ds % self.n_uav != 0:
            raise ValueError(
                f"n_ds must be divisible by n_uav so every UAV serves the same number of DS: "
                f"got n_ds={self.n_ds}, n_uav={self.n_uav}"
            )
        self.ds_per_uav = self.n_ds // self.n_uav
        self.n_agents = self.n_sut + self.n_uav

        self.delta_T = self._finite_float("delta_T", 1.0, lower=1.0e-9)
        self.T = self._finite_float("T", 600.0, lower=self.delta_T)
        self.max_steps = max(1, int(np.ceil(self.T / self.delta_T)))
        self.current_step = 0

        self.map_size = self._as_float_array(
            self._arg("map_size", [6000.0, 6000.0, 600.0]), 3, [6000.0, 6000.0, 600.0]
        )
        self.sut_height_m = self._finite_float("sut_height_m", 10.0, lower=0.0)
        self.pos_sut = np.asarray(
            [0.5 * self.map_size[0], 0.5 * self.map_size[1], self.sut_height_m],
            dtype=float,
        )
        self.ds_height_m = self._finite_float("ds_height_m", 1.5, lower=0.0)
        self.uav_altitude_m = self._finite_float("uav_altitude_m", 100.0, lower=1.0)
        self.uav_min_distance_to_sut_m = self._finite_float("uav_min_distance_to_sut_m", 1000.0, lower=0.0)
        self.ds_uav_service_radius_m = self._finite_float("ds_uav_service_radius_m", 100.0, lower=0.0)

        # Link parameters. DS->UAV is modeled for diagnostics/admission feasibility,
        # but admission itself is strictly cache-state determined.
        self.c_light = 3.0e8
        self.N0 = 10.0 ** ((-174.0 - 30.0) / 10.0)
        self.f_uav_sut = self._finite_float("f_uav_sut_hz", 2.4e9, lower=1.0)
        self.f_sut_sat = self._finite_float("f_sut_sat_hz", 14.0e9, lower=1.0)
        self.f_ds_uav = self._finite_float("f_ds_uav_hz", 2.4e9, lower=1.0)
        self.B_uav_sut = self._finite_float("B_uav_sut", 2.0e6, lower=1.0)
        self.B_sut_sat = self._finite_float("B_sut_sat", 1.0e5, lower=1.0)
        self.B_ds_uav = self._finite_float("B_ds_uav", 20.0e6, lower=1.0)
        self.p_uav_sut_w = self._finite_float("p_uav_sut_w", 0.2, lower=1.0e-12)
        self.p_sut_sat_w = self._finite_float("p_sut_sat_w", 2.0, lower=1.0e-12)
        self.p_ds_uav_w = self._finite_float("p_ds_uav_w", 0.1, lower=1.0e-12)
        self.g_uav_sut_tx_db = self._finite_float("g_uav_sut_tx_db", 0.0)
        self.g_uav_sut_rx_db = self._finite_float("g_uav_sut_rx_db", 0.0)
        self.g_sut_sat_tx_db = self._finite_float("g_sut_sat_tx_db", 35.0)
        self.g_sut_sat_rx_db = self._finite_float("g_sut_sat_rx_db", 20.0)
        self.g_ds_uav_tx_db = self._finite_float("g_ds_uav_tx_db", 0.0)
        self.g_ds_uav_rx_db = self._finite_float("g_ds_uav_rx_db", 0.0)
        self.nf_uav_sut_db = self._finite_float("nf_uav_sut_db", 7.0)
        self.nf_sut_sat_db = self._finite_float("nf_sut_sat_db", 3.0)
        self.nf_ds_uav_db = self._finite_float("nf_ds_uav_db", 7.0)
        self.sigma_uav_sut_db = self._finite_float("sigma_uav_sut", 3.0, lower=0.0)
        self.sigma_sut_sat_db = self._finite_float("sigma_sut_sat", 1.0, lower=0.0)
        self.sigma_ds_uav_db = self._finite_float("sigma_ds_uav", 4.0, lower=0.0)
        self.atg_los_a = self._finite_float("atg_los_a", 12.08, lower=1.0e-9)
        self.atg_los_b = self._finite_float("atg_los_b", 0.11, lower=1.0e-9)
        self.atg_eta_los_db = self._finite_float("atg_eta_los_db", 1.6, lower=0.0)
        self.atg_eta_nlos_db = self._finite_float("atg_eta_nlos_db", 23.0, lower=0.0)
        self.satellite_altitude_m = self._finite_float("satellite_altitude_m", 550.0e3, lower=1.0)
        self.satellite_slant_range_m = self._finite_float(
            "satellite_slant_range_m", self.satellite_altitude_m, lower=1.0
        )

        self.beta_sat_lower_bound = self._finite_float("beta_sat_lower_bound", 0.02, lower=0.0)
        self.beta_sat_lower_bound = min(self.beta_sat_lower_bound, 1.0 / float(self.n_uav))
        self.beta_sut_sat = np.ones(self.n_uav, dtype=float) / float(self.n_uav)

        # Semantic stream/model parameters.
        self.use_semcom_registry = self._as_bool(self._arg("use_semcom_registry", False))
        self.semantic_model_set = str(self._arg("semantic_model_set", "ran_adaptive_snr10_mu_sweep_4modes"))
        self.semantic_registry_path = str(self._arg("semantic_registry_path", "") or "")
        self.semantic_profile_path = str(self._arg("semantic_profile_path", "") or "")
        self.semantic_library = SemanticModeLibrary.from_config(
            enabled=self.use_semcom_registry,
            set_name=self.semantic_model_set,
            registry_path=self.semantic_registry_path or None,
            profile_path=self.semantic_profile_path or None,
        )
        if self.semantic_library.enabled:
            self.n_semantic_modes = int(self.semantic_library.num_modes)
        else:
            self.n_semantic_modes = max(1, int(self._arg("n_semantic_modes", self._arg("M_sc", 4))))
        self.raw_video_h = max(1, int(self._arg("raw_video_h", 720)))
        self.raw_video_w = max(1, int(self._arg("raw_video_w", 1280)))
        self.raw_video_l = max(1, int(self._arg("raw_video_l", 32)))
        self.raw_video_channels = max(1, int(self._arg("raw_video_channels", 3)))
        self.H = max(8, int(self._arg("video_h", self._arg("sci_input_h", 256))))
        self.W = max(8, int(self._arg("video_w", self._arg("sci_input_w", 256))))
        self.L_video = max(1, int(self._arg("video_l", self._arg("sci_input_l", 16))))
        self.video_channels = max(1, int(self._arg("video_channels", self._arg("sci_input_channels", 1))))
        self.raw_sample_bits = self._finite_float("raw_sample_bits", 8.0, lower=1.0e-9)
        self.sci_bits_per_sample = self._finite_float("sci_bits_per_sample", 8.0, lower=1.0e-9)
        self.raw_video_samples = int(
            self.raw_video_h * self.raw_video_w * self.raw_video_l * self.raw_video_channels
        )
        self.raw_video_bits = float(self.raw_video_samples) * self.raw_sample_bits
        self.sci_input_samples = int(self.H * self.W * self.L_video * self.video_channels)
        self.sci_input_bits = float(self.sci_input_samples) * self.raw_sample_bits
        self.sci_sensor_channels = max(0, int(self._arg("sci_sensor_channels", 24)))
        if self.sci_sensor_channels > 0:
            self.sci_num_samples = int(self.H * self.W * self.sci_sensor_channels)
            self.sci_sampling_ratio = float(self.sci_num_samples) / max(float(self.sci_input_samples), 1.0e-12)
        else:
            default_sci_ratio = 1.0 / float(self.L_video)
            self.sci_sampling_ratio = float(
                np.clip(self._finite_float("sci_sampling_ratio", default_sci_ratio, lower=1.0e-12), 1.0e-12, 1.0)
            )
            self.sci_num_samples = int(np.ceil(float(self.sci_input_samples) * self.sci_sampling_ratio))
        self.sci_access_bits = float(self.sci_num_samples) * self.sci_bits_per_sample
        self.sci_compression_ratio = self.sci_access_bits / max(self.sci_input_bits, 1.0e-12)
        self.end_to_end_access_ratio = self.sci_access_bits / max(self.raw_video_bits, 1.0e-12)
        self.H_s = max(1, int(np.ceil(self.H / 8.0)))
        self.W_s = max(1, int(np.ceil(self.W / 8.0)))
        self.C_m = int(
            self.semantic_library.latent_channels[0]
            if self.semantic_library.enabled
            else self._arg("semantic_channels", 48)
        )
        self.side_mode_bits = int(np.ceil(np.log2(max(self.n_semantic_modes, 2))))
        self.side_info_bits = 2.0 * float(self.H_s * self.W_s) + float(self.side_mode_bits)
        if self.semantic_library.enabled:
            self.mode_rate_levels = self.semantic_library.rate_level_proxy.astype(float)
            self.mode_quality_bias = self.semantic_library.quality_bias_db.astype(float)
            self.mode_content_sensitivity = self.semantic_library.content_sensitivity.astype(float)
        else:
            self.mode_rate_levels = self._mode_vector("semantic_rate_level", 1.0, 4.0)
            self.mode_quality_bias = self._mode_vector("semantic_quality_bias_db", -1.5, 1.5)
            self.mode_content_sensitivity = self._mode_vector("semantic_content_sensitivity", 0.6, 0.2)
        self.n_z_base_scale = self._finite_float("n_z_base_scale", 12.0, lower=1.0e-9)
        self.snr_bucket_values_db = np.asarray([0.0, 5.0, 10.0, 15.0], dtype=float)
        self.mu_comm_levels = np.asarray([1.0e-4, 5.0e-4, 1.0e-3, 2.0e-3], dtype=float)
        self.semantic_mode_selection = str(self._arg("semantic_mode_selection", "snr_bucket_mu"))
        if self.semantic_mode_selection not in ("snr_bucket_mu", "all_modes"):
            raise ValueError("semantic_mode_selection must be snr_bucket_mu or all_modes")
        self.n_mu_modes = (
            self.n_semantic_modes if self.semantic_mode_selection == "all_modes"
            else int(np.clip(int(self._arg("n_mu_modes", 4)), 1, 4))
        )
        self.use_direct_ds_logits = self._as_bool(self._arg("use_direct_ds_logits", False))
        self.scheduling_weight_transform = str(self._arg("scheduling_weight_transform", "softmax")).lower()
        self.scheduling_weight_dim = 3
        self.ds_priority_dim = self.ds_per_uav if self.use_direct_ds_logits else 0
        self.semantic_mode_lookup_by_bucket_mu = self._build_semantic_mode_lookup()

        self.actor_observe_instruction = self._as_bool(self._arg("actor_observe_instruction", True))
        self.num_instructions = max(1, int(self._arg("num_instructions", 3)))
        names = self._parse_str_list(
            self._arg(
                "instruction_names",
                self._arg("instruction_mode_names", ["balance", "aoi", "quality"]),
            )
        )
        if len(names) < self.num_instructions:
            names.extend([f"instruction_{i}" for i in range(len(names), self.num_instructions)])
        self.instruction_names = names[: self.num_instructions]
        self.use_instruction_constraints = self._as_bool(self._arg("use_instruction_constraints", False))
        self.instruction_mode_strategy = str(self._arg("instruction_mode_strategy", "fixed")).strip().lower()
        self.fixed_instruction_id = self._clip_instruction_id(
            self._arg("fixed_instruction_id", self._arg("instruction_mode_fixed", 0))
        )
        self.instruction_candidates = self._instruction_id_array(
            self._arg("instruction_candidates", list(range(self.num_instructions))),
            list(range(self.num_instructions)),
        )
        self.instruction_switch_step = max(0, int(self._arg("instruction_switch_step", 200)))
        self.instruction_switch_min_step = max(0, int(self._arg("instruction_switch_min_step", 200)))
        self.instruction_switch_max_step = max(
            self.instruction_switch_min_step,
            int(self._arg("instruction_switch_max_step", 400)),
        )
        self.instruction_before_id = self._clip_instruction_id(
            self._arg("instruction_before_id", self._arg("instruction_before_mode", 0))
        )
        self.instruction_before_candidates = self._instruction_id_array(
            self._arg("instruction_before_candidates", [self.instruction_before_id]),
            [self.instruction_before_id],
        )
        self.instruction_after_id_raw = self._arg(
            "instruction_after_id",
            self._arg("instruction_after_mode", 1),
        )
        self.instruction_after_id = self._clip_instruction_id(self.instruction_after_id_raw)
        self.instruction_after_id_candidates = self._instruction_id_array(
            self._arg(
                "instruction_after_id_candidates",
                self._arg("instruction_after_mode_candidates", [1, 2]),
            ),
            [1, 2],
        )
        self.instruction_after_candidates = self._instruction_id_array(
            self._arg("instruction_after_candidates", self.instruction_after_id_candidates),
            self.instruction_after_id_candidates,
        )
        self.allow_same_instruction_after_switch = self._as_bool(
            self._arg("allow_same_instruction_after_switch", True)
        )
        self.instruction_after_mode_strategy = str(self._arg("instruction_after_mode_strategy", "")).strip().lower()
        self.include_instruction_id_in_obs = self._as_bool(self._arg("include_instruction_id_in_obs", True))
        self.include_instruction_constraints_in_obs = self._as_bool(
            self._arg("include_instruction_constraints_in_obs", True)
        )
        self.Q_min_eval = self._finite_float("Q_min_eval", self._arg("Q_min", 30.0))
        self.A_limit_context_ref = self._finite_float("A_limit_context_ref", 10.0, lower=1.0e-9)
        self.Q_req_by_instruction_snr_bucket = self._instruction_float_matrix(
            self._arg(
                "Q_req_by_instruction_snr_bucket",
                [
                    [27.336, 30.719, 34.103, 37.486],
                    [26.650, 30.033, 33.417, 36.800],
                    [27.897, 31.280, 34.664, 38.047],
                ],
            ),
            4,
            float(self.Q_min_eval),
        )
        self.A_limit_by_instruction = self._instruction_float_vector(
            self._arg("A_limit_by_instruction", [7.5, 5.5, 10.0]),
            10.0,
        )
        self.Lambda_budget_ratio_by_instruction = self._instruction_float_vector(
            self._arg("Lambda_budget_ratio_by_instruction", [1.0, 0.9, 1.2]),
            1.0,
        )
        self.constraint_penalty_Q = self._finite_float("constraint_penalty_Q", 1.0, lower=0.0)
        self.constraint_penalty_A = self._finite_float("constraint_penalty_A", 1.5, lower=0.0)
        self.constraint_penalty_A_by_instruction = self._instruction_float_vector(
            self._arg(
                "constraint_penalty_A_by_instruction",
                [self.constraint_penalty_A] * self.num_instructions,
            ),
            self.constraint_penalty_A,
        )
        self.constraint_penalty_Lambda = self._finite_float("constraint_penalty_Lambda", 0.1, lower=0.0)
        self.eta_recv_aoi_bonus = self._finite_float("eta_recv_aoi_bonus", 0.1, lower=0.0)
        self.use_snr_bucket_mode_mask = self._as_bool(self._arg("use_snr_bucket_mode_mask", True))
        self.use_quality_feasibility_mask = self._as_bool(self._arg("use_quality_feasibility_mask", True))
        self.use_load_feasibility_mask = self._as_bool(self._arg("use_load_feasibility_mask", False))
        self.current_instruction_id = int(self.fixed_instruction_id)
        self._episode_instruction_after_id = int(self.instruction_after_id)

        # Q_max is the quality-gain ceiling (gain=1 at Q_max). "Q_ref" is the
        # legacy key, still accepted so old config.json runs keep loading.
        self.Q_max = self._finite_float("Q_max", self._arg("Q_ref", 45.0))
        self.Lambda_ref = self._finite_float(
            "Lambda_ref",
            max(1.0, 0.25 * self.delta_T * min(self.B_uav_sut, self.B_sut_sat)),
            lower=1.0e-9,
        )
        self.A_max = self._finite_float("A_max", 600.0, lower=1.0)

        self.Q_min = self._finite_float("Q_min", 30.0)
        self.omega_Q = self._finite_float("omega_Q", 0.50)
        self.omega_Lambda = self._finite_float("omega_Lambda", 0.10)
        self.omega_A = self._finite_float("omega_A", 0.40)
        self.aoi_mean_weight = self._finite_float("aoi_mean_weight", 0.4, lower=0.0)
        self.aoi_max_weight = self._finite_float("aoi_max_weight", 0.3, lower=0.0)
        self.aoi_tail_weight = self._finite_float("aoi_tail_weight", 0.3, lower=0.0)
        self.aoi_tail_threshold = self._finite_float("aoi_tail_threshold", 10.0, lower=0.0)
        self.enable_episode_trace = self._as_bool(self._arg("enable_episode_trace", False))

        self.initial_cache_prob = float(np.clip(self._finite_float("initial_cache_prob", 1.0), 0.0, 1.0))
        self.initial_aoi = self._finite_float("initial_aoi", 1.0, lower=0.0)
        self.content_feature_dim = max(1, int(self._arg("content_feature_dim", 3)))

        self.pos_uav = np.zeros((self.n_uav, 3), dtype=float)
        self.pos_ds = np.zeros((self.n_ds, 3), dtype=float)
        self.owner_uav = np.zeros(self.n_ds, dtype=int)
        self.owner_mask = np.zeros((self.n_uav, self.n_ds), dtype=bool)
        self.ds_by_uav: List[np.ndarray] = []

        self.q_cache = np.zeros((self.n_uav, self.n_ds), dtype=np.int8)
        self.tau_cache = np.full((self.n_uav, self.n_ds), self.INVALID_TAU, dtype=int)
        self.A_rcc = np.full(self.n_ds, self.initial_aoi, dtype=float)
        self.psi = np.zeros((self.n_uav, self.n_ds, self.content_feature_dim), dtype=float)
        self.gamma_uav_sut = np.ones(self.n_uav, dtype=float)
        self.gamma_sut_sat = 1.0
        self.gamma_bh = np.ones(self.n_uav, dtype=float)
        self.B_bh = np.zeros(self.n_uav, dtype=float)
        self.Phi_bh = np.zeros(self.n_uav, dtype=float)
        self.Lambda_sem = np.zeros((self.n_uav, self.n_ds, self.n_semantic_modes), dtype=float)
        self.Q_hat_rec = np.zeros_like(self.Lambda_sem)
        self.M_feas = np.zeros_like(self.Lambda_sem, dtype=bool)
        self.n_z = np.zeros_like(self.Lambda_sem)
        self.L_z = np.zeros_like(self.Lambda_sem)

        self.last_selected_modes = np.full(self.n_uav, -1, dtype=int)
        self.last_chi = np.zeros_like(self.Lambda_sem, dtype=np.int8)
        self.last_y = np.zeros((self.n_uav, self.n_ds), dtype=np.int8)
        self.last_admission = np.zeros((self.n_uav, self.n_ds), dtype=np.int8)
        self.last_lambda_usage = np.zeros(self.n_uav, dtype=float)
        self.last_quality_gain = 0.0
        self.last_load_cost = 0.0
        self.last_A_fair = 0.0
        self.last_quality_term = 0.0
        self.last_load_term = 0.0
        self.last_aoi_term = 0.0
        self.last_aoi_mean_term = 0.0
        self.last_aoi_max_term = 0.0
        self.last_aoi_tail_term = 0.0
        self.last_aoi_state_term = 0.0
        self.last_fresh_gain_term = 0.0
        self.last_quality_contrib = 0.0
        self.last_load_penalty = 0.0
        self.last_aoi_penalty = 0.0
        self.last_utility = 0.0
        self.last_slot_snapshot: Dict[str, Any] = {}
        self.last_effective_snr_db = np.zeros(self.n_uav, dtype=float)
        self.last_snr_bucket_ids = np.zeros(self.n_uav, dtype=int)
        self.last_selected_mu_ids = np.full(self.n_uav, -1, dtype=int)
        self.last_scheduling_alphas = np.zeros((self.n_uav, self.scheduling_weight_dim), dtype=float)
        self.last_direct_ds_priority_logits = np.zeros((self.n_uav, self.ds_per_uav), dtype=float)
        self.last_per_uav_scheduled_count = np.zeros(self.n_uav, dtype=int)
        self.last_per_uav_mean_aoi = np.zeros(self.n_uav, dtype=float)
        self.last_per_uav_max_aoi = np.zeros(self.n_uav, dtype=float)
        self.last_all_mu_infeasible_count = 0
        self.last_instruction_id = int(self.current_instruction_id)
        self.last_instruction_name = self.instruction_names[self.last_instruction_id]
        self.last_A_limit_g = float(self.A_limit_by_instruction[self.last_instruction_id])
        self.last_Lambda_budget_ratio_g = float(self.Lambda_budget_ratio_by_instruction[self.last_instruction_id])
        self.last_Lambda_budget_g = float(self.last_Lambda_budget_ratio_g * self.Lambda_ref * self.n_uav)
        self.last_Q_req_gb = np.zeros(self.n_uav, dtype=float)
        self.last_base_reward = 0.0
        self.last_total_reward = 0.0
        self.last_quality_violation_bucketwise = 0.0
        self.last_aoi_violation_max = 0.0
        self.last_load_violation = 0.0
        self.last_recv_aoi_bonus = 0.0
        self.last_fallback_to_best_feasible_mode_count = 0
        self.last_selected_mode_q_hat = np.zeros(self.n_uav, dtype=float)
        self.last_selected_mode_lambda_sem = np.zeros(self.n_uav, dtype=float)
        self.last_reward_components: Dict[str, float] = {}
        self.last_sut_raw_action = np.zeros(self.n_uav, dtype=float)
        uav_raw_dim = self.n_mu_modes + (
            self.ds_priority_dim if self.use_direct_ds_logits else self.scheduling_weight_dim
        )
        self.last_uav_raw_actions = np.zeros(
            (self.n_uav, uav_raw_dim),
            dtype=float,
        )

        # Action encoding:
        # SUT: simplex logits for beta_sut_sat over UAVs.
        # UAV: masked categorical mu_comm selection inside the current
        # effective-SNR bucket + either per-DS priority logits or the legacy
        # 3-D scheduling-control weights.
        # The final DS set is produced by environment-level feasibility
        # projection and budget-aware greedy selection, so PPO log_prob is over
        # the policy action vector, not over the final DS set.
        self.sut_act_slices = {"beta_logits": slice(0, self.n_uav)}
        mu_slice = slice(0, self.n_mu_modes)
        self.uav_act_slices = {"mu_logits": mu_slice, "mode_logits": mu_slice}
        if self.use_direct_ds_logits:
            priority_slice = slice(self.n_mu_modes, self.n_mu_modes + self.ds_per_uav)
            self.uav_act_slices["ds_priority_logits"] = priority_slice
        else:
            weight_slice = slice(self.n_mu_modes, self.n_mu_modes + self.scheduling_weight_dim)
            self.uav_act_slices["scheduling_weights"] = weight_slice
        self.sut_act_dim_total = self.n_uav
        self.uav_act_dim_total = self.n_mu_modes + (
            self.ds_per_uav if self.use_direct_ds_logits else self.scheduling_weight_dim
        )
        self.common_act_dim = max(self.sut_act_dim_total, self.uav_act_dim_total)

        self.sut_instruction_obs_dim = 0
        self.uav_instruction_obs_dim = 0
        self.share_instruction_obs_dim = 0
        if self.use_instruction_constraints:
            if self.include_instruction_id_in_obs:
                self.sut_instruction_obs_dim += self.num_instructions
                self.uav_instruction_obs_dim += self.num_instructions
                self.share_instruction_obs_dim += self.num_instructions
            if self.include_instruction_constraints_in_obs:
                self.sut_instruction_obs_dim += 2
                self.uav_instruction_obs_dim += 4 + 3
                self.share_instruction_obs_dim += 2 + self.n_uav

        self.sut_obs_dim = 1 + 3 * self.n_uav + self.sut_instruction_obs_dim
        self.uav_obs_dim = (
            1
            + 3 * self.ds_per_uav
            + self.n_semantic_modes
            + self.ds_per_uav * self.n_semantic_modes
            + self.n_uav
            + self.uav_instruction_obs_dim
        )
        self.obs_dim_common = max(self.sut_obs_dim, self.uav_obs_dim)
        self.share_obs_dim = (
            self.n_uav
            + 1
            + self.n_uav
            + self.n_ds
            + self.n_ds
            + self.n_ds
            + self.n_uav * self.n_semantic_modes
            + self.n_ds * self.n_semantic_modes
            + self.share_instruction_obs_dim
        )

        self.observation_space = [
            Box(low=-np.inf, high=np.inf, shape=(self.obs_dim_common,), dtype=np.float32)
            for _ in range(self.n_agents)
        ]
        share_box = Box(low=-np.inf, high=np.inf, shape=(self.share_obs_dim,), dtype=np.float32)
        self.share_observation_space = [share_box for _ in range(self.n_agents)]
        self.action_space = [
            Box(low=-1.0, high=1.0, shape=(self.sut_act_dim_total,), dtype=np.float32)
        ] + [
            Box(low=-1.0, high=1.0, shape=(self.uav_act_dim_total,), dtype=np.float32)
            for _ in range(self.n_uav)
        ]
        self._annotate_action_space_specs()

    def _arg(self, key: str, default: Any) -> Any:
        if isinstance(self.args, dict):
            return self.args.get(key, default)
        return getattr(self.args, key, default)

    def _finite_float(self, key: str, default: float, lower: float | None = None) -> float:
        try:
            value = float(self._arg(key, default))
        except Exception:
            value = float(default)
        if not np.isfinite(value):
            value = float(default)
        if lower is not None:
            value = max(value, float(lower))
        return value

    @staticmethod
    def _as_float_array(raw: Any, size: int, default: Iterable[float]) -> np.ndarray:
        try:
            arr = np.asarray(raw, dtype=float).reshape(-1)
        except Exception:
            arr = np.asarray(default, dtype=float).reshape(-1)
        if arr.size < size:
            arr = np.pad(arr, (0, size - arr.size), mode="edge")
        arr = arr[:size]
        fallback = np.asarray(default, dtype=float).reshape(-1)[:size]
        arr = np.where(np.isfinite(arr), arr, fallback)
        return arr.astype(float)

    @staticmethod
    def _parse_str_list(raw: Any) -> List[str]:
        if isinstance(raw, str):
            text = raw.strip()
            if not text:
                return []
            try:
                raw = ast.literal_eval(text)
            except Exception:
                raw = [part.strip() for part in text.split(",")]
        if not isinstance(raw, (list, tuple)):
            return []
        return [str(item).strip() for item in raw if str(item).strip()]

    def _clip_instruction_id(self, raw: Any) -> int:
        try:
            gid = int(raw)
        except Exception:
            gid = 0
        return int(np.clip(gid, 0, max(self.num_instructions - 1, 0)))

    def _instruction_id_array(self, raw: Any, default: Iterable[int]) -> np.ndarray:
        try:
            arr = np.asarray(raw, dtype=int).reshape(-1)
        except Exception:
            arr = np.asarray(list(default), dtype=int).reshape(-1)
        if arr.size == 0:
            arr = np.asarray(list(default), dtype=int).reshape(-1)
        if arr.size == 0:
            arr = np.asarray([0], dtype=int)
        arr = np.clip(arr, 0, max(self.num_instructions - 1, 0))
        return np.unique(arr.astype(int))

    def _instruction_float_vector(self, raw: Any, default_value: float) -> np.ndarray:
        default = np.full(self.num_instructions, float(default_value), dtype=float)
        try:
            arr = np.asarray(raw, dtype=float).reshape(-1)
        except Exception:
            arr = default
        if arr.size == 0:
            arr = default
        if arr.size < self.num_instructions:
            arr = np.pad(arr, (0, self.num_instructions - arr.size), mode="edge")
        arr = arr[: self.num_instructions]
        return np.where(np.isfinite(arr), arr, default).astype(float)

    def _instruction_float_matrix(self, raw: Any, width: int, default_value: float) -> np.ndarray:
        default = np.full((self.num_instructions, width), float(default_value), dtype=float)
        try:
            arr = np.asarray(raw, dtype=float)
        except Exception:
            arr = default
        if arr.ndim == 0:
            arr = arr.reshape(1, 1)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.size == 0:
            arr = default
        if arr.shape[0] < self.num_instructions:
            pad_rows = np.repeat(arr[-1:, :], self.num_instructions - arr.shape[0], axis=0)
            arr = np.vstack([arr, pad_rows])
        if arr.shape[1] < width:
            pad_cols = np.repeat(arr[:, -1:], width - arr.shape[1], axis=1)
            arr = np.hstack([arr, pad_cols])
        arr = arr[: self.num_instructions, :width]
        return np.where(np.isfinite(arr), arr, default).astype(float)

    @staticmethod
    def _as_bool(raw: Any) -> bool:
        if isinstance(raw, bool):
            return raw
        if isinstance(raw, (int, float)):
            return bool(raw)
        text = str(raw).strip().lower()
        if text in ("1", "true", "yes", "y", "on"):
            return True
        if text in ("0", "false", "no", "n", "off"):
            return False
        return bool(raw)

    def _mode_vector(self, key: str, first: float, last: float) -> np.ndarray:
        default = np.linspace(first, last, self.n_semantic_modes)
        raw = self._arg(key, default)
        return self._as_float_array(raw, self.n_semantic_modes, default)

    def _build_semantic_mode_lookup(self) -> np.ndarray:
        """Map runtime SNR bucket and mu_id to the 16-mode global id.

        The current registry order is SNR-major and mu-minor. This explicit
        lookup keeps the environment correct if a future registry preserves the
        same metadata but changes JSON ordering.
        """
        if self.semantic_mode_selection == "all_modes":
            # Every registered, measured codec is a selectable action. In a
            # partial checkpoint inventory, never alias missing SNR/mu entries
            # to an unrelated checkpoint through the historical bucket lookup.
            return np.tile(np.arange(self.n_semantic_modes, dtype=int), (4, 1))
        lookup = np.full((4, 4), -1, dtype=int)
        if self.semantic_library.enabled and self.semantic_library.num_modes > 0:
            for mode_id in range(self.semantic_library.num_modes):
                snr = float(self.semantic_library.trained_snr_db[mode_id])
                mu = float(self.semantic_library.mu_comm[mode_id])
                bucket_id = int(np.argmin(np.abs(self.snr_bucket_values_db - snr)))
                mu_id = int(np.argmin(np.abs(self.mu_comm_levels - mu)))
                if (
                    abs(float(self.snr_bucket_values_db[bucket_id]) - snr) <= 1.0e-6
                    and abs(float(self.mu_comm_levels[mu_id]) - mu) <= 1.0e-12
                ):
                    lookup[bucket_id, mu_id] = int(mode_id)

        for bucket_id in range(4):
            for mu_id in range(4):
                if lookup[bucket_id, mu_id] < 0:
                    fallback = 4 * bucket_id + mu_id
                    lookup[bucket_id, mu_id] = int(min(max(fallback, 0), self.n_semantic_modes - 1))
        return lookup

    def _annotate_action_space_specs(self):
        sut_spec = {
            "kind": "ic_semantic_video_hybrid",
            "agent_role": "sut",
            "action_dim": int(self.sut_act_dim_total),
            "categorical_groups": [],
            "continuous_groups": [
                {
                    "name": "beta_logits",
                    "kind": "simplex",
                    "parts": [{"name": "beta_logits", "start": 0, "stop": self.n_uav}],
                }
            ],
        }
        self.action_space[0].hybrid_action_spec = sut_spec
        self.action_space[0].action_head_kind = sut_spec["kind"]
        self.action_space[0].hybrid_available_dim = int(self.common_act_dim)

        for n in range(self.n_uav):
            spec = {
                "kind": "ic_semantic_video_hybrid",
                "agent_role": "uav",
                "uav_idx": int(n),
                "action_dim": int(self.uav_act_dim_total),
                "categorical_groups": [
                    {
                        "name": "mu_logits",
                        "start": 0,
                        "stop": self.n_mu_modes,
                        "num_groups": 1,
                        "group_size": self.n_mu_modes,
                    }
                ],
                "continuous_groups": [],
            }
            if self.use_direct_ds_logits:
                spec["continuous_groups"].append(
                    {
                        "name": "ds_priority_logits",
                        "kind": "gaussian",
                        "parts": [
                            {
                                "name": "ds_priority_logits",
                                "start": self.n_mu_modes,
                                "stop": self.n_mu_modes + self.ds_per_uav,
                            }
                        ],
                    }
                )
            else:
                spec["continuous_groups"].append(
                    {
                        "name": "scheduling_weights",
                        "kind": "gaussian",
                        "parts": [
                            {
                                "name": "scheduling_weights",
                                "start": self.n_mu_modes,
                                "stop": self.n_mu_modes + self.scheduling_weight_dim,
                            }
                        ],
                    }
                )
            self.action_space[1 + n].hybrid_action_spec = spec
            self.action_space[1 + n].action_head_kind = spec["kind"]
            self.action_space[1 + n].hybrid_available_dim = int(self.common_act_dim)

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        z = np.asarray(x, dtype=float).reshape(-1)
        if z.size == 0:
            return z
        z = z - np.max(z)
        e = np.exp(np.clip(z, -60.0, 60.0))
        s = float(np.sum(e))
        return e / s if s > 1.0e-12 else np.ones_like(e) / float(e.size)

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(x, dtype=float), -60.0, 60.0)))

    @staticmethod
    def _softplus(x: np.ndarray) -> np.ndarray:
        z = np.clip(np.asarray(x, dtype=float), -60.0, 60.0)
        return np.log1p(np.exp(-np.abs(z))) + np.maximum(z, 0.0)

    def _transform_scheduling_weights(self, raw: np.ndarray) -> np.ndarray:
        raw = np.asarray(raw, dtype=float).reshape(-1)
        if raw.size < self.scheduling_weight_dim:
            raw = np.pad(raw, (0, self.scheduling_weight_dim - raw.size))
        raw = raw[: self.scheduling_weight_dim]
        if self.scheduling_weight_transform == "softplus":
            alpha = self._softplus(raw)
            total = float(np.sum(alpha))
            return alpha / total if total > 1.0e-12 else np.ones(self.scheduling_weight_dim) / float(self.scheduling_weight_dim)
        return self._softmax(raw)

    def _snr_to_bucket_id(self, snr_db: float) -> int:
        snr = float(snr_db)
        if snr < 5.0:
            return 0
        if snr < 10.0:
            return 1
        if snr < 15.0:
            return 2
        return 3

    def _effective_snr_db_for_uav(self, n: int) -> float:
        # Bucket by the end-to-end backhaul SNR (gamma_bh) that actually drives
        # the semantic mode quality/load tables, instead of the conservative
        # min-link approximation, so the SNR bucket and the trained profile stay
        # consistent.
        gamma_eff = float(self.gamma_bh[n])
        return 10.0 * float(np.log10(max(gamma_eff, 1.0e-12)))

    def _effective_snr_bucket_for_uav(self, n: int) -> int:
        return self._snr_to_bucket_id(self._effective_snr_db_for_uav(n))

    def _global_mode_id(self, snr_bucket_id: int, mu_id: int) -> int:
        bucket = int(np.clip(snr_bucket_id, 0, self.semantic_mode_lookup_by_bucket_mu.shape[0] - 1))
        mu = int(np.clip(mu_id, 0, self.semantic_mode_lookup_by_bucket_mu.shape[1] - 1))
        return int(np.clip(self.semantic_mode_lookup_by_bucket_mu[bucket, mu], 0, self.n_semantic_modes - 1))

    def _bucket_mode_ids(self, snr_bucket_id: int) -> np.ndarray:
        bucket = int(np.clip(snr_bucket_id, 0, self.semantic_mode_lookup_by_bucket_mu.shape[0] - 1))
        return np.asarray(
            [self._global_mode_id(bucket, mu_id) for mu_id in range(self.n_mu_modes)],
            dtype=int,
        )

    def _instruction_one_hot(self, gid: int) -> np.ndarray:
        out = np.zeros(self.num_instructions, dtype=float)
        out[self._clip_instruction_id(gid)] = 1.0
        return out

    def _sample_instruction_id(self, candidates: np.ndarray, default_gid: int) -> int:
        arr = np.asarray(candidates, dtype=int).reshape(-1)
        if arr.size == 0:
            return self._clip_instruction_id(default_gid)
        return int(self.rng_instruction.choice(np.clip(arr, 0, max(self.num_instructions - 1, 0))))

    def _sample_after_instruction_id(self, before_gid: int) -> int:
        candidates = np.asarray(self.instruction_after_candidates, dtype=int).reshape(-1)
        if not self.allow_same_instruction_after_switch:
            candidates = candidates[candidates != int(before_gid)]
        if candidates.size == 0:
            candidates = np.asarray(self.instruction_after_candidates, dtype=int).reshape(-1)
        return self._sample_instruction_id(candidates, self.instruction_after_id)

    def _get_instruction_id_for_slot(self, slot_t: int) -> int:
        if not self.use_instruction_constraints:
            return int(self.fixed_instruction_id)
        strategy = self.instruction_mode_strategy
        if strategy == "fixed":
            return int(self.fixed_instruction_id)
        if strategy == "random_fixed":
            return int(self.current_instruction_id)
        if strategy in ("switch_once", "random_switch_once"):
            if int(slot_t) < int(self.instruction_switch_step):
                return int(self.instruction_before_id)
            return int(self.instruction_after_id)
        return int(self.fixed_instruction_id)

    def _reset_instruction_schedule(self) -> None:
        strategy = self.instruction_mode_strategy
        if strategy == "random_switch_once":
            lo = max(0, int(self.instruction_switch_min_step))
            hi = max(lo, int(self.instruction_switch_max_step))
            hi = min(hi, max(int(self.max_steps) - 1, 0))
            lo = min(lo, hi)
            self.instruction_switch_step = int(self.rng_instruction.integers(lo, hi + 1))
            self.instruction_before_id = self._sample_instruction_id(
                self.instruction_before_candidates,
                self.instruction_before_id,
            )
            self.instruction_after_id = self._sample_after_instruction_id(self.instruction_before_id)
            self._episode_instruction_after_id = int(self.instruction_after_id)
            self.current_instruction_id = self._get_instruction_id_for_slot(0)
            self._refresh_instruction_context_from_current()
            return

        if strategy == "random_fixed":
            self.current_instruction_id = self._sample_instruction_id(
                self.instruction_candidates,
                self.fixed_instruction_id,
            )
        else:
            self.current_instruction_id = int(self.fixed_instruction_id)

        after_mode = self.instruction_after_mode_strategy
        raw_after_text = str(self.instruction_after_id_raw).strip().lower()
        if after_mode in ("random", "random_fixed", "random_after") or raw_after_text in (
            "random",
            "random_after",
        ):
            self._episode_instruction_after_id = self._sample_instruction_id(
                self.instruction_after_id_candidates,
                self.instruction_after_id,
            )
        else:
            self._episode_instruction_after_id = int(self.instruction_after_id)

        if strategy == "switch_once":
            self.current_instruction_id = int(self.instruction_before_id)
            self.instruction_after_id = int(self._episode_instruction_after_id)
        self._refresh_instruction_context_from_current()

    def _update_instruction_for_slot(self, slot_t: int, refresh_last: bool = True) -> None:
        self.current_instruction_id = self._clip_instruction_id(self._get_instruction_id_for_slot(slot_t))
        if refresh_last:
            self._refresh_instruction_context_from_current()

    def _instruction_context_for_buckets(self, bucket_ids: np.ndarray | None = None) -> Tuple[int, str, float, float, float, np.ndarray]:
        gid = self._clip_instruction_id(self.current_instruction_id)
        name = self.instruction_names[gid]
        a_limit = float(self.A_limit_by_instruction[gid])
        lambda_ratio = float(self.Lambda_budget_ratio_by_instruction[gid])
        lambda_budget = float(lambda_ratio * self.Lambda_ref * self.n_uav)
        if bucket_ids is None:
            bucket_ids = np.asarray([self._effective_snr_bucket_for_uav(n) for n in range(self.n_uav)], dtype=int)
        bucket_ids = np.asarray(bucket_ids, dtype=int).reshape(-1)
        q_req = np.zeros(self.n_uav, dtype=float)
        for n in range(self.n_uav):
            bucket = int(bucket_ids[n]) if n < bucket_ids.size else 0
            q_req[n] = self._q_req_for_instruction_bucket(gid, bucket)
        return gid, name, a_limit, lambda_ratio, lambda_budget, q_req

    def _refresh_instruction_context_from_current(self) -> None:
        gid, name, a_limit, lambda_ratio, lambda_budget, q_req = self._instruction_context_for_buckets()
        self.last_instruction_id = int(gid)
        self.last_instruction_name = str(name)
        self.last_A_limit_g = float(a_limit)
        self.last_Lambda_budget_ratio_g = float(lambda_ratio)
        self.last_Lambda_budget_g = float(lambda_budget)
        self.last_Q_req_gb = q_req.astype(float)

    def _q_req_for_instruction_bucket(self, gid: int, bucket_id: int) -> float:
        g = self._clip_instruction_id(gid)
        b = int(np.clip(bucket_id, 0, self.Q_req_by_instruction_snr_bucket.shape[1] - 1))
        return float(self.Q_req_by_instruction_snr_bucket[g, b])

    def feasible_modes_for_stream(
        self,
        n: int,
        k: int,
        gid: int | None = None,
        bucket_id: int | None = None,
    ) -> np.ndarray:
        """Return modes satisfying the shared instruction quality rule.

        The active scientific feasibility rule is Q_hat_rec >= Q_req(g, bucket).
        M_feas/Q_min is kept as a legacy diagnostic and reward-normalization
        reference, but it is not used to decide scheduled-mode feasibility.
        """
        if gid is None:
            gid = self.current_instruction_id
        if bucket_id is None:
            bucket_id = self._effective_snr_bucket_for_uav(n)
        q_req = self._q_req_for_instruction_bucket(int(gid), int(bucket_id))
        feasible = np.asarray(self.Q_hat_rec[n, k, :] >= q_req - 1.0e-9, dtype=bool)
        feasible &= bool(self.owner_mask[n, k])
        return feasible

    def candidate_mask(
        self,
        n: int,
        mode_id: int,
        q_state: np.ndarray,
        q_req_gb: float,
        include_load_budget: bool = True,
    ) -> np.ndarray:
        candidate = self.owner_mask[n] & (q_state[n] > 0)
        candidate &= self.Q_hat_rec[n, :, mode_id] >= float(q_req_gb) - 1.0e-9
        if include_load_budget:
            candidate &= self.Lambda_sem[n, :, mode_id] <= self.Phi_bh[n] + 1.0e-9
        return candidate

    def _instruction_candidate_mask(
        self,
        n: int,
        mode_id: int,
        q_state: np.ndarray,
        q_req_gb: float,
    ) -> np.ndarray:
        return self.candidate_mask(n, mode_id, q_state, q_req_gb, include_load_budget=True)

    def _base_candidate_mask(self, n: int, mode_id: int, q_state: np.ndarray) -> np.ndarray:
        bucket_id = self._effective_snr_bucket_for_uav(n)
        gid = self._clip_instruction_id(self.current_instruction_id)
        q_req_gb = self._q_req_for_instruction_bucket(gid, bucket_id)
        return self.candidate_mask(n, mode_id, q_state, q_req_gb, include_load_budget=True)

    def _derived_schedule_scores(
        self,
        n: int,
        mode_id: int,
        bucket_id: int,
        alpha: np.ndarray,
        q_t: np.ndarray,
        A_t: np.ndarray,
        q_req_gb: float,
        A_limit_g: float,
        use_instruction_quality: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        del use_instruction_quality
        candidate = self._instruction_candidate_mask(n, mode_id, q_t, q_req_gb)
        scores = np.full(self.n_ds, -np.inf, dtype=float)
        if not np.any(candidate):
            return scores, candidate
        bucket_modes = self._bucket_mode_ids(bucket_id)
        q_bucket = self.Q_hat_rec[n][:, bucket_modes]
        q_best_bucket = np.max(q_bucket, axis=1)
        q_worst_bucket = np.min(q_bucket, axis=1)
        norm_aoi = np.asarray(A_t, dtype=float) / max(float(A_limit_g), 1.0e-9)
        norm_q = (self.Q_hat_rec[n, :, mode_id] - q_worst_bucket) / (
            q_best_bucket - q_worst_bucket + 1.0e-9
        )
        norm_q = np.clip(norm_q, 0.0, 1.0)
        norm_load = self.Lambda_sem[n, :, mode_id] / max(float(self.Lambda_ref), 1.0e-9)
        scores[candidate] = (
            float(alpha[0]) * norm_aoi[candidate]
            + float(alpha[1]) * norm_q[candidate]
            - float(alpha[2]) * norm_load[candidate]
        )
        return scores, candidate

    def _direct_ds_priority_scores(
        self,
        n: int,
        mode_id: int,
        ds_priority_logits: np.ndarray,
        q_t: np.ndarray,
        q_req_gb: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        candidate = self._instruction_candidate_mask(n, mode_id, q_t, q_req_gb)
        scores = np.full(self.n_ds, -np.inf, dtype=float)
        ds = self.ds_by_uav[n]
        logits = np.asarray(ds_priority_logits, dtype=float).reshape(-1)
        if logits.size < ds.size:
            logits = np.pad(logits, (0, ds.size - logits.size), constant_values=-np.inf)
        scores[ds] = logits[: ds.size]
        scores[~candidate] = -np.inf
        return scores, candidate

    def _project_beta(self, raw: np.ndarray) -> np.ndarray:
        beta = self._softmax(raw)
        lb = float(self.beta_sat_lower_bound)
        beta = lb + (1.0 - lb * self.n_uav) * beta
        beta = np.clip(beta, lb, 1.0)
        beta = beta / max(float(np.sum(beta)), 1.0e-12)
        beta = beta.astype(float)
        self._assert_beta_constraints(beta)
        return beta

    def _assert_beta_constraints(self, beta: np.ndarray, tol: float = 1.0e-8) -> None:
        beta = np.asarray(beta, dtype=float).reshape(-1)
        if beta.size != self.n_uav:
            raise AssertionError(f"beta size {beta.size} != n_uav {self.n_uav}")
        if not np.all(np.isfinite(beta)):
            raise AssertionError("beta_sut_sat contains non-finite values")
        if np.any(beta < -tol):
            raise AssertionError(f"beta_sut_sat has negative entries: {beta.tolist()}")
        if np.any(beta > 1.0 + tol):
            raise AssertionError(f"beta_sut_sat entries exceed 1: {beta.tolist()}")
        if abs(float(np.sum(beta)) - 1.0) > tol:
            raise AssertionError(f"beta_sut_sat must sum to 1.0, got {float(np.sum(beta))}")
        lb = float(self.beta_sat_lower_bound)
        if lb > 0.0 and np.any(beta < lb - tol):
            raise AssertionError(f"beta_sut_sat violates lower bound {lb}: {beta.tolist()}")

    def _assert_common_uav_modes(self, chi: np.ndarray) -> None:
        for n in range(self.n_uav):
            used = np.argwhere(np.asarray(chi[n]) > 0)
            if used.size == 0:
                continue
            modes = np.unique(used[:, 1])
            if modes.size > 1:
                raise AssertionError(f"UAV {n} scheduled multiple semantic modes in one slot: {modes.tolist()}")

    def _assert_scheduled_feasibility(self, chi: np.ndarray, q_req_gb: np.ndarray, tol: float = 1.0e-8) -> None:
        q_req_gb = np.asarray(q_req_gb, dtype=float).reshape(-1)
        for n, k, m in np.argwhere(np.asarray(chi) > 0):
            q_req = float(q_req_gb[n]) if n < q_req_gb.size else self._q_req_for_instruction_bucket(
                self.current_instruction_id, self._effective_snr_bucket_for_uav(int(n))
            )
            q_hat = float(self.Q_hat_rec[n, k, m])
            if q_hat + tol < q_req:
                raise AssertionError(
                    f"Scheduled infeasible mode n={n}, k={k}, m={m}: Q_hat={q_hat:.6f} < Q_req={q_req:.6f}"
                )

    def _normalize_actions_by_agent(self, actions) -> Tuple[np.ndarray, List[np.ndarray]]:
        if isinstance(actions, np.ndarray) and actions.dtype != object:
            arr = np.asarray(actions, dtype=float)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            if arr.shape[0] != self.n_agents:
                raise ValueError(f"actions rows mismatch: got {arr.shape[0]}, expected {self.n_agents}")
            seq = [arr[i] for i in range(self.n_agents)]
        elif isinstance(actions, (list, tuple)):
            seq = list(actions)
            if len(seq) != self.n_agents:
                raise ValueError(f"actions len mismatch: got {len(seq)}, expected {self.n_agents}")
        else:
            raise ValueError(f"Unsupported actions type: {type(actions)}")

        sut = np.asarray(seq[0], dtype=float).reshape(-1)
        if sut.size < self.sut_act_dim_total:
            sut = np.pad(sut, (0, self.sut_act_dim_total - sut.size))
        uav = []
        for n in range(self.n_uav):
            a = np.asarray(seq[1 + n], dtype=float).reshape(-1)
            if a.size < self.uav_act_dim_total:
                a = np.pad(a, (0, self.uav_act_dim_total - a.size))
            uav.append(a[: self.uav_act_dim_total])
        return sut[: self.sut_act_dim_total], uav

    def seed(self, seed=None):
        self._base_seed = int(np.random.SeedSequence(seed).entropy)
        self._episode_index = -1

    def _reset_episode_random_streams(self):
        self._episode_index += 1
        streams = np.random.SeedSequence([self._base_seed, self._episode_index]).spawn(5)
        (self.rng_topology, self.rng_channel, self.rng_content,
         self.rng_instruction, self.rng_cache) = [np.random.default_rng(s) for s in streams]

    def reset(self):
        self._reset_episode_random_streams()
        self.current_step = 0
        self._sample_topology()
        self.beta_sut_sat = np.ones(self.n_uav, dtype=float) / float(self.n_uav)

        self.q_cache.fill(0)
        if self.initial_cache_prob > 0.0:
            init = self.rng_cache.random((self.n_uav, self.n_ds)) < self.initial_cache_prob
            self.q_cache = (init & self.owner_mask).astype(np.int8)
        self.tau_cache.fill(self.INVALID_TAU)
        self.tau_cache[self.q_cache > 0] = 0
        self.A_rcc = np.full(self.n_ds, self.initial_aoi, dtype=float)

        self.last_selected_modes = np.full(self.n_uav, -1, dtype=int)
        self.last_chi = np.zeros_like(self.Lambda_sem, dtype=np.int8)
        self.last_y = np.zeros((self.n_uav, self.n_ds), dtype=np.int8)
        self.last_admission = np.zeros((self.n_uav, self.n_ds), dtype=np.int8)
        self.last_lambda_usage = np.zeros(self.n_uav, dtype=float)
        self.last_quality_gain = 0.0
        self.last_load_cost = 0.0
        self.last_A_fair = float(np.max(self.A_rcc) / self.A_max)
        self.last_quality_term = 0.0
        self.last_load_term = 0.0
        self.last_aoi_mean_term = float(np.mean(self.A_rcc / self.A_max)) if self.n_ds > 0 else 0.0
        self.last_aoi_max_term = float(np.max(self.A_rcc / self.A_max)) if self.n_ds > 0 else 0.0
        self.last_aoi_tail_term = float(
            np.mean(np.maximum((self.A_rcc - self.aoi_tail_threshold) / self.A_max, 0.0))
        ) if self.n_ds > 0 else 0.0
        self.last_aoi_state_term = (
            self.aoi_mean_weight * self.last_aoi_mean_term
            + self.aoi_max_weight * self.last_aoi_max_term
            + self.aoi_tail_weight * self.last_aoi_tail_term
        )
        self.last_aoi_term = self.last_aoi_state_term
        self.last_fresh_gain_term = 0.0
        self.last_quality_contrib = 0.0
        self.last_load_penalty = 0.0
        self.last_aoi_penalty = 0.0
        self.last_utility = 0.0
        self.last_effective_snr_db = np.zeros(self.n_uav, dtype=float)
        self.last_snr_bucket_ids = np.zeros(self.n_uav, dtype=int)
        self.last_selected_mu_ids = np.full(self.n_uav, -1, dtype=int)
        self.last_scheduling_alphas = np.zeros((self.n_uav, self.scheduling_weight_dim), dtype=float)
        self.last_direct_ds_priority_logits = np.zeros((self.n_uav, self.ds_per_uav), dtype=float)
        self.last_per_uav_scheduled_count = np.zeros(self.n_uav, dtype=int)
        self.last_per_uav_mean_aoi = np.zeros(self.n_uav, dtype=float)
        self.last_per_uav_max_aoi = np.zeros(self.n_uav, dtype=float)
        self.last_all_mu_infeasible_count = 0
        self.last_base_reward = 0.0
        self.last_total_reward = 0.0
        self.last_quality_violation_bucketwise = 0.0
        self.last_aoi_violation_max = 0.0
        self.last_load_violation = 0.0
        self.last_recv_aoi_bonus = 0.0
        self.last_fallback_to_best_feasible_mode_count = 0
        self.last_selected_mode_q_hat = np.zeros(self.n_uav, dtype=float)
        self.last_selected_mode_lambda_sem = np.zeros(self.n_uav, dtype=float)
        self.last_reward_components = {}

        self._sample_content_features()
        self._update_channels()
        self._update_semantic_tables()
        self._reset_instruction_schedule()
        return self._build_obs_multi(), self._build_share_obs_multi(), self._build_action_available_masks()

    def _sample_xy_outside_sut(self, count: int, min_distance_m: float) -> np.ndarray:
        xy = np.zeros((count, 2), dtype=float)
        filled = 0
        center = self.pos_sut[:2]
        batch = max(64, 4 * count)
        attempts = 0
        while filled < count and attempts < 1000:
            candidates = self.rng_topology.uniform(
                low=np.asarray([0.0, 0.0], dtype=float),
                high=self.map_size[:2],
                size=(batch, 2),
            )
            dist = np.linalg.norm(candidates - center[None, :], axis=1)
            valid = candidates[dist >= min_distance_m]
            take = min(count - filled, valid.shape[0])
            if take > 0:
                xy[filled : filled + take] = valid[:take]
                filled += take
            attempts += 1
        if filled < count:
            raise RuntimeError(
                f"Cannot sample {count} UAV positions at least {min_distance_m}m from the centered SUT "
                f"inside map_size={self.map_size.tolist()}"
            )
        return xy

    def _sample_xy_around_uav(self, center_xy: np.ndarray, count: int, radius_m: float) -> np.ndarray:
        xy = np.zeros((count, 2), dtype=float)
        if count <= 0:
            return xy
        center = np.asarray(center_xy, dtype=float).reshape(2)
        filled = 0
        batch = max(64, 4 * count)
        attempts = 0
        while filled < count and attempts < 1000:
            r = radius_m * np.sqrt(self.rng_topology.uniform(0.0, 1.0, size=batch))
            theta = self.rng_topology.uniform(0.0, 2.0 * np.pi, size=batch)
            candidates = np.column_stack((center[0] + r * np.cos(theta), center[1] + r * np.sin(theta)))
            valid = candidates[
                (candidates[:, 0] >= 0.0)
                & (candidates[:, 0] <= self.map_size[0])
                & (candidates[:, 1] >= 0.0)
                & (candidates[:, 1] <= self.map_size[1])
            ]
            take = min(count - filled, valid.shape[0])
            if take > 0:
                xy[filled : filled + take] = valid[:take]
                filled += take
            attempts += 1
        if filled < count:
            raise RuntimeError(
                f"Cannot sample {count} DS positions within {radius_m}m of UAV at {center.tolist()} "
                f"inside map_size={self.map_size.tolist()}"
            )
        return xy

    def _sample_topology(self):
        self.pos_sut[:] = np.asarray(
            [0.5 * self.map_size[0], 0.5 * self.map_size[1], self.sut_height_m],
            dtype=float,
        )

        self.pos_uav[:, :2] = self._sample_xy_outside_sut(self.n_uav, self.uav_min_distance_to_sut_m)
        self.pos_uav[:, 2] = self.uav_altitude_m

        self.owner_uav = np.repeat(np.arange(self.n_uav, dtype=int), self.ds_per_uav)
        self.owner_mask = np.zeros((self.n_uav, self.n_ds), dtype=bool)
        self.owner_mask[self.owner_uav, np.arange(self.n_ds)] = True
        self.ds_by_uav = [np.where(self.owner_uav == n)[0].astype(int) for n in range(self.n_uav)]

        for n, ds_idx in enumerate(self.ds_by_uav):
            self.pos_ds[ds_idx, :2] = self._sample_xy_around_uav(
                self.pos_uav[n, :2],
                int(ds_idx.size),
                self.ds_uav_service_radius_m,
            )
        self.pos_ds[:, 2] = self.ds_height_m

    def _sample_content_features(self):
        self.psi = self.rng_content.random((self.n_uav, self.n_ds, self.content_feature_dim))
        self.psi *= self.owner_mask[:, :, None]

    def _sample_new_content_features(self, admission: np.ndarray):
        admitted = (np.asarray(admission, dtype=np.int8) > 0) & self.owner_mask
        # Draw potential content for every (slot, UAV, DS), including when no
        # admission happens. Different policies consume identical exogenous RNG.
        potential = self.rng_content.random(self.psi.shape)
        self.psi[admitted] = potential[admitted]

    def _path_loss_fspl_db(self, dist_m: np.ndarray, fc_hz: float, sigma_db: float) -> np.ndarray:
        d = np.maximum(np.asarray(dist_m, dtype=float), 1.0)
        fspl = 20.0 * np.log10(d) + 20.0 * np.log10(float(fc_hz)) + 20.0 * np.log10(4.0 * np.pi / self.c_light)
        if sigma_db > 0.0:
            fspl = fspl + self.rng_channel.normal(0.0, sigma_db, size=np.shape(fspl))
        return fspl

    def _path_loss_atg_db(
        self, pos_air: np.ndarray, pos_ground: np.ndarray, fc_hz: float, sigma_db: float
    ) -> np.ndarray:
        air = np.asarray(pos_air, dtype=float)
        ground = np.asarray(pos_ground, dtype=float)
        d3 = np.maximum(np.linalg.norm(air - ground, axis=-1), 1.0)
        d2 = np.linalg.norm(air[..., :2] - ground[..., :2], axis=-1)
        elev_deg = np.degrees(np.arctan2(np.abs(air[..., 2] - ground[..., 2]), d2 + 1.0e-9))
        p_los = 1.0 / (1.0 + self.atg_los_a * np.exp(-self.atg_los_b * (elev_deg - self.atg_los_a)))
        fspl = 20.0 * np.log10(d3) + 20.0 * np.log10(float(fc_hz)) + 20.0 * np.log10(4.0 * np.pi / self.c_light)
        loss = fspl + p_los * self.atg_eta_los_db + (1.0 - p_los) * self.atg_eta_nlos_db
        if sigma_db > 0.0:
            loss = loss + self.rng_channel.normal(0.0, sigma_db, size=np.shape(loss))
        return loss

    def _snr_from_path_loss(
        self,
        p_tx_w: float,
        pl_db: np.ndarray,
        bandwidth_hz: float,
        noise_figure_db: float,
        link_gain_db: float = 0.0,
    ) -> np.ndarray:
        noise = self.N0 * float(bandwidth_hz) * (10.0 ** (float(noise_figure_db) / 10.0))
        gain = 10.0 ** ((float(link_gain_db) - np.asarray(pl_db, dtype=float)) / 10.0)
        return np.maximum(float(p_tx_w) * gain / max(noise, 1.0e-30), 1.0e-12)

    def _update_channels(self):
        pl_uav_sut = self._path_loss_atg_db(
            self.pos_uav, self.pos_sut[None, :], self.f_uav_sut, self.sigma_uav_sut_db
        )
        self.gamma_uav_sut = self._snr_from_path_loss(
            self.p_uav_sut_w,
            pl_uav_sut,
            self.B_uav_sut,
            self.nf_uav_sut_db,
            self.g_uav_sut_tx_db + self.g_uav_sut_rx_db,
        )

        pl_sut_sat = self._path_loss_fspl_db(
            np.asarray([self.satellite_slant_range_m]), self.f_sut_sat, self.sigma_sut_sat_db
        )[0]
        self.gamma_sut_sat = float(
            self._snr_from_path_loss(
                self.p_sut_sat_w,
                pl_sut_sat,
                self.B_sut_sat,
                self.nf_sut_sat_db,
                self.g_sut_sat_tx_db + self.g_sut_sat_rx_db,
            )
        )

        pl_ds_uav = self._path_loss_atg_db(
            self.pos_uav[:, None, :], self.pos_ds[None, :, :], self.f_ds_uav, self.sigma_ds_uav_db
        )
        self.gamma_ds_uav = self._snr_from_path_loss(
            self.p_ds_uav_w,
            pl_ds_uav,
            self.B_ds_uav,
            self.nf_ds_uav_db,
            self.g_ds_uav_tx_db + self.g_ds_uav_rx_db,
        )
        self.rate_ds_uav = self.B_ds_uav * np.log2(1.0 + self.gamma_ds_uav)

        self.gamma_bh = (
            self.gamma_uav_sut
            * self.gamma_sut_sat
            / (self.gamma_uav_sut + self.gamma_sut_sat + 1.0)
        )
        self.gamma_bh = np.maximum(self.gamma_bh, 1.0e-12)
        self._update_backhaul_budget()

    def _update_backhaul_budget(self):
        """Apply SUT allocation without advancing the observed channel state."""
        self.B_bh = np.minimum(self.B_uav_sut, self.beta_sut_sat * self.B_sut_sat)
        # Channel-use budget model: Lambda_sem is expressed in equivalent
        # channel uses, not raw bits. If Lambda_sem is later redefined as bit
        # load, this budget should become delta_T * B_bh * log2(1 + gamma_bh).
        self.Phi_bh = self.delta_T * self.B_bh

    def _update_semantic_tables(self):
        psi_mean = np.mean(self.psi, axis=-1)
        if self.semantic_library.enabled:
            tables = self.semantic_library.predict_mode_tables(
                gamma_bh=self.gamma_bh,
                psi_mean=psi_mean,
                h_s=self.H_s,
                w_s=self.W_s,
                side_info_bits=self.side_info_bits,
            )
            self.n_z = tables["n_z"]
            self.L_z = tables["L_z"]
            self.Lambda_sem = tables["Lambda_sem"]
            self.Q_hat_rec = tables["Q_hat_rec"]
            self.M_feas = (self.Q_hat_rec >= self.Q_min) & self.owner_mask[:, :, None]
            return

        snr_db = 10.0 * np.log10(np.maximum(self.gamma_bh, 1.0e-12))
        for n in range(self.n_uav):
            for m in range(self.n_semantic_modes):
                avg_rate_level = float(np.clip(self.mode_rate_levels[m], 1.0, 4.0))
                n_z_val = self.n_z_base_scale * avg_rate_level * float(self.H_s * self.W_s)
                self.n_z[n, :, m] = n_z_val
                self.L_z[n, :, m] = np.ceil(n_z_val / 2.0)
                side_uses = self.side_info_bits / np.log2(1.0 + self.gamma_bh[n])
                self.Lambda_sem[n, :, m] = self.L_z[n, :, m] + side_uses
                self.Q_hat_rec[n, :, m] = (
                    26.0
                    + 2.1 * snr_db[n]
                    + 1.5 * float(m)
                    + self.mode_quality_bias[m]
                    - self.mode_content_sensitivity[m] * psi_mean[n]
                )
        self.M_feas = (self.Q_hat_rec >= self.Q_min) & self.owner_mask[:, :, None]

    def _build_obs_multi(self) -> np.ndarray:
        obs = [self._pad_obs(self._build_sut_obs())]
        for n in range(self.n_uav):
            obs.append(self._pad_obs(self._build_uav_obs(n)))
        return np.asarray(obs, dtype=np.float32)

    def _pad_obs(self, obs: np.ndarray) -> np.ndarray:
        out = np.zeros(self.obs_dim_common, dtype=np.float32)
        v = np.asarray(obs, dtype=np.float32).reshape(-1)
        out[: min(out.size, v.size)] = v[: out.size]
        return out

    def _sut_instruction_obs(self) -> np.ndarray:
        if not self.use_instruction_constraints:
            return np.zeros(0, dtype=float)
        if not self.actor_observe_instruction:
            return np.zeros(self.sut_instruction_obs_dim, dtype=float)
        gid, _, a_limit, lambda_ratio, _, _ = self._instruction_context_for_buckets()
        parts = []
        if self.include_instruction_id_in_obs:
            parts.append(self._instruction_one_hot(gid))
        if self.include_instruction_constraints_in_obs:
            parts.append(np.asarray([a_limit / self.A_limit_context_ref, lambda_ratio], dtype=float))
        return np.concatenate(parts) if parts else np.zeros(0, dtype=float)

    def _uav_instruction_obs(self, n: int) -> np.ndarray:
        if not self.use_instruction_constraints:
            return np.zeros(0, dtype=float)
        bucket_id = self._effective_snr_bucket_for_uav(n)
        gid, _, a_limit, lambda_ratio, _, q_req = self._instruction_context_for_buckets(
            np.asarray([self._effective_snr_bucket_for_uav(i) for i in range(self.n_uav)], dtype=int)
        )
        parts = []
        if self.include_instruction_id_in_obs:
            parts.append(self._instruction_one_hot(gid))
        if self.include_instruction_constraints_in_obs:
            bucket_one_hot = np.zeros(4, dtype=float)
            bucket_one_hot[int(np.clip(bucket_id, 0, 3))] = 1.0
            parts.append(bucket_one_hot)
            parts.append(
                np.asarray(
                    [
                        q_req[n] / max(float(self.Q_max), 1.0e-9),
                        a_limit / self.A_limit_context_ref,
                        lambda_ratio,
                    ],
                    dtype=float,
                )
            )
        result = np.concatenate(parts) if parts else np.zeros(0, dtype=float)
        if not self.actor_observe_instruction:
            # Preserve the physical SNR bucket; remove only task ID/targets.
            cursor = self.num_instructions if self.include_instruction_id_in_obs else 0
            result[:cursor] = 0.0
            if self.include_instruction_constraints_in_obs:
                result[cursor + 4:] = 0.0
        return result

    def _share_instruction_obs(self) -> np.ndarray:
        if not self.use_instruction_constraints:
            return np.zeros(0, dtype=float)
        buckets = np.asarray([self._effective_snr_bucket_for_uav(n) for n in range(self.n_uav)], dtype=int)
        gid, _, a_limit, lambda_ratio, _, q_req = self._instruction_context_for_buckets(buckets)
        parts = []
        if self.include_instruction_id_in_obs:
            parts.append(self._instruction_one_hot(gid))
        if self.include_instruction_constraints_in_obs:
            parts.append(np.asarray([a_limit / self.A_limit_context_ref, lambda_ratio], dtype=float))
            parts.append(q_req / max(float(self.Q_max), 1.0e-9))
        return np.concatenate(parts) if parts else np.zeros(0, dtype=float)

    def _build_sut_obs(self) -> np.ndarray:
        h_pend = np.zeros(self.n_uav, dtype=float)
        a_max_n = np.zeros(self.n_uav, dtype=float)
        for n in range(self.n_uav):
            ds = self.ds_by_uav[n]
            if ds.size > 0:
                a_max_n[n] = np.max(self.A_rcc[ds]) / self.A_max
                bucket_id = self._effective_snr_bucket_for_uav(n)
                gid = self._clip_instruction_id(self.current_instruction_id)
                for k in ds:
                    feasible_modes = self.feasible_modes_for_stream(n, int(k), gid=gid, bucket_id=bucket_id)
                    if self.q_cache[n, k] > 0 and np.any(feasible_modes):
                        h_pend[n] += float(np.min(self.Lambda_sem[n, k, feasible_modes]))
        return np.concatenate(
            [
                self.gamma_uav_sut / 100.0,
                np.asarray([self.gamma_sut_sat / 100.0]),
                h_pend / self.Lambda_ref,
                a_max_n,
                self._sut_instruction_obs(),
            ]
        )

    def _build_uav_obs(self, n: int) -> np.ndarray:
        ds = self.ds_by_uav[n]
        tau_local = self.tau_cache[n, ds]
        cached_age = np.where(
            tau_local >= 0,
            np.maximum(0.0, float(self.current_step) - tau_local.astype(float)) / self.A_max,
            0.0,
        )
        lambda_by_mode = np.mean(self.Lambda_sem[n, ds, :], axis=0) / self.Lambda_ref
        return np.concatenate(
            [
                np.asarray([self.Phi_bh[n] / self.Lambda_ref], dtype=float),
                self.q_cache[n, ds].astype(float),
                cached_age,
                self.A_rcc[ds] / self.A_max,
                lambda_by_mode,
                self.Q_hat_rec[n, ds, :].reshape(-1) / self.Q_max,
                np.eye(self.n_uav, dtype=float)[n],
                self._uav_instruction_obs(n),
            ]
        )

    def _build_share_obs_multi(self) -> np.ndarray:
        q_owned = np.zeros(self.n_ds, dtype=float)
        age_owned = np.zeros(self.n_ds, dtype=float)
        q_hat_owned = np.zeros((self.n_ds, self.n_semantic_modes), dtype=float)
        lambda_by_uav_mode = np.zeros((self.n_uav, self.n_semantic_modes), dtype=float)
        for n in range(self.n_uav):
            ds = self.ds_by_uav[n]
            q_owned[ds] = self.q_cache[n, ds].astype(float)
            tau_local = self.tau_cache[n, ds]
            age_owned[ds] = np.where(
                tau_local >= 0,
                np.maximum(0.0, float(self.current_step) - tau_local.astype(float)) / self.A_max,
                0.0,
            )
            q_hat_owned[ds, :] = self.Q_hat_rec[n, ds, :] / self.Q_max
            lambda_by_uav_mode[n, :] = np.mean(self.Lambda_sem[n, ds, :], axis=0) / self.Lambda_ref

        state = np.concatenate(
            [
                self.gamma_uav_sut / 100.0,
                np.asarray([self.gamma_sut_sat / 100.0]),
                self.beta_sut_sat,
                q_owned,
                age_owned,
                self.A_rcc / self.A_max,
                lambda_by_uav_mode.reshape(-1),
                q_hat_owned.reshape(-1),
                self._share_instruction_obs(),
            ]
        ).astype(np.float32)
        if state.size != self.share_obs_dim:
            fixed = np.zeros(self.share_obs_dim, dtype=np.float32)
            fixed[: min(fixed.size, state.size)] = state[: fixed.size]
            state = fixed
        return np.tile(state[None, :], (self.n_agents, 1)).astype(np.float32)

    def _build_action_available_masks(self) -> np.ndarray:
        masks = np.ones((self.n_agents, self.common_act_dim), dtype=np.float32)
        masks[0, self.sut_act_dim_total :] = 0.0
        for n in range(self.n_uav):
            row = np.zeros(self.common_act_dim, dtype=np.float32)
            bucket_id = self._effective_snr_bucket_for_uav(n)
            gid = self._clip_instruction_id(self.current_instruction_id)
            q_req_gb = self._q_req_for_instruction_bucket(gid, bucket_id)
            mu_ok = np.zeros(self.n_mu_modes, dtype=bool)
            for mu_id in range(self.n_mu_modes):
                m = self._global_mode_id(bucket_id, mu_id)
                # Actors act simultaneously. The current SUT allocation is
                # not yet known; step() enforces the newly allocated budget.
                feasible_streams = self.candidate_mask(
                    n, m, self.q_cache, q_req_gb, include_load_budget=False
                )
                mu_ok[mu_id] = bool(np.any(feasible_streams))
            if not np.any(mu_ok):
                # Keep the categorical distribution well-defined when no
                # stream is feasible. step() still enforces feasibility, so
                # this UAV effectively performs no-op in that case.
                mu_ok[:] = True
            row[self.uav_act_slices["mu_logits"]] = mu_ok.astype(np.float32)
            if self.use_direct_ds_logits:
                row[self.uav_act_slices["ds_priority_logits"]] = 1.0
            else:
                row[self.uav_act_slices["scheduling_weights"]] = 1.0
            masks[1 + n] = row
        return masks

    def step(self, actions):
        slot_t = int(self.current_step)
        q_t = self.q_cache.copy()
        tau_t = self.tau_cache.copy()
        A_t = self.A_rcc.copy()

        sut_raw, uav_raw = self._normalize_actions_by_agent(actions)
        self.last_sut_raw_action = sut_raw.copy()
        self.last_uav_raw_actions = np.asarray(uav_raw, dtype=float).copy()
        self.beta_sut_sat = self._project_beta(sut_raw[self.sut_act_slices["beta_logits"]])
        self._update_backhaul_budget()
        self._update_semantic_tables()
        self._update_instruction_for_slot(slot_t)
        self.last_slot_snapshot = {
            "q_cache_pre": q_t.copy(),
            "tau_cache_pre": tau_t.copy(),
            "A_rcc_pre": A_t.copy(),
            "beta_sut_sat": self.beta_sut_sat.copy(),
            "gamma_uav_sut": self.gamma_uav_sut.copy(),
            "gamma_sut_sat": float(self.gamma_sut_sat),
            "gamma_bh": self.gamma_bh.copy(),
            "B_bh": self.B_bh.copy(),
            "Phi_bh": self.Phi_bh.copy(),
            "Lambda_sem": self.Lambda_sem.copy(),
            "L_z": self.L_z.copy(),
            "n_z": self.n_z.copy(),
            "Q_hat_rec": self.Q_hat_rec.copy(),
            "M_feas": self.M_feas.copy(),
        }

        chi = np.zeros_like(self.last_chi, dtype=np.int8)
        selected_modes = np.full(self.n_uav, -1, dtype=int)
        selected_mu_ids = np.full(self.n_uav, -1, dtype=int)
        snr_bucket_ids = np.zeros(self.n_uav, dtype=int)
        effective_snr_db = np.zeros(self.n_uav, dtype=float)
        scheduling_alphas = np.zeros((self.n_uav, self.scheduling_weight_dim), dtype=float)
        direct_ds_priority_logits = np.zeros((self.n_uav, self.ds_per_uav), dtype=float)
        lambda_usage = np.zeros(self.n_uav, dtype=float)
        all_mu_infeasible_count = 0
        fallback_to_best_feasible_mode_count = 0
        gid = self._clip_instruction_id(self.current_instruction_id)
        A_limit_g = float(self.A_limit_by_instruction[gid])
        constraint_penalty_A_g = float(self.constraint_penalty_A_by_instruction[gid])
        Lambda_budget_ratio_g = float(self.Lambda_budget_ratio_by_instruction[gid])
        Lambda_budget_g = float(Lambda_budget_ratio_g * self.Lambda_ref * self.n_uav)
        Q_req_gb = np.zeros(self.n_uav, dtype=float)

        for n in range(self.n_uav):
            mu_logits = uav_raw[n][self.uav_act_slices["mu_logits"]]
            mu_order = list(np.argsort(mu_logits)[::-1])
            effective_snr_db[n] = self._effective_snr_db_for_uav(n)
            bucket_id = self._snr_to_bucket_id(effective_snr_db[n])
            snr_bucket_ids[n] = bucket_id
            Q_req_gb[n] = self._q_req_for_instruction_bucket(gid, bucket_id)
            if self.use_direct_ds_logits:
                raw_priority = uav_raw[n][self.uav_act_slices["ds_priority_logits"]]
                if raw_priority.size < self.ds_per_uav:
                    raw_priority = np.pad(raw_priority, (0, self.ds_per_uav - raw_priority.size))
                direct_ds_priority_logits[n] = raw_priority[: self.ds_per_uav]
            else:
                raw_alpha = uav_raw[n][self.uav_act_slices["scheduling_weights"]]
                scheduling_alphas[n] = self._transform_scheduling_weights(raw_alpha)
            chosen_mode = None
            chosen_mu = None
            chosen_streams: List[int] = []
            bucket_has_feasible = False
            for mu_id in mu_order:
                mu_id = int(np.clip(mu_id, 0, self.n_mu_modes - 1))
                m = self._global_mode_id(bucket_id, mu_id)
                if self.use_direct_ds_logits:
                    score_by_ds, candidate = self._direct_ds_priority_scores(
                        n,
                        m,
                        direct_ds_priority_logits[n],
                        q_t,
                        Q_req_gb[n],
                    )
                else:
                    score_by_ds, candidate = self._derived_schedule_scores(
                        n,
                        m,
                        bucket_id,
                        scheduling_alphas[n],
                        q_t,
                        A_t,
                        Q_req_gb[n],
                        A_limit_g,
                    )
                if not np.any(candidate):
                    continue
                bucket_has_feasible = True
                budget_left = float(self.Phi_bh[n])
                ordered_ds = np.where(candidate)[0]
                ordered_ds = ordered_ds[np.argsort(score_by_ds[ordered_ds])[::-1]]
                selected = []
                for k in ordered_ds:
                    load = float(self.Lambda_sem[n, k, m])
                    if load <= budget_left + 1.0e-9:
                        selected.append(int(k))
                        budget_left -= load
                if selected:
                    chosen_mode = int(m)
                    chosen_mu = int(mu_id)
                    chosen_streams = selected
                    break
            if not bucket_has_feasible:
                all_mu_infeasible_count += 1
            if chosen_mode is None and self.use_instruction_constraints and self.use_quality_feasibility_mask:
                best = None
                for mu_id in range(self.n_mu_modes):
                    m = self._global_mode_id(bucket_id, mu_id)
                    score_by_ds, candidate = self._derived_schedule_scores(
                        n,
                        m,
                        bucket_id,
                        scheduling_alphas[n],
                        q_t,
                        A_t,
                        Q_req_gb[n],
                        A_limit_g,
                        use_instruction_quality=False,
                    )
                    if not np.any(candidate):
                        continue
                    q_vals = self.Q_hat_rec[n, candidate, m]
                    mode_rank = (float(np.mean(q_vals)), float(np.max(q_vals)))
                    ordered_ds = np.where(candidate)[0]
                    ordered_ds = ordered_ds[np.argsort(score_by_ds[ordered_ds])[::-1]]
                    budget_left = float(self.Phi_bh[n])
                    selected = []
                    for k in ordered_ds:
                        load = float(self.Lambda_sem[n, k, m])
                        if load <= budget_left + 1.0e-9:
                            selected.append(int(k))
                            budget_left -= load
                    if not selected:
                        continue
                    if best is None or mode_rank > best[0]:
                        best = (mode_rank, int(m), int(mu_id), selected)
                if best is not None:
                    _, chosen_mode, chosen_mu, chosen_streams = best
                    fallback_to_best_feasible_mode_count += 1
            if chosen_mode is None:
                continue
            selected_modes[n] = chosen_mode
            selected_mu_ids[n] = int(chosen_mu)
            for k in chosen_streams:
                chi[n, k, chosen_mode] = 1
                lambda_usage[n] += float(self.Lambda_sem[n, k, chosen_mode])

        self._assert_common_uav_modes(chi)
        self._assert_scheduled_feasibility(chi, Q_req_gb)
        if np.any(lambda_usage > self.Phi_bh + 1.0e-8):
            raise AssertionError(
                f"Backhaul budget violation: usage={lambda_usage.tolist()}, Phi_bh={self.Phi_bh.tolist()}"
            )

        y = np.sum(chi, axis=2).astype(np.int8)
        y = np.minimum(y, q_t).astype(np.int8)

        A_next = A_t.copy()
        for k in range(self.n_ds):
            n = int(self.owner_uav[k])
            if y[n, k] > 0:
                gen_tau = tau_t[n, k]
                if gen_tau < 0:
                    gen_tau = slot_t
                A_next[k] = min(float(slot_t - gen_tau + 1), self.A_max)
            else:
                A_next[k] = min(float(A_t[k] + 1.0), self.A_max)

        q_next = np.zeros_like(q_t, dtype=np.int8)
        tau_next = np.full_like(tau_t, self.INVALID_TAU, dtype=int)
        # One-slot sensing/acquisition pipeline: a served stream frees the
        # cache at the end of this slot, and its replacement is acquired in a
        # later slot. Thus a DS served in slot t is not immediately schedulable
        # again in slot t+1 under the default model.
        admission = ((1 - q_t) * self.owner_mask.astype(np.int8)).astype(np.int8)
        for n in range(self.n_uav):
            for k in self.ds_by_uav[n]:
                q_next[n, k] = int(q_t[n, k] - y[n, k] + admission[n, k])
                if admission[n, k] == 1:
                    tau_next[n, k] = slot_t
                elif q_t[n, k] == 1 and y[n, k] == 0:
                    tau_next[n, k] = int(tau_t[n, k])
                elif q_next[n, k] == 0:
                    tau_next[n, k] = self.INVALID_TAU

        q_min = self.Q_min_eval
        denom_q = max(float(self.Q_max - q_min), 1.0e-9)
        quality_gain = 0.0
        load_cost = 0.0
        quality_violations = []
        for n in range(self.n_uav):
            for k in self.ds_by_uav[n]:
                for m in range(self.n_semantic_modes):
                    if chi[n, k, m] <= 0:
                        continue
                    quality_gain += max((float(self.Q_hat_rec[n, k, m]) - q_min) / denom_q, 0.0)
                    load_cost += float(self.Lambda_sem[n, k, m]) / self.Lambda_ref
                    bucket_id = int(snr_bucket_ids[n])
                    bucket_modes = self._bucket_mode_ids(bucket_id)
                    q_bucket = self.Q_hat_rec[n, k, bucket_modes]
                    q_best_bucket = float(np.max(q_bucket))
                    q_worst_bucket = float(np.min(q_bucket))
                    vq = max(
                        0.0,
                        (float(Q_req_gb[n]) - float(self.Q_hat_rec[n, k, m]))
                        / max(q_best_bucket - q_worst_bucket, 1.0e-9),
                    )
                    quality_violations.append(vq)
        A_norm = A_next / max(float(self.A_max), 1.0e-9)
        A_fair = float(np.max(A_norm)) if self.n_ds > 0 else 0.0
        A_mean = float(np.mean(A_norm)) if self.n_ds > 0 else 0.0
        A_tail = (
            float(
                np.mean(
                    np.maximum(
                        (A_next - self.aoi_tail_threshold) / max(float(self.A_max), 1.0e-9),
                        0.0,
                    )
                )
            )
            if self.n_ds > 0
            else 0.0
        )
        aoi_state_term = float(
            self.aoi_mean_weight * A_mean + self.aoi_max_weight * A_fair + self.aoi_tail_weight * A_tail
        )
        fresh_gain = 0.0
        for n in range(self.n_uav):
            for k in self.ds_by_uav[n]:
                if y[n, k] > 0:
                    fresh_gain += max(float(A_t[k] - A_next[k]), 0.0) / max(float(self.A_max), 1.0e-9)
        fresh_gain = float(fresh_gain / max(float(self.n_ds), 1.0))
        # Total effective semantic quality gain normalized by the number of
        # DSs. This is not the average reconstruction quality of scheduled
        # streams; use avg_quality_gain in info for per-scheduled-stream views.
        quality_term = float(quality_gain / max(float(self.n_ds), 1.0))
        load_term = float(load_cost / max(float(self.n_uav), 1.0))
        aoi_term = aoi_state_term
        quality_contrib = float(self.omega_Q * quality_term)
        load_penalty = float(-self.omega_Lambda * load_term)
        aoi_penalty = float(-self.omega_A * aoi_term)
        # fresh_gain is logged as a diagnostic for served-stream freshness
        # reduction, but it is not an extra reward bonus. Objective pressure is
        # concentrated in the three fixed reward weights below.
        base_reward = quality_contrib + load_penalty + aoi_penalty
        quality_violation_bucketwise = float(np.mean(quality_violations)) if quality_violations else 0.0
        max_aoi_after = float(np.max(A_next)) if self.n_ds > 0 else 0.0
        aoi_violation_max = max(0.0, (max_aoi_after - A_limit_g) / max(A_limit_g, 1.0e-9))
        used_lambda = float(np.sum(lambda_usage))
        load_violation = max(0.0, (used_lambda - Lambda_budget_g) / max(Lambda_budget_g, 1.0e-9))
        aoi_reduction = float(np.sum(np.maximum(A_t - A_next, 0.0)))
        recv_aoi_bonus = float(self.eta_recv_aoi_bonus * aoi_reduction / max(A_limit_g, 1.0e-9))
        if self.use_instruction_constraints:
            reward = (
                base_reward
                - self.constraint_penalty_Q * quality_violation_bucketwise
                - constraint_penalty_A_g * aoi_violation_max
                - self.constraint_penalty_Lambda * load_violation
                + recv_aoi_bonus
            )
        else:
            reward = base_reward
        reward_components = {
            "reward_quality": float(quality_contrib),
            "reward_load_cost": float(load_penalty),
            "reward_aoi_penalty": float(aoi_penalty),
            "penalty_quality_violation": float(-self.constraint_penalty_Q * quality_violation_bucketwise)
            if self.use_instruction_constraints
            else 0.0,
            "penalty_aoi_violation": float(-constraint_penalty_A_g * aoi_violation_max)
            if self.use_instruction_constraints
            else 0.0,
            "penalty_load_violation": float(-self.constraint_penalty_Lambda * load_violation)
            if self.use_instruction_constraints
            else 0.0,
            "bonus_aoi_reduction": float(recv_aoi_bonus) if self.use_instruction_constraints else 0.0,
        }
        reward_component_sum = float(sum(reward_components.values()))
        if abs(reward_component_sum - float(reward)) > 1.0e-8:
            raise AssertionError(
                f"Reward component sum mismatch: components={reward_component_sum}, reward={float(reward)}"
            )

        self.q_cache = q_next
        self.tau_cache = tau_next
        self.A_rcc = A_next
        self._sample_new_content_features(admission)
        self.last_chi = chi
        self.last_y = y
        self.last_admission = admission
        self.last_selected_modes = selected_modes
        self.last_selected_mu_ids = selected_mu_ids
        self.last_snr_bucket_ids = snr_bucket_ids
        self.last_effective_snr_db = effective_snr_db
        self.last_scheduling_alphas = scheduling_alphas
        self.last_direct_ds_priority_logits = direct_ds_priority_logits
        self.last_per_uav_scheduled_count = np.zeros(self.n_uav, dtype=int)
        self.last_per_uav_mean_aoi = np.zeros(self.n_uav, dtype=float)
        self.last_per_uav_max_aoi = np.zeros(self.n_uav, dtype=float)
        self.last_selected_mode_q_hat = np.zeros(self.n_uav, dtype=float)
        self.last_selected_mode_lambda_sem = np.zeros(self.n_uav, dtype=float)
        for n in range(self.n_uav):
            ds = self.ds_by_uav[n]
            self.last_per_uav_scheduled_count[n] = int(np.sum(y[n, ds])) if ds.size else 0
            self.last_per_uav_mean_aoi[n] = float(np.mean(A_next[ds])) if ds.size else 0.0
            self.last_per_uav_max_aoi[n] = float(np.max(A_next[ds])) if ds.size else 0.0
            mode_id = int(selected_modes[n])
            scheduled_ds = ds[y[n, ds] > 0]
            if mode_id >= 0 and scheduled_ds.size > 0:
                self.last_selected_mode_q_hat[n] = float(np.mean(self.Q_hat_rec[n, scheduled_ds, mode_id]))
                self.last_selected_mode_lambda_sem[n] = float(np.mean(self.Lambda_sem[n, scheduled_ds, mode_id]))
        self.last_all_mu_infeasible_count = int(all_mu_infeasible_count)
        self.last_fallback_to_best_feasible_mode_count = int(fallback_to_best_feasible_mode_count)
        self.last_lambda_usage = lambda_usage
        self.last_quality_gain = float(quality_gain)
        self.last_load_cost = float(load_cost)
        self.last_A_fair = float(A_fair)
        self.last_quality_term = float(quality_term)
        self.last_load_term = float(load_term)
        self.last_aoi_mean_term = float(A_mean)
        self.last_aoi_max_term = float(A_fair)
        self.last_aoi_tail_term = float(A_tail)
        self.last_aoi_state_term = float(aoi_state_term)
        self.last_fresh_gain_term = float(fresh_gain)
        self.last_aoi_term = float(aoi_term)
        self.last_quality_contrib = float(quality_contrib)
        self.last_load_penalty = float(load_penalty)
        self.last_aoi_penalty = float(aoi_penalty)
        self.last_instruction_id = int(gid)
        self.last_instruction_name = self.instruction_names[gid]
        self.last_A_limit_g = float(A_limit_g)
        self.last_Lambda_budget_ratio_g = float(Lambda_budget_ratio_g)
        self.last_Lambda_budget_g = float(Lambda_budget_g)
        self.last_Q_req_gb = Q_req_gb.copy()
        self.last_base_reward = float(base_reward)
        self.last_total_reward = float(reward)
        self.last_quality_violation_bucketwise = float(quality_violation_bucketwise)
        self.last_aoi_violation_max = float(aoi_violation_max)
        self.last_load_violation = float(load_violation)
        self.last_recv_aoi_bonus = float(recv_aoi_bonus)
        self.last_reward_components = reward_components
        self.last_utility = float(reward)

        self.current_step += 1
        self._update_channels()
        self._update_semantic_tables()
        self._update_instruction_for_slot(self.current_step, refresh_last=False)

        done_env = self.current_step >= self.max_steps
        obs = self._build_obs_multi()
        share_obs = self._build_share_obs_multi()
        rewards = np.full((self.n_agents, 1), float(reward), dtype=np.float32)
        dones = np.full(self.n_agents, bool(done_env), dtype=bool)
        info = self._build_info(slot_t, reward)
        infos = [dict(info) for _ in range(self.n_agents)]
        available_actions = self._build_action_available_masks()
        return obs, share_obs, rewards, dones, infos, available_actions

    def _build_info(self, slot_t: int, reward: float) -> Dict[str, Any]:
        scheduled = np.argwhere(self.last_y > 0)
        scheduled_count = int(np.sum(self.last_y))
        avg_quality_gain = float(self.last_quality_gain / max(1, scheduled_count))
        avg_aoi = float(np.mean(self.A_rcc)) if self.n_ds > 0 else 0.0
        max_aoi = float(np.max(self.A_rcc)) if self.n_ds > 0 else 0.0
        avg_lambda_usage = float(np.mean(self.last_lambda_usage)) if self.n_uav > 0 else 0.0
        slot = self.last_slot_snapshot if isinstance(self.last_slot_snapshot, dict) else {}
        beta_slot = np.asarray(slot.get("beta_sut_sat", self.beta_sut_sat), dtype=float)
        gamma_uav_sut_slot = np.asarray(slot.get("gamma_uav_sut", self.gamma_uav_sut), dtype=float)
        gamma_sut_sat_slot = float(slot.get("gamma_sut_sat", self.gamma_sut_sat))
        gamma_bh_slot = np.asarray(slot.get("gamma_bh", self.gamma_bh), dtype=float)
        B_bh_slot = np.asarray(slot.get("B_bh", self.B_bh), dtype=float)
        Phi_bh_slot = np.asarray(slot.get("Phi_bh", self.Phi_bh), dtype=float)
        Lambda_sem_slot = np.asarray(slot.get("Lambda_sem", self.Lambda_sem), dtype=float)
        L_z_slot = np.asarray(slot.get("L_z", self.L_z), dtype=float)
        n_z_slot = np.asarray(slot.get("n_z", self.n_z), dtype=float)
        Q_hat_rec_slot = np.asarray(slot.get("Q_hat_rec", self.Q_hat_rec), dtype=float)
        M_feas_slot = np.asarray(slot.get("M_feas", self.M_feas), dtype=bool)
        q_cache_pre_slot = np.asarray(slot.get("q_cache_pre", self.q_cache), dtype=np.int8)
        tau_cache_pre_slot = np.asarray(slot.get("tau_cache_pre", self.tau_cache), dtype=int)
        A_rcc_pre_slot = np.asarray(slot.get("A_rcc_pre", self.A_rcc), dtype=float)
        avg_phi_budget = float(np.mean(Phi_bh_slot)) if self.n_uav > 0 else 0.0
        mode_usage_by_bucket_mu = np.zeros((4, self.n_mu_modes), dtype=int)
        quality_sum_by_mu = np.zeros(self.n_mu_modes, dtype=float)
        load_sum_by_mu = np.zeros(self.n_mu_modes, dtype=float)
        count_by_mu = np.zeros(self.n_mu_modes, dtype=int)
        for n, k in scheduled:
            mode_ids = np.where(self.last_chi[int(n), int(k)] > 0)[0]
            if mode_ids.size == 0:
                continue
            mode_id = int(mode_ids[0])
            bucket_id = int(self.last_snr_bucket_ids[int(n)])
            mu_id = int(self.last_selected_mu_ids[int(n)])
            mode_usage_by_bucket_mu[bucket_id, mu_id] += 1
            quality_sum_by_mu[mu_id] += float(Q_hat_rec_slot[int(n), int(k), mode_id])
            load_sum_by_mu[mu_id] += float(Lambda_sem_slot[int(n), int(k), mode_id])
            count_by_mu[mu_id] += 1
        quality_by_mu = np.divide(
            quality_sum_by_mu,
            np.maximum(count_by_mu, 1),
            out=np.zeros_like(quality_sum_by_mu),
            where=count_by_mu > 0,
        )
        load_by_mu = np.divide(
            load_sum_by_mu,
            np.maximum(count_by_mu, 1),
            out=np.zeros_like(load_sum_by_mu),
            where=count_by_mu > 0,
        )
        mode_distribution_by_instruction = np.zeros((self.num_instructions, self.n_semantic_modes), dtype=int)
        mu_id_distribution_by_instruction = np.zeros((self.num_instructions, self.n_mu_modes), dtype=int)
        mode_distribution_by_snr_bucket_instruction = np.zeros(
            (self.num_instructions, 4, self.n_semantic_modes),
            dtype=int,
        )
        gid_slot = self._clip_instruction_id(self.last_instruction_id)
        for n in range(self.n_uav):
            mu_id = int(self.last_selected_mu_ids[n])
            if 0 <= mu_id < self.n_mu_modes:
                mu_id_distribution_by_instruction[gid_slot, mu_id] += 1
        for n, k in scheduled:
            mode_ids = np.where(self.last_chi[int(n), int(k)] > 0)[0]
            if mode_ids.size == 0:
                continue
            mode_id = int(mode_ids[0])
            bucket_id = int(self.last_snr_bucket_ids[int(n)])
            if 0 <= mode_id < self.n_semantic_modes:
                mode_distribution_by_instruction[gid_slot, mode_id] += 1
                mode_distribution_by_snr_bucket_instruction[gid_slot, int(np.clip(bucket_id, 0, 3)), mode_id] += 1
        info = {
            "system_model": "fixed_multi_objective_semantic_video_acquisition_dec_pomdp",
            "reward_type": "fixed_multi_objective",
            "slot": int(slot_t),
            "reward": float(reward),
            "utility": float(reward),
            "instruction_mode_strategy": str(self.instruction_mode_strategy),
            "instruction_switch_step": int(self.instruction_switch_step),
            "instruction_before_id": int(self.instruction_before_id),
            "instruction_after_id": int(self.instruction_after_id),
            "instruction_schedule": {
                "strategy": str(self.instruction_mode_strategy),
                "switch_step": int(self.instruction_switch_step),
                "before_id": int(self.instruction_before_id),
                "before_name": str(self.instruction_names[self._clip_instruction_id(self.instruction_before_id)]),
                "after_id": int(self.instruction_after_id),
                "after_name": str(self.instruction_names[self._clip_instruction_id(self.instruction_after_id)]),
            },
            "instruction_id": int(self.last_instruction_id),
            "instruction_name": str(self.last_instruction_name),
            "is_instruction_switch_step": bool(int(slot_t) == int(self.instruction_switch_step)),
            "A_limit_g": float(self.last_A_limit_g),
            "Lambda_budget_ratio_g": float(self.last_Lambda_budget_ratio_g),
            "Lambda_budget_g": float(self.last_Lambda_budget_g),
            "Q_req_gb": self.last_Q_req_gb.copy(),
            "Q_min_eval": float(self.Q_min_eval),
            "Q_min": float(self.Q_min),
            "feasibility_rule": "Q_hat_rec >= Q_req(instruction, snr_bucket)",
            "quality_feasibility_reference": "Q_req",
            "lambda_constraint": "sum(beta_sut_sat)=1 with beta_sut_sat>=beta_sat_lower_bound",
            "beta_sat_lower_bound": float(self.beta_sat_lower_bound),
            "omega_Q": float(self.omega_Q),
            "omega_Lambda": float(self.omega_Lambda),
            "omega_A": float(self.omega_A),
            "A_max": float(self.A_max),
            "aoi_mean_weight": float(self.aoi_mean_weight),
            "aoi_max_weight": float(self.aoi_max_weight),
            "aoi_tail_weight": float(self.aoi_tail_weight),
            "aoi_tail_threshold": float(self.aoi_tail_threshold),
            "raw_video_shape_hwl": np.asarray(
                [self.raw_video_h, self.raw_video_w, self.raw_video_l],
                dtype=int,
            ),
            "raw_video_channels": int(self.raw_video_channels),
            "raw_sample_bits": float(self.raw_sample_bits),
            "raw_video_bits": float(self.raw_video_bits),
            "sci_input_shape_hwl": np.asarray([self.H, self.W, self.L_video], dtype=int),
            "sci_input_channels": int(self.video_channels),
            "sci_input_bits": float(self.sci_input_bits),
            "sci_sensor_channels": int(self.sci_sensor_channels),
            "sci_sampling_ratio": float(self.sci_sampling_ratio),
            "sci_num_samples": int(self.sci_num_samples),
            "sci_bits_per_sample": float(self.sci_bits_per_sample),
            "sci_access_bits": float(self.sci_access_bits),
            "sci_compression_ratio": float(self.sci_compression_ratio),
            "end_to_end_access_ratio": float(self.end_to_end_access_ratio),
            "beta_sut_sat": beta_slot.copy(),
            "projected_beta": beta_slot.copy(),
            "gamma_uav_sut": gamma_uav_sut_slot.copy(),
            "gamma_sut_sat": gamma_sut_sat_slot,
            "gamma_bh": gamma_bh_slot.copy(),
            "B_bh": B_bh_slot.copy(),
            "Phi_bh": Phi_bh_slot.copy(),
            "q_cache": self.q_cache.copy(),
            "q_cache_state": "post_state",
            "q_cache_pre": q_cache_pre_slot.copy(),
            "tau_cache": self.tau_cache.copy(),
            "tau_cache_state": "post_state",
            "tau_cache_pre": tau_cache_pre_slot.copy(),
            "A_rcc": self.A_rcc.copy(),
            "A_rcc_state": "post_state",
            "A_rcc_pre": A_rcc_pre_slot.copy(),
            "A_fair": float(self.last_A_fair),
            "avg_aoi": avg_aoi,
            "max_aoi": max_aoi,
            "mean_AoI": avg_aoi,
            "max_AoI": max_aoi,
            "admission": self.last_admission.copy(),
            "scheduled_streams": scheduled.copy(),
            "selected_ds": [self.ds_by_uav[n][self.last_y[n, self.ds_by_uav[n]] > 0].astype(int).tolist() for n in range(self.n_uav)],
            "scheduled_count": scheduled_count,
            "per_uav_scheduled_count": self.last_per_uav_scheduled_count.copy(),
            "per_uav_mean_aoi": self.last_per_uav_mean_aoi.copy(),
            "per_uav_max_aoi": self.last_per_uav_max_aoi.copy(),
            "selected_modes": self.last_selected_modes.copy(),
            "selected_global_mode_ids": self.last_selected_modes.copy(),
            "selected_mode_Q_hat": self.last_selected_mode_q_hat.copy(),
            "selected_mode_Lambda_sem": self.last_selected_mode_lambda_sem.copy(),
            "selected_mu_ids": self.last_selected_mu_ids.copy(),
            "effective_snr_db": self.last_effective_snr_db.copy(),
            "snr_bucket_ids": self.last_snr_bucket_ids.copy(),
            "snr_bucket_values_db": self.snr_bucket_values_db.copy(),
            "semantic_mode_lookup_by_bucket_mu": self.semantic_mode_lookup_by_bucket_mu.copy(),
            "use_direct_ds_logits": bool(self.use_direct_ds_logits),
            "direct_ds_priority_logits": self.last_direct_ds_priority_logits.copy(),
            "scheduling_alphas": self.last_scheduling_alphas.copy(),
            "all_mu_infeasible_count": int(self.last_all_mu_infeasible_count),
            "fallback_to_best_feasible_mode_count": int(self.last_fallback_to_best_feasible_mode_count),
            "mode_usage_by_snr_bucket_mu": mode_usage_by_bucket_mu.copy(),
            "mode_distribution_by_instruction": mode_distribution_by_instruction.copy(),
            "mu_id_distribution_by_instruction": mu_id_distribution_by_instruction.copy(),
            "mode_distribution_by_snr_bucket_instruction": mode_distribution_by_snr_bucket_instruction.copy(),
            "mu_selection_distribution": np.bincount(
                self.last_selected_mu_ids[self.last_selected_mu_ids >= 0],
                minlength=self.n_mu_modes,
            ).astype(int),
            "scheduled_count_by_mu": count_by_mu.copy(),
            "quality_by_mu": quality_by_mu.copy(),
            "load_by_mu": load_by_mu.copy(),
            "chi": self.last_chi.copy(),
            "y": self.last_y.copy(),
            "Lambda_sem": Lambda_sem_slot.copy(),
            "L_z": L_z_slot.copy(),
            "n_z": n_z_slot.copy(),
            "Q_hat_rec": Q_hat_rec_slot.copy(),
            "M_feas": M_feas_slot.copy(),
            "M_feas_reference": "legacy_Q_min_diagnostic",
            "semantic_registry_enabled": bool(self.semantic_library.enabled),
            **(self.semantic_library.diagnostic() if self.semantic_library.enabled else {}),
            "lambda_usage": self.last_lambda_usage.copy(),
            "avg_lambda_usage": avg_lambda_usage,
            "Phi_budget": Phi_bh_slot.copy(),
            "avg_phi_budget": avg_phi_budget,
            "quality_gain": float(self.last_quality_gain),
            "avg_quality_gain": avg_quality_gain,
            "load_cost": float(self.last_load_cost),
            "quality_term": float(self.last_quality_term),
            "load_term": float(self.last_load_term),
            "aoi_term": float(self.last_aoi_term),
            "aoi_mean_term": float(self.last_aoi_mean_term),
            "aoi_max_term": float(self.last_aoi_max_term),
            "aoi_tail_term": float(self.last_aoi_tail_term),
            "aoi_state_term": float(self.last_aoi_state_term),
            "fresh_gain_term": float(self.last_fresh_gain_term),
            "quality_violation_bucketwise": float(self.last_quality_violation_bucketwise),
            "aoi_violation_max": float(self.last_aoi_violation_max),
            "load_violation": float(self.last_load_violation),
            "recv_aoi_bonus": float(self.last_recv_aoi_bonus),
            "base_reward": float(self.last_base_reward),
            "total_reward": float(self.last_total_reward),
            "reward_components": dict(self.last_reward_components),
            "reward_component_sum": float(sum(self.last_reward_components.values())),
            "reward_quality": float(self.last_reward_components.get("reward_quality", 0.0)),
            "reward_load_cost": float(self.last_reward_components.get("reward_load_cost", 0.0)),
            "reward_aoi_penalty": float(self.last_reward_components.get("reward_aoi_penalty", 0.0)),
            "penalty_quality_violation": float(
                self.last_reward_components.get("penalty_quality_violation", 0.0)
            ),
            "penalty_aoi_violation": float(self.last_reward_components.get("penalty_aoi_violation", 0.0)),
            "penalty_load_violation": float(self.last_reward_components.get("penalty_load_violation", 0.0)),
            "bonus_aoi_reduction": float(self.last_reward_components.get("bonus_aoi_reduction", 0.0)),
            "quality_contrib": float(self.last_quality_contrib),
            "load_penalty": float(self.last_load_penalty),
            "aoi_penalty": float(self.last_aoi_penalty),
            "backhaul_rate": B_bh_slot * np.log2(1.0 + gamma_bh_slot),
            "sum_backhaul_rate": float(np.sum(B_bh_slot * np.log2(1.0 + gamma_bh_slot))),
        }
        if self.enable_episode_trace:
            info["trace"] = self._build_system_trace(info)
        return info

    def _mode_trace_info(self, mode_id: int) -> Dict[str, Any]:
        if mode_id < 0:
            return {
                "snr_bucket_id": -1,
                "snr_bucket_db": None,
                "mu_id": -1,
                "mu_comm": None,
                "trained_snr_db": None,
            }
        bucket_id = int(np.clip(mode_id // 4, 0, 3))
        mu_id = int(np.clip(mode_id % 4, 0, self.n_mu_modes - 1))
        trained_snr = float(self.snr_bucket_values_db[bucket_id])
        mu_comm = float(self.mu_comm_levels[mu_id])
        if self.semantic_library.enabled and mode_id < self.semantic_library.num_modes:
            trained_snr = float(self.semantic_library.trained_snr_db[mode_id])
            mu_comm = float(self.semantic_library.mu_comm[mode_id])
        if self.semantic_mode_selection == "all_modes":
            bucket_id = self._snr_to_bucket_id(trained_snr)
            mu_id = mode_id
        return {
            "snr_bucket_id": bucket_id,
            "snr_bucket_db": float(self.snr_bucket_values_db[bucket_id]),
            "mu_id": mu_id,
            "mu_comm": mu_comm,
            "trained_snr_db": trained_snr,
        }

    def _build_system_trace(self, info: Dict[str, Any]) -> Dict[str, Any]:
        """Build a JSON-friendly one-slot system trace from executed decisions.

        This is diagnostics-only. It reuses the slot snapshot and executed
        scheduling variables, and does not affect dynamics, rewards, masks, or
        policy inputs.
        """
        slot_t = int(info["slot"])
        q_pre = np.asarray(info["q_cache_pre"], dtype=int)
        q_post = np.asarray(info["q_cache"], dtype=int)
        tau_pre = np.asarray(info["tau_cache_pre"], dtype=int)
        tau_post = np.asarray(info["tau_cache"], dtype=int)
        A_pre = np.asarray(info["A_rcc_pre"], dtype=float).reshape(-1)
        A_post = np.asarray(info["A_rcc"], dtype=float).reshape(-1)
        chi = np.asarray(info["chi"], dtype=int)
        y = np.asarray(info["y"], dtype=int)
        admission = np.asarray(info["admission"], dtype=int)
        Lambda_sem = np.asarray(info["Lambda_sem"], dtype=float)
        L_z = np.asarray(info["L_z"], dtype=float)
        Q_hat_rec = np.asarray(info["Q_hat_rec"], dtype=float)
        beta = np.asarray(info["projected_beta"], dtype=float).reshape(-1)
        Phi_bh = np.asarray(info["Phi_bh"], dtype=float).reshape(-1)
        B_bh = np.asarray(info["B_bh"], dtype=float).reshape(-1)
        gamma_uav_sut = np.asarray(info["gamma_uav_sut"], dtype=float).reshape(-1)
        lambda_usage = np.asarray(info["lambda_usage"], dtype=float).reshape(-1)
        direct_ds_priority_logits = np.asarray(info.get("direct_ds_priority_logits", []), dtype=float)
        selected_modes = np.asarray(info["selected_global_mode_ids"], dtype=int).reshape(-1)
        selected_mu = np.asarray(info["selected_mu_ids"], dtype=int).reshape(-1)
        snr_bucket_ids = np.asarray(info["snr_bucket_ids"], dtype=int).reshape(-1)
        effective_snr_db = np.asarray(info["effective_snr_db"], dtype=float).reshape(-1)
        q_req_gb = np.asarray(info.get("Q_req_gb", np.zeros(self.n_uav)), dtype=float).reshape(-1)
        q_min = float(info["Q_min"])
        denom_q = max(float(self.Q_max - q_min), 1.0e-9)
        total_predicted_quality = 0.0
        quality_violations = 0
        cache_violations = 0
        scheduled_detail_by_uav: List[List[Dict[str, Any]]] = [[] for _ in range(self.n_uav)]

        scheduled_pairs = np.argwhere(y > 0)
        for n_raw, k_raw in scheduled_pairs:
            n = int(n_raw)
            k = int(k_raw)
            mode_ids = np.where(chi[n, k] > 0)[0] if chi.ndim == 3 else np.asarray([], dtype=int)
            mode_id = int(mode_ids[0]) if mode_ids.size else -1
            lambda_sem = float(Lambda_sem[n, k, mode_id]) if mode_id >= 0 else 0.0
            q_hat = float(Q_hat_rec[n, k, mode_id]) if mode_id >= 0 else 0.0
            l_z = float(L_z[n, k, mode_id]) if mode_id >= 0 and mode_id < L_z.shape[2] else 0.0
            side_uses = max(lambda_sem - l_z, 0.0)
            q_req = float(q_req_gb[n]) if n < q_req_gb.size else 0.0
            feasible = bool(mode_id >= 0 and q_hat + 1.0e-9 >= q_req)
            total_predicted_quality += q_hat
            if not feasible:
                quality_violations += 1
            if int(q_pre[n, k]) <= 0:
                cache_violations += 1
            scheduled_detail_by_uav[n].append(
                {
                    "ds_id": k,
                    "position": self.pos_ds[k].astype(float).tolist(),
                    "mode": mode_id,
                    "selected_mode_info": self._mode_trace_info(mode_id),
                    "lambda_sem": lambda_sem,
                    "q_hat_rec": q_hat,
                    "q_req": q_req,
                    "q_min": q_min,
                    "quality_feasible": feasible,
                    "quality_feasibility_reference": "Q_req",
                    "side_info_bits": float(self.side_info_bits),
                    "side_info_channel_uses": float(side_uses),
                    "latent_channel_uses": float(l_z),
                    "normalized_quality_gain": float(max((q_hat - q_min) / denom_q, 0.0)),
                    "normalized_semantic_load": float(lambda_sem / max(float(self.Lambda_ref), 1.0e-9)),
                    "aoi_before": float(A_pre[k]) if k < A_pre.size else 0.0,
                    "aoi_after": float(A_post[k]) if k < A_post.size else 0.0,
                    "cache_before": int(q_pre[n, k]),
                    "cache_after": int(q_post[n, k]),
                    "tau_cache_before": int(tau_pre[n, k]),
                    "tau_cache_after": int(tau_post[n, k]),
                    "admitted_after_slot": bool(admission[n, k] > 0),
                }
            )

        uavs = []
        for n in range(self.n_uav):
            local_ds = np.asarray(self.ds_by_uav[n], dtype=int)
            raw_action = self.last_uav_raw_actions[n] if n < self.last_uav_raw_actions.shape[0] else np.zeros(0)
            mode_id = int(selected_modes[n]) if n < selected_modes.size else -1
            local_states = []
            for k_raw in local_ds:
                k = int(k_raw)
                local_states.append(
                    {
                        "ds_id": k,
                        "cache_before": int(q_pre[n, k]),
                        "cache_after": int(q_post[n, k]),
                        "tau_cache_before": int(tau_pre[n, k]),
                        "tau_cache_after": int(tau_post[n, k]),
                        "aoi_before": float(A_pre[k]) if k < A_pre.size else 0.0,
                        "aoi_after": float(A_post[k]) if k < A_post.size else 0.0,
                        "scheduled": bool(y[n, k] > 0),
                        "admitted_after_slot": bool(admission[n, k] > 0),
                    }
                )
            q_vals = [stream["q_hat_rec"] for stream in scheduled_detail_by_uav[n]]
            uavs.append(
                {
                    "uav_id": int(n),
                    "position": self.pos_uav[n].astype(float).tolist(),
                    "gamma_uav_sut": float(gamma_uav_sut[n]) if n < gamma_uav_sut.size else 0.0,
                    "effective_snr_db": float(effective_snr_db[n]) if n < effective_snr_db.size else 0.0,
                    "snr_bucket_id": int(snr_bucket_ids[n]) if n < snr_bucket_ids.size else -1,
                    "Q_req_gb": float(q_req_gb[n]) if n < q_req_gb.size else 0.0,
                    "raw_action": {
                        "vector": raw_action.astype(float).tolist(),
                        "mu_logits": raw_action[self.uav_act_slices["mu_logits"]].astype(float).tolist()
                        if raw_action.size >= self.uav_act_dim_total
                        else [],
                        "direct_ds_priority_logits": raw_action[self.uav_act_slices["ds_priority_logits"]]
                        .astype(float)
                        .tolist()
                        if self.use_direct_ds_logits and raw_action.size >= self.uav_act_dim_total
                        else [],
                        "scheduling_weights": raw_action[self.uav_act_slices["scheduling_weights"]]
                        .astype(float)
                        .tolist()
                        if (not self.use_direct_ds_logits) and raw_action.size >= self.uav_act_dim_total
                        else [],
                    },
                    "direct_ds_priority_logits": direct_ds_priority_logits[n].astype(float).tolist()
                    if direct_ds_priority_logits.ndim == 2 and n < direct_ds_priority_logits.shape[0]
                    else [],
                    "scheduling_alphas": self.last_scheduling_alphas[n].astype(float).tolist(),
                    "selected_mode": mode_id,
                    "selected_mu_id": int(selected_mu[n]) if n < selected_mu.size else -1,
                    "selected_mode_info": self._mode_trace_info(mode_id),
                    "selected_ds_ids": [int(stream["ds_id"]) for stream in scheduled_detail_by_uav[n]],
                    "scheduled_ds": scheduled_detail_by_uav[n],
                    "local_ds_state": local_states,
                    "cached_ds_before": local_ds[q_pre[n, local_ds] > 0].astype(int).tolist(),
                    "cached_ds_after": local_ds[q_post[n, local_ds] > 0].astype(int).tolist(),
                    "admitted_ds": local_ds[admission[n, local_ds] > 0].astype(int).tolist(),
                    "scheduled_count": int(np.sum(y[n, local_ds])),
                    "per_uav_mean_aoi": float(self.last_per_uav_mean_aoi[n]),
                    "per_uav_max_aoi": float(self.last_per_uav_max_aoi[n]),
                    "total_lambda_usage": float(lambda_usage[n]) if n < lambda_usage.size else 0.0,
                    "avg_q_hat_rec": float(np.mean(q_vals)) if q_vals else 0.0,
                    "backhaul_constraint_satisfied": bool(
                        n < lambda_usage.size and n < Phi_bh.size and lambda_usage[n] <= Phi_bh[n] + 1.0e-9
                    ),
                }
            )

        backhaul_violations = int(np.sum(lambda_usage > Phi_bh + 1.0e-9)) if lambda_usage.size == Phi_bh.size else 0
        mode_mapping_violations = 0
        for n, mode_id in enumerate(selected_modes):
            if mode_id < 0:
                continue
            expected = self._global_mode_id(int(snr_bucket_ids[n]), int(selected_mu[n]))
            if int(mode_id) != int(expected):
                mode_mapping_violations += 1
        total_scheduled = int(info["scheduled_count"])
        return {
            "episode": None,
            "t": slot_t,
            "done": bool(slot_t + 1 >= self.max_steps),
            "reward_type": str(info["reward_type"]),
            "instruction": {
                "id": int(info.get("instruction_id", 0)),
                "name": str(info.get("instruction_name", "")),
                "A_limit_g": float(info.get("A_limit_g", 0.0)),
                "Lambda_budget_ratio_g": float(info.get("Lambda_budget_ratio_g", 0.0)),
                "Lambda_budget_g": float(info.get("Lambda_budget_g", 0.0)),
                "Q_req_gb": q_req_gb.astype(float).tolist(),
            },
            "sut": {
                "position": self.pos_sut.astype(float).tolist(),
                "raw_action": self.last_sut_raw_action.astype(float).tolist(),
                "beta_sut_sat": beta.astype(float).tolist(),
                "lambda_sut_sat": (beta * float(self.B_sut_sat)).astype(float).tolist(),
                "B_bh": B_bh.astype(float).tolist(),
                "Phi_bh": Phi_bh.astype(float).tolist(),
                "gamma_sut_sat": float(info["gamma_sut_sat"]),
                "gamma_uav_sut": gamma_uav_sut.astype(float).tolist(),
            },
            "uavs": uavs,
            "global_state": {
                "q_cache_before": q_pre.astype(int).tolist(),
                "q_cache_after": q_post.astype(int).tolist(),
                "tau_cache_before": tau_pre.astype(int).tolist(),
                "tau_cache_after": tau_post.astype(int).tolist(),
                "A_rcc_before": A_pre.astype(float).tolist(),
                "A_rcc_after": A_post.astype(float).tolist(),
                "y": y.astype(int).tolist(),
                "admission": admission.astype(int).tolist(),
                "selected_global_mode_ids": selected_modes.astype(int).tolist(),
                "selected_mu_ids": selected_mu.astype(int).tolist(),
                "snr_bucket_ids": np.asarray(info["snr_bucket_ids"], dtype=int).tolist(),
                "Q_req_gb": q_req_gb.astype(float).tolist(),
            },
            "slot_summary": {
                "reward": float(info["reward"]),
                "base_reward": float(info.get("base_reward", info["reward"])),
                "total_reward": float(info.get("total_reward", info["reward"])),
                "reward_quality": float(info["quality_contrib"]),
                "reward_load": float(info["load_penalty"]),
                "reward_aoi": float(info["aoi_penalty"]),
                "quality_term": float(info["quality_term"]),
                "load_term": float(info["load_term"]),
                "aoi_term": float(info["aoi_term"]),
                "aoi_tail_term": float(info["aoi_tail_term"]),
                "aoi_state_term": float(info["aoi_state_term"]),
                "fresh_gain_term": float(info["fresh_gain_term"]),
                "mean_aoi": float(info["mean_AoI"]),
                "max_aoi": float(info["max_AoI"]),
                "total_scheduled_count": total_scheduled,
                "total_semantic_load": float(np.sum(lambda_usage)),
                "total_predicted_quality": float(total_predicted_quality),
                "avg_predicted_quality": float(total_predicted_quality / max(total_scheduled, 1)),
                "total_served_ds": total_scheduled,
                "quality_violation_bucketwise": float(info.get("quality_violation_bucketwise", 0.0)),
                "aoi_violation_max": float(info.get("aoi_violation_max", 0.0)),
                "load_violation": float(info.get("load_violation", 0.0)),
                "recv_aoi_bonus": float(info.get("recv_aoi_bonus", 0.0)),
                "fallback_to_best_feasible_mode_count": int(
                    info.get("fallback_to_best_feasible_mode_count", 0)
                ),
                "backhaul_constraint_satisfied": bool(backhaul_violations == 0),
                "constraint_violations": {
                    "backhaul": backhaul_violations,
                    "quality": int(quality_violations),
                    "cache": int(cache_violations),
                    "mode_mapping": int(mode_mapping_violations),
                },
                "feasibility_masks_removed_actions": {
                    "all_mu_infeasible_count": int(info["all_mu_infeasible_count"]),
                    "note": "Mode/cache/quality/load feasibility is enforced inside env.step before greedy scheduling.",
                },
            },
        }

    def render(self, mode="human"):
        print(
            f"Step {self.current_step}: reward_type=fixed_multi_objective, "
            f"scheduled={int(np.sum(self.last_y))}, reward={self.last_utility:.4f}"
        )

    def close(self):
        return None
