from __future__ import annotations

import numpy as np

from harl.envs.uav_escs.SC.uav_escs_env_sc import SCUAVEnv
from harl.utils.configs_tools import get_defaults_yaml_args


def make_env(**overrides):
    _, env_args = get_defaults_yaml_args("happo", "uav_escs_sc")
    env_args.update(
        {
            "n_uav": 1,
            "n_ds": 2,
            "T": 4.0,
            "initial_cache_prob": 1.0,
            "use_semcom_registry": False,
            "B_sut_sat": 5.0e6,
            "B_uav_sut": 5.0e6,
            "Q_req_by_instruction_snr_bucket": [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            "A_limit_by_instruction": [7.5, 5.5, 10.0],
            "instruction_mode_strategy": "fixed",
            "fixed_instruction_id": 1,
        }
    )
    env_args.update(overrides)
    env = SCUAVEnv(env_args)
    env.seed(123)
    return env


def greedy_actions(env):
    sut = np.zeros(env.sut_act_dim_total, dtype=np.float32)
    uav = np.zeros(env.uav_act_dim_total, dtype=np.float32)
    uav[env.uav_act_slices["mu_logits"]] = np.asarray([5.0, 1.0, 0.0, -1.0], dtype=np.float32)
    if "scheduling_weights" in env.uav_act_slices:
        uav[env.uav_act_slices["scheduling_weights"]] = np.asarray([2.0, 1.0, 0.0], dtype=np.float32)
    return [sut, uav]


def step_once(env):
    env.reset()
    return env.step(greedy_actions(env))[-2][0]


def test_lambda_projection_constraint():
    env = make_env()
    env.reset()
    beta = env._project_beta(np.asarray([100.0], dtype=float))
    assert np.all(np.isfinite(beta))
    assert np.all(beta >= env.beta_sat_lower_bound - 1.0e-8)
    assert np.all(beta <= 1.0 + 1.0e-8)
    assert abs(float(np.sum(beta)) - 1.0) < 1.0e-8


def test_cache_update_and_availability():
    env = make_env()
    info = step_once(env)
    q_pre = np.asarray(info["q_cache_pre"], dtype=int)
    q_post = np.asarray(info["q_cache"], dtype=int)
    y = np.asarray(info["y"], dtype=int)
    admission = ((1 - q_pre) * env.owner_mask.astype(int)).astype(int)
    expected = (q_pre - y + admission) * env.owner_mask.astype(int)
    assert np.array_equal(q_post, expected)
    assert np.all(y <= q_pre)


def test_aoi_update_success_and_failure():
    env = make_env()
    info = step_once(env)
    y = np.asarray(info["y"], dtype=int)
    tau_pre = np.asarray(info["tau_cache_pre"], dtype=int)
    a_pre = np.asarray(info["A_rcc_pre"], dtype=float)
    a_post = np.asarray(info["A_rcc"], dtype=float)
    slot = int(info["slot"])
    for k in range(env.n_ds):
        n = int(env.owner_uav[k])
        if y[n, k] > 0:
            tau = int(tau_pre[n, k]) if int(tau_pre[n, k]) >= 0 else slot
            expected = min(float(slot - tau + 1), env.A_max)
        else:
            expected = min(float(a_pre[k] + 1.0), env.A_max)
        assert abs(float(a_post[k]) - expected) < 1.0e-8


def test_backhaul_budget_qreq_feasibility_and_common_mode():
    env = make_env()
    info = step_once(env)
    chi = np.asarray(info["chi"], dtype=int)
    lambda_sem = np.asarray(info["Lambda_sem"], dtype=float)
    phi = np.asarray(info["Phi_bh"], dtype=float)
    q_hat = np.asarray(info["Q_hat_rec"], dtype=float)
    q_req = np.asarray(info["Q_req_gb"], dtype=float)
    for n in range(env.n_uav):
        assert float(np.sum(chi[n] * lambda_sem[n])) <= float(phi[n]) + 1.0e-8
        used = np.argwhere(chi[n] > 0)
        if used.size:
            assert np.unique(used[:, 1]).size <= 1
        for k, m in used:
            assert float(q_hat[n, k, m]) + 1.0e-8 >= float(q_req[n])


def test_instruction_mapping_balance_aoi_quality():
    env = make_env()
    assert env.instruction_names[:3] == ["balance", "aoi", "quality"]
    assert env._q_req_for_instruction_bucket(1, 0) == 0.0
    assert env._q_req_for_instruction_bucket(2, 0) == 0.0
    assert float(env.A_limit_by_instruction[0]) == 7.5
    assert float(env.A_limit_by_instruction[1]) == 5.5
    assert float(env.A_limit_by_instruction[2]) == 10.0


def test_semantic_load_formula_and_reward_component_sum():
    env = make_env()
    info = step_once(env)
    gamma = np.asarray(info["gamma_bh"], dtype=float)
    assert np.all(gamma > 0.0)
    assert np.all(np.isfinite(info["Lambda_sem"]))
    expected = np.asarray(info["L_z"], dtype=float) + env.side_info_bits / np.maximum(
        np.log2(1.0 + gamma)[:, None, None], 1.0e-12
    )
    assert np.allclose(np.asarray(info["Lambda_sem"], dtype=float), expected)
    components = [
        "reward_quality",
        "reward_load_cost",
        "reward_aoi_penalty",
        "penalty_quality_violation",
        "penalty_aoi_violation",
        "penalty_load_violation",
        "bonus_aoi_reduction",
    ]
    total = sum(float(info[name]) for name in components)
    assert abs(total - float(info["reward"])) < 1.0e-8
    assert abs(float(info["reward_component_sum"]) - float(info["reward"])) < 1.0e-8
