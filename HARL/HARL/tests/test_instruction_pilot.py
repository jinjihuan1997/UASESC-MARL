"""Regression checks for causal state timing and paired instruction experiments."""
import json
import numpy as np
import pytest
from harl.envs.uav_escs.SC.uav_escs_env_sc import SCUAVEnv
from harl.utils.configs_tools import get_defaults_yaml_args


def make_env(**kw):
    _, args = get_defaults_yaml_args("happo", "uav_escs_sc")
    args.update(T=24., instruction_switch_min_step=8, instruction_switch_max_step=16,
                use_semcom_registry=False, semantic_profile_path="", n_semantic_modes=4)
    args.update(kw)
    env = SCUAVEnv(args)
    env.seed(481)
    env.reset()
    return env


def actions(env, mode=0):
    result = [np.zeros(env.sut_act_dim_total)]
    for _ in range(env.n_uav):
        a = np.zeros(env.uav_act_dim_total)
        a[mode] = 1.
        result.append(a)
    return result


def test_observed_channel_is_executed_and_budget_uses_current_action():
    e = make_env()
    for _ in range(24):
        observed = e.gamma_bh.copy()
        act = actions(e)
        act[0] = np.asarray([-1., 0., 1.])
        e.step(act)
        np.testing.assert_array_equal(e.last_slot_snapshot["gamma_bh"], observed)
        np.testing.assert_allclose(e.last_slot_snapshot["Phi_bh"],
                                   e.delta_T * np.minimum(e.B_uav_sut, e._project_beta(act[0]) * e.B_sut_sat))


def test_policies_share_exogenous_trajectories_across_episodes():
    a, b = make_env(), make_env()
    for episode in range(3):
        if episode:
            a.reset(); b.reset()
        np.testing.assert_array_equal(a.pos_ds, b.pos_ds)
        for _ in range(24):
            np.testing.assert_array_equal(a.gamma_bh, b.gamma_bh)
            assert a.current_instruction_id == b.current_instruction_id
            a.step(actions(a, 0))
            b.step(actions(b, 3))
            assert json.dumps(a.rng_content.bit_generator.state) == json.dumps(b.rng_content.bit_generator.state)


def test_actor_ablation_keeps_critic_masks_and_dynamics_identical():
    a, b = make_env(actor_observe_instruction=False), make_env(actor_observe_instruction=True)
    for _ in range(24):
        np.testing.assert_array_equal(a._build_share_obs_multi(), b._build_share_obs_multi())
        np.testing.assert_array_equal(a._build_action_available_masks(), b._build_action_available_masks())
        assert not np.array_equal(a._build_obs_multi(), b._build_obs_multi())
        assert not a._sut_instruction_obs().any()
        np.testing.assert_array_equal(a._uav_instruction_obs(0)[3:7], b._uav_instruction_obs(0)[3:7])
        ar, br = a.step(actions(a)), b.step(actions(b))
        np.testing.assert_array_equal(ar[2], br[2])
        np.testing.assert_array_equal(a.last_chi, b.last_chi)


def test_all_modes_has_no_missing_checkpoint_aliases():
    e = make_env(use_semcom_registry=False, n_semantic_modes=5, semantic_mode_selection="all_modes")
    assert e.n_mu_modes == 5
    for bucket in range(4):
        np.testing.assert_array_equal(e._bucket_mode_ids(bucket), np.arange(5))
    e.Q_req_by_instruction_snr_bucket[:] = -100.
    e.step(actions(e, 4))
    assert e.last_chi.shape[-1] == 5


def test_actor_mask_does_not_exclude_modes_due_to_previous_budget():
    e = make_env()
    e.Q_hat_rec[:] = 100.
    e.Phi_bh[:] = 0.
    masks = e._build_action_available_masks()
    # All cache/quality-feasible modes stay available before SUT acts.
    assert masks[1:, :e.n_mu_modes].all()


def test_missing_requested_policy_fails(tmp_path):
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("pilot_eval", Path(__file__).parents[1] / "examples/evaluate_instruction_constraints.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with pytest.raises(FileNotFoundError, match="missing actor weights"):
        module._load_actors(algo="happo", algo_args={}, env=make_env(), model_dir=tmp_path, device="cpu")
