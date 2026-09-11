"""Scientific interface, causality, constraints, and reward regression checks."""
from pathlib import Path
import copy
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from protocol import VERSION, TABLE_SHA, METHODS, SCENARIOS, configurations, sha, activate_runtime
activate_runtime()
import numpy as np
from harl.envs.uav_escs.SC.uav_escs_env_sc import SCUAVEnv
from harl.envs.uav_escs.SC.rules import fixed_mode_actions, mode_choice, rule_actions
from evaluate import external_metrics, aggregate


class ComparisonTests(unittest.TestCase):
    def env(self, name="IC_HAPPO", **overrides):
        args = configurations(steps=8000)[name]["env_args"]
        args.update(overrides)
        env = SCUAVEnv(args)
        env.seed(2718)
        env.reset()
        self.addCleanup(env.close)
        return env

    def test_final_table_is_unchanged_and_rows_are_not_snr_groups(self):
        self.assertEqual(sha(VERSION/"inputs/profile.npz"), TABLE_SHA)
        env = self.env()
        np.testing.assert_array_equal(env.semantic_mode_lookup_by_bucket_mu, np.tile(np.arange(16), (4, 1)))
        with np.load(VERSION/"inputs/profile.npz", allow_pickle=True) as table:
            self.assertEqual(str(table["mode_names"][0]), "sci0m1_rate1")
            self.assertEqual(str(table["mode_names"][4]), "sci1m2_rate1")
            self.assertTrue(np.isnan(table["trained_snr_db"]).all())
            for n in range(env.n_uav):
                snr = 10*np.log10(env.gamma_bh[n])
                for m in range(16):
                    q = np.interp(snr, table["snr_grid_db"], table["q_hat_mean"][m])
                    l = np.interp(snr, table["snr_grid_db"], table["bar_ls_main_mean"][m])
                    np.testing.assert_allclose(env.Q_hat_rec[n, :, m], q)
                    np.testing.assert_allclose(env.Lambda_sem[n, :, m], l+env.side_info_bits/np.log2(1+env.gamma_bh[n]))
        with self.assertRaises(ValueError):
            self.env(semantic_mode_selection="snr_bucket_mu")

    def test_only_trainable_roles_have_heads(self):
        expected = {"HAPPO_fixed_mode_rule": [0], "HAPPO_equal_resources": [1, 2, 3]}
        for name, _, _ in METHODS:
            env = self.env(name)
            self.assertEqual(env.active_agent_ids, expected.get(name, [0, 1, 2, 3]))
            self.assertEqual(env._build_obs_multi().shape, (env.n_agents, 219))
            self.assertEqual(env._build_share_obs_multi().shape, (env.n_agents, 632))
            self.assertEqual(env._build_action_available_masks().shape, (env.n_agents, 16))
            for physical, space in zip(env.active_agent_ids, env.action_space):
                self.assertEqual(space.shape, (3,) if physical == 0 else (16,))
                if physical > 0:
                    self.assertEqual(space.hybrid_action_spec["continuous_groups"], [])
            self.assertNotIn("Lambda_budget_ratio_by_instruction", env.args)
            self.assertFalse(hasattr(env, "Lambda_budget_ratio_by_instruction"))
            self.assertEqual(env._sut_instruction_obs().size, 4)
            self.assertEqual(env._uav_instruction_obs(0).size, 9)
            self.assertEqual(env._share_instruction_obs().size, 7)

    def test_hidden_instruction_only_masks_explicit_actor_features(self):
        a, b = self.env(), self.env("HAPPO_hidden_instruction")
        np.testing.assert_array_equal(a._build_share_obs_multi(), b._build_share_obs_multi())
        np.testing.assert_array_equal(a._build_action_available_masks(), b._build_action_available_masks())
        np.testing.assert_array_equal(a._build_sut_obs()[:-4], b._build_sut_obs()[:-4])
        np.testing.assert_array_equal(b._sut_instruction_obs(), np.zeros(4))
        for n in range(3):
            np.testing.assert_array_equal(a._build_uav_obs(n)[:-9], b._build_uav_obs(n)[:-9])
            np.testing.assert_array_equal(a._uav_instruction_obs(n)[3:7], b._uav_instruction_obs(n)[3:7])
            np.testing.assert_array_equal(b._uav_instruction_obs(n)[[0, 1, 2, 7, 8]], np.zeros(5))

    def test_future_schedule_is_not_an_actor_or_critic_input(self):
        a = self.env(explicit_instruction_schedule=[[0, 0], [200, 1]])
        b = self.env(explicit_instruction_schedule=[[0, 0], [500, 2]])
        np.testing.assert_array_equal(a._build_obs_multi(), b._build_obs_multi())
        np.testing.assert_array_equal(a._build_share_obs_multi(), b._build_share_obs_multi())
        np.testing.assert_array_equal(a._build_action_available_masks(), b._build_action_available_masks())

    def test_instruction_distribution_and_all_directions(self):
        env = self.env()
        counts = np.zeros((3, 3), dtype=int)
        occupancy = np.zeros(3)
        switches = []
        for _ in range(6000):
            env._reset_episode_random_streams()
            env._reset_instruction_schedule()
            a, b, t = env.instruction_before_id, env.instruction_after_id, env.instruction_switch_step
            counts[a, b] += 1
            occupancy[a] += t
            occupancy[b] += 600-t
            switches.append(t)
        self.assertEqual(int(np.trace(counts)), 0)
        self.assertEqual(np.count_nonzero(counts), 6)
        self.assertTrue(all(100 <= t <= 500 for t in switches))
        self.assertEqual((min(switches), max(switches)), (100, 500))
        np.testing.assert_allclose(occupancy/occupancy.sum(), [1/3]*3, atol=.025, rtol=0)

    def test_snr_and_multi_switch_boundaries(self):
        env = self.env()
        for snr, bucket in [(4.999, 0), (5, 1), (9.999, 1), (10, 2), (14.999, 2), (15, 3)]:
            self.assertEqual(env._snr_to_bucket_id(snr), bucket)
        for schedule in SCENARIOS.values():
            env.explicit_instruction_schedule = schedule
            env._reset_instruction_schedule()
            for t, g in schedule:
                self.assertEqual(env._get_instruction_id_for_slot(t), g)
                if t:
                    self.assertNotEqual(env._get_instruction_id_for_slot(t-1), g)
        env.explicit_instruction_schedule = [[0, 0], [200, 1], [200, 2]]
        with self.assertRaises(ValueError):
            env._reset_instruction_schedule()

    def test_mean_reward_normalization_and_common_external_metric(self):
        a, b = self.env(), self.env("HAPPO_no_task_aux_reward")
        nonzero_bonus = False
        for _ in range(35):
            action = [np.zeros(3), *fixed_mode_actions(a)]
            xa, xb = a.step(action), b.step(action)
            np.testing.assert_array_equal(a.A_rcc, b.A_rcc)
            np.testing.assert_array_equal(a.q_cache, b.q_cache)
            ia, ib = xa[4][0], xb[4][0]
            expected = .1*np.mean(np.maximum(ia["A_rcc_pre"]-ia["A_rcc"], 0))/ia["A_limit_g"]
            self.assertAlmostEqual(ia["recv_aoi_bonus"], expected, places=12)
            self.assertAlmostEqual(ia["common_evaluation_reward"], ib["common_evaluation_reward"], places=12)
            self.assertAlmostEqual(ib["total_reward"], ib["base_reward"], places=12)
            self.assertAlmostEqual(sum(ib["reward_components"].values()), ib["total_reward"], places=12)
            external_metrics(a, ia)
            external_metrics(b, ib)
            nonzero_bonus |= expected > 0
        self.assertTrue(nonzero_bonus)

    def test_deterministic_aoi_tie_break_and_real_round_robin(self):
        env = self.env(Q_req_by_instruction_snr_bucket=[[0]*4]*3, scheduler="round_robin")
        env.rr_cursor[:] = 3
        # One packet per UAV; channel kept fixed so the expected cursor is exact.
        def capacity():
            env.Phi_bh[:] = [1.1*env.Lambda_sem[n, env.ds_by_uav[n][0], 0] for n in range(3)]
        with patch.object(env, "_update_backhaul_budget", capacity), patch.object(env, "_update_channels", lambda: None):
            for target in (3, 4, 5):
                action = [np.zeros(3), *[np.eye(16)[0] for _ in range(3)]]
                result = env.step(action)
                for n in range(3):
                    self.assertEqual(result[4][0]["selected_ds"][n], [int(env.ds_by_uav[n][target])])
                    self.assertEqual(env.rr_cursor[n], target+1)
        env = self.env(Q_req_by_instruction_snr_bucket=[[0]*4]*3)
        env.A_rcc[:] = 7
        ds = env.ds_by_uav[0]
        env.A_rcc[ds[2]] = 20
        scores, candidate = env._derived_schedule_scores(0, 0, env.q_cache, env.A_rcc, 0)
        order = np.flatnonzero(candidate)
        order = order[np.lexsort((order, -scores[order]))]
        self.assertEqual(order[0], ds[2])
        self.assertEqual(order[1], ds[0])

    def test_fixed_resources_and_rule_do_not_train_dummy_roles(self):
        env = self.env("HAPPO_equal_resources")
        env.step([np.eye(16)[0] for _ in range(3)])
        np.testing.assert_allclose(env.beta_sut_sat, [1/3]*3)
        env = self.env("HAPPO_fixed_mode_rule")
        expected = fixed_mode_actions(env)
        env.step([np.array([1, -1, 0])])
        np.testing.assert_array_equal(env.last_uav_raw_actions, expected)
        with self.assertRaises(ValueError):
            env.step([np.zeros(16)])

    def test_rules_do_not_read_hidden_current_budget_or_future(self):
        env = self.env()
        obs, masks = env._build_obs_multi(), env._build_action_available_masks()
        for kind in ("R_fixed", "G_local_greedy", "Random", "RoundRobin"):
            a = rule_actions(env, obs, masks, kind, np.random.default_rng(9))
            with patch.object(env, "Phi_bh", np.full(3, 1e12)), patch.object(env, "Q_hat_rec", np.zeros_like(env.Q_hat_rec)):
                b = rule_actions(env, obs, masks, kind, np.random.default_rng(9))
            for x, y in zip(a, b):
                np.testing.assert_array_equal(x, y)

    def test_action_dependent_cache_does_not_change_external_stream(self):
        a, b = self.env(), self.env("HAPPO_equal_resources")
        for _ in range(40):
            np.testing.assert_array_equal(a.gamma_uav_sut, b.gamma_uav_sut)
            self.assertEqual(a.gamma_sut_sat, b.gamma_sut_sat)
            self.assertEqual(a.current_instruction_id, b.current_instruction_id)
            a.step([np.array([1, -1, 0]), *[np.eye(16)[15] for _ in range(3)]])
            b.step([np.eye(16)[0] for _ in range(3)])

    def test_happo_mappo_use_distinct_runners_and_shared_hyperparameters(self):
        from harl.runners import RUNNER_REGISTRY
        self.assertEqual(RUNNER_REGISTRY["happo"].__name__, "OnPolicyHARunner")
        self.assertEqual(RUNNER_REGISTRY["mappo"].__name__, "OnPolicyMARunner")
        configs = configurations(steps=8000)
        self.assertEqual(configs["IC_HAPPO"]["algo_args"], configs["IC_MAPPO"]["algo_args"])
        for config in configs.values():
            self.assertIsNone(config["algo_args"]["train"]["model_dir"])
            self.assertFalse(config["algo_args"]["algo"]["instruction_conditioned_critic"])
            self.assertFalse(config["algo_args"]["algo"]["use_instruction_adv_norm"])
        with self.assertRaises(ValueError):
            configurations(steps=8001)


if __name__ == "__main__":
    unittest.main()
