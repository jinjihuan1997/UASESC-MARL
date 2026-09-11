"""Calibration, interface verification, and paired full-episode evaluations."""
import argparse
import copy
import subprocess
import ast
from runtime import OUT, REFERENCE, h, FIELDS, summarize, audit_trace, predict_reward, np, torch, verify
from online_policy import UAVGreedy, SUTGreedy, TreeScorePredictor, sut_candidates, score_features


def local_modes(policy, obs, masks):
    # No concatenation of the three UAVs' inputs into a centralized policy.
    return [policy.act(obs[:, i], masks[:, i]) for i in range(1, 4)]


def calibrate(folder, candidates, protocol):
    policy = UAVGreedy(candidates)
    features, labels, seed_ids, gids = [], [], [], []
    rng = np.random.default_rng(protocol['calibration_behavior_seed'])
    seeds = protocol['fit_seeds'] + protocol['validation_seeds']
    for scene, schedule in protocol['scenarios'].items():
        env, obs, _, masks = h.make_env(h.config('joint_continue', 218), seeds, schedule)
        for slot in range(600):
            resource_actions = sut_candidates(obs[:, 0])
            candidate_actions, rewards = [], []
            for family in range(3):
                preview = copy.copy(env)
                post, _, post_masks = preview.allocate_resources(resource_actions[:, family])
                actions = [resource_actions[:, family]] + local_modes(policy, post, post_masks)
                rewards.append(predict_reward(preview, actions))
                candidate_actions.append(actions)
            scores = torch.stack(rewards, -1)
            features.append(h.arr(score_features(obs[:, 0])).reshape(-1, 30))
            labels.append(h.arr(scores).reshape(-1)*100)
            seed_ids.append(np.repeat(seeds, 3))
            gids.append(np.repeat(h.arr(env.context()[0]), 3))
            chosen = torch.as_tensor(rng.integers(0, 3, size=env.count), dtype=torch.int64)
            actions = [torch.stack([a[i] for a in candidate_actions], 1)[torch.arange(env.count), chosen] for i in range(4)]
            obs, _, masks, _, values = h.checked_step(env, actions)
            np.testing.assert_allclose(h.arr(scores[torch.arange(env.count), chosen]), values['common_reward'], atol=1e-9, rtol=0)
        print(folder.name, 'CALIBRATION', scene, flush=True)
    np.savez_compressed(folder / 'calibration_data.npz', X=np.concatenate(features), y=np.concatenate(labels),
                        seeds=np.concatenate(seed_ids), gid=np.concatenate(gids))
    h.write(folder / 'calibration.json', dict(state='complete', episodes=len(seeds)*len(protocol['scenarios']),
        physical_steps=600*len(seeds)*len(protocol['scenarios']), counterfactual_labels=3*600*len(seeds)*len(protocol['scenarios']),
        dataset_sha256=h.digest(folder/'calibration_data.npz'), candidate_score_vs_actual_step_verified=True))


def interface_test(folder, candidates, protocol, sut_policy, uav_policy):
    env, obs, _, masks = h.make_env(h.config('joint_continue', 218), protocol['validation_seeds'], [[0, 2]])
    with np.load(folder / 'predictor_validation.npz') as z:
        np.testing.assert_allclose(h.arr(sut_policy.predictor.predict(torch.as_tensor(z['X']))), z['native_prediction'], atol=1e-9, rtol=0)
    input_hashes = {}
    for slot in range(80):
        sut_obs = obs[:, 0].clone()
        action, _ = sut_policy.act(sut_obs)
        # Change every UAV observation while holding the SUT input fixed.
        corrupted = obs.clone()
        corrupted[:, 1:] = 99999
        np.testing.assert_array_equal(h.arr(action), h.arr(sut_policy.act(corrupted[:, 0])[0]))
        post, _, post_masks = env.allocate_resources(action)
        modes = local_modes(uav_policy, post, post_masks)
        for i in range(1, 4):
            changed = post.clone()
            for other in range(4):
                if other != i:
                    changed[:, other] = -99999
            np.testing.assert_array_equal(h.arr(modes[i-1]), h.arr(uav_policy.act(changed[:, i], post_masks[:, i])))
        # Permuting the evaluation batch cannot affect another episode's action.
        order = torch.arange(env.count-1, -1, -1)
        np.testing.assert_array_equal(h.arr(action[order]), h.arr(sut_policy.act(sut_obs[order])[0]))
        obs, _, masks, _, _ = h.checked_step(env, [action]+modes)
    source = (OUT / 'online_policy.py').read_text()
    tree = ast.parse(source)
    forbidden = {'env', 'critic', 'share_obs', 'global_state', 'tau', 'noise_us', 'noise_sat', 'instructions', 'potential_content'}
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    assert not names.intersection(forbidden), names.intersection(forbidden)
    h.write(folder / 'interface_audit.json', dict(state='PASS', validation_states=80*len(protocol['validation_seeds']),
        sut_reads_only_own_observation=True, each_uav_reads_only_own_post_budget_observation_and_mask=True,
        no_online_environment_handle=True, no_other_episode_access=True, portable_predictor_matches_sklearn=True,
        static_source_forbidden_access_check=True))


def evaluate(folder, label, sut_policy, uav_policy, protocol):
    output = folder / 'evaluation' / label
    output.mkdir(parents=True, exist_ok=True)
    reference_pairing = h.read(REFERENCE / 'evaluation/rules/R_myopic/summary.json')['pairing']
    parts, per, seed_scores, hashes, family_counts, pairing = [], {}, {}, {}, {}, {}
    for scene, schedule in protocol['scenarios'].items():
        env, obs, _, masks = h.make_env(h.config('joint_continue', 218), protocol['evaluation_seeds'], schedule)
        external = h.external_hashes(env)
        assert external == reference_pairing[scene]
        records, all_modes, fractions, ages, families = [], [], [], [], []
        for slot in range(600):
            resource, family = sut_policy.act(obs[:, 0])
            post, _, post_masks = env.allocate_resources(resource)
            actions = [resource] + local_modes(uav_policy, post, post_masks)
            obs, _, masks, info, values = h.checked_step(env, actions)
            values['instruction_id'] = h.arr(info['gid'])
            records.append(np.column_stack([values[k] for k in FIELDS]))
            all_modes.append(h.arr(info['mode']))
            fractions.append(h.arr(env.beta))
            ages.append(h.arr(env.aoi).astype(np.uint16))
            families.append(h.arr(family))
        trace, age = np.stack(records), np.stack(ages)
        audit_trace(trace, age)
        file = output / f'{scene}.npz'
        np.savez_compressed(file, trace=trace, fields=np.asarray(FIELDS), seeds=np.asarray(protocol['evaluation_seeds']),
                            modes=np.stack(all_modes), resource_fractions=np.stack(fractions), aoi_after=age,
                            resource_family=np.stack(families))
        per[scene] = summarize(trace)
        parts.append(trace)
        seed_scores[scene] = trace[..., 0].mean(0).tolist()
        hashes[str(file.relative_to(OUT))] = h.digest(file)
        pairing[scene] = external
        family_counts[scene] = np.bincount(np.stack(families).ravel(), minlength=3).tolist()
        for j, (start, gid) in enumerate(schedule):
            end = schedule[j+1][0] if j+1 < len(schedule) else 600
            per[f'{scene}/segment_{j}_g{gid}'] = summarize(trace[start:end])
            if j:
                per[f'{scene}/first20_after_{start}'] = summarize(trace[start:min(start+20, end)])
        h.write(output / 'status.json', dict(state='evaluating', completed_episodes=len(parts)*env.count))
        print(folder.name, label, scene, round(100*per[scene]['common_reward'], 5), flush=True)
    merged = np.concatenate(parts)
    per['overall'] = summarize(merged)
    for gid in range(3):
        per[f'true_instruction_{gid}'] = summarize(merged[merged[..., 5] == gid])
    result = dict(method=f'{folder.name}/{label}', per_scope=per, per_eval_seed_scores=seed_scores,
                  resource_family_counts=family_counts, trace_hashes=hashes, pairing=pairing,
                  audit=dict(state='PASS', episodes=260, slots=156000, independent_reward_and_physics=True))
    h.write(output / 'results.json', result)
    h.write(output / 'status.json', dict(state='complete', completed_episodes=260, results_sha256=h.digest(output/'results.json')))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--modes', type=int, choices=(3, 16), required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    protocol = verify()
    folder = OUT / f'modes_{args.modes}'
    folder.mkdir(exist_ok=True)
    candidates = [0, 5, 10] if args.modes == 3 else list(range(16))
    assert not (folder/'evaluation').exists(), 'Preserve existing evaluation; do not silently rerun.'
    if not (folder/'calibration_data.npz').exists():
        calibrate(folder, candidates, protocol)
    else:
        assert h.digest(folder/'calibration_data.npz') == h.read(folder/'calibration.json')['dataset_sha256']
    subprocess.run([protocol['fit_python'], str(OUT/'fit_predictor.py'), '--branch', str(folder)], check=True)
    fit = h.read(folder / 'fit.json')
    assert h.digest(folder/'predictor.npz') == fit['predictor_sha256']
    uav_policy = UAVGreedy(candidates)
    fitted = SUTGreedy(TreeScorePredictor(folder/'predictor.npz'))
    static = SUTGreedy(static_family_by_instruction=fit['static_family_by_instruction'])
    interface_test(folder, candidates, protocol, fitted, uav_policy)
    for name, policy in [('fitted_sut', fitted), ('calibrated_sut', static)]:
        evaluate(folder, name, policy, uav_policy, protocol)
    verify()
    h.write(folder / 'status.json', dict(state='complete', evaluation_episodes=520, calibration_episodes=130,
        training_kind='offline_one_step_reward_regression_only; no RL retraining',
        predictor_sha256=h.digest(folder/'predictor.npz')))
    print('BRANCH COMPLETE', args.modes, flush=True)


if __name__ == '__main__':
    main()
