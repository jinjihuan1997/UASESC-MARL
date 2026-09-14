"""One physical step per slot, using the original staged tensor environment."""
from support import *


@torch.no_grad()
def execute(seed, scenario, variant, action_seed, env_seeds, device='cpu', steps=600, probe=False):
    start = time.monotonic()
    env, (obs, state, available), cfg = make_env(seed, env_seeds, scenario, device)
    learned = variant in 'ABCDEF'
    actors = FrozenActors(cfg, env, seed) if learned else None
    streams = ActionStreams(action_seed, device)
    external = exogenous(env)
    legacy_external = frozen.external_hashes(env)
    observations = {}; records = {}
    def add(key, value):
        records.setdefault(key, []).append(array(value) if torch.is_tensor(value) else np.asarray(value).copy())
    for slot in range(steps):
        assert env.step_index == slot and not env.allocation_pending
        before_state = state[:, 0].clone()
        before_phase = {key: getattr(env, key).clone() for key in (
            'q', 'tau', 'aoi', 'psi', 'gamma_us', 'gamma_sat', 'gamma_du', 'rr_cursor', 'switched')}
        previous_beta = env.beta.clone()
        previous_reward_record = getattr(env, 'last', None)
        rule = frozen.simple_rule_actions(env, variant, obs) if not learned else None
        if variant == 'E':
            resource = torch.full((env.count, env.U), 2 / env.U - 1, dtype=torch.float32, device=env.device)
            sut_stats = None
        elif learned:
            resource, sut_stats = actors.act(0, obs[:, 0], available[:, 0], variant not in ('B', 'D'), streams)
        else:
            resource = rule[0]; sut_stats = None
        post, _, post_masks = env.allocate_resources(resource)
        assert env.step_index == slot and env.allocation_pending
        assert getattr(env, 'last', None) is previous_reward_record
        for key, value in before_phase.items():
            assert torch.equal(value, getattr(env, key)), f'Allocation mutated {key}'
        # This is an audit calculation only; it is not reapplied to the environment.
        raw = ((resource.double() + 1) / 2).clamp_min(0)
        raw = torch.where(raw.sum(-1, keepdim=True) > 0,
                          raw / raw.sum(-1, keepdim=True).clamp_min(1e-12), torch.full_like(raw, 1/3))
        expected_beta = env.p.beta_sat_lower_bound + (1-3*env.p.beta_sat_lower_bound)*raw
        expected_beta = expected_beta / expected_beta.sum(-1, keepdim=True)
        torch.testing.assert_close(env.beta, expected_beta, atol=1e-12, rtol=0)
        np.testing.assert_allclose(array(post[:, 1:, 0]), array(env.budget / env.p.Lambda_ref), atol=1e-7, rtol=1e-6)
        np.testing.assert_allclose(array(before_state[:, 4:7]), array(previous_beta), atol=1e-7, rtol=1e-6)
        choices = []; uav_stats = []
        if variant == 'F':
            # Reuse only the existing mode rule. Its suggested SUT action is discarded.
            choices = frozen.simple_rule_actions(env, 'R_instruction', post)[1:]
        elif not learned:
            choices = rule[1:]
        else:
            for i in range(1, 4):
                a, stats = actors.act(i, post[:, i], post_masks[:, i], variant not in ('C', 'D'), streams)
                choices.append(a); uav_stats.append(stats)
        sampled_modes = torch.stack(choices, 1).argmax(-1)
        add('resource_raw', resource); add('resource_fractions', env.beta)
        add('previous_resource_fractions', previous_beta)
        add('sampled_modes', sampled_modes)
        add('sut_concentration', sut_stats['concentration'] if sut_stats else np.full((env.count, 3), np.nan))
        add('sut_entropy', sut_stats['entropy'] if sut_stats else np.full(env.count, np.nan))
        add('mode_probabilities', np.stack([s['probabilities'] for s in uav_stats], 1) if uav_stats else np.full((env.count, 3, 16), np.nan))
        add('mode_entropy', np.stack([s['entropy'].reshape(env.count) for s in uav_stats], 1) if uav_stats else np.full((env.count, 3), np.nan))
        add('sampled_log_prob', np.column_stack([sut_stats['logp'] if sut_stats else np.full(env.count, np.nan)] +
            ([s['logp'] for s in uav_stats] if uav_stats else [np.full(env.count, np.nan)]*3)))
        if probe:
            add('critic_input_preallocation', before_state)
            add('uav_observations_used', post[:, 1:]); add('uav_masks_used', post_masks[:, 1:])
            add('uav_actions_sampled', torch.stack(choices, 1))
        # Original independent physical and reward checker runs on every new step.
        obs, state, available, info, values = frozen.checked_step(env, [resource] + choices)
        assert env.step_index == slot + 1 and not env.allocation_pending
        values['instruction_id'] = array(info['gid'])
        add('trace', np.column_stack([values[k] for k in FIELDS]))
        add('training_reward', info['common_reward'].float())
        add('done', np.full(env.count, env.step_index >= env.p.max_steps, dtype=bool))
        add('executed_modes', info['mode']); add('budgets', info['budget'])
        add('usage', info['usage']); add('unused_budgets', info['budget']-info['usage'])
        add('deliveries_uav', info['served'].sum(-1)); add('served_ds', info['served'])
        add('quality_predicted_uav', info['quality']); add('aoi_after', env.aoi.to(torch.int16))
    if actors: actors.verify()
    data = {k: np.stack(v) for k, v in records.items()}
    data.update(fields=np.asarray(FIELDS), env_seeds=np.asarray(env_seeds), parent_seed=np.asarray(seed),
                variant=np.asarray(variant), scenario=np.asarray(scenario), action_seed=np.asarray(action_seed),
                stream_seeds=np.asarray(streams.seeds), rng_initial=np.stack([array(x) for x in streams.initial]),
                rng_final=np.stack([array(x) for x in streams.states]), external_hashes=np.asarray(external),
                legacy_external_hashes=np.asarray(legacy_external))
    if steps == 600:
        assert data['done'][-1].all() and not data['done'][:-1].any()
    meta = dict(parent_seed=seed, variant=variant, scenario=scenario, action_seed=action_seed,
                environment_seeds=env_seeds, steps=steps, episodes=len(env_seeds) if steps == 600 else 0,
                device=device, elapsed_seconds=time.monotonic()-start,
                all_step_physics_reward_checks_passed=True, allocation_no_clock_cache_reward_advance=True,
                original_actors_unchanged=True, external_hashes=external, legacy_external_hashes=legacy_external,
                numeric_sha256=digest_arrays(data), action_stream_seeds=streams.seeds,
                instruction_strategy=cfg['env_args']['instruction_mode_strategy'] if probe else 'explicit_evaluation')
    return data, meta


def batch_identity(seed, scenario, variant, action_seed, env_seeds, probe=False):
    m = read(HERE / 'manifest.json')
    return dict(protocol_sha256=m['protocol_sha256'], diagnostic_code_sha256=m['diagnostic_code_sha256'],
                seed=seed, scenario=scenario, variant=variant, action_seed=action_seed,
                environment_seeds=env_seeds, probe=probe, steps=600, reference_commit=REFERENCE_COMMIT)


def run_batch(folder, seed, scenario, variant, action_seed, env_seeds, device, probe=False):
    folder = Path(folder); file = folder / 'trajectory.npz'; marker = folder / 'complete.json'
    identity = batch_identity(seed, scenario, variant, action_seed, env_seeds, probe)
    if marker.exists():
        previous = read(marker)
        assert previous['identity'] == identity and previous['trace_sha256'] == sha(file)
        assert previous['state'] == 'complete' and previous['episodes'] == len(env_seeds)
        return previous
    assert not file.exists(), f'Unmarked trajectory requires explicit inspection: {file}'
    data, meta = execute(seed, scenario, variant, action_seed, env_seeds, device, probe=probe)
    meta.update(state='complete', identity=identity, trace_sha256=save_npz(file, data), completed_utc=stamp())
    write(marker, meta)
    print(json.dumps(dict(completed=str(folder.relative_to(HERE)), episodes=meta['episodes'], seconds=meta['elapsed_seconds'])), flush=True)
    return meta
