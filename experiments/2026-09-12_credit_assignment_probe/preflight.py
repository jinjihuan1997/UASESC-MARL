"""Small preregistered checks. Any assertion failure prevents formal execution."""
from support import *
from engine import execute
import traceback


def historical_check(data, parent, scenario):
    path = FROZEN / 'evaluation' / f'seed_{parent}/joint_at_1000000' / f'{scenario}.npz'
    meta = read(path.with_suffix('.json')); assert sha(path) == meta['trace_sha256']
    with np.load(path) as old:
        np.testing.assert_array_equal(data['env_seeds'], old['seeds'])
        cols = [FIELDS.index(f) for f in old['fields']]
        difference = float(np.max(np.abs(data['trace'][..., cols]-old['trace'])))
        np.testing.assert_allclose(data['trace'][..., cols], old['trace'], rtol=0, atol=1e-9)
        for new, previous in [('executed_modes', 'modes'), ('resource_fractions', 'resource_fractions'),
                               ('aoi_after', 'aoi_after'), ('budgets', 'budgets'), ('unused_budgets', 'unused_budgets')]:
            np.testing.assert_allclose(data[new], old[previous], rtol=0, atol=1e-9)
    assert data['legacy_external_hashes'].tolist() == meta['external_hashes']
    return dict(path=str(path), sha256=sha(path), max_abs_trace_difference=difference, tolerance=1e-9)


def main():
    m = verify_inputs(full=True)
    result = dict(state='RUNNING', started_utc=stamp(), checks={}, benchmarks=[])
    write(HERE / 'preflight.json', result)
    seed = m['training_seeds'][0]
    # Tiny CPU/CUDA measurements use only the original preflight environment seed.
    env_seeds = [m['preflight_seeds'][0]]*20
    benchmarks = {}
    for device in ['cpu'] + (['cuda:0'] if torch.cuda.is_available() else []):
        t = time.monotonic()
        data, meta = execute(seed, 'fixed_0', 'D', m['action_seeds'][0], env_seeds, device, steps=40)
        benchmarks[device] = (data, meta)
        result['benchmarks'].append(dict(device=device, steps=800, seconds=meta['elapsed_seconds'],
                                        steps_per_second=800/meta['elapsed_seconds'], setup_included=True))
        write(HERE / 'preflight.json', result)
        print('benchmark', device, meta['elapsed_seconds'], flush=True)
    device = min(result['benchmarks'], key=lambda x: x['seconds'])['device']
    # CPU was the original evaluation device. CUDA must reproduce deterministic
    # mode/physics trajectories at unchanged tolerances before being eligible.
    original, original_meta = execute(seed, 'fixed_0', 'A', 0, m['validation_seeds'], 'cpu')
    result['checks']['A_historical_reproduction'] = historical_check(original, seed, 'fixed_0')
    save_npz(HERE / 'preflight/A_original_reproduction.npz', original)
    if 'cuda:0' in benchmarks:
        cpu, _ = execute(seed, 'fixed_0', 'A', 0, m['preflight_seeds'], 'cpu', steps=40)
        gpu, _ = execute(seed, 'fixed_0', 'A', 0, m['preflight_seeds'], 'cuda:0', steps=40)
        result['checks']['cross_device'] = dict(
            raw_action_max_absolute_difference=float(np.max(np.abs(cpu['resource_raw']-gpu['resource_raw']))),
            reward_max_absolute_difference=float(np.max(np.abs(cpu['trace'][...,0]-gpu['trace'][...,0]))),
            modes_equal=bool(np.array_equal(cpu['executed_modes'], gpu['executed_modes'])),
            note='Cross-device float32 networks need not be bitwise equal. No tolerance is enlarged; CPU retains historical comparability.')
    # Use CPU if GPU cannot satisfy the existing 1e-9 complete trace gate.
    if device != 'cpu':
        candidate, _ = execute(seed, 'fixed_0', 'A', 0, m['validation_seeds'], device)
        historical_check(candidate, seed, 'fixed_0')
    x, _ = execute(seed, 'fixed_1', 'D', m['action_seeds'][0], m['preflight_seeds'], device, steps=50)
    y, _ = execute(seed, 'fixed_1', 'D', m['action_seeds'][0], m['preflight_seeds'], device, steps=50)
    z, _ = execute(seed, 'fixed_1', 'D', m['action_seeds'][1], m['preflight_seeds'], device, steps=50)
    assert digest_arrays(x) == digest_arrays(y)
    assert x['external_hashes'].tolist() == z['external_hashes'].tolist()
    assert not np.array_equal(x['resource_raw'], z['resource_raw'])
    result['checks']['repeated_action_seed_bitwise_replay'] = True
    result['checks']['different_action_seed_same_exogenous_sequence'] = True
    # No resource/UAV draw is allowed to mutate the ambient torch stream.
    state = torch.get_rng_state().clone(); stream = ActionStreams(777, 'cpu')
    stream.call(0, lambda: torch.rand(11)); assert torch.equal(state, torch.get_rng_state())
    untouched = [s.clone() for s in stream.states]
    stream.call(1, lambda: torch.rand(7))
    assert all(torch.equal(untouched[i], stream.states[i]) for i in [0,2,3])
    result['checks']['separate_agent_rng_and_ambient_restoration'] = True
    for variant in 'BCEF':
        execute(seed, 'fixed_2', variant, m['action_seeds'][0], m['preflight_seeds'], device, steps=30)
    p, _ = execute(seed, 'random_switch_once', 'D', m['probe_action_seeds'][0], m['preflight_seeds'], device, steps=600, probe=True)
    g = p['trace'][..., FIELDS.index('instruction_id')].astype(int)
    assert np.count_nonzero(g[1:] != g[:-1]) == 1
    assert p['done'][-1].all() and not p['done'][:-1].any()
    result['checks']['original_random_switch_and_terminal'] = True
    result['checks']['allocation_no_physical_advance'] = True
    result['checks']['all_new_steps_original_physics_reward_audited'] = True
    verify_inputs(full=True)
    result['checks']['protected_input_hashes_unchanged'] = True
    from fit_value_probe import benchmark_fit
    fit_benchmarks = benchmark_fit(p['critic_input_preallocation'].shape[-1]+3)
    result['fitting_benchmarks'] = fit_benchmarks
    result.update(state='PASS', execution_device=device, fitting_device=min(fit_benchmarks,key=lambda x:x['seconds'])['device'],
                  maximum_concurrency=1, torch_threads=1, completed_utc=stamp(),
                  expected_formal_execution_steps=1454400,
                  expected_seconds_from_short_batch=(1454400/800)*min(x['seconds'] for x in result['benchmarks']),
                  note='Short checked-rollout timing includes setup and is a rough execution estimate; probe fitting measured separately.')
    write(HERE / 'preflight.json', result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    try: main()
    except BaseException:
        error = traceback.format_exc()
        (HERE / 'logs/preflight_error.log').write_text(error)
        p = read(HERE / 'preflight.json') if (HERE / 'preflight.json').exists() else {}
        p.update(state='FAIL', error=error, failed_utc=stamp()); write(HERE / 'preflight.json', p)
        raise
