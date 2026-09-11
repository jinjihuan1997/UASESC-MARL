"""Read-only verification of the four completed allocation probes."""
import hashlib
import json
from pathlib import Path

import torch
from common import ROOT, verify_reference


PROBES = {
    'cpu3_gpu1': ROOT/'results/allocation_4_vs_5_80k/cpu3_gpu1',
    'cpu3_gpu2': ROOT/'results/allocation_4_vs_5_80k/cpu3_gpu2',
    'cpu5_gpu2': ROOT/'results/allocation_7_80k/cpu5_gpu2',
    'cpu6_gpu1': ROOT/'results/allocation_7_cpu6_gpu1_80k/cpu6_gpu1',
}


def read(path):
    return json.loads(path.read_text())


def main():
    verify_reference()
    summaries, runs, comparisons = {}, {}, []
    for name, folder in PROBES.items():
        result = read(folder/'status.json')
        assert result['state'] == 'complete' and result['steps_per_method'] == 80_000
        assert result['total_steps'] == 560_000 and len(result['methods']) == 7
        assert abs(result['effective_steps_per_second']-560_000/result['wall_seconds']) < 1e-9
        assert result['formal_after']['state'] == 'training'
        assert not result['formal_after']['failed_jobs']
        assert not result['formal_after']['thermal_derated']
        for method, item in result['methods'].items():
            path = folder/method
            status, config, manifest = (read(path/f'{key}.json') for key in ('status', 'config', 'manifest'))
            assert status['state'] == 'complete' and status['completed_steps'] == 80_000
            assert status['device'] == status['environment_device'] == item['device']
            assert config['algo_args']['train']['num_env_steps'] == 80_000
            assert config['algo_args']['train']['n_rollout_threads'] == 10
            assert config['algo_args']['train']['episode_length'] == 400
            assert config['algo_args']['train']['model_dir'] is None
            assert manifest['seed'] == 1 and manifest['batch'] == 4000
            assert all(a != b for a, b in zip(manifest['initial_actor_hashes'], status['final_actor_hashes']))
            assert manifest['initial_critic_hash'] != status['final_critic_hash']
            assert len((path/'training_metrics.jsonl').read_text().splitlines()) == 20
            for source, expected in manifest['sources'].items():
                assert hashlib.sha256((path/'source'/source).read_bytes()).hexdigest() == expected
            for source, expected in manifest['reference']['input_hashes'].items():
                assert hashlib.sha256((path/'source'/source).read_bytes()).hexdigest() == expected
            for file, expected in status['checkpoint_hashes'].items():
                assert hashlib.sha256((path/file).read_bytes()).hexdigest() == expected
                state = torch.load(path/file, map_location='cpu', weights_only=True)
                assert all(torch.isfinite(v).all() for v in state.values())
            if method in runs:
                previous = runs[method]
                assert manifest['initial_actor_hashes'] == previous['manifest']['initial_actor_hashes']
                assert manifest['initial_critic_hash'] == previous['manifest']['initial_critic_hash']
                same_device = status['device'] == previous['status']['device']
                comparison = dict(profile=name, reference_profile=previous['profile'], method=method,
                                  same_device=same_device)
                if same_device:
                    for file in status['checkpoint_hashes']:
                        current = torch.load(path/file, map_location='cpu', weights_only=True)
                        prior = torch.load(previous['path']/file, map_location='cpu', weights_only=True)
                        assert current.keys() == prior.keys()
                        assert all(torch.equal(current[k], prior[k]) for k in current)
                    comparison['all_actor_critic_and_valuenorm_tensors_equal'] = True
                else:
                    comparison['note'] = 'CPU/CUDA policy RNG differs; trained weight equality is not asserted.'
                comparisons.append(comparison)
            else:
                runs[method] = dict(profile=name, path=path, status=status, manifest=manifest)
        summaries[name] = {key: result[key] for key in (
            'allocation', 'wall_seconds', 'effective_steps_per_second', 'linear_70m_hours',
            'cpu_temperature_peak_c', 'ram_available_min_mib', 'gpu_memory_peak_mib',
            'gpu_utilization_mean_percent')}
    print(json.dumps(dict(passed=True, methods_per_profile=7, steps_per_method=80_000,
        total_probe_steps=2_240_000, profiles=summaries,
        fastest_tested_profile=min(summaries, key=lambda name: summaries[name]['wall_seconds']),
        cross_profile_comparisons=comparisons,
        limitations=[
            'Each resource profile was measured once for a short engineering probe, with original formal jobs active.',
            'Memory and GPU utilization measurements include background jobs and the desktop.',
            '18 same-device comparisons are tensor-identical; three changed-device comparisons are not paired learning trajectories.',
            'This does not establish the globally optimal allocation or full-budget learning stability.',
        ]), indent=2))


if __name__ == '__main__':
    main()
