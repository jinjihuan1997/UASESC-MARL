"""Read-only checks of frozen inputs, complete checkpoints and seed statistics."""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
from common import stamp
from multiseed_protocol import GPU_METHOD, read, verify_model, verify_run
from training_checkpoint import digest, load_checkpoint


def close(actual, expected):
    if expected is None:
        assert actual is None
    else:
        assert math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-10), (actual, expected)


def tensors_equal(left, right):
    assert left.keys() == right.keys()
    assert all(torch.equal(left[k], right[k]) for k in left)


def verify(run, prepared_only=False):
    run = Path(run).resolve()
    manifest = verify_run(run)
    assert len(set(manifest['seeds'])) == 3
    assert manifest['total_training_steps'] == 21*manifest['steps_per_method']
    assert manifest['total_evaluation_jobs'] == 25
    assert {(j['seed'], j['method']) for j in manifest['jobs']} == {
        (seed, method) for seed in manifest['seeds'] for method in manifest['method_order']}
    for job in manifest['jobs']:
        config = read(run/job['config'])
        train = config['algo_args']['train']
        assert config['algo_args']['seed']['seed'] == job['seed']
        assert config['main_args']['exp_name'] == job['method']
        assert train['model_dir'] is None
        assert (train['n_rollout_threads'], train['episode_length']) == (10, 400)
        assert train['num_env_steps'] == manifest['steps_per_method']
        assert job['device'] == ('cuda:0' if job['method'] == GPU_METHOD else 'cpu')
        assert config['algo_args']['device']['cuda'] == (job['device'] != 'cpu')
    if prepared_only:
        return dict(passed=True, state='prepared_inputs_verified', utc=stamp(), seeds=manifest['seeds'],
            jobs=21, training_steps=manifest['total_training_steps'], trained_results_verified=False)
    for job in manifest['jobs']:
        folder = run/job['output']
        status = verify_model(folder, manifest['steps_per_method'])
        state, checkpoint = load_checkpoint(folder/'checkpoints')
        assert checkpoint['selected'] == 'current'
        assert state['update'] == manifest['steps_per_method']//4000
        assert state['total_updates'] == state['update']
        assert state['runtime'] == manifest['runtime']
        assert state['identity']['run_manifest_sha256'] == digest(run/'manifest.json')
        assert state['identity']['config_sha256'] == digest(run/job['config'])
        assert state['device'] == job['device'] == status['device']
        steps_per_env = manifest['steps_per_method']//10
        assert state['step_index'] == steps_per_env % 600
        assert state['episode_indices'] == [steps_per_env//600]*10
        assert [s['seed'] for s in state['source_episodes']] == [job['seed']+1000*i for i in range(10)]
        for i, actor in enumerate(state['actors']):
            tensors_equal(actor, torch.load(folder/f'actor_agent{i}.pt', map_location='cpu', weights_only=True))
        tensors_equal(state['critic'], torch.load(folder/'critic_agent.pt', map_location='cpu', weights_only=True))
        tensors_equal(state['normalizer'], torch.load(folder/'value_normalizer.pt', map_location='cpu', weights_only=True))
        for weights in state['actors']+[state['critic'], state['normalizer']]:
            assert all(torch.isfinite(v).all() for v in weights.values())
        rows = [json.loads(line) for line in (folder/'training_metrics.jsonl').read_text().splitlines()]
        assert [r['update'] for r in rows] == list(range(1, state['update']+1))
        assert [r['steps'] for r in rows] == [4000*u for u in range(1, state['update']+1)]
    items = [j['id'] for j in manifest['jobs']]+['rules/'+name for name in manifest['rules']]
    episode_count = len(manifest['scenarios'])*len(manifest['evaluation_seeds'])
    pairing, seed_means = None, {}
    checked_slots = 0
    for item in items:
        folder = run/'evaluation'/item
        status, summary = read(folder/'status.json'), read(folder/'summary.json')
        assert status['state'] == 'complete' and status['completed_episodes'] == episode_count
        assert digest(folder/'summary.json') == status['summary_sha256']
        assert len(summary['pairing']) == episode_count
        if pairing is None:
            pairing = summary['pairing']
        assert summary['pairing'] == pairing
        rewards, aois = [], []
        for scenario in manifest['scenarios']:
            for seed in manifest['evaluation_seeds']:
                stem = f'{scenario}_seed{seed}'
                episode = read(folder/'episodes'/f'{stem}.json')
                trace = folder/'traces'/f'{stem}.csv'
                assert digest(trace) == episode['trace_sha256']
                with trace.open(newline='') as stream:
                    rows = list(csv.DictReader(stream))
                assert len(rows) == episode['steps'] == 600
                assert [int(row['slot']) for row in rows] == list(range(600))
                for row in rows:
                    assert all(float(row[key]) == 0 for key in ('quality_violations', 'budget_violations', 'cache_violations'))
                    assert math.isfinite(float(row['common_reward']))
                reward = float(np.mean([float(row['common_reward']) for row in rows]))
                aoi = float(np.mean([float(row['mean_aoi']) for row in rows]))
                close(episode['common_reward'], reward)
                close(episode['mean_aoi'], aoi)
                rewards.append(reward)
                aois.append(aoi)
                checked_slots += 600
        seed_means[item] = dict(common_reward=float(np.mean(rewards)), mean_aoi=float(np.mean(aois)))
        for metric in ('common_reward', 'mean_aoi'):
            close(summary['overall'][metric], seed_means[item][metric])
    report = read(run/'report/comparison.json')
    verification = read(run/'report/verification.json')
    for name, sha in verification['files'].items():
        assert digest(run/'report'/name) == sha
    for method in manifest['method_order']:
        for metric in ('common_reward', 'mean_aoi'):
            values = [seed_means[f'seed_{seed}/{method}'][metric] for seed in manifest['seeds']]
            entry = report['learned'][method][metric]
            assert entry['n'] == 3
            close(entry['mean'], float(np.mean(values)))
            close(entry['sample_std'], float(np.std(values, ddof=1)))
    for rule in manifest['rules']:
        for metric in ('common_reward', 'mean_aoi'):
            close(report['rules'][rule][metric], seed_means[f'rules/{rule}'][metric])
    assert checked_slots == manifest['total_evaluation_slots']
    return dict(passed=True, state='complete_run_verified', utc=stamp(), purpose=manifest['purpose'],
        seeds=manifest['seeds'], training_jobs_verified=21, full_checkpoints_verified=21,
        evaluation_jobs_verified=25, checked_evaluation_slots=checked_slots,
        paired_trajectories=episode_count, seed_mean_and_sample_std_independently_recalculated=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--prepared-only', action='store_true')
    args = parser.parse_args()
    print(json.dumps(verify(args.run, args.prepared_only), indent=2))
