"""Replay frozen pilots to separate original feasibility design from strict input ablation.

No training or input mutation. Each invocation imports one frozen runtime only.
"""
from pathlib import Path
import argparse
import copy
import csv
import hashlib
import json
import sys


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    sys.path.insert(0, str(run / 'source'))
    from common import verify_reference
    from evaluate import load_actors, external_metrics
    from harl.envs.uav_escs.SC.uav_escs_env_sc import SCUAVEnv
    import numpy as np
    import torch
    torch.set_num_threads(1)
    verify_reference()
    manifest = json.loads((run / 'manifest.json').read_text())
    seed = manifest['evaluation_seeds'][0]
    records = []
    for method in manifest['methods']:
        config = json.loads((run / 'configs' / f'{method}.json').read_text())
        status = json.loads((run / 'jobs' / method / 'status.json').read_text())
        for name, expected in status['checkpoint_hashes'].items():
            assert sha(run / 'jobs' / method / name) == expected
        env_args = copy.deepcopy(config['env_args'])
        env_args['explicit_instruction_schedule'] = [[0, 0]]
        env = SCUAVEnv(env_args)
        actors = load_actors(config, env, run / 'jobs' / method)
        episodes = json.loads((run / 'evaluation' / method / 'summary.json').read_text())['episodes']
        for g in range(3):
            scenario = f'fixed_{g}'
            saved = next(e for e in episodes if e['seed'] == seed and e['scenario'] == scenario)
            csv_path = run / 'evaluation' / method / f'{scenario}_{seed}.csv'
            assert sha(csv_path) == saved['trace_sha256']
            with csv_path.open() as stream:
                rows = list(csv.DictReader(stream))
            env.explicit_instruction_schedule = [[0, g]]
            env.seed(seed)
            obs, _, available = env.reset()
            tape = hashlib.sha256()
            for item in [env.pos_uav, env.q_cache, env.tau_cache]:
                tape.update(item.tobytes())
            rnn = np.zeros((1, 1, 256), np.float32)
            masks = np.ones((1, 1), np.float32)
            delivering = overridden = bad_quality = pending = 0
            allowed = []
            for t, row in enumerate(rows):
                tape.update(env.gamma_bh.tobytes())
                tape.update(np.asarray([env.current_instruction_id]).tobytes())
                allowed.extend(available[1:, :16].sum(axis=1).tolist())
                with torch.no_grad():
                    actions = [actor.act(obs[i:i+1], rnn, masks, available[i:i+1], deterministic=True)[0].numpy().reshape(-1)
                               for i, actor in enumerate(actors)]
                proposed = np.asarray([int(np.argmax(a[:16])) for a in actions[1:]])
                has_cache = np.asarray([env.q_cache[n, env.ds_by_uav[n]].any() for n in range(3)])
                bad = np.asarray([env.Q_hat_rec[n, env.ds_by_uav[n][0], proposed[n]] <
                                  env._q_req_for_instruction_bucket(g, env._effective_snr_bucket_for_uav(n)) - 1e-9
                                  for n in range(3)])
                pending += int(has_cache.sum())
                bad_quality += int((has_cache & bad).sum())
                obs, _, _, _, info, available = env.step(actions)
                np.testing.assert_array_equal(env.last_selected_modes, json.loads(row['modes']))
                np.testing.assert_allclose(env.beta_sut_sat, json.loads(row['fractions']), rtol=0, atol=1e-12)
                metrics = external_metrics(env, info[0])
                assert abs(metrics['common_reward'] - float(row['common_reward'])) < 1e-9
                active = env.last_selected_modes >= 0
                delivering += int(active.sum())
                overridden += int((active & (proposed != env.last_selected_modes)).sum())
            assert tape.hexdigest() == saved['external_sha256']
            item = dict(method=method, instruction=g, seed=seed, slots=len(rows),
                        delivering_uav_slots=delivering, proposal_overrides=overridden,
                        override_fraction=overridden / delivering if delivering else None,
                        bad_quality_proposals=bad_quality, pending_uav_slots=pending,
                        allowed_mode_count_min=min(allowed), allowed_mode_count_max=max(allowed),
                        existing_trace_reproduced=True)
            records.append(item)
            print(json.dumps(item), flush=True)
    # This enumerates executed choices at a common state, not policy performance.
    env = SCUAVEnv(env_args)
    choices = []
    for g in range(3):
        env.explicit_instruction_schedule = [[0, g]]
        env.seed(seed)
        env.reset()
        for m in range(16):
            probe = copy.deepcopy(env)
            actions = [np.full(probe.sut_act_dim_total, -1.0 / 3.0)]
            for n in range(3):
                action = np.full(probe.uav_act_dim_total, -1.0)
                action[m] = 1.0
                actions.append(action)
            probe.step(actions)
            choices.append(dict(instruction=g, proposal=m, executed=probe.last_selected_modes.tolist()))
    result = dict(run=str(run), manifest_sha256=sha(run / 'manifest.json'), replay=records,
                  common_initial_state_equal_allocation_choices=choices,
                  caveat='One evaluation seed, three fixed instructions, both methods. These are distinct trained pilots; their rates alone do not identify a causal training effect. Original pre-September source was inspected, not retrained or evaluated here.')
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')


if __name__ == '__main__':
    main()
