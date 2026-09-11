"""Evaluate the existing 810974 10M base; never train or alter source artifacts."""
import os
import sys
from pathlib import Path

for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
sys.dont_write_bytecode = True
OUTPUT = Path(__file__).resolve().parent
EXPERIMENTS = OUTPUT.parent
BASE = EXPERIMENTS / '2026-09-11_three_new_seeds_training/base'
LONG = EXPERIMENTS / '2026-09-10_sequential_long_training'
CONTINUATION = EXPERIMENTS / '2026-09-11_three_new_seeds_training/continuation'
sys.path.insert(0, str(BASE))
import helpers as h
sys.path.insert(0, str(LONG))
from evaluation import FIELDS, summarize
from aggregate import audit_trace
import numpy as np
import torch


def main():
    torch.set_num_threads(1)
    manifest = h.verify()
    model = BASE / 'jobs/seed_810974/joint/milestones/steps_10000000'
    status = h.read(model / 'status.json')
    assert status['state'] == 'complete' and status['completed_steps'] == 10000000
    hashes = {str(model / f): sha for f, sha in status['checkpoint_hashes'].items()}
    for path, sha in hashes.items():
        assert h.digest(Path(path)) == sha
    cfg = h.config('joint', 810974)
    pairing = h.read(CONTINUATION / 'evaluation/rules/R_myopic/summary.json')['pairing']
    folder = OUTPUT / 'evaluation/seed_810974/joint_at_10000000'
    folder.mkdir(parents=True, exist_ok=True)
    per, scores, traces, parts = {}, {}, {}, []
    actors = None
    for scene, schedule in manifest['scenarios'].items():
        file = folder / f'{scene}.npz'
        meta = folder / f'{scene}.json'
        if meta.exists():
            record = h.read(meta)
            assert h.digest(file) == record['trace_sha256']
            assert record['external_hashes'] == pairing[scene]
            with np.load(file) as z:
                data, ages = z['trace'], z['aoi_after']
                np.testing.assert_array_equal(z['seeds'], manifest['evaluation_seeds'])
                np.testing.assert_array_equal(z['fields'], FIELDS)
        else:
            env, obs, _, masks = h.make_env(cfg, manifest['evaluation_seeds'], schedule)
            external = h.external_hashes(env)
            assert external == pairing[scene]
            if actors is None:
                actors = h.load_actors(cfg, env, model)
            records, modes, fractions, age_rows = [], [], [], []
            for slot in range(600):
                actions = h.learned_actions(env, actors, obs, masks, 'joint')
                obs, _, masks, info, values = h.checked_step(env, actions)
                values['instruction_id'] = h.arr(info['gid'])
                records.append(np.column_stack([values[f] for f in FIELDS]))
                modes.append(h.arr(info['mode']))
                fractions.append(h.arr(env.beta))
                age_rows.append(h.arr(env.aoi).astype(np.uint16))
            data, ages = np.stack(records), np.stack(age_rows)
            np.savez_compressed(file, trace=data, fields=np.asarray(FIELDS),
                                seeds=np.asarray(manifest['evaluation_seeds']),
                                modes=np.stack(modes), resource_fractions=np.stack(fractions),
                                aoi_after=ages)
            record = dict(overall=summarize(data),
                          by_evaluation_seed=[summarize(data[:, i]) for i in range(env.count)],
                          external_hashes=external, trace_sha256=h.digest(file))
            h.write(meta, record)
        assert data.shape == (600, 20, len(FIELDS))
        audit_trace(data, ages)
        expected_g = [next(g for t, g in reversed(schedule) if t <= slot) for slot in range(600)]
        np.testing.assert_array_equal(data[..., 5], np.broadcast_to(np.asarray(expected_g)[:, None], (600, 20)))
        per[scene] = summarize(data)
        scores[scene] = data[..., 0].mean(0).tolist()
        traces[str(file.relative_to(OUTPUT))] = h.digest(file)
        parts.append(data)
        for i, (start, g) in enumerate(schedule):
            end = schedule[i + 1][0] if i + 1 < len(schedule) else 600
            per[f'{scene}/segment_{i}_g{g}'] = summarize(data[start:end])
            if i:
                per[f'{scene}/first20_after_{start}'] = summarize(data[start:min(start + 20, end)])
        print(f'{scene}: {len(parts)}/13 reward_x100={100 * per[scene]["common_reward"]:.6f}', flush=True)
    merged = np.concatenate(parts)
    per['overall'] = summarize(merged)
    for g in range(3):
        per[f'true_instruction_{g}'] = summarize(merged[merged[..., 5] == g])
    for path, sha in hashes.items():
        assert h.digest(Path(path)) == sha
    h.verify()
    h.write(OUTPUT / 'base_810974_results.json', dict(
        item='seed_810974/joint_at_10000000', per_scope=per,
        per_eval_seed_scores=scores, trace_hashes=traces, model_hashes=hashes,
        model_directory=str(model), config=str(BASE / 'configs/seed_810974/joint.json'),
        pairing=pairing, audit=dict(state='PASS', episodes=260, slots=156000,
                                   independent_reward_and_physics=True,
                                   paired_exogenous_sequences=True, source_and_weights_unchanged=True)))
    print('BASE EVALUATION PASS', per['overall'], flush=True)


if __name__ == '__main__':
    main()
