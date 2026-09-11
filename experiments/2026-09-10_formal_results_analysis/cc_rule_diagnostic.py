"""One declared post-hoc CC rule; frozen environment and paired evaluation seeds."""
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent.parent
RUN = PROJECT/'experiments/2026-09-09_cc_comparison/runs/three_seed_cc_20260909'
sys.path.insert(0, str(RUN/'source'))
import numpy as np
import torch
from common import stamp, write
from multiseed_protocol import read, verify_run
from training_checkpoint import digest
from formal_eval import summarize
from cc_metrics import external_reference, metrics, array
from tensor_env import TensorCCEnv


def main():
    torch.set_num_threads(1)
    manifest = verify_run(RUN)
    config = read(RUN/'configs/seed_85/CC_IC_HAPPO.json')
    formal = read(RUN/'report/per_seed_results.json')['seed_85/CC_IC_HAPPO']
    out = HERE/'cc_rule_diagnostic'
    out.mkdir(exist_ok=True)
    identity = dict(manifest_sha256=digest(RUN/'manifest.json'),
                    protocol_sha256=digest(HERE/'PROTOCOL.md'), script_sha256=digest(Path(__file__)),
                    method='CC_R_fixed_diagnostic', label='POST_HOC_DIAGNOSTIC_NOT_TRAINED')
    if (out/'identity.json').exists():
        assert read(out/'identity.json') == identity, 'Diagnostic inputs changed'
    write(out/'identity.json', identity)
    summaries = []
    start = time.monotonic()
    for scenario, schedule in manifest['scenarios'].items():
        args = dict(config['env_args'], instruction_mode_strategy='explicit_evaluation',
                    explicit_instruction_schedule=schedule)
        for seed in manifest['evaluation_seeds']:
            key = f'{scenario}_seed{seed}'
            summary_file = out/'episodes'/f'{key}.json'
            trace_file = out/'traces'/f'{key}.csv'
            if summary_file.exists():
                value = read(summary_file)
                assert digest(trace_file) == value['trace_sha256']
                summaries.append(value)
                continue
            env = TensorCCEnv(args, count=1, seed=seed, device='cpu')
            env.reset()
            ref, trace = external_reference(config, seed, schedule)
            np.testing.assert_array_equal(array(env.pos_ds)[0], ref.pos_ds)
            np.testing.assert_array_equal(array(env.pos_uav)[0], ref.pos_uav)
            np.testing.assert_array_equal(array(env.q)[0], ref.q_cache[env.p.owner_mask].reshape(env.U, env.K))
            delivery_hash = hashlib.sha256(array(env.delivery_uniforms).tobytes()).hexdigest()
            assert delivery_hash == formal['delivery_pairing'][f'{scenario}/{seed}']
            rows = []
            for slot in range(env.p.max_steps):
                gid = next(g for t, g in reversed(schedule) if t <= slot)
                assert int(env.instructions[slot, 0]) == gid
                np.testing.assert_allclose(array(env.gamma_us)[0], ref.gamma_uav_sut, rtol=1e-12, atol=1e-10)
                np.testing.assert_allclose(float(env.gamma_sat[0]), ref.gamma_sut_sat, rtol=1e-12, atol=1e-10)
                trace.update(ref.gamma_uav_sut.tobytes())
                trace.update(np.asarray([ref.gamma_sut_sat, gid], dtype=np.float64).tobytes())
                mode = env.fixed_mode()
                actions = [torch.zeros_like(env.beta)] + [mode[:, u] for u in range(env.U)]
                _, _, _, done, info, _ = env.step(actions, auto_reset=False)
                assert bool(done.all()) == (slot == env.p.max_steps-1)
                np.testing.assert_allclose(array(env.beta), 1/env.U, rtol=0, atol=1e-15)
                row = dict(slot=slot, instruction_id=gid, **metrics(env, info, slot))
                row.update(proposed_sut_logits=json.dumps(array(actions[0])[0].tolist()),
                    proposed_modes=json.dumps([int(a[0].argmax()) for a in actions[1:]]),
                    executed_modes=json.dumps(array(info['mode'])[0].tolist()),
                    selected_ds=json.dumps(np.flatnonzero(array(info['served'])[0]).tolist()),
                    attempted_ds=json.dumps(np.flatnonzero(array(info['attempted'])[0]).tolist()),
                    resource_fractions=json.dumps(array(env.beta)[0].tolist()))
                rows.append(row)
                if slot < env.p.max_steps-1:
                    ref._update_channels()
            external_hash = trace.hexdigest()
            assert external_hash == formal['pairing'][f'{scenario}/{seed}']
            trace_file.parent.mkdir(parents=True, exist_ok=True)
            temp = trace_file.with_suffix('.tmp')
            with temp.open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            temp.replace(trace_file)
            value = dict(method=identity['method'], training_seed=None, scenario=scenario, seed=seed,
                steps=len(rows), external_trajectory_sha256=external_hash, delivery_trajectory_sha256=delivery_hash,
                trace_sha256=digest(trace_file), **summarize(rows))
            value['by_instruction'] = {str(g): dict(steps=len(part), **summarize(part))
                for g in range(3) if (part := [r for r in rows if r['instruction_id'] == g])}
            write(summary_file, value)
            summaries.append(value)
            write(out/'status.json', dict(state='running', updated_utc=stamp(),
                completed_episodes=len(summaries), total_episodes=260, elapsed_seconds=time.monotonic()-start))
        print(f'{scenario}: {len(summaries)}/260; elapsed {time.monotonic()-start:.1f}s', flush=True)
    result = dict(item_id='rules/CC_R_fixed_diagnostic', method=identity['method'], training_seed=None,
        overall=summarize(summaries), by_scenario={s:summarize([r for r in summaries if r['scenario']==s])
        for s in manifest['scenarios']}, pairing={f"{r['scenario']}/{r['seed']}":r['external_trajectory_sha256']
        for r in summaries}, delivery_pairing={f"{r['scenario']}/{r['seed']}":r['delivery_trajectory_sha256'] for r in summaries})
    write(out/'summary.json', result)
    write(out/'status.json', dict(state='complete', updated_utc=stamp(), completed_episodes=len(summaries),
        total_episodes=260, elapsed_seconds=time.monotonic()-start, all_transition_reward_checks_passed=True,
        all_external_and_delivery_hashes_paired=True, summary_sha256=digest(out/'summary.json')))
    print(json.dumps(result['overall'], indent=2), flush=True)


if __name__ == '__main__':
    main()
