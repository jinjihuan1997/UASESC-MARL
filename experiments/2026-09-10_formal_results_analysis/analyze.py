"""Evidence audit and descriptive paired analysis; writes only beside this script."""
import csv
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
sys.dont_write_bytecode = True
import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent.parent
RUNS = {
    'SC': PROJECT/'experiments/2026-09-09_instruction_long_training/runs/three_seed_sc_20260909',
    'CC': PROJECT/'experiments/2026-09-09_cc_comparison/runs/three_seed_cc_20260909',
}
MEANS = ['common_reward', 'base_reward', 'training_reward', 'recv_aoi_bonus', 'mean_aoi',
         'max_aoi', 'p95_aoi', 'aoi_exceedance_fraction', 'max_aoi_violation', 'infeasible_uav_fraction']
SUMS = ['deliveries', 'predicted_quality_sum', 'channel_uses', 'quality_violations',
        'budget_violations', 'cache_violations']
METRICS = ['common_reward', 'mean_aoi', 'delivered_predicted_psnr', 'deliveries_per_slot',
           'channel_uses_per_slot', 'aoi_exceedance_fraction', 'infeasible_uav_fraction',
           'channel_uses_per_delivery', 'failure_fraction']


def read(p):
    return json.loads(Path(p).read_text())


def digest(p):
    with Path(p).open('rb') as stream:
        h = hashlib.sha256()
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def write(p, value):
    Path(p).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def aggregate(rows):
    weights = np.asarray([r.get('steps', 1) for r in rows], dtype=float)
    result = {k: float(np.average([r[k] for r in rows], weights=weights)) for k in MEANS}
    for key in SUMS + (['attempts', 'failed_packets'] if 'attempts' in rows[0] else []):
        result[key] = sum(r[key] for r in rows)
    result['steps'] = int(weights.sum())
    result['delivered_predicted_psnr'] = result['predicted_quality_sum']/result['deliveries'] if result['deliveries'] else None
    result['deliveries_per_slot'] = result['deliveries']/weights.sum()
    result['channel_uses_per_slot'] = result['channel_uses']/weights.sum()
    result['channel_uses_per_delivery'] = result['channel_uses']/result['deliveries'] if result['deliveries'] else None
    result['failure_fraction'] = result.get('failed_packets', 0)/result.get('attempts', result['deliveries']) if result['deliveries'] else None
    return result


def assert_aggregate(actual, reported):
    for key in MEANS + SUMS + ['delivered_predicted_psnr', 'deliveries_per_slot', 'channel_uses_per_slot']:
        if reported[key] is None:
            assert actual[key] is None
        else:
            np.testing.assert_allclose(actual[key], reported[key], rtol=1e-10, atol=1e-8, err_msg=key)


def stats(values):
    values = [float(v) for v in values]
    return dict(n=len(values), mean=float(np.mean(values)), sd=float(np.std(values, ddof=1)) if len(values)>1 else None,
                min=min(values), max=max(values), values=values, positive_count=sum(v>0 for v in values))


def metric_stats(rows):
    return {k: stats([r[k] for r in rows]) for k in METRICS}


def parse_trace(path):
    rows = []
    with path.open(newline='') as stream:
        for row in csv.DictReader(stream):
            for key in MEANS + SUMS + ['slot', 'instruction_id', 'attempts', 'failed_packets']:
                if key in row:
                    row[key] = float(row[key])
            rows.append(row)
    return rows


def mode_summary(rows):
    prop = np.asarray([json.loads(r['proposed_modes']) for r in rows])
    executed = np.asarray([json.loads(r['executed_modes']) for r in rows])
    beta = np.asarray([json.loads(r['resource_fractions']) for r in rows])
    active = executed >= 0
    return dict(uav_decisions=int(active.size), active_decisions=int(active.sum()),
        proposed_counts={str(k):int(v) for k,v in sorted(Counter(prop.reshape(-1)).items())},
        executed_counts={str(k):int(v) for k,v in sorted(Counter(executed.reshape(-1)).items())},
        override_fraction_when_executed=float(np.mean(prop[active] != executed[active])),
        no_execution_fraction=float(np.mean(~active)), mean_absolute_beta_deviation_from_equal=float(np.abs(beta-1/3).mean()),
        beta_below_two_cheapest_cc_packets_fraction=float(np.mean(beta < 19680/60000-1e-12)),
        cc_mode3_below_two_packet_budget_fraction=float(np.mean(beta[executed==3] < 19680/60000-1e-12)) if np.any(executed==3) else None)


def main():
    results, records, fixed_modes, switching, training = {}, {}, {}, {}, {}
    audit = dict(created_utc=datetime.now(timezone.utc).isoformat(), script_sha256=digest(__file__),
                 protocol_sha256=digest(HERE/'PROTOCOL.md'), families={}, errors=[])
    canonical_pairing = None
    for family, run in RUNS.items():
        manifest, state = read(run/'manifest.json'), read(run/'status.json')
        assert state['state'] == 'complete'
        checked = 0
        for name, expected in manifest['input_hashes'].items():
            assert digest(run/name) == expected, f'Input changed: {name}'
            checked += 1
        for name, expected in state['report']['files'].items():
            assert digest(run/'report'/name) == expected, f'Report changed: {name}'
            checked += 1
        for job in manifest['jobs']:
            out = run/job['output']
            job_state = read(out/'status.json')
            assert job_state['state'] == 'complete' and job_state['completed_steps'] == 10000000
            for name, expected in job_state['checkpoint_hashes'].items():
                assert digest(out/name) == expected, f'Model changed: {job["id"]}/{name}'
                checked += 1
            logs = [json.loads(line) for line in (out/'training_metrics.jsonl').read_text().splitlines()]
            # Duplicate updates can be replayed after a clean checkpoint restore.
            logs = list({r['update']:r for r in logs}.values())
            assert max(r['steps'] for r in logs) == 10000000
            training[f'{family}/{job["id"]}'] = [dict(end_steps=end,
                mean_training_reward=float(np.mean([r['mean_training_reward'] for r in logs if end-2000000 < r['steps'] <= end])))
                for end in range(2000000, 10000001, 2000000)]
        reported = read(run/'report/per_seed_results.json')
        count, slots, fixed_slots = 0, 0, 0
        for item, summary in reported.items():
            label = f'{family}/{item}'
            if canonical_pairing is None:
                canonical_pairing = summary['pairing']
            assert summary['pairing'] == canonical_pairing
            episodes, mode_rows, switch_rows = [], defaultdict(list), defaultdict(list)
            for path in sorted((run/'evaluation'/item/'episodes').glob('*.json')):
                ep = read(path)
                trace = path.parent.parent/'traces'/f'{path.stem}.csv'
                assert digest(trace) == ep['trace_sha256'], f'Trace changed: {trace}'
                assert ep['external_trajectory_sha256'] == canonical_pairing[f"{ep['scenario']}/{ep['seed']}"]
                assert ep['steps'] == 600
                assert all(ep[k] == 0 for k in ['quality_violations','budget_violations','cache_violations'])
                episodes.append(ep)
                scenario = ep['scenario']
                selected_switch = scenario in ('switch300_1_to_2','switch300_2_to_1') and summary['method'] in (
                    'IC_HAPPO','HAPPO_hidden_instruction','R_fixed','CC_IC_HAPPO')
                if scenario.startswith('fixed_') or selected_switch:
                    rows = parse_trace(trace)
                    assert len(rows) == 600
                    assert_aggregate(aggregate(rows), ep)
                    if scenario.startswith('fixed_'):
                        mode_rows[scenario].extend(rows)
                        fixed_slots += len(rows)
                    if selected_switch or (scenario in ('fixed_1','fixed_2') and summary['method'] in (
                        'IC_HAPPO','HAPPO_hidden_instruction','R_fixed','CC_IC_HAPPO')):
                        for window, lo, hi in [('before',250,300),('after',300,350),('late',550,600)]:
                            switch_rows[f'{scenario}/{window}'].append(aggregate(rows[lo:hi]))
                count += 1
                slots += ep['steps']
            assert len(episodes) == 260
            overall = aggregate(episodes)
            assert_aggregate(overall, summary['overall'])
            scenarios = {}
            for scenario in manifest['scenarios']:
                scenarios[scenario] = aggregate([ep for ep in episodes if ep['scenario'] == scenario])
                assert_aggregate(scenarios[scenario], summary['by_scenario'][scenario])
            results[label] = dict(family=family, method=summary['method'], training_seed=summary['training_seed'],
                                  overall=overall, by_scenario=scenarios)
            records[label] = episodes
            fixed_modes[label] = {s:mode_summary(rows) for s, rows in mode_rows.items()}
            switching[label] = {s:aggregate(rows) for s, rows in switch_rows.items()}
            print(f'verified {label}: 260 episodes', flush=True)
        audit['families'][family] = dict(run=str(run), manifest_sha256=digest(run/'manifest.json'),
            verified_input_report_model_files=checked, training_jobs=len(manifest['jobs']),
            training_steps=sum(read(run/j['output']/'status.json')['completed_steps'] for j in manifest['jobs']),
            trace_hashes=count, evaluation_slots=slots, fixed_trace_slots_reaggregated=fixed_slots)
    diagnostic = HERE/'cc_rule_diagnostic'
    if (diagnostic/'status.json').exists() and read(diagnostic/'status.json')['state'] == 'complete':
        state, value = read(diagnostic/'status.json'), read(diagnostic/'summary.json')
        assert digest(diagnostic/'summary.json') == state['summary_sha256']
        assert value['pairing'] == canonical_pairing
        episodes, mode_rows = [], defaultdict(list)
        formal_delivery = read(RUNS['CC']/'report/per_seed_results.json')['seed_85/CC_IC_HAPPO']['delivery_pairing']
        assert value['delivery_pairing'] == formal_delivery
        for p in sorted((diagnostic/'episodes').glob('*.json')):
            ep = read(p)
            trace = diagnostic/'traces'/f'{p.stem}.csv'
            assert digest(trace) == ep['trace_sha256']
            rows = parse_trace(trace)
            assert_aggregate(aggregate(rows), ep)
            episodes.append(ep)
            if ep['scenario'].startswith('fixed_'):
                mode_rows[ep['scenario']].extend(rows)
        assert len(episodes) == 260
        assert_aggregate(aggregate(episodes), value['overall'])
        label = 'CC/rules/CC_R_fixed_diagnostic'
        results[label] = dict(family='CC', method='CC_R_fixed_diagnostic', training_seed=None,
            overall=aggregate(episodes), by_scenario={s:aggregate([ep for ep in episodes if ep['scenario']==s]) for s in manifest['scenarios']})
        fixed_modes[label] = {s:mode_summary(rows) for s, rows in mode_rows.items()}
        audit['posthoc_diagnostic'] = state
    else:
        raise RuntimeError('Wait for the declared diagnostic to complete before publishing analysis')
    grouped = defaultdict(list)
    for value in results.values():
        grouped[value['method']].append(value)
    method_summary, instruction_contrasts = {}, {}
    for method, values in grouped.items():
        values.sort(key=lambda v: v['training_seed'] or -1)
        method_summary[method] = dict(overall=metric_stats([v['overall'] for v in values]),
            by_scenario={s:metric_stats([v['by_scenario'][s] for v in values]) for s in manifest['scenarios']})
        instruction_contrasts[method] = {key:stats([v['by_scenario']['fixed_2'][key]-v['by_scenario']['fixed_1'][key]
            for v in values]) for key in METRICS}
    pairs = [
        ('IC_HAPPO','HAPPO_hidden_instruction'), ('IC_MAPPO','MAPPO_hidden_instruction'),
        ('IC_HAPPO','HAPPO_fixed_mode_rule'), ('IC_HAPPO','HAPPO_equal_resources'),
        ('IC_HAPPO','HAPPO_no_task_aux_reward'), ('IC_HAPPO','R_fixed'),
        ('IC_MAPPO','R_fixed'), ('IC_HAPPO','IC_MAPPO'),
        ('IC_HAPPO','CC_IC_HAPPO'), ('IC_MAPPO','CC_IC_MAPPO'),
        ('CC_IC_HAPPO','CC_R_fixed_diagnostic'), ('CC_IC_MAPPO','CC_R_fixed_diagnostic'),
        ('IC_HAPPO','CC_R_fixed_diagnostic'), ('R_fixed','CC_R_fixed_diagnostic')]
    paired = {}
    for first, second in pairs:
        left = sorted(grouped[first], key=lambda v:v['training_seed'] or -1)
        right_map = {v['training_seed']:v for v in grouped[second]}
        right = [right_map.get(v['training_seed'], right_map.get(None)) for v in left]
        assert all(v is not None for v in right)
        paired[f'{first} - {second}'] = {}
        for scope in ['overall']+list(manifest['scenarios']):
            part = [(a['overall'],b['overall']) if scope=='overall' else (a['by_scenario'][scope],b['by_scenario'][scope]) for a,b in zip(left,right)]
            paired[f'{first} - {second}'][scope] = {k:stats([a[k]-b[k] for a,b in part]) for k in METRICS}
    # Exact row equivalence and Pareto dominance on every stored SNR knot.
    with np.load(RUNS['SC']/'source/reference/inputs/profile.npz', allow_pickle=False) as table:
        q, load = table['q_hat_mean'], table['bar_ls_total_mean']
        groups, dominated = [], {}
        for i in range(len(q)):
            if not any(i in g for g in groups):
                groups.append([j for j in range(i,len(q)) if np.array_equal(q[i],q[j]) and np.array_equal(load[i],load[j])])
            dominated[str(i)] = [j for j in range(len(q)) if np.all(q[j]>=q[i]-1e-12) and np.all(load[j]<=load[i]+1e-12)
                                and (np.any(q[j]>q[i]+1e-12) or np.any(load[j]<load[i]-1e-12))]
        profile = dict(equivalent_rows=groups, dominated_by=dominated, nondominated_modes=[i for i in range(len(q)) if not dominated[str(i)]])
    audit['state'] = 'PASS'
    audit['formal_episodes'] = sum(f['trace_hashes'] for f in audit['families'].values())
    audit['formal_slots'] = sum(f['evaluation_slots'] for f in audit['families'].values())
    audit['paired_external_scenario_seed_conditions'] = len(canonical_pairing)
    write(HERE/'audit.json', audit)
    write(HERE/'analysis.json', dict(methods=method_summary, paired_differences=paired, quality_minus_aoi_instruction=instruction_contrasts,
        fixed_mode_diagnostics=fixed_modes, switch_windows=switching, training_windows=training, sc_profile_geometry=profile,
        note='Descriptive post-hoc analysis. Only three training seeds; no significance or convergence claims.'))
    write(HERE/'independent_per_seed_results.json', results)
    fields = ['method', 'scope', 'n']+[f'{k}_{suffix}' for k in METRICS for suffix in ('mean','sd','min','max')]
    with (HERE/'metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for method, value in method_summary.items():
            for scope in ['overall','fixed_0','fixed_1','fixed_2']:
                part = value['overall'] if scope=='overall' else value['by_scenario'][scope]
                writer.writerow(dict(method=method, scope=scope, n=part['mean_aoi']['n'],
                    **{f'{k}_{suffix}':part[k][suffix] for k in METRICS for suffix in ('mean','sd','min','max')}))
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == '__main__':
    main()
