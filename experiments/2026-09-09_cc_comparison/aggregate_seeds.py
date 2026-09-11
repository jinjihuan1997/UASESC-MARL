"""Report seed-level variation separately from paired evaluation episodes."""
import argparse
import csv
from pathlib import Path
import statistics
import uuid

from common import stamp, write
from multiseed_protocol import read, verify_model, verify_run
from training_checkpoint import digest


METRICS = ['common_reward', 'base_reward', 'mean_aoi', 'max_aoi', 'p95_aoi',
    'aoi_exceedance_fraction', 'max_aoi_violation', 'infeasible_uav_fraction',
    'delivered_predicted_psnr', 'deliveries_per_slot', 'channel_uses_per_slot']


def stats(values):
    if any(value is None for value in values):
        return dict(mean=None, sample_std=None, n=len(values), note='At least one seed had no delivered-quality value')
    return dict(mean=statistics.mean(values), sample_std=statistics.stdev(values) if len(values) > 1 else None,
                n=len(values))


def aggregate_run(run):
    run = Path(run).resolve()
    manifest = verify_run(run)
    if (run/'report').exists():
        verification = read(run/'report/verification.json')
        for name, sha in verification['files'].items():
            if digest(run/'report'/name) != sha:
                raise RuntimeError('Existing report changed; preserving it')
        return verification
    results, responses, reference_pairing, delivery_pairing = {}, {}, None, None
    items = [job['id'] for job in manifest['jobs']]+[f'rules/{name}' for name in manifest['rules']]
    for job in manifest['jobs']:
        verify_model(run/job['output'], manifest['steps_per_method'])
    for item in items:
        path = run/'evaluation'/item
        status = read(path/'status.json')
        if status['state'] != 'complete' or digest(path/'summary.json') != status['summary_sha256']:
            raise RuntimeError(f'Incomplete or changed evaluation: {item}')
        result = read(path/'summary.json')
        if reference_pairing is None:
            reference_pairing = result['pairing']
        if result['pairing'] != reference_pairing:
            raise AssertionError(f'Unpaired exogenous evaluation trajectories: {item}')
        if delivery_pairing is None: delivery_pairing=result['delivery_pairing']
        if result['delivery_pairing']!=delivery_pairing: raise AssertionError('Unpaired CC delivery draws')
        episodes = []
        for episode in (path/'episodes').glob('*.json'):
            row = read(episode)
            if digest(path/'traces'/episode.with_suffix('.csv').name) != row['trace_sha256']:
                raise RuntimeError('A committed trace changed')
            episodes.append(row)
        if len(episodes) != len(manifest['scenarios'])*len(manifest['evaluation_seeds']):
            raise RuntimeError('Evaluation episode count differs from the frozen protocol')
        responses[item] = {}
        for scenario, schedule in manifest['scenarios'].items():
            if not scenario.startswith('switch300_'):
                continue
            before, after = str(schedule[0][1]), str(schedule[1][1])
            selected = [e for e in episodes if e['scenario'] == scenario]
            responses[item][scenario] = {}
            for metric in ['mean_aoi','delivered_predicted_psnr','common_reward','deliveries_per_slot']:
                pre = [e['by_instruction'][before][metric] for e in selected]
                post = [e['by_instruction'][after][metric] for e in selected]
                defined=all(v is not None for v in pre+post)
                responses[item][scenario][metric] = dict(before_mean=statistics.mean(pre) if defined else None,
                    after_mean=statistics.mean(post) if defined else None, mean_change=statistics.mean(b-a for a,b in zip(pre,post)) if defined else None,
                    evaluation_episodes=len(selected))
        results[item] = result
    learned, scenarios, paired = {}, {}, {}
    for method in manifest['method_order']:
        rows = [results[f'seed_{seed}/{method}'] for seed in manifest['seeds']]
        learned[method] = {metric: stats([row['overall'][metric] for row in rows]) for metric in METRICS}
        scenarios[method] = {scenario: {metric: stats([row['by_scenario'][scenario][metric] for row in rows])
            for metric in METRICS} for scenario in manifest['scenarios']}
        if method != 'CC_IC_HAPPO':
            differences = [results[f"seed_{seed}/CC_IC_HAPPO"]['overall']['common_reward']-
                row['overall']['common_reward']
                for seed, row in zip(manifest['seeds'], rows)]
            paired[method] = dict(per_seed=dict(zip(map(str, manifest['seeds']), differences)), **stats(differences))
    rules = {name: results[f'rules/{name}']['overall'] for name in manifest['rules']}
    for name in manifest['rules']:
        differences = [results[f'seed_{seed}/CC_IC_HAPPO']['overall']['common_reward']-
            rules[name]['common_reward'] for seed in manifest['seeds']]
        paired[name] = dict(per_seed=dict(zip(map(str,manifest['seeds']),differences)), **stats(differences))
    destination = run/('report.tmp.'+uuid.uuid4().hex[:12])
    destination.mkdir()
    write(destination/'per_seed_results.json', results)
    write(destination/'switch_response.json', dict(results=responses,
        note='Descriptive before/after means over equal 300-slot phases; not a causal advantage over another controller.'))
    write(destination/'comparison.json', dict(seeds=manifest['seeds'], learned=learned, rules=rules,
        by_scenario=scenarios, paired_common_reward_differences=paired,
        difference_direction='CC_IC_HAPPO minus CC comparator; positive favors CC_IC_HAPPO',
        note='Each mean/SD uses three training-seed means. Evaluation episodes are paired, not additional training seeds. Rules are evaluated once. No significance or convergence claim is inferred.'))
    flat = []
    for method in manifest['method_order']:
        row = dict(method=method, training_seed_count=3)
        for metric in METRICS:
            row[metric+'_mean'] = learned[method][metric]['mean']
            row[metric+'_sample_std'] = learned[method][metric]['sample_std']
        flat.append(row)
    for method in manifest['rules']:
        row = dict(method=method, training_seed_count=0)
        for metric in METRICS:
            row[metric+'_mean'] = rules[method][metric]
            row[metric+'_sample_std'] = None
        flat.append(row)
    with (destination/'comparison.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    lines = ['三训练种子CC对比', '', f"用途：{manifest['purpose']}；种子：{manifest['seeds']}；每方法每种子{manifest['steps_per_method']:,}步。",
        '各学习方法报告三个训练种子均值的平均值 ± 样本标准差；CC 的两种算法间对比，SC/CC 合并对比需等待双方完整评估。', '',
        '| 方法 | 共同奖励↑ | 平均AoI↓ | AoI超限比例↓ | 预测PSNR↑ |', '|---|---:|---:|---:|---:|']
    metrics = ['common_reward', 'mean_aoi', 'aoi_exceedance_fraction', 'delivered_predicted_psnr']
    for method in manifest['method_order']:
        values = [learned[method][metric] for metric in metrics]
        formatted = [f"{v['mean']:.5f} ± {v['sample_std']:.5f}" if v['mean'] is not None else 'N/A' for v in values]
        lines.append('| '+method+' | '+' | '.join(formatted)+' |')
    for method in manifest['rules']:
        formatted = [f'{rules[method][metric]:.5f}' if rules[method][metric] is not None else 'N/A' for metric in metrics]
        lines.append('| '+method+' | '+' | '.join(formatted)+' |')
    lines += ['', '质量值来自CC实测/解析平均表的条件成功质量。传输失败照常扣资源，AoI不按成功接收更新；共同奖励独立重算。',
              '三个训练种子可观察训练波动，不能据此自动认定统计显著、稳定收敛或方法必然占优。']
    (destination/'comparison.md').write_text('\n'.join(lines)+'\n')
    verification = dict(state='complete', utc=stamp(), purpose=manifest['purpose'], training_jobs=len(manifest['jobs']),
        training_steps=manifest['total_training_steps'], evaluation_items=len(items),
        evaluation_slots=manifest['total_evaluation_slots'], paired_external_trajectories=len(reference_pairing),
        all_executed_constraints_passed=True,
        files={p.name: digest(p) for p in destination.iterdir() if p.is_file()})
    write(destination/'verification.json', verification)
    destination.rename(run/'report')
    return verification


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    print(aggregate_run(args.run))
