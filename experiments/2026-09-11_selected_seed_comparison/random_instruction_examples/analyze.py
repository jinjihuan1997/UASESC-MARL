"""One random paired AoI/quality episode: frozen traces and optional exact replay."""
import json
import os
import sys
import hashlib
from pathlib import Path

for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
sys.dont_write_bytecode = True
OUT = Path(__file__).resolve().parent
PARENT = OUT.parent
EXPERIMENTS = PARENT.parent
RUNTIME = EXPERIMENTS / '2026-09-10_resource_selector_training'
sys.path.insert(0, str(RUNTIME))
import helpers as h
from evaluation import FIELDS, summarize
from aggregate import audit_trace
from rule_tools import myopic_actions
import numpy as np
import torch


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    torch.set_num_threads(1)
    selected = h.read(OUT / 'selection.json')
    assert digest(PARENT / 'comparison.json') == selected['comparison_sha256']
    data = h.read(PARENT / 'comparison.json')
    frozen = h.read(PARENT / 'source_hashes.json')
    seed, eval_seed = selected['training_seed'], selected['evaluation_seed']
    index = data['evaluation_seeds'].index(eval_seed)
    result, hashes = {}, {}
    weights = h.config('joint_continue', seed)['env_args']['reward_weights_by_instruction']
    for scene in selected['scenarios']:
        result[scene] = {}
        for name, row in data['methods'].items():
            entry = row['by_training_seed'].get(str(seed), row['by_training_seed'].get('rule'))
            if entry is None:
                continue
            path = Path(entry['source']) / f'{scene}.npz'
            hashes[str(path)] = digest(path)
            assert hashes[str(path)] == frozen[str(path)]
            with np.load(path) as z:
                np.testing.assert_array_equal(z['seeds'], data['evaluation_seeds'])
                np.testing.assert_array_equal(z['fields'], FIELDS)
                trace = z['trace'][:, index:index + 1]
                ages = z['aoi_after'][:, index:index + 1]
                modes = z['modes'][:, index]
                fractions = z['resource_fractions'][:, index]
                choice = z['selector_choices'][:, index] if name == 'selector' else None
            audit_trace(trace, ages)
            x = summarize(trace)
            age_mean, age_max, age_tail = [x[k] for k in ('age_mean_cost', 'age_max_cost', 'age_tail_cost')]
            components = dict(quality_credit=100*x['quality_credit'],
                              age_mean_cost=100*age_mean, age_max_cost=100*age_max,
                              age_tail_cost=100*age_tail, total_age_cost=100*(age_mean+age_max+age_tail),
                              resource_cost=100*x['resource_cost'], score=100*x['common_reward'])
            assert x['service_violation_cost'] == 0 and x['recv_aoi_bonus'] == 0
            np.testing.assert_allclose(components['quality_credit'] - components['total_age_cost'] - components['resource_cost'],
                                       components['score'], atol=1e-10, rtol=0)
            gid = int(scene.rsplit('_', 1)[1])
            age_flat = ages.reshape(600, -1).astype(float)
            physical = dict(mean_instantaneous_max_aoi=float(age_flat.max(-1).mean()),
                            mean_excess_aoi_above4=float(np.maximum(age_flat-4, 0).mean()),
                            delivered_blocks=int(trace[..., FIELDS.index('deliveries')].sum()),
                            total_channel_uses=float(trace[..., FIELDS.index('channel_uses')].sum()),
                            resource_mean_per_uav=fractions.mean(0).tolist(),
                            equal_fraction=float((choice == 1).mean()) if choice is not None else None)
            independently = 100 * (weights[gid][0] * (x['delivered_predicted_psnr']-21)*x['deliveries_per_slot']/360
                                   - weights[gid][1]*(.4*x['mean_aoi']+.3*physical['mean_instantaneous_max_aoi']+
                                                       .3*physical['mean_excess_aoi_above4'])/8
                                   - .02*x['channel_uses_per_slot']/60000)
            np.testing.assert_allclose(independently, components['score'], atol=1e-10, rtol=0)
            result[scene][name] = dict(label=row['label'], category=row['category'], metrics=x,
                                      components_x100=components, physical=physical,
                                      trace_path=str(path), evaluation_index=index)
        rl, rule = [result[scene][k]['components_x100'] for k in ('selector', 'R_myopic')]
        print(scene, flush=True)
        for name, row in result[scene].items():
            if row['category'] == 'main':
                print(name, json.dumps(row['metrics']), flush=True)
        print('selector components', rl, 'myopic components', rule, flush=True)

    replay = []
    if '--replay' in sys.argv:
        for scene in selected['scenarios']:
            for method in ('selector', 'R_myopic'):
                replay.append(replay_scene(method, scene, seed, data, selected, hashes))
    contributions = {}
    for gid in range(3):
        scope = f'true_instruction_{gid}'
        slots = data['methods']['selector']['by_training_seed'][str(seed)]['per_scope'][scope]['slots']
        weight = slots / data['methods']['selector']['by_training_seed'][str(seed)]['per_scope']['overall']['slots']
        rl = 100*data['methods']['selector']['mean_all_three'][scope]['common_reward']
        rule = 100*data['methods']['R_myopic']['mean_all_three'][scope]['common_reward']
        contributions[str(gid)] = dict(slot_weight=weight, selector_score=rl, myopic_score=rule,
                                       difference=rl-rule, contribution=(rl-rule)*weight)
    expected = 100*(data['methods']['selector']['mean_all_three']['overall']['common_reward']-
                    data['methods']['R_myopic']['mean_all_three']['overall']['common_reward'])
    np.testing.assert_allclose(sum(x['contribution'] for x in contributions.values()), expected, atol=1e-10, rtol=0)
    h.write(OUT / 'results.json', dict(selection=selected, per_scene=result,
        aggregate_instruction_contributions_x100=contributions, exact_replays=replay,
        source_hashes=hashes, audit=dict(state='PASS', independent_formula_recomputed=True,
                                      full_episode_slots=600, no_training=True, no_parameter_changes=True)))
    report(selected, result, contributions, replay)
    h.write(OUT / 'artifact_hashes.json', {p.name: digest(p) for p in
        [OUT / 'selection.json', OUT / 'results.json', OUT / 'REPORT.md', OUT / 'analyze.py']})


def replay_scene(method, scene, seed, data, selected, hashes):
    h.verify()
    cfg = h.config('selector' if method == 'selector' else 'joint_continue', seed)
    entry = data['methods'][method]['by_training_seed'].get(str(seed), data['methods'][method]['by_training_seed'].get('rule'))
    env, obs, _, masks = h.make_env(cfg, data['evaluation_seeds'], data['scenarios'][scene])
    expected = h.read(Path(entry['source']) / 'summary.json')['pairing'][scene]
    assert h.external_hashes(env) == expected
    actors = None
    if method == 'selector':
        model = Path(entry['model'])
        for file, sha in h.read(model / 'status.json')['checkpoint_hashes'].items():
            assert digest(model / file) == sha
            hashes[str(model / file)] = sha
        actors = h.load_actors(cfg, env, model)
    records, modes, fractions, ages = [], [], [], []
    for slot in range(600):
        if actors:
            actions = h.learned_actions(env, actors, obs, masks, 'selector')
        else:
            actions, prediction = myopic_actions(env)
        obs, _, masks, info, values = h.checked_step(env, actions)
        if actors is None:
            np.testing.assert_allclose(h.arr(prediction), values['common_reward'], atol=1e-9, rtol=0)
        values['instruction_id'] = h.arr(info['gid'])
        records.append(np.column_stack([values[f] for f in FIELDS]))
        modes.append(h.arr(info['mode']))
        fractions.append(h.arr(env.beta))
        ages.append(h.arr(env.aoi).astype(np.uint16))
    with np.load(Path(entry['source']) / f'{scene}.npz') as z:
        for key, arr in [('trace', records), ('modes', modes), ('resource_fractions', fractions), ('aoi_after', ages)]:
            np.testing.assert_array_equal(np.stack(arr), z[key])
    h.verify()
    print(f'EXACT REPLAY PASS {method}/{scene} all 20 paired evaluation seeds', flush=True)
    return dict(method=method, scenario=scene, state='EXACT_MATCH', episodes=20,
                requested_evaluation_seed=selected['evaluation_seed'],
                reason_for_batch20='Keep original numerical batch execution identical; only the frozen random episode is illustrated.')


def report(selected, result, contributions, replay):
    lines = ['# 随机实际回合：AoI与质量指令', '',
        f'随机一次抽得训练种子 {selected["training_seed"]}、评估种子 {selected["evaluation_seed"]}，分别查看fixed_1和fixed_2完整600时隙。两条指令使用同一外部测试种子，各方法共享相同初始条件及外生轨迹。选择算法和随机熵在读取回合结果前保存在selection.json，未重抽。', '',
        '这是已有仿真系统的实际运行轨迹，不是随机生成指标，也不是实物网络或真实视频解码实验。PSNR仍为质量表预测的已交付块均值。', '',
        '专项得分仍是该指令权重下的综合奖励，并非单独AoI或PSNR。当前：分数×100 = 质量交付加分 − AoI综合扣分 − 资源扣分。AoI综合扣分同时含平均AoI、每时隙最大AoI及超过4的尾部AoI。', '',
        '## 为什么总体胜出、某些专项落后', '',
        '| 当前指令 | 三种子平均选择器分数 | 一步择优分数 | 指令内分差 | 时隙占比 | 对总体分差的贡献 |',
        '|---|---:|---:|---:|---:|---:|']
    for gid, label in enumerate(['均衡', 'AoI', '质量']):
        x = contributions[str(gid)]
        lines.append(f'| {label} | {x["selector_score"]:.4f} | {x["myopic_score"]:.4f} | {x["difference"]:+.4f} | {100*x["slot_weight"]:.3f}% | {x["contribution"]:+.4f} |')
    lines += ['', f'总体差值为 {sum(x["contribution"] for x in contributions.values()):+.4f}。均衡时隙获得的加分抵消了另外两类的负差。这里按13种场景的实际指令时隙加权，不把三种固定指令分数直接等权平均。', '']
    for scene, label in [('fixed_1', '全程AoI指令'), ('fixed_2', '全程质量指令')]:
        lines += [f'## {label}：一个600时隙回合', '',
                  '| 方法 | 综合分数×100 ↑ | 平均AoI ↓ | 交付预测PSNR/dB ↑ | 交付块数 | 资源使用/时隙 |',
                  '|---|---:|---:|---:|---:|---:|']
        for row in result[scene].values():
            if row['category'] == 'main':
                x = row['metrics']
                lines.append(f'| {row["label"]} | {100*x["common_reward"]:.4f} | {x["mean_aoi"]:.4f} | {x["delivered_predicted_psnr"]:.4f} | {row["physical"]["delivered_blocks"]} | {x["channel_uses_per_slot"]:.2f} |')
        lines += ['', '### 选择器与一步择优的实际计分', '',
                  '| 方法 | 质量交付加分 | AoI扣分 | 其中平均AoI扣分 | 其中最大AoI扣分 | 其中尾部扣分 | 资源扣分 | 最终分数 |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|']
        for method in ('selector', 'R_myopic'):
            row = result[scene][method]
            x = row['components_x100']
            lines.append('| '+row['label']+' | '+' | '.join(f'{x[k]:.6f}' for k in
                ['quality_credit', 'total_age_cost', 'age_mean_cost', 'age_max_cost', 'age_tail_cost', 'resource_cost', 'score'])+' |')
        a, b = [result[scene][k]['components_x100'] for k in ('selector', 'R_myopic')]
        lines += ['', f'选择器减一步择优：质量项 {a["quality_credit"]-b["quality_credit"]:+.6f}；AoI项贡献 {b["total_age_cost"]-a["total_age_cost"]:+.6f}；资源项贡献 {b["resource_cost"]-a["resource_cost"]:+.6f}；合计 {a["score"]-b["score"]:+.6f}。', '']
    lines += ['## 核验范围', '',
              '所有抽取方法的轨迹哈希、字段、评估种子及逐时隙奖励拆项已检查；从PSNR、交付量、AoI和资源使用独立重算回合得分一致。其他历史诊断变体也抽取并保存在results.json，未冒充本轮重新训练的方法。', '']
    for row in replay:
        lines += [f'- {row["method"]}/{row["scenario"]}：重新运行20环境原始批次，与既有轨迹、动作、资源分配、AoI逐元素完全一致。']
    lines += ['', '单次随机案例用于理解实际取舍，不能代替多种子汇总。此处的训练种子来自此前筛选出的三个种子，仍保留该选择条件。', '']
    (OUT / 'REPORT.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
