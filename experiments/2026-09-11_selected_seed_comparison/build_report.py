"""Recompute selected-seed comparisons from frozen, paired evaluation traces."""
import copy
from evaluate_base import OUTPUT, EXPERIMENTS, BASE, LONG, CONTINUATION, h, np, FIELDS, summarize, audit_trace

SEEDS = [810974, 85, 218]
SELECTOR = EXPERIMENTS / '2026-09-10_resource_selector_training'
PILOT = EXPERIMENTS / '2026-09-10_sequential_mdp_training'
DIAG = EXPERIMENTS / '2026-09-10_post_long_diagnosis'
MANIFEST = h.read(CONTINUATION / 'manifest.json')
SCENES = MANIFEST['scenarios']
PAIRING = h.read(CONTINUATION / 'evaluation/rules/R_myopic/summary.json')['pairing']
SOURCES, VERIFIED, ROWS = {}, {}, {}


def remember(path, expected=None):
    sha = h.digest(path)
    if expected is not None:
        assert sha == expected, str(path)
    VERIFIED[str(path)] = sha
    return sha


def env_signature(cfg):
    args = copy.deepcopy(cfg['env_args'])
    for key in ('semantic_profile_path', 'semantic_registry_path'):
        args[key] = remember(h.Path(args[key]))
    return args


def verify_root(root):
    manifest = h.read(root / 'manifest.json')
    assert manifest['scenarios'] == SCENES
    assert manifest['evaluation_seeds'] == MANIFEST['evaluation_seeds']
    for file, sha in manifest['input_hashes'].items():
        remember(root / file, sha)
    for file, sha in h.read(root / 'report/artifact_hashes.json').items():
        remember(root / file, sha)
    assert h.read(root / 'report/audit.json')['state'] == 'PASS'
    SOURCES[str(root)] = h.read(root / 'report/results.json')


def extract(root, item, model=None, config=None, diagnostic=False, new_base=False):
    folder = root / 'evaluation' / item
    if model is not None:
        status = h.read(model / 'status.json')
        assert status['state'] == 'complete'
        remember(model / 'status.json')
        for file, sha in status['checkpoint_hashes'].items():
            remember(model / file, sha)
        assert env_signature(h.read(config)) == REFERENCE_ENV
        remember(config)
    if new_base:
        base = h.read(OUTPUT / 'base_810974_results.json')
        expected = base['per_scope']
        summary = None
    else:
        status = h.read(folder / 'status.json')
        assert status['state'] == 'complete'
        remember(folder / 'summary.json', status['summary_sha256'])
        summary = h.read(folder / 'summary.json')
        expected = summary if diagnostic else SOURCES[str(root)]['per_item'][item]
    per, parts, scores, selections = {}, [], {}, []
    for scene, schedule in SCENES.items():
        file = folder / f'{scene}.npz'
        if new_base:
            meta = h.read(folder / f'{scene}.json')
        elif diagnostic:
            meta = summary[scene]
        else:
            meta = summary['scenarios'][scene]
        remember(file, meta['trace_sha256'])
        assert meta['external_hashes'] == PAIRING[scene]
        with np.load(file) as z:
            data, age = z['trace'], z['aoi_after']
            np.testing.assert_array_equal(z['fields'], FIELDS)
            np.testing.assert_array_equal(z['seeds'], MANIFEST['evaluation_seeds'])
            if 'selector_choices' in z and 'selector_at_' in item:
                selections.append((z['selector_choices'], data[..., 5]))
        assert data.shape == (600, 20, len(FIELDS))
        audit_trace(data, age)
        gids = [next(g for t, g in reversed(schedule) if t <= slot) for slot in range(600)]
        np.testing.assert_array_equal(data[..., 5], np.broadcast_to(np.asarray(gids)[:, None], (600, 20)))
        per[scene] = summarize(data)
        parts.append(data)
        scores[scene] = data[..., 0].mean(0).tolist()
        for j, (start, g) in enumerate(schedule):
            end = schedule[j + 1][0] if j + 1 < len(schedule) else 600
            per[f'{scene}/segment_{j}_g{g}'] = summarize(data[start:end])
            if j:
                per[f'{scene}/first20_after_{start}'] = summarize(data[start:min(start + 20, end)])
    merged = np.concatenate(parts)
    per['overall'] = summarize(merged)
    for g in range(3):
        per[f'true_instruction_{g}'] = summarize(merged[merged[..., 5] == g])
    for scope, metrics in per.items():
        if scope in expected:
            for key, value in metrics.items():
                np.testing.assert_allclose(value, expected[scope][key], rtol=1e-10, atol=1e-9)
    behavior = None
    if selections:
        choices = np.concatenate([x[0] for x in selections])
        gids = np.concatenate([x[1] for x in selections])
        behavior = dict(equal_fraction=float((choices == 1).mean()),
                        equal_fraction_by_instruction={str(g): float((choices[gids == g] == 1).mean()) for g in range(3)})
    return dict(source=str(folder), model=str(model) if model else None,
                per_scope=per, per_eval_seed_scores=scores, selector_behavior=behavior)


def add(method, label, budget, category, seed, entry):
    if method not in ROWS:
        ROWS[method] = dict(label=label, training_steps=budget, category=category, by_training_seed={})
    ROWS[method]['by_training_seed'][str(seed)] = entry


def score_cells(row, scope='overall'):
    by_seed = row['by_training_seed']
    return [f'{100 * by_seed.get(str(seed), by_seed.get("rule"))["per_scope"][scope]["common_reward"]:.4f}'
            if str(seed) in by_seed or 'rule' in by_seed else '未训练/未评估' for seed in SEEDS]


def main():
    global REFERENCE_ENV
    REFERENCE_ENV = env_signature(h.read(BASE / 'configs/seed_810974/joint.json'))
    for root in (LONG, SELECTOR, CONTINUATION, PILOT):
        verify_root(root)
        assert env_signature(h.read(root / 'configs' / ('seed_810974' if root == CONTINUATION else 'seed_85') /
                                         ('joint_continue.json' if root in (SELECTOR, CONTINUATION) else 'joint.json'))) == REFERENCE_ENV
    for root in (LONG, SELECTOR, PILOT):
        for method in ('R_single', 'R_instruction', 'R_myopic'):
            assert SOURCES[str(root)]['per_item'][f'rules/{method}'] == SOURCES[str(CONTINUATION)]['per_item'][f'rules/{method}']
    for seed in SEEDS:
        root = CONTINUATION if seed == 810974 else SELECTOR
        for arm, label in [('selector', '选择器 RL（冻结基础策略）'), ('joint_continue', '联合 RL 继续训练')]:
            item = f'seed_{seed}/{arm}_at_2000000'
            add(arm, label, 12000000, 'main', seed, extract(root, item,
                root / f'jobs/seed_{seed}/{arm}/milestones/steps_2000000', root / f'configs/seed_{seed}/{arm}.json'))
        root = OUTPUT if seed == 810974 else LONG
        model_root = BASE if seed == 810974 else LONG
        add('joint', '原始联合 RL', 10000000, 'main', seed, extract(root, f'seed_{seed}/joint_at_10000000',
            model_root / f'jobs/seed_{seed}/joint/milestones/steps_10000000',
            model_root / f'configs/seed_{seed}/joint.json', new_base=seed == 810974))
        if seed != 810974:
            add('staged', '分阶段 RL（60万步均分，再940万步联合）', 10000000, 'main', seed,
                extract(LONG, f'seed_{seed}/staged_at_10000000', LONG / f'jobs/seed_{seed}/staged/milestones/steps_10000000',
                        LONG / f'configs/seed_{seed}/staged.json'))
    for method, label in [('R_myopic', '一步择优规则'), ('R_instruction', '按指令切换规则'), ('R_single', '单一固定规则'),
                          ('R_equal_myopic', '均分资源＋一步择优模式'), ('R_equal_instruction', '均分资源＋指令选模'),
                          ('R_equal_single', '均分资源＋单一模式')]:
        root = PILOT if method.startswith('R_equal') else CONTINUATION
        add(method, label, 0, 'main', 'rule', extract(root, f'rules/{method}'))
    for seed in (85, 218):
        for arm, label in [('joint', '100万步联合 RL'), ('staged', '100万步分阶段 RL（30万＋70万）'),
                           ('mode_only', '100万步均分资源、仅训练 UAV 模式')]:
            add(f'pilot_{arm}', label, 1000000, 'pilot', seed, extract(PILOT, f'seed_{seed}/{arm}',
                PILOT / f'jobs/seed_{seed}/{arm}', PILOT / f'configs/seed_{seed}/{arm}.json'))
    for root in (DIAG, DIAG / 'targeted_followup'):
        for file, sha in h.read(root / 'artifact_hashes.json').items():
            remember(root / file, sha)
        protocol = h.read(root / 'protocol.json')
        assert protocol['scenarios'] == SCENES and protocol['evaluation_seeds'] == MANIFEST['evaluation_seeds']
        labels = dict(rule_modes_rl_resources='规则选模式＋RL 分配资源',
                      rl_modes_rule_resources='RL 选模式＋指令规则分配资源',
                      quality_rule_modes_only='仅质量指令改成规则选模式',
                      quality_rule_resources_only='仅质量指令改成均分资源',
                      alias_probability_aggregation='合并物理等价模式的概率后选模')
        for arm in protocol['variants']:
            for seed in (85, 218):
                add(arm, labels[arm], 10000000, 'diagnostic', seed, extract(root, f'seed_{seed}/{arm}',
                    LONG / f'jobs/seed_{seed}/joint/milestones/steps_10000000',
                    LONG / f'configs/seed_{seed}/joint.json', diagnostic=True))
    for row in ROWS.values():
        entries = list(row['by_training_seed'].values())
        complete = 'rule' in row['by_training_seed'] or all(str(s) in row['by_training_seed'] for s in SEEDS)
        row['selected_three_complete'] = complete
        row['mean_all_three'] = ({scope: {k: float(np.mean([e['per_scope'][scope][k] for e in entries]))
                                          for k in entries[0]['per_scope'][scope] if k != 'slots'}
                                  for scope in entries[0]['per_scope']} if complete else None)
        row['sample_sd_all_three'] = ({scope: {k: float(np.std([e['per_scope'][scope][k] for e in entries], ddof=1))
                                                for k in entries[0]['per_scope'][scope] if k != 'slots'}
                                        for scope in entries[0]['per_scope']} if complete and len(entries) == 3 else None)
    paired = {}
    selected = ROWS['selector']
    for method, row in ROWS.items():
        if method == 'selector' or not row['selected_three_complete']:
            continue
        paired[method] = {scope: [selected['by_training_seed'][str(s)]['per_scope'][scope]['common_reward'] -
                                  row['by_training_seed'].get(str(s), row['by_training_seed'].get('rule'))['per_scope'][scope]['common_reward']
                                  for s in SEEDS] for scope in selected['mean_all_three']}
    gaps = [
        dict(method='HAPPO_hidden_instruction', status='三个所选种子均缺当前最终协议的重新训练结果；旧模型不直接并表'),
        dict(method='IC_MAPPO / MAPPO_hidden_instruction', status='三个所选种子均缺当前最终协议结果'),
        dict(method='HAPPO_equal_resources', status='当前协议只有85、218等旧种子的100万步mode_only试验，缺三种子正式长训练'),
        dict(method='HAPPO_fixed_mode_rule', status='有85、218的部署替换诊断；缺按当前最终协议重新训练的三种子模型'),
        dict(method='HAPPO_no_task_aux_reward', status='旧版消融，缺当前协议结果；当前共同奖励已关闭额外违约罚与接收奖励，需先重新明确消融含义'),
        dict(method='CC_IC_HAPPO / CC_IC_MAPPO', status='已存在旧版训练；缺当前奖励和决策时序下这三个种子的对齐结果'),
        dict(method='staged', status='有85、218的1000万步结果；810974未训练')]
    episode_count = 260 * sum(len(r['by_training_seed']) for r in ROWS.values())
    audit = dict(state='PASS', selected_training_seeds=SEEDS, evaluated_existing_model_this_turn=810974,
                 new_evaluation_episodes=260, new_training_steps=0,
                 audited_unique_evaluation_items=episode_count // 260, audited_episodes=episode_count,
                 audited_slots=episode_count * 600, same_environment_and_profiles=True,
                 paired_exogenous_sequences=True, independent_reward_recomputed=True,
                 summary_vs_trace_verified=True, verified_file_count=len(VERIFIED),
                 selection_basis='best selector final scores among seven completed training seeds; reused diagnostic evaluation split')
    h.write(OUTPUT / 'comparison.json', dict(selected_training_seeds=SEEDS, score_display_scale=100,
        scenarios=SCENES, evaluation_seeds=MANIFEST['evaluation_seeds'], methods=ROWS,
        selector_minus_method_raw_reward=paired, missing_current_protocol=gaps, audit=audit))
    h.write(OUTPUT / 'source_hashes.json', VERIFIED)
    h.write(OUTPUT / 'audit.json', audit)
    write_markdown(gaps, audit, paired)
    h.write(OUTPUT / 'artifact_hashes.json', {str(p.relative_to(OUTPUT)): h.digest(p) for p in
        [OUTPUT / 'comparison.json', OUTPUT / 'source_hashes.json', OUTPUT / 'audit.json', OUTPUT / 'REPORT.md',
         OUTPUT / 'base_810974_results.json', OUTPUT / 'evaluate_base.py', OUTPUT / 'build_report.py']})
    print('REPORT PASS', audit, flush=True)
    print((OUTPUT / 'REPORT.md').read_text().split('## 各训练种子的完整场景')[0], flush=True)


def write_markdown(gaps, audit, paired):
    lines = ['# 所选最优种子的方法对比：810974、85、218', '',
        '这些是从已完成的7个训练种子中，按最终选择器分数选出的3个。此表用于比较所选模型；评估集已反复用于诊断和选种子，不能作为未筛选随机三种子的泛化证据。', '',
        '评估统一为13种指令场景（含全程固定、途中切换和多次切换）×20个环境种子（20262501–20262520）×600时隙。每种模型260回合。奖励原值统一乘100显示，越大越好；乘100不改变排名。', '',
        '总体奖励对13种场景等权平均，再对三个训练种子等权平均。规则不需要训练，三个种子列复用同一组配对规则评估。PSNR是平均质量表给出的已交付块预测值；每个模型先按交付数加权，再对模型取均值。', '',
        '本次仅补评估已有的810974原始1000万步模型，没有重新训练、改变环境或改奖励。', '',
        '## 当前同口径主对比', '',
        '| 方法 | 每模型训练步数 | 810974 | 85 | 218 | 所选三种子均值 |',
        '|---|---:|---:|---:|---:|---:|']
    main = {k: r for k, r in ROWS.items() if r['category'] == 'main'}
    for method, row in main.items():
        mean = f'{100 * row["mean_all_three"]["overall"]["common_reward"]:.4f}' if row['selected_three_complete'] else '缺810974，不计算'
        lines.append('| ' + ' | '.join([row['label'], f'{row["training_steps"]:,}', *score_cells(row), mean]) + ' |')
    delta = paired['R_myopic']['overall']
    lines += ['', f'选择器相对一步择优规则的分差依次为 {", ".join(f"{100*x:+.4f}" for x in delta)}，三种子平均 {100*np.mean(delta):+.4f}。这3个所选模型的综合分数均更高；不表示每条指令、每个场景都胜出。', '',
              '选择器先训练基础联合策略1000万步，再冻结4个基础actor、训练二选一资源选择器200万步。joint_continue使用相同1000万步父模型，继续联合更新200万步。1000万步的joint和staged作为预算不同的参照。', '',
              '## 所选三种子的物理指标均值', '',
              '| 方法 | 分数×100 ↑ | 平均AoI/时隙 ↓ | 交付预测PSNR/dB ↑ | 交付数/时隙 | 信道使用次数/时隙 |',
              '|---|---:|---:|---:|---:|---:|']
    for row in main.values():
        if row['mean_all_three']:
            x = row['mean_all_three']['overall']
            lines.append(f'| {row["label"]} | {100*x["common_reward"]:.4f} | {x["mean_aoi"]:.4f} | {x["delivered_predicted_psnr"]:.4f} | {x["deliveries_per_slot"]:.4f} | {x["channel_uses_per_slot"]:.2f} |')
    for title, scopes in [('全程固定指令下的平均分数', ['fixed_0', 'fixed_1', 'fixed_2']),
                           ('含切换场景、按当前真实指令汇总的平均分数', ['true_instruction_0', 'true_instruction_1', 'true_instruction_2'])]:
        lines += ['', '## ' + title, '', '| 方法 | 均衡 | AoI | 质量 |', '|---|---:|---:|---:|']
        for row in main.values():
            if row['mean_all_three']:
                lines.append('| ' + row['label'] + ' | ' + ' | '.join(f'{100*row["mean_all_three"][s]["common_reward"]:.4f}' for s in scopes) + ' |')
    lines += ['', '## 选择器的实际行为', '', '| 种子 | 所有时隙选择均分资源比例 | 均衡指令 | AoI指令 | 质量指令 |', '|---|---:|---:|---:|---:|']
    for seed in SEEDS:
        x = ROWS['selector']['by_training_seed'][str(seed)]['selector_behavior']
        lines.append(f'| {seed} | {100*x["equal_fraction"]:.3f}% | ' + ' | '.join(f'{100*x["equal_fraction_by_instruction"][str(g)]:.3f}%' for g in range(3)) + ' |')
    lines += ['', '810974在这些评估中始终选择均分资源，优势对应“均分资源＋冻结的RL模式策略”；该种子的结果不能证明资源选择器学会了随指令切换资源方案。UAV模式策略仍然可见指令。', '',
              '## 同口径但缺810974的先前试验', '',
              '以下单独列出已有结果，不补造810974数值，不用两个种子的均值冒充三个种子均值。100万步试验预算更小；部署诊断只替换控制组件，没有重新训练。', '',
              '| 方法 | 性质 | 810974 | 85 | 218 |', '|---|---|---:|---:|---:|']
    for row in ROWS.values():
        if row['category'] != 'main':
            lines.append('| ' + ' | '.join([row['label'], '100万步短训练' if row['category'] == 'pilot' else '1000万步冻结模型的部署诊断', *score_cells(row)]) + ' |')
    lines += ['', '## 尚缺的正式方法', '', '| 方法 | 状态 |', '|---|---|']
    for gap in gaps:
        lines.append(f'| {gap["method"]} | {gap["status"]} |')
    lines += ['', '因此当前不能声称“原计划所有SC、CC、MAPPO和隐藏指令方法均已完成这三个种子的正式对比”。旧版结果保留在各自目录，不跨版本拼接排名。', '',
              '## 各训练种子的完整场景与物理指标', '']
    for seed in SEEDS:
        lines += [f'### 训练种子 {seed}', '']
        for scope in ['overall', *SCENES, 'true_instruction_0', 'true_instruction_1', 'true_instruction_2']:
            lines += [f'#### {scope}', '', '| 方法 | 分数×100 | AoI | 交付预测PSNR | 交付数/时隙 | 信道使用次数/时隙 |', '|---|---:|---:|---:|---:|---:|']
            for row in ROWS.values():
                entry = row['by_training_seed'].get(str(seed), row['by_training_seed'].get('rule'))
                if entry:
                    x = entry['per_scope'][scope]
                    lines.append(f'| {row["label"]} | {100*x["common_reward"]:.4f} | {x["mean_aoi"]:.4f} | {x["delivered_predicted_psnr"]:.4f} | {x["deliveries_per_slot"]:.4f} | {x["channel_uses_per_slot"]:.2f} |')
            lines.append('')
    lines += ['## 来源与核验', '', f'重新核验{audit["audited_unique_evaluation_items"]}组评估、{audit["audited_episodes"]}回合、{audit["audited_slots"]}时隙：轨迹哈希、模型哈希、环境与表格内容、外生序列配对、逐时隙奖励分量及汇总一致性均通过。', '',
              '完整精度、每个评估种子的分数、切换后20时隙指标、训练种子标准差及模型绝对路径见comparison.json。输入和模型哈希见source_hashes.json；本报告和生成脚本哈希见artifact_hashes.json。', '']
    for method, row in ROWS.items():
        for seed, entry in row['by_training_seed'].items():
            lines += [f'- {method} / {seed}: {entry["source"]}' + (f'；模型 {entry["model"]}' if entry['model'] else '')]
    (OUTPUT / 'REPORT.md').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
