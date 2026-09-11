"""Export the requested simple-baseline view from existing paired evaluations.

This performs no training, fitting, candidate reward search, or simulation.
Historical full comparison files remain authoritative for their original scope.
"""
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
SOURCE = ROOT / 'experiments/2026-09-11_selected_seed_comparison/comparison.json'
RULE_SOURCE = ROOT / 'experiments/2026-09-10_sequential_mdp_training/rule_selection.json'

DEFINITIONS = {
    'R_equal_single': ('均分资源＋固定模式', '始终均分资源，优先模式0'),
    'R_single': ('紧急度分配＋固定模式', '按各UAV缓存AoI总和分资源，优先模式0'),
    'R_equal_instruction': ('均分资源＋指令选模', '始终均分；均衡/AoI/质量分别优先模式5/0/5'),
    'R_instruction': ('按指令切换简单规则', '均衡/质量：均分＋模式5；AoI：按缓存AoI总和分配＋模式0'),
    'joint': ('原始联合RL', '1000万步，资源和模式均由RL输出'),
    'selector': ('当前选择器RL', '基础1000万步＋冻结基础策略后训练二选一资源选择器200万步'),
}


def main():
    data = json.loads(SOURCE.read_text())
    selection = json.loads(RULE_SOURCE.read_text())
    assert selection['single'] == 'm0_urgency'
    assert selection['by_instruction'] == ['m5_equal', 'm0_urgency', 'm5_equal']
    assert selection['equal']['single'] == 'm0_equal'
    assert selection['equal']['by_instruction'] == ['m5_equal', 'm0_equal', 'm5_equal']
    scale = data['score_display_scale']
    assert scale == 100
    rows = {}
    for method, (label, description) in DEFINITIONS.items():
        source = data['methods'][method]
        metrics = source['mean_all_three']
        # Verify the exported mean against all actual model entries and scenes.
        actual = list(source['by_training_seed'].values())
        expected = sum(item['per_scope'][scene]['common_reward']
                       for item in actual for scene in data['scenarios'])
        expected /= len(actual) * len(data['scenarios'])
        assert math.isclose(expected, metrics['overall']['common_reward'], abs_tol=1e-12)
        overall = metrics['overall']
        decomposed = (overall['quality_credit'] - overall['age_mean_cost']
                      - overall['age_max_cost'] - overall['age_tail_cost']
                      - overall['resource_cost'] - overall['service_violation_cost']
                      + overall['recv_aoi_bonus'])
        assert math.isclose(decomposed, expected, abs_tol=1e-12)
        rows[method] = dict(label=label, rule=description,
            score=scale * expected,
            score_by_fixed_instruction={str(g): scale * metrics[f'fixed_{g}']['common_reward']
                                        for g in range(3)},
            overall_metrics=overall,
            sources={key: item['source'] for key, item in source['by_training_seed'].items()})
    output = dict(status='EXISTING_RESULTS_REEXPORTED_AND_CHECKED', training_started=False,
        main_rule_methods=[m for m in DEFINITIONS if m.startswith('R_')],
        historical_methods_not_in_this_view=['R_myopic', 'R_equal_myopic',
                                             'observation_matched_fitted_greedy',
                                             'observation_matched_local_greedy'],
        source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in [SOURCE, RULE_SOURCE]},
        selected_training_seeds=data['selected_training_seeds'],
        evaluation_seeds=data['evaluation_seeds'], scenarios=data['scenarios'], methods=rows)
    (HERE / 'comparison.json').write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n')
    lines = ['# 简单基线对比与动作空间', '',
        '本表重新整理已有评估，没有重新训练或运行仿真。当前主规则对照采用以下四个简单方法；不包含单步奖励搜索或得分预测器。', '',
        '规则中的模式和指令映射来自原独立校准集，此次未重新挑选。模式只是优先选择，所有方法沿用同一质量/预算可行性回退和AoI优先缓存调度。', '',
        '| 方法 | 实际规则 | 综合分数↑ | 均衡 | AoI | 质量 |',
        '|---|---|---:|---:|---:|---:|']
    for row in rows.values():
        scores = row['score_by_fixed_instruction']
        lines.append(f"| {row['label']} | {row['rule']} | {row['score']:.4f} | "
                     + ' | '.join(f'{scores[str(g)]:.4f}' for g in range(3)) + ' |')
    lines += ['', '综合分数为13场景等权平均奖励×100；专项列为全程固定该指令。20个环境种子，每回合600时隙。RL均值对应此前从7个训练种子中筛选的810974、85、218，不是新的随机三种子验证。简单规则每个只有一套评估，不能当作三次独立训练。', '',
        '当前RL综合得分超过这些简单规则，但AoI及质量专项并非全面领先。此表仅支持相对列出方法的结论。历史完整实验保留在原目录。', '',
        '## 当前动作空间（已核对实现）', '',
        '- SUT输出3个连续资源份额，总和1；每个UAV至少0.05，最大可到0.90（边界是可行集合闭包）。实现使用simplex/Dirichlet动作头。',
        '- 每个UAV在获得本时隙实际预算之后，从16个模式中离散选择1个；使用masked categorical，不是16个独立连续动作。16维数组是接口编码。',
        '- 3个UAV有16³=4096种名义模式组合，实际还受各自质量和预算掩码限制；再配连续资源份额。每个UAV每时隙选一个模式，适用于其本时隙交付的缓存块。',
        '- RL不直接选择逐DS服务顺序、飞行位置或视频；缓存服务顺序由环境AoI调度器决定。模式是固定平均质量载荷表的条目。',
        '- 选择器续训时只有“原SUT分配/均分资源”两个可学习选项，四个基础actor冻结；改变资源仍可间接改变UAV输入和最终模式。', '',
        '下一轮训练建议见同目录training_plan.md。该建议未执行，未改动当前训练程序、权重、环境或奖励。', '']
    (HERE / 'comparison.md').write_text('\n'.join(lines))
    print(json.dumps({'status': output['status'], 'rows_verified': len(rows),
                      'training_started': False}, ensure_ascii=False))


if __name__ == '__main__':
    main()
