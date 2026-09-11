"""Checkpoint evidence for extra training, paired with the runtime information audit."""
import json
import hashlib
import statistics
from pathlib import Path

OUT = Path(__file__).resolve().parent
EXPERIMENTS = OUT.parent
SELECTED = [810974, 85, 218]


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    hashes, rows = {}, []
    roots = [EXPERIMENTS / name for name in [
        '2026-09-10_resource_selector_training',
        '2026-09-11_additional_seed_training/continuation',
        '2026-09-11_three_new_seeds_training/continuation']]
    for root in roots:
        for file, digest in read(root / 'manifest.json')['input_hashes'].items():
            assert sha(root / file) == digest
        for file, digest in read(root / 'report/artifact_hashes.json').items():
            assert sha(root / file) == digest
            hashes[str(root / file)] = digest
        data = read(root / 'report/results.json')['per_item']
        for seed in read(root / 'manifest.json')['seeds']:
            row = dict(seed=seed, selected=seed in SELECTED, by_method={})
            for arm in ('selector', 'joint_continue'):
                path = root / f'jobs/seed_{seed}/{arm}/training_metrics.jsonl'
                hashes[str(path)] = sha(path)
                logs = [json.loads(line) for line in path.read_text().splitlines()]
                assert len(logs) == 500 and logs[-1]['steps'] == 2000000
                expected_actors = ['selector'] if arm == 'selector' else [0, 1, 2, 3]
                assert all(log['trained_actor_ids'] == expected_actors for log in logs)
                checkpoints = {}
                for step in (1000000, 2000000):
                    result = data[f'seed_{seed}/{arm}_at_{step}']
                    checkpoints[str(step)] = {scope: 100*result[scope]['common_reward']
                                              for scope in ('overall', 'fixed_0', 'fixed_1', 'fixed_2')}
                    model = root / f'jobs/seed_{seed}/{arm}/milestones/steps_{step}'
                    for file, digest in read(model / 'status.json')['checkpoint_hashes'].items():
                        assert sha(model / file) == digest
                        hashes[str(model / file)] = digest
                first_model = root / f'jobs/seed_{seed}/{arm}/milestones/steps_1000000/status.json'
                last_model = root / f'jobs/seed_{seed}/{arm}/milestones/steps_2000000/status.json'
                if arm == 'selector':
                    assert read(first_model)['actor_hashes'] == read(last_model)['actor_hashes']
                tail = logs[-50:]
                row['by_method'][arm] = dict(checkpoints=checkpoints,
                    score_change_1m_to_2m=checkpoints['2000000']['overall']-checkpoints['1000000']['overall'],
                    trained_actor_ids=expected_actors, final_actor_lr=logs[-1]['actor_learning_rates'],
                    final_entropy_coefficient=logs[-1]['entropy_coef'],
                    last50_mean_policy_entropy=statistics.mean(log['actor_loss_entropy_grad_ratio'][1] for log in tail),
                    last50_mean_actor_gradient_norm=statistics.mean(log['actor_loss_entropy_grad_ratio'][2] for log in tail),
                    source=str(root))
            rows.append(row)
    selected_rows = sorted((r for r in rows if r['selected']), key=lambda x: SELECTED.index(x['seed']))
    means = {population: {arm: {str(step): statistics.mean(r['by_method'][arm]['checkpoints'][str(step)]['overall'] for r in population_rows)
                               for step in (1000000, 2000000)} for arm in ('selector', 'joint_continue')}
             for population, population_rows in [('selected_three', selected_rows), ('all_seven', rows)]}
    comparison = read(EXPERIMENTS / '2026-09-11_selected_seed_comparison/comparison.json')
    diag = EXPERIMENTS / '2026-09-10_post_long_diagnosis/targeted_followup'
    headroom_folder = diag / 'evaluation/seed_218/quality_rule_resources_only'
    diagnostic_status = read(headroom_folder / 'status.json')
    assert diagnostic_status['state'] == 'complete'
    assert sha(headroom_folder / 'summary.json') == diagnostic_status['summary_sha256']
    diagnostic_result = read(headroom_folder / 'summary.json')
    for scene, metrics in diagnostic_result.items():
        if scene != 'overall':
            path = headroom_folder / f'{scene}.npz'
            assert sha(path) == metrics['trace_sha256']
            hashes[str(path)] = metrics['trace_sha256']
    hashes[str(headroom_folder / 'summary.json')] = diagnostic_status['summary_sha256']
    headroom = dict(training_seed=218,
        selector_score=100*comparison['methods']['selector']['by_training_seed']['218']['per_scope']['overall']['common_reward'],
        feasible_quality_only_equal_score=100*diagnostic_result['overall']['common_reward'],
        rule='Use equal resources only for current quality instruction; otherwise original frozen SUT output. Same frozen UAV actors.',
        source=str(headroom_folder),
        interpretation='A known feasible policy in the same binary resource-choice family is better on this diagnostic evaluation; not a learned improvement or an independent-test guarantee.')
    joint_10_to_12 = {str(s): {method: 100*comparison['methods'][method]['by_training_seed'][str(s)]['per_scope']['overall']['common_reward']
                              for method in ('joint', 'joint_continue')} for s in SELECTED}
    long_root = EXPERIMENTS / '2026-09-10_sequential_long_training'
    for file, digest in read(long_root / 'report/artifact_hashes.json').items():
        assert sha(long_root / file) == digest
        hashes[str(long_root / file)] = digest
    long = read(long_root / 'report/results.json')
    early_curve = {step: 100*long['grouped_by_steps'][step]['joint']['overall']['common_reward']['mean']
                   for step in ('1000000', '3000000', '5000000', '10000000')}
    runtime = read(OUT / 'runtime_audit.json')
    assert runtime['state'] == 'PASS'
    for file, digest in runtime['source_hashes'].items():
        assert sha(Path(file)) == digest
    report = dict(state='PASS', selected_seeds=SELECTED, rows=rows, means=means,
                  joint_10m_to_12m=joint_10_to_12, feasible_headroom_example=headroom, original_three_joint_curve_x100=early_curve,
                  original_curve_seeds=[85, 218, 966], source_hashes=hashes,
                  limitations=['Two selector evaluation checkpoints do not prove convergence.',
                               'Training remains stochastic while reported deployment uses deterministic actions.',
                               'All evaluations use the existing repeatedly consulted diagnostic split.',
                               'No matched-information ablation has yet quantified causal contributions to the rule gap.'])
    (OUT / 'training_review.json').write_text(json.dumps(report, indent=2)+'\n')
    lines = ['# 训练提升空间与一步择优规则的信息审计', '',
        '结论：原始RL在早期加长训练时确实改善，但目前所选三个选择器模型从追加100万到200万步没有持续提高的迹象；不能由此证明已收敛，也不能承诺继续训练会超过规则。一步择优未读取未来信息，但使用全局当前状态和已知精确单步模型，与分散执行actor的能力条件不同。', '',
        '## 训练多久的现有证据', '',
        '| 训练种子 | 选择器追加100万步 | 选择器追加200万步 | 再训练100万步的分差 |',
        '|---|---:|---:|---:|']
    for row in selected_rows:
        x = row['by_method']['selector']
        lines.append(f'| {row["seed"]} | {x["checkpoints"]["1000000"]["overall"]:.6f} | {x["checkpoints"]["2000000"]["overall"]:.6f} | {x["score_change_1m_to_2m"]:+.6f} |')
    lines += ['', f'所选三种子选择器均值：{means["selected_three"]["selector"]["1000000"]:.6f} → {means["selected_three"]["selector"]["2000000"]:.6f}。', '',
        '| 训练种子 | 原始联合1000万步 | 联合续训至1200万步 |', '|---|---:|---:|']
    for s in SELECTED:
        x = joint_10_to_12[str(s)]
        lines.append(f'| {s} | {x["joint"]:.6f} | {x["joint_continue"]:.6f} |')
    lines += ['', '早期原始三种子（85、218、966）的联合RL均值，100/300/500/1000万步依次为：'+
              ' → '.join(f'{v:.4f}' for v in early_curve.values())+'。说明早期增加训练曾有效；当前续训结果并非普遍的“训练越长越差”。', '',
              f'全部七种子选择器100万→200万步均值：{means["all_seven"]["selector"]["1000000"]:.6f} → {means["all_seven"]["selector"]["2000000"]:.6f}。较差种子771611、779709仍在改善，不能把三个好种子的部署平台误说成全部种子训练已收敛。', '',
              '## 是否存在实际可改进空间', '',
              f'存在一个已核验的具体例子：种子218的当前选择器为{headroom["selector_score"]:.6f}；使用同一组冻结基础actor，仅在质量指令采用均分、其余指令沿用旧SUT输出，已有完整回合得分为{headroom["feasible_quality_only_equal_score"]:.6f}，高出{headroom["feasible_quality_only_equal_score"]-headroom["selector_score"]:+.6f}。', '',
              '这个诊断策略只需要当前指令和原先已有的两个资源候选，没有额外未来信息，证明在该种子和开发评估集上，即使不扩大二选一动作集合，也存在更好的可执行选择方式。但它是人工设定路由的诊断，不能冒充RL学习结果，也不能证明延长训练必然学到或在独立测试上仍改善。', '',
              '## 当前选择器继续训练能改变什么', '',
              '- 基础SUT和3个UAV actor已冻结，当前只有二选一选择器和critic更新。选项为旧SUT资源分配或均分资源。更长训练不能直接学习新的UAV模式映射，但可通过资源选择改变已有模式策略的输入和状态轨迹。',
              '- 两个检查点的最终确定性行为几乎不变，不等于参数不更新。所选三种子最后50次更新的策略熵约0.50–0.62，二分类最大熵约0.693，梯度仍非零。尤其810974始终选择均分，仅说明概率较大的选项没有变，不代表该选项概率已经为1。',
              '- 学习率由1e-4降到1e-5。已结束的任务直接resume不会增加训练预算；进一步训练需要独立版本的预算与学习率计划。',
              '- 当前证据不能确定最优策略上限，也不能把“还有梯度”当成“长期分数会继续上涨”。继续训练应作为受控实验，报告固定验证集全部检查点；最终结论需未用于反复选择的新测试集。', '',
              '## 一步择优实际上做什么', '',
              '每时隙枚举3个模式偏好（0、5、10）×3种资源分配（均分、按缓存数量、按缓存AoI总和）=9个候选。对各候选应用同样的可行性回退和缓存服务顺序，用当前缓存、生成时间、RCC AoI、当前质量载荷表预测本时隙结束后的奖励，再取最大者。模式偏好相同不意味着所有UAV实际执行模式相同，可行性回退仍生效。', '',
              'next_aoi是由当前状态与候选动作算出来的结果，不是读取未来轨迹。当前SC采用固定平均质量表，质量交付和缓存/AoI转移在给定当前状态和动作后是确定的；未来信道只在推进下一时隙时更新。因此它能够与仿真器当步奖励精确一致。', '',
              '## 动态信息访问核验', '',
              f'覆盖4个指令场景、每场景9个时点、20个环境种子，共{runtime["sampled_states"]}个状态、{runtime["tested_candidate_state_pairs"]}组候选评分。', '',
              '- 只允许读取当前状态和公开参数的代理接口：九候选评分及最终动作与原实现完全一致。指令数组仅允许访问当前下标，禁止读取下一条指令。',
              '- 扰乱所有未来信道、未来内容和未来指令后：所有九候选的当前评分及最终动作均不变。',
              '- 候选评分前后所有环境张量哈希和时钟一致：评分未偷偷推进环境，也未修改缓存。',
              '- 源码追溯确认当前质量/载荷只依赖当前CSI与固定profile。当前CSI在策略决策前可用是RL和规则共同采用的建模假设。', '',
              '## 与RL的能力条件哪里不同', '',
              '| 能力或信息 | 部署RL | 当前一步择优 |', '|---|---|---|',
              '| 当前指令、当前CSI、固定profile | 可用；每个UAV有自己的模式质量和载荷 | 可用；直接读取全局数组 |',
              '| 每个DS的缓存、生成时间、RCC AoI | UAV看自己所属DS；SUT主要看每UAV汇总 | 同一决策器同时查看全部DS细节 |',
              '| 每个候选的精确当前奖励 | actor执行时没有模拟九候选 | 使用完整的单步状态转移和奖励公式计算 |',
              '| 全局critic | 训练使用，部署丢弃 | 不需critic，直接使用当前全局状态和模型 |',
              '| 未来信道、未来指令、未来内容 | 不可见 | 审计未读取 |', '',
              '注意：质量载荷数组本身来自当前CSI与公开固定表，因此SUT没有直接输入这些数组，并不等于信息理论上无法重算它们。真正被当前SUT汇总压缩掉的内容包括每DS缓存时间与AoI的对应关系；模型知识也是与纯actor执行不同的能力。', '',
              '构造性信息差证明：从第50时隙的一个实际状态出发，在保持缓存占用、所有AoI、每UAV最大缓存年龄和整个SUT观测完全不变的前提下，重新排列缓存块生成时间。一种状态的一步择优选择m5_urgency，另一种选择m5_equal。时间戳保持当前有效范围且缓存块比上次交付更新。该例证明SUT观测压缩确实丢失了会影响择优动作的细节；替代状态是人工构造，不声称两者都出现在已运行轨迹中，且UAV局部观测会变化。', '',
              '## 是否符合实际，论文怎么定位', '',
              '它没有未来信息泄漏，也没有利用测试结果临时修改权重；但现实现直接读取仿真器的全局内部当前状态，并假设单步模型完全正确。若中心能够获得同步状态反馈，或可靠跟踪这些状态，并掌握同一模型，则可以实现类似的在线控制；现代码尚未实现或计入状态上报、反馈时延、估计误差和在线模型计算成本。', '',
              'Manuscript/main.tex第175行允许SUT接收低维状态报告，第1364–1368行明确只交换汇总信息、不交换每DS细节。因而当前直接全局读状态的规则，不能默认为遵守了与该部署方案完全相同的信息接口。原稿其他公式也可能仍是旧版，本次仅引用其部署信息约束，没有将旧数值当成当前实验。', '',
              '该基线可标为“集中式、已知模型的一步贪心参照”。它只优化九个候选的一步奖励，不是完整动作空间或长期回报的数学上界；既有RL总体胜出也说明不能把它称为性能上限。', '',
              '可部署替代实现不一定必须集中传送全部DS状态：也可先广播候选分配，由每个UAV针对候选回报预测质量增益、资源、AoI和/最大/尾部等摘要，再在SUT汇总评分。这需要额外候选评估和控制交互，不是现有的一次固定汇总消息，尚未实现和核算。', '',
              '## 建议的下一步', '',
              '1. 保留现有强基线。新增严格通过RL现有可见信息接口决策的规则，或新增取得同等全局当前信息的集中式RL对照；据此量化信息差和模型知识的影响，而不是直接削弱原基线。',
              '2. 当前三个选择器暂不直接翻倍长训。若检验纯续训收益，可冻结环境/奖励、预设追加100–200万步和固定检查点，用独立验证数据判断；这只能验证是否继续改善，不能预先保证提升。',
              '3. 若目标是提高AoI/质量专项，更应先诊断基础UAV模式策略和SUT的信息输入。只训练二选一资源选择器不能直接修正基础模式策略；增加可观测摘要或小学习率模式微调应各自独立对照，并保持旧模型和强基线。',
              '4. 不把当前落后全部归因于信息差：精确模型、有限候选枚举、actor的信息压缩和优化难度均可能影响结果，尚无消融定量区分各自贡献。', '',
              '本次新增的是只读分析和隔离环境诊断脚本，未改变原始训练代码、奖励、环境、模型或论文。源文件、模型、日志及报告哈希见JSON附件。', '']
    (OUT / 'REPORT.md').write_text('\n'.join(lines))
    (OUT / 'artifact_hashes.json').write_text(json.dumps({p.name: sha(p) for p in
        [OUT / 'REPORT.md', OUT / 'training_review.json', OUT / 'runtime_audit.json',
         OUT / 'training_review.py', OUT / 'audit_runtime.py']}, indent=2)+'\n')
    print('TRAINING REVIEW PASS', json.dumps(means), flush=True)


if __name__ == '__main__':
    main()
