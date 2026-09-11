"""Summarize all completed diagnostics, including the adverse LR result."""
from diagnose import *

def mean(rows,key):return float(np.mean([r[key] for r in rows]))

def main():
    verify()
    frozen=read(OUT/'results.json');r=frozen['results'];sig=read(OUT/'training_signal.json')
    continuation={str(s):read(OUT/f'continuation/seed_{s}/results.json') for s in SEEDS}
    components=read(OUT/'continuation_components.json')['results']
    coordination=read(OUT/'coordination.json')['results']
    noise=read(OUT/'execution_noise.json')
    probes=np.load(OUT/'probes.npz');idx=read(OUT/'probe_index.json')
    comparison={}
    for sc in SCENARIOS:
        comparison[sc]={}
        for v in VARIANTS:
            rows=[r[f'{s}/{sc}/{v}']['whole'] for s in SEEDS]
            comparison[sc][v]={k:mean(rows,k) for k in ['common_reward','psnr','mean_aoi','low_fraction','middle_fraction','max_aoi','quality_credit','age_max_cost']}
        for v in ['final_lr','restart_lr']:
            rows=[continuation[str(s)][v][sc] for s in SEEDS]
            comparison[sc][v]={k:mean(rows,k) for k in ['common_reward','psnr','mean_aoi','low_fraction','max_aoi','quality_credit','age_max_cost']}
        for v in ['old_resources_new_modes','new_resources_old_modes']:
            rows=[components[f'{s}/{sc}/{v}'] for s in SEEDS]
            comparison[sc][v]={k:mean(rows,k) for k in ['common_reward','psnr','mean_aoi','low_fraction','max_aoi','quality_credit','age_max_cost']}
    mechanism={}
    for seed in SEEDS:
        keep=np.array([i['seed']==seed and i['gid']==2 for i in idx])
        pr=probes['probabilities'][keep];cf=probes['counterfactual_probabilities'][keep]
        delta=probes['one_step_reward_by_agent_group'][keep,...,4]-probes['one_step_reward_by_agent_group'][keep,...,0]
        mid=pr[...,[5,9,13]].sum(-1)
        mechanism[str(seed)]=dict(quality_one_uav_counterfactual_count=int(delta.size),middle_vs_low_immediate_gain=float(delta.mean()),
            positive_fraction=float((delta>0).mean()),low_probability=float(pr[...,[0,4,8,12]].sum(-1).mean()),middle_probability=float(mid.mean()),
            conditional_joint_all_middle_sampling_probability=float(mid.prod(-1).mean()),
            matched_state_command_middle_probability_difference=float((cf[:,2][...,[5,9,13]].sum(-1)-cf[:,1][...,[5,9,13]].sum(-1)).mean()))
    patterns=['000','005','050','500','055','505','550','555']
    cm={p:{k:mean([coordination[f'{s}/{p}'] for s in SEEDS],k) for k in
        ['common_reward','quality_credit','age_mean_cost','age_max_cost','age_tail_cost','resource_cost','mean_aoi','max_aoi','psnr']} for p in patterns}
    # Joint gain minus sum of one-UAV gains. Components close algebraically.
    synergy={k:cm['555'][k]-sum(cm[p][k] for p in ['005','050','500'])+2*cm['000'][k]
        for k in ['common_reward','quality_credit','age_mean_cost','age_max_cost','age_tail_cost','resource_cost']}
    np.testing.assert_allclose(synergy['common_reward'],synergy['quality_credit']-sum(synergy[k] for k in ['age_mean_cost','age_max_cost','age_tail_cost','resource_cost']),atol=1e-12)
    audit={};saved_trace_slots=0
    for p in OUT.rglob('*.npz'):
        with np.load(p) as z:
            if 'trace' not in z:continue
            fields=list(z['fields']);a=z['trace'];v={k:a[...,i] for i,k in enumerate(fields)}
            rebuilt=v['quality_credit']-v['age_mean_cost']-v['age_max_cost']-v['age_tail_cost']-v['resource_cost']-v['service_violation_cost']
            np.testing.assert_allclose(rebuilt,v['common_reward'],atol=1e-9,rtol=0)
            saved_trace_slots+=a.shape[0]*a.shape[1]
    for f,h in frozen['protocol']['model_hashes'].items():assert digest(REF/f)==h
    for s in SEEDS:
        from training_checkpoint import load_checkpoint
        state,entry=load_checkpoint(REF/f'jobs/ref8/seed_{s}/IC_HAPPO/checkpoints')
        assert entry==read(OUT/'continuation_protocol.json')['checkpoints'][str(s)]
        tiny=read(OUT/f'continuation/seed_{s}/final_lr/identity.json');restart=read(OUT/f'continuation/seed_{s}/restart_lr/identity.json')
        assert tiny['initial_actor_hashes']==restart['initial_actor_hashes']
        for arm in ['final_lr','restart_lr']:
            status=read(OUT/f'continuation/seed_{s}/{arm}/status.json');assert status['state']=='complete' and status['completed_steps']==200000
            for f,h in status['output_hashes'].items():assert digest(OUT/f'continuation/seed_{s}/{arm}'/f)==h
    audit.update(state='PASS',original_inputs_and_models_unchanged=True,initial_continuation_models_paired=True,
        all_saved_trace_rewards_recomputed=True,saved_trace_slots=saved_trace_slots,
        original_replay_match=frozen['original_replay_match'],diagnostic_training_steps=1200000,
        evaluation_slots_checked_during_step=1584000,frozen_stochastic_collection_steps=60000,
        total_new_environment_steps=2844000,formal_results_replaced=False)
    write(OUT/'summary.json',dict(state='PASS',comparison=comparison,mechanism=mechanism,coordination=cm,synergy=synergy,audit=audit))
    lines=['# 模式学习根因诊断','',
      '结论：当前最明确的失败环节是模式策略没有稳定学会按指令协同选档。奖励和模式动作的已检查实现未发现新的计算错误。最大 AoI 项造成了可测量的协同收益；但尚不能把训练失败唯一归因于某一个超参数或一个代码缺陷。',
      '', '本次使用参考值8、三个原100万步模型、20个配对评估种子；只检查固定质量、固定AoI、质量→AoI三种场景。所有表内得分乘100，越大越好。不是原13场景正式总表。',
      '', '## 1. 排除了什么','',
      '- 指令确实进入 actor。保持物理状态不变，只换指令输入，各模型的模式概率都会改变，但区分较弱。',
      '- 模式头是16类离散分布；不是连续动作取整。训练保存的16维分摊 log probability 求和与真实分类 log probability 一致。',
      '- 同一奖励下存在更好的模式策略。原模型质量状态的单架无人机低档→中档干预，绝大多数单步得分提高；这不是长期最优的证明。',
      '- 随机抽样评估、合并重复档位概率，都未提高质量任务得分。',
      '- 原固定模型在先前那组质量→AoI回放中，最高偏好与执行模式完全一致。该结论不能外推为随机训练时从不发生预算筛选。',
      '', '| 原冻结模型评估方式 | 质量指令分数 | AoI指令分数 | 质量→AoI分数 |', '|---|---:|---:|---:|']
    for name,label in [('deterministic','原RL'),('all_sample','按训练方式全部抽样'),('group_argmax','合并重复模式概率'),('rule_mode_det_resources','只替换模式，保留RL资源'),('rule_both','完整指令规则')]:
        lines.append('| '+label+' | '+' | '.join(f'{100*comparison[sc][name]["common_reward"]:.4f}' for sc in SCENARIOS)+' |')
    lines+=['','## 2. 直接检查网络偏好','', '| 训练种子 | 质量指令下低档概率 | 中档概率 | 单架换中档即时得分提高的比例 | 同状态换指令带来的中档概率差 |', '|---|---:|---:|---:|---:|']
    for s in SEEDS:
        v=mechanism[str(s)]
        lines.append(f'| {s} | {100*v["low_probability"]:.2f}% | {100*v["middle_probability"]:.2f}% | {100*v["positive_fraction"]:.2f}% | {100*v["matched_state_command_middle_probability_difference"]:.2f}个百分点 |')
    lines+=['','概率按物理等价模式合计：低档0/4/8/12，中档5/9/13。单步比较保持当前状态、其他无人机动作和当前资源不变；质量状态合计8100次单架模式干预比较。',
      '', '## 3. 为什么需要协同，而不是随便一架提档','',
      '保持原SUT资源策略和奖励不变，枚举三架无人机的全部8种固定低/中档组合。下表是三种子均值；每个组合从相同初态运行完整600槽，资源策略根据各自轨迹继续决策。',
      '', '| 三架模式 | 质量加分 | 最大AoI扣分 | 净分 |', '|---|---:|---:|---:|']
    for p in patterns:
        v=cm[p];lines.append(f'| {p} | {100*v["quality_credit"]:.4f} | {100*v["age_max_cost"]:.4f} | {100*v["common_reward"]:.4f} |')
    lines+=['', '0是低档，5是中档。只让一架用中档，就可能把全系统最大AoI的扣分抬起来，却只得到该架的质量收益；多架一起升级可以获得更多质量收益，而最大AoI扣分不会按架数线性增加。',
      f'三架一起提档相对全低档的收益，比三次单架提档收益之和多 {100*synergy["common_reward"]:.4f} 分。其中最大AoI项贡献 {100*(-synergy["age_max_cost"]):.4f} 分，其他项略有抵消。这量化了动作间的耦合；不能据此宣称已经证明某个严格局部最优或唯一优化失败原因。',
      '', '## 4. 恢复学习率的受控续训没有成功','',
      '原100万步结束时 actor 学习率为4×10⁻⁷，是初始1×10⁻⁴的1/250，并非严格为0。三个种子各从完全相同的模型、优化器状态和随机数状态分成两支，每支续训20万步；一支维持最终学习率，另一支恢复初始学习率。critic 同比例调整。物理参数、奖励、熵系数、网络结构、指令规则均不变。',
      '', '| 对照 | 质量指令分数 | AoI指令分数 | 质量→AoI分数 |', '|---|---:|---:|---:|']
    for name,label in [('deterministic','续训前'),('final_lr','维持极小学习率+20万步'),('restart_lr','恢复初始学习率+20万步'),('old_resources_new_modes','只换成续训后的模式策略'),('new_resources_old_modes','只换成续训后的资源策略')]:
        lines.append('| '+label+' | '+' | '.join(f'{100*comparison[sc][name]["common_reward"]:.4f}' for sc in SCENARIOS)+' |')
    lines+=['','恢复学习率分支在三个种子的这三种场景中均低于对应极小学习率分支。不能据此排除更长训练、其他学习率日程的作用；它否定了“恢复初始学习率再跑这20万步即可改善”的本次方案。拆换控制器显示，这次退步主要来自模式策略，不能归咎于资源策略。',
      '', '例如种子966续训后，在质量任务中三架无人机的低档比例变成100%、0%、100%；同时质量与AoI任务的PSNR差只有0.0164 dB。它更接近固定分工，未学成清晰的指令条件协同。',
      '', '## 5. 训练反馈中还存在哪些困难','',
      '最终模型的随机训练采样中，资源份额的抽样标准差约0.15；GAE优势标准差约1.71–2.86。全系统奖励同时受到三架模式、资源随机分配、后续状态和指令切换的影响。单架模式的即时收益远小于这一多步反馈的波动，但两者时间口径不同，不能直接把这个比例当作正式信噪比。',
      '在真实随机训练状态中，一架从低档换中档的即时质量任务收益仍有约80%–84%的比较为正；所以问题也不能单纯解释为确定性评估与训练分布不同。优势采样方向不稳定、固定熵奖励、旧资源预算观测均是后续需要拆分验证的因素，尚未被本次实验唯一确定为根因。',
      '', '随机训练中原始中档动作被执行器改选的比例（按实际预算检查，和确定性评估分开报告）：','', '| 种子 | 质量指令 | AoI指令 |', '|---|---:|---:|']
    for s in SEEDS:
        n=noise['results'][str(s)];lines.append(f'| {s} | {100*n["2"]["raw_middle_replaced_fraction"]:.2f}% | {100*n["1"]["raw_middle_replaced_fraction"]:.2f}% |')
    lines+=['', '这些比例来自各模型的一批4000步新采样，不代表完整100万步历史比例。替换比例较小，不支持将其视为主要失败原因。UAV观测包含上次资源预算，SUT在同一时隙重新抽样资源；两者不同会让模式收益更不稳定。这是当前并行动作设计的耦合，不能直接称作动作编码错误。',
      '', '## 6. 建议的下一项受控验证','',
      '保持参考值8和现有奖励不变，先固定资源分配，只训练模式策略，并与同预算联合训练配对比较。若模式单独训练能稳定做到质量任务协同中档、AoI任务低档，再释放资源策略；若仍不能，重点检查模式优势估计与探索日程。此处是下一项建议，本次没有启动这项新训练。',
      '不建议现在把失败归因于质量/AoI权重不公平，再继续按测试结果调权重；同一奖励下已有更好的可行模式组合。最大AoI项若符合论文任务目标，应先保留并解决协同学习；若研究目标本身要改变，应作为新的建模选择重新预注册比较。',
      '', '## 7. 完成与证据边界','',
      '- 已完成：86.4万槽冻结策略诊断、6万槽冻结模型随机采样、6个20万步续训分支、续训后配对评估和拆分回放、8种模式组合枚举。新增环境步合计284.4万，其中实际梯度训练120万步。',
      '- 冻结回放和全部新增评估共158.4万槽经过逐步物理/奖励检查；所有保存轨迹又重新加总奖励分项。原模型和配置哈希未变，原报告未修改。',
      '- 这是针对既有测试种子的诊断，包含事后追加的控制器拆分和组合枚举；不能作为独立测试集上的论文改进结论。',
      '- 本次所有进程已结束；新模型保留在本目录continuation中，未替换正式结果。',
      '', f'汇总数字：[summary.json]({OUT}/summary.json)。冻结试验：[protocol.json]({OUT}/protocol.json)。续训控制协议：[continuation_protocol.json]({OUT}/continuation_protocol.json)。']
    (OUT/'DIAGNOSIS.md').write_text('\n'.join(lines)+'\n')
    write(OUT/'audit.json',audit)
    # stdout logs can still grow until the caller exits; hash closed artifacts only.
    write(OUT/'artifact_hashes.json',{str(p.relative_to(OUT)):digest(p) for p in OUT.rglob('*') if p.is_file() and p.name!='artifact_hashes.json' and p.suffix!='.log'})
    print(json.dumps(dict(audit=audit,synergy=synergy,comparison=comparison),ensure_ascii=False),flush=True)

if __name__=='__main__':main()
