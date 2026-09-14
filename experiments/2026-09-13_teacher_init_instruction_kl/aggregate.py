"""Deterministic Stage A numerical aggregation; obey the all-student stop gate."""
from study import *
from gate_check import gate_scores,GROUPS,ci
from audit_trajectories import load_arrays
from collections import Counter,defaultdict
import argparse

def cost_ledger():
 rows=[json.loads(line) for p in sorted((HERE/'costs').glob('*.jsonl')) for line in p.read_text().splitlines()]
 by={};identities=Counter((x['kind'],x['identity']) for x in rows)
 keys=['physical_steps','complete_episodes','gradient_steps','critic_updates','forward_example_evaluations','examples_presented']
 for kind in sorted(set(x['kind'] for x in rows)):
  values=[x for x in rows if x['kind']==kind];by[kind]={k:sum(x.get(k,0) for x in values) for k in keys};by[kind].update(records=len(values),teacher_queries=sum(sum(x.get('teacher_queries',[])) for x in values))
 return dict(by_kind=by,total_physical_steps=sum(v['physical_steps'] for v in by.values()),total_teacher_queries=sum(v['teacher_queries'] for v in by.values()),formal_supervised_optimizer_steps=by['supervised_fit']['gradient_steps'],prototype_preflight_optimizer_steps=by['preflight_gradient']['gradient_steps'],RL_environment_steps=0,critic_warmup_environment_steps=0,critic_optimizer_updates=0,repeated_offline_audit_identities={f'{k}/{i}':n for (k,i),n in identities.items() if n>1},forward_accounting=dict(formal_collection_student_forward_examples=8*432000,gate_student_forward_examples=8*936000,formal_supervision_forward_examples=4*by['supervised_fit']['examples_presented'],offline_replay_forward_examples=by['offline_trajectory_audit']['forward_example_evaluations'],prototype_preflight_forward_total=None,prototype_preflight_forward_note='Physical steps and optimizer steps are exact; the heterogeneous prototype finite/MC/rollback tests did not instrument a complete forward-call counter. No missing prototype work is counted as zero.'))

def diagnostic_summary(items):
 out={}
 for g in ['0','1','2']:
  rows=[x[g] for x in items if g in x];n=sum(x['slots'] for x in rows)
  if not n:continue
  w=lambda k:np.sum([np.asarray(x[k])*x['slots'] for x in rows],axis=0)/n
  uavs=[]
  for u in range(3):
   nv=sum(x['uavs'][u]['valid_samples'] for x in rows)
   uv={k:sum((x['uavs'][u][k] or 0)*x['uavs'][u]['valid_samples'] for x in rows)/nv if nv else None for k in ['original_mode_accuracy','equivalent_mode_accuracy','teacher_cross_entropy']}
   uv.update(valid_samples=nv,skip_reason_counts=np.sum([x['uavs'][u]['skip_reason_counts'] for x in rows],0).tolist(),original_entropy=sum(x['uavs'][u]['original_entropy']*x['slots'] for x in rows)/n,equivalence_entropy=sum(x['uavs'][u]['equivalence_entropy']*x['slots'] for x in rows)/n);uavs.append(uv)
  out[g]=dict(slots=n,**{k:w(k).tolist() for k in ['share_absolute_error_mean','budget_absolute_error_mean','smoothing_share_bias_mean','smoothing_budget_bias_mean']},student_total_concentration_mean=float(w('student_total_concentration_mean')),uavs=uavs)
 return out

def main():
 m=verify();g=read(HERE/'gate_report.json');assert g['state']=='STOP_AFTER_A', 'This completed run stopped at A; B aggregation must not invent absent RL results.'
 seal=read(HERE/'gate_execution_seal.json')
 for p,h in seal['sources'].items():assert sha(HERE/p)==h
 for seed,h in seal['A3_checkpoints'].items():assert sha(HERE/f'students/{seed}/A3/checkpoint.pt')==h
 scores=gate_scores();boot=np.random.default_rng(m['bootstrap_seed']).integers(0,10,(4000,10));physical={};metadata={};audits={};diags={};traces={};arrays={};actual_seeds=set()
 for label,modes in [('teacher',['D'])]+[(str(s),['D','S0','S1','S2']) for s in m['students']]:
  for mode in modes:
   for scene in m['scenarios']:
    key=f'{label}/{mode}/{scene}';p=HERE/f'gate/{key}.npz';z=load_arrays(p);meta=read(p.with_suffix('.json'));a=read(HERE/f'audits/gate/{key}.json')
    assert a['state']=='PASS' and a['identity']['trace_sha256']==sha(p);audits['gate/'+key]=a['physical'];traces['gate/'+key]=sha(p);actual_seeds.update(z['seeds'].tolist())
    v={f:z['trace'][:,:,i] for i,f in enumerate(FIELDS)};arrays[key]={k:v[k] for k in ['common_reward','instruction_id','quality_credit','age_mean_cost','age_max_cost','age_tail_cost','resource_cost','service_violation_cost','recv_aoi_bonus']}
    stats=prior_auditor().compact_stats(z,m)
    for gid,d in stats.items():
     take=v['instruction_id']==int(gid);age=z['aoi_after'][take].astype(float)
     d['by_uav_aoi']=[dict(mean=float(age[:,u].mean()),mean_slot_max=float(age[:,u].max(-1).mean()),p95_all_ds_slots=float(np.quantile(age[:,u],.95)),excess_above4_mean=float(np.maximum(age[:,u]-4,0).mean())) for u in range(3)]
     d['no_delivery_fraction_by_uav']=(z['served'][take].sum(-1)==0).mean(0).tolist()
    physical[key]=stats;metadata[key]=dict(episode_scores_x100=(v['common_reward'].mean(0)*100).tolist(),episode_mean_reward_raw=v['common_reward'].mean(0).tolist(),score_x100=float(v['common_reward'].mean()*100),predicted_PSNR_mean=float(v['predicted_quality_sum'].sum()/v['deliveries'].sum()),complete_episodes=10)
    diags[key]=a['diagnostics']
 for p in sorted((HERE/'data').rglob('batch_*.npz')):
  key=str(p.relative_to(HERE));a=read(HERE/'audits'/p.relative_to(HERE).with_suffix('.json'));assert a['state']=='PASS' and a['identity']['trace_sha256']==sha(p)
  audits[key]=a['physical'];traces[key]=sha(p);actual_seeds.update(a['seeds'])
 assert len(audits)==211 and sum(x['physical_slots'] for x in audits.values())==1518000
 for p in (HERE/'preflight').rglob('*.npz'):
  with np.load(p,allow_pickle=False) as z:
   if 'seeds' in z.files:actual_seeds.update(z['seeds'].tolist())
 for p in (HERE/'costs').glob('*.jsonl'):
  for line in p.read_text().splitlines():actual_seeds.update(json.loads(line).get('seeds',[]))
 assert not actual_seeds&set(m['reserved']);assert set(m['gate_dev']).isdisjoint(m['known_development_seeds'])
 for kind in ['A0','dagger','warmup','B','preflight']:assert not set(m['allowed_environment_seeds'][kind])&set(m['known_development_seeds']+m['gate_dev']+m['reserved'])
 diagnostics={};training={};gap={};carry={}
 reward_fields=['quality_credit','age_mean_cost','age_max_cost','age_tail_cost','resource_cost','service_violation_cost','recv_aoi_bonus'];sign={k:1 if k in ['quality_credit','recv_aoi_bonus'] else -1 for k in reward_fields}
 for seed in m['students']:
  label=str(seed);diagnostics[label]={mode:diagnostic_summary([diags[f'{label}/{mode}/{scene}'] for scene in m['scenarios']]) for mode in ['D','S0','S1','S2']}
  training[label]={}
  for stage in ['A0','A1','A2','A3']:
   folder=HERE/f'students/{seed}/{stage}';st=read(folder/'status.json');rows=[json.loads(x) for x in (folder/'fit.jsonl').read_text().splitlines()];assert len(rows)==(20 if stage=='A0' else 10) and st['epochs']==len(rows)
   training[label][stage]=dict(**st,first_epoch=rows[0],last_epoch=rows[-1],checkpoint= str((folder/'checkpoint.pt').relative_to(HERE)))
  assert training[label]['A3']['optimizer_steps']==[6360]*4
  gap[label]={};carry[label]={}
  for mode in ['D','S_mean']:
   parts=[];post=[]
   for scene,sc in m['scenarios'].items():
    teacher=arrays[f'teacher/D/{scene}'];student=arrays[f'{label}/D/{scene}'] if mode=='D' else {k:np.mean([arrays[f'{label}/S{r}/{scene}'][k] for r in range(3)],0) for k in teacher}
    gid=teacher['instruction_id'];np.testing.assert_array_equal(student['instruction_id'],gid)
    parts.append({str(gidvalue):{f:((student[f]-teacher[f])*(gid==gidvalue)).mean(0)*100 for f in ['common_reward',*reward_fields]} for gidvalue in range(3)})
    seen=np.maximum.accumulate(gid==2,axis=0);cm=seen&(gid!=2)
    if cm.any():post.append((scene,(student['common_reward']-teacher['common_reward'])[cm].reshape(-1,10).mean(0)*100,int(cm[:,0].sum())))
   entry={str(gidvalue):{f:ci(np.stack([p[str(gidvalue)][f] for p in parts]).mean(0),boot) for f in ['common_reward',*reward_fields]} for gidvalue in range(3)}
   total=sum(entry[str(i)]['common_reward']['mean'] for i in range(3));source=scores[label]['D'] if mode=='D' else np.mean([scores[label][f'S{r}'] for r in range(3)],0)
   np.testing.assert_allclose(total,(source-scores['teacher']['D']).mean(),atol=1e-12,rtol=0)
   for j in range(3):np.testing.assert_allclose(sum(sign[f]*entry[str(j)][f]['mean'] for f in reward_fields),entry[str(j)]['common_reward']['mean'],atol=1e-12,rtol=0)
   gap[label][mode]=entry;carry[label][mode]={scene:dict(delta_score_x100=ci(x,boot),post_quality_slots=n,interpretation='Full student-versus-teacher trajectories differ in both current policy and past states; not a uniquely identified quality carryover cost.') for scene,x,n in post}
 costs=cost_ledger();assert costs['by_kind']['A0_collection']['physical_steps']==72000 and costs['by_kind']['dagger_collection']['physical_steps']==432000 and costs['by_kind']['gate_student']['physical_steps']==936000
 assert costs['formal_supervised_optimizer_steps']==76320 and costs['RL_environment_steps']==0
 numeric_training={s:{stage:{k:v for k,v in d.items() if k not in ['first_epoch','last_epoch']} | {name:{k:v for k,v in d[name].items() if k!='elapsed_seconds'} for name in ['first_epoch','last_epoch']} for stage,d in stages.items()} for s,stages in training.items()}
 result=dict(state='COMPLETED_STOPPED_AFTER_A',gate=g,units='mean original common reward per physical slot x100; raw episode means retained',teacher='frozen greedy_modes_16',students=m['students'],episodes=metadata,supervised_training=numeric_training,diagnostics=diagnostics,costs=costs,stage_B=dict(state='NOT_RUN_STAGE_A_GATE_FAILED',critic_warmup_steps=0,RL_environment_steps=0,KL_protection_triggered=None,HAPPO_effect=None,reason='0/3 students pass the fixed deterministic score gate; no conditional critic warmup or joint RL is authorized.'),interpretation_limits=['gate_dev is development admission data, not independent final test','three fixed initializations condition all confidence intervals','teacher imitation gains are supervised learning, not pure RL','predicted PSNR derives from the fixed mean profile, not decoded video measurements'])
 report=HERE/'report';write(report/'results.json',result);write(report/'physical_statistics.json',physical);write(report/'diagnostics_by_scene.json',diags);write(report/'gap_decomposition.json',gap);write(report/'post_quality_comparison.json',carry)
 audit=dict(state='PASS',manifest_sha256=sha(HERE/'manifest.json'),gate_execution_seal_sha256=sha(HERE/'gate_execution_seal.json'),protected_files_checked=len(m['protected_sha256']),protected_files_unchanged=True,formal_physical_reconstruction_slots=1518000,independent_profile_and_executor_traces=211,checks=audits,trace_sha256=traces,all_new_trajectories_original_checked_every_slot=True,physical_input_label_and_saved_policy_replay='PASS_FULL_FORMAL_COLLECTION_AND_GATE',maximum_reward_reconstruction_error=max(x['max_reward_reconstruction_error'] for x in audits.values()),tolerances_unchanged=True,actual_environment_seeds=sorted(actual_seeds),reserved_intersection=sorted(actual_seeds&set(m['reserved'])),reserved_final_test_used=False,new_RL_training_steps=0,new_supervised_optimizer_updates=76320,new_critic_optimizer_updates=0,prototype_preflight_optimizer_updates=262,student_inference_teacher_queries=0,teacher_is_frozen_local_observation_predictor=True,stage_B_integration_preflight='NOT_RUN_STAGE_A_GATE_FAILED',conditional_modules='Warmup and joint training entries are delivered and gate guarded. Their full integration is not executed or claimed validated after the Stage A stop.',costs=costs)
 write(report/'audit.json',audit)
 make_report(result,physical,gap,report)
 print('aggregated',g['state'],'formal physics checks',len(audits),flush=True)

def make_report(r,physical,gap,report):
 g=r['gate'];students=r['students'];teacher=g['teacher_scores'];num=lambda x:f'{x:.6f}';rows=[]
 for s in students:
  d=g['students'][str(s)];rows.append(f"| {s} | {num(d['D_scores']['overall_13']['mean'])} | {num(d['D_minus_teacher']['overall_13']['mean'])} | {num(d['S_scores']['overall_13']['mean'])} | {num(d['S_minus_D']['overall_13']['mean'])} | 未通过 |")
 lines=['# 教师初始化与按指令KL对照：阶段A停止报告','', '**结论：固定A0—A3预算及完整准入评估已完成，0/3学生通过；依协议停止，不进行critic预热和600万步联合HAPPO。** 所有物理、预算条件标签及保存策略重放检查通过。停止原因是闭环分数未达到预设门槛，不是训练崩溃。','',f"新gate_dev教师综合分为 **{num(teacher['overall_13']['mean'])}**。本表每分均为原每槽共同奖励平均×100；13场景和10环境等权。旧20开发环境上的历史分数没有搬到本表。",'', '| 新学生初始化 | 确定性D综合 | D−教师 | 采样S综合（3次平均） | S−D | 准入 |','|---|---:|---:|---:|---:|---|',*rows,'', 'D综合允许落后不超过0.10分；三个学生分别落后约0.231、0.214、0.193分，全部不通过。三者随机执行相对D的退步均通过事先规定的随机准入门槛；不能把这次停止描述为“随机执行完全崩溃”。','', '## 三个固定任务与配对区间','', '| 学生 | 任务 | 教师 | 学生D | D−教师（95%条件区间） | S−D（95%条件区间） |','|---|---|---:|---:|---|---|']
 names={'balance_fixed':'均衡','aoi_fixed':'AoI','quality_fixed':'质量'}
 for s in students:
  d=g['students'][str(s)]
  for k,name in names.items():
   td=d['D_minus_teacher'][k];sd=d['S_minus_D'][k];interval=lambda z:f"{num(z['mean'])} [{num(z['ci95'][0])}, {num(z['ci95'][1])}]"
   lines.append(f"| {s} | {name} | {num(teacher[k]['mean'])} | {num(d['D_scores'][k]['mean'])} | {interval(td)} | {interval(sd)} |")
 lines+=['', '固定任务D门槛均为不低于教师−0.20分。所有学生固定均衡失败；前两个学生固定AoI也失败，第三个固定AoI通过；三个学生固定质量均通过。综合配对区间、三个采样重复和13个场景逐回合原始分数详见results.json，未按显著性筛选比较。','', 'bootstrap为4000次环境种子成组重采样，每次保留其13场景、三个网络和所有方法；采样重复先在同一学生/场景/环境内平均。区间只以这三个网络和本轮开发准入条件为前提，不是随机训练总体的稳定性结论，也不是统计等价证明。','', '## 资源误差、预算条件选模及随机执行','', '以下为全部13场景中按当槽真实指令归组、按实际时隙数加权的确定性学生诊断。教师标签在学生已经执行的资源预算下只读查询，UAV一致率只统计有缓存且有物理可行模式的样本。资源误差对照同一学生SUT观测上的教师原始确定性动作，不仅对照平滑目标。','', '| 学生 | 当前指令 | 资源份额绝对误差（UAV平均，百分点） | 预算绝对误差（UAV平均） | 模式等价组一致率（3 UAV平均） | Dirichlet总浓度平均 |','|---|---|---:|---:|---:|---:|']
 for s in students:
  for k,d in r['diagnostics'][str(s)]['D'].items():
   lines.append(f"| {s} | {['均衡','AoI','质量'][int(k)]} | {np.mean(d['share_absolute_error_mean'])*100:.4f} | {np.mean(d['budget_absolute_error_mean']):.3f} | {np.mean([u['equivalent_mode_accuracy'] for u in d['uavs']])*100:.3f}% | {d['student_total_concentration_mean']:.3f} |")
 lines+=['', '模式仍是原16类，统计等价组来自原profile逐值相等，不改变动作空间。原编号准确率、等价组准确率、跳过原因、分类熵及逐UAV结果均保存。资源监督目标仍是总浓度1000、平滑1e−4的预定Dirichlet分布；实际学生浓度及资源误差不是这个预设目标本身。','', '质量任务已经较好接近教师；均衡和AoI还存在资源与选模误差。预算是连续量，但可交付块数存在离散门槛，因此较小的份额误差也可能改变交付数。budget_threshold_diagnostics.json在保存的学生状态上固定实际执行模式，比较学生预算和教师预算可支持的块数，并给出原始例子；这是离线敏感性分析，不把反事实容量当成实际教师交付，更不把它当作根因的因果比例。','', '监督时每条UAV标签均使用实际执行资源后的局部输入和原掩码，全部504,000个采集槽经过重放核验，没有发现错配标签或物理违规。纯局部教师接口复现、保存加载及动作/logp重放也通过。这使“这轮实现把预算A和预算B的标签混用了”缺乏支持。','', '随机执行在已有确定性误差之外进一步降低回报，但未越过本轮允许的随机退步门槛。SUT连续抽样与UAV分类抽样同时开启，本轮没有分别固定一侧做新的消融，不能将随机损失唯一归因于某一侧。训练数据在逐轮进入学生自身状态后仍有闭环差距；现有证据也不能唯一分开函数拟合精度、状态覆盖与局部决策门槛的作用。','', '## 完整切换回合与奖励分项','', 'gap_decomposition.json按照13场景实际当槽指令占比累加D−教师及S−教师的奖励分项，保留正负贡献，并核对其总和等于综合分差。不是把三个固定任务分差三等分。physical_statistics.json保存每场景、执行方式、当前指令和UAV的资源、16类模式/等价组、交付、预测质量与AoI。','', 'post_quality_comparison.json报告质量结束后的非质量时段，但学生与教师当时使用的策略本身也不同，不能把差异全称为质量阶段遗留成本。完整回合分数已经包含这些影响，不额外扣减。PSNR始终标为平均质量表给出的交付加权预测值，未解码真实视频。','', '## 对本轮研究问题的回答','', '1. **神经策略能否在自身闭环接近教师？** 质量任务接近；本固定预算下，三个模型均未达到综合及均衡任务的预设工程准入标准，不能写成完整学会了教师。', '2. **资源与模式误差是什么？** 上表及逐UAV统计显示两者均存在；预算条件标签本身核验正确。资源平滑偏差另列在results诊断中，不与实际拟合误差混淆。', '3. **随机执行是否保留能力？** 有进一步退步，但全部达到预设S相对D门槛；仍不能视为与教师等价。', '4. **原HAPPO改善还是破坏监督起点？** 未检验。阶段B未获准，新增RL环境步为0。', '5. **按指令KL是否触发并减少退步？** 正式训练未运行，因此没有触发率、保护收益或稳定性结论。解析KL和回滚的原型预检通过不等于正式方法有效。', '6. **稳定性是否伴随更高回报？** 未检验B0/B1，不能以KL较小或停在一个固定学生替代回报证据。', '7. **是否超过仅监督学生或教师？** 没有HAPPO结果可与仅监督学生比较；三个监督学生综合均低于同口径教师。greedy_modes_3、C1/C3与简单规则保留冻结引用索引；它们没有在本gate_dev补评，不能拿旧开发分数证明本次胜负。', '8. **下一步建议？** 暂不扩大这批起点的联合RL。优先研究均衡/AoI下资源拟合精度与交付门槛、同预算模式误差，及连续/分类抽样分别造成的损失；本轮不自动追加这些实验。不由本次未过关推断神经网络不可能学会，也不推断初始化是唯一根因。','', '## 实际执行、恢复与核验账本','', '| 计算项目 | 实际预算 |','|---|---:|', '| 公共教师采集 | 120回合，72,000步，只计一次 |','| 三学生三轮自身状态采集 | 720回合，432,000步 |','| 正式监督拟合 | 每学生50轮；合计76,320个actor optimizer.step |','| 准入评估（含教师和采样重复） | 1690回合，1,014,000步 |','| 原型预检及教师复现 | 169,280物理步，262个原型优化器更新 |','| critic预热、critic更新、阶段B RL | 全部为0，按准入条件未运行 |','',f"实际物理计算合计 **{r['costs']['total_physical_steps']:,}步**。教师接口查询（按单样本/单actor计，包括离线重放、原型和诊断）合计 **{r['costs']['total_teacher_queries']:,}次**。详细成本及重复离线审核在results/audit账本中保留；监督不是零训练成本，也不是纯RL成果。异构原型预检没有完整记录全部前向调用数，该项明确为null，不假报为0；其物理步与优化器步已计入。",'', 'A1首次恢复遇到CPU shuffle Generator拒绝CUDA ByteTensor，发生在新增A1更新之前。AMENDMENT_001仅把保存的随机字节送回CPU，原数据、种子、权重、Adam动量和预算均保留；未重跑A0来增加训练。原日志与修复前后源码哈希保留。另一次离线审核烟测首次使用了错误文件后缀编号，未产生物理或梯度步，随后按实际batch_00路径执行。','', '旧模型、质量表、配置、报告和论文共2702个受保护输入结束时逐个SHA-256复核。全部211份正式采集/准入轨迹共1,518,000槽经过独立NumPy信道/profile/预算/缓存/AoI/调度/奖励重建；新轨迹原执行中每槽原核验器也通过。所有实际环境seed数组及日志与reserved集合交集为空。两次完整数值聚合及报告文本哈希核验见reproducibility.json。','', '条件warmup_critic.py、train_joint.py及KL模块已交付并设准入守卫；由于阶段A停止，没有运行完整阶段B集成或声称它已正式验证。没有自动追加训练、改奖励、换种子、使用最终测试、修改论文或推送GitHub。','']
 threshold=read(report/'budget_threshold_diagnostics.json')
 example=next(x for x in threshold['examples'] if x['scene']=='fixed_1')
 position=lines.index('## 对本轮研究问题的回答')
 lines[position:position]=['一个可重放的预算门槛例子：学生'+str(example['student'])+'，固定AoI，环境种子'+str(example['environment_seed'])+'，第'+str(example['slot'])+'槽，UAV '+str(example['uav'])+'。执行模式'+str(example['executed_mode'])+f"的预测单块载荷为{example['predicted_load']:.3f}，4块需要{4*example['predicted_load']:.3f}；学生预算{example['student_budget']:.3f}只能交付3块。同一学生SUT输入下的教师预算为{example['teacher_budget_at_same_student_SUT_observation']:.3f}，固定该模式时可容纳4块。这个例子说明预算误差如何跨越交付门槛；它不是把教师完整闭环多交付1块作为已经运行的事实。",'']
 position=lines.index('## 实际执行、恢复与核验账本')
 lines[position:position]=['按完整13场景实际指令占比分解确定性综合差距：','', '| 学生 | 均衡时段贡献 | AoI时段贡献 | 质量时段贡献 | 合计D−教师 |','|---|---:|---:|---:|---:|',*[f"| {s} | {gap[str(s)]['D']['0']['common_reward']['mean']:.6f} | {gap[str(s)]['D']['1']['common_reward']['mean']:.6f} | {gap[str(s)]['D']['2']['common_reward']['mean']:.6f} | {g['students'][str(s)]['D_minus_teacher']['overall_13']['mean']:.6f} |" for s in students],'', '上述是实际奖励差的分组账目，不能解释为训练根因的因果贡献比例。','']
 lines+=['已有教师还包含历史预测器拟合成本，来源见冻结observation_matched_greedy实验；本轮没有重拟合，也没有把它称为无成本。并发离线审核中有17份轨迹被重复重放，额外前向和教师查询均计入账本；没有重复采集物理回合或重复监督更新。','']
 temp=report/'REPORT.md.tmp';temp.write_text('\n'.join(lines));temp.replace(report/'REPORT.md')

if __name__=='__main__':guard();main()
