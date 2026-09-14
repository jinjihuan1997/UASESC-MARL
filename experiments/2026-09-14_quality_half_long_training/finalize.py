from train_support import *

def report():
 r=read(HERE/'report/results.json');pairs=read(HERE/'report/paired_differences.json');audit=read(HERE/'report/audit.json');groups=['overall_13','balance_fixed','aoi_fixed','quality_fixed','switch_10']
 lines=['# 半质量权重下联合HAPPO长期续训结果','',
 '三个固定种子从2026-09-14_quality_half_retraining各自100万步完整断点继续到累计1000万步；各新增900万，合计新增2700万RL环境步，累计3000万步。全部原优化器、ValueNorm、环境与RNG状态保留。模型、奖励、学习率和其他设置不变，未换种子或按结果选择检查点。',
 '', '统一新权重下的共同奖励均值×100，越大越好。综合按13场景×20开发环境×三个固定种子等权。200/400/600/800万步仅用于固定三任务过程观察，主结果固定1000万步。','',
 '| 方法 | 综合 | 固定均衡 | 固定AoI | 固定质量 | 切换场景 |','|---|---:|---:|---:|---:|---:|']
 for method in r['methods']:lines.append('| '+method+' | '+' | '.join(f"{r['scores'][method]['conditional_mean'][g]['mean']:.6f}" for g in groups)+' |')
 lines+=['','## 长训练相对100万步起点','', '| 种子 | 1000万步综合 | 100万步综合 | 差值 | 95%配对区间 |','|---|---:|---:|---:|---|']
 for seed in r['seeds']:
  a=r['scores']['long_rl_Qhalf']['by_parent'][str(seed)]['groups']['overall_13']['mean'];b=r['scores']['retrained_rl_Qhalf']['by_parent'][str(seed)]['groups']['overall_13']['mean'];d=pairs['long_rl_Qhalf_minus_retrained_rl_Qhalf']['by_parent'][str(seed)]['groups']['overall_13'];lines.append(f"| {seed} | {a:.6f} | {b:.6f} | {d['mean']:+.6f} | [{d['ci95'][0]:+.6f}, {d['ci95'][1]:+.6f}] |")
 lines+=['','## 全部配对参照','', '| 长训练减去参照 | 综合分差 | 95%区间 |','|---|---:|---|']
 for method in r['methods']:
  if method=='long_rl_Qhalf':continue
  d=pairs[f'long_rl_Qhalf_minus_{method}']['conditional_mean']['overall_13'];lines.append(f"| {method} | {d['mean']:+.6f} | [{d['ci95'][0]:+.6f}, {d['ci95'][1]:+.6f}] |")
 lines+=['','## 结论','']
 for method,label in [('retrained_rl_Qhalf','新权重100万步起点'),('original_rl','旧权重训练的冻结RL'),('G_equal_local16_Qhalf_local','按新权重选模的均分局部贪心'),('greedy_modes_16_Qhalf_local','按新权重选模的同观测强贪心')]:
  d=pairs[f'long_rl_Qhalf_minus_{method}']['conditional_mean']['overall_13'];lines.append(f"相对{label}，长期训练平均分差{d['mean']:+.6f}，配对区间[{d['ci95'][0]:+.6f}, {d['ci95'][1]:+.6f}]。"+('平均表现较好。' if d['mean']>0 else '平均表现仍较低。')+('区间跨零，不能称稳定胜出或统计等价。' if d['ci95'][0]<=0<=d['ci95'][1] else ''))
 d=pairs['long_rl_Qhalf_minus_retrained_rl_Qhalf']['conditional_mean']['overall_13'];g=pairs['long_rl_Qhalf_minus_G_equal_local16_Qhalf_local']['conditional_mean']['overall_13']
 lines.append('在本轮固定模型和开发环境中，'+('延长训练改善了平均回报。' if d['mean']>0 else '延长训练未改善平均回报，不支持仅靠继续加步数来解决当前差距。')+('仍落后同目标均分局部贪心，应先检查分任务/资源与模式取舍，而不是自动再延长。' if g['mean']<0 else '对同目标均分局部贪心的优势仍需按逐种子及区间审视，不能外推全局最优。'))
 lines+=['', '训练步数增加不能证明初始化或奖励权重是唯一根因；本轮无其他算法改动。所有种子、负结果和固定检查点均保留。强贪心_Qhalf_local的SUT预测器仍按旧目标拟合，本轮未拟合新预测器；同时保留无需预测器的G_equal_local16_Qhalf_local。',
 '按环境种子成簇bootstrap4000次，保留全部13场景和固定模型关联。区间只描述当前三个固定模型与已使用开发集，不能冒称独立最终测试或训练总体结论。PSNR是固定平均质量表给出的交付加权预测值，不是实测视频解码质量。',
 '', '## 核验、成本与文件','',
 '正式新增2700万RL步，新增actor优化器更新270000次、critic更新67500次；原100万步起点三模型累计300万步单列。新增评估1500完整回合/900000步；认证复用旧参照6760回合，未将复用算新增采集。预检实际16000物理步、160次actor更新、40次critic更新，不计入正式预算。',
 '全部新增训练物理步在线核验；每种子60批完整训练轨迹共360000步进行离线重建，三模型合计1080000步（新增训练覆盖4%）。所有评估轨迹逐槽核验并完整独立重建。初始状态迁移、下一轮更新和跨600槽终止的恢复均通过逐值检查。',
 f"全部{audit['protected_files']}个受保护输入结束时哈希一致，实际训练派生种子和评估轨迹seed数组与reserved_final_test交集为空。最终测试未使用。两次完整聚合数值哈希见reproducibility.json。",
 'results.json保存全方法、逐种子、全场景和中间固定任务；paired_differences.json保存全部分差与区间；gap_decomposition.json保存奖励分项差；physical_statistics.json保存交付、质量、AoI、资源和模式；audit.json保存核验及成本；../jobs与../evaluation保存模型、完整恢复状态与可重算轨迹。',
 '', '完成本轮固定预算后停止，不自动增加训练、改奖励、换种子、做最终测试、修改论文或推送GitHub。']
 textfile(HERE/'report/REPORT.md','\n'.join(lines)+'\n')

def main():
 guard();verify();files=['results.json','paired_differences.json','physical_statistics.json','gap_decomposition.json','audit.json','REPORT.md'];initial=None
 for f,h in read(HERE/'analysis_seal.json')['sources'].items():assert sha(HERE/f)==h
 for repeat in range(2):
  subprocess.run([sys.executable,str(HERE/'aggregate.py')],check=True);report();hashes={f:sha(HERE/'report'/f) for f in files}
  if initial is None:initial=hashes
  else:assert hashes==initial,'Aggregation/report not deterministic'
 verify();before=read(HERE/'git_status_start.json');after={}
 for directory in before:
  after[directory]={}
  for key,args in [('head',['rev-parse','HEAD']),('status',['status','--porcelain=v1'])]:
   r=subprocess.run(['git','-C',directory,*args],capture_output=True,text=True,env={**os.environ,'GIT_OPTIONAL_LOCKS':'0'});after[directory][key]=dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)
 write(HERE/'git_status_end.json',after);write(HERE/'report/reproducibility.json',dict(state='PASS',complete_aggregation_runs=2,deterministic_output_sha256=initial,protected_inputs_unchanged=True,git_status_unchanged=before==after))
 write(HERE/'status.json',dict(state='complete',completed_additional_steps=27000000,completed_cumulative_steps=30000000,completed_evaluation_steps=900000,report=str(HERE/'report/REPORT.md'),updated_utc=stamp(),reserved_final_test_used=False))
if __name__=='__main__':main()
