"""Repeat all aggregation and produce a deterministic Chinese report."""
from train_support import *

def report():
 r=read(HERE/'report/results.json');d=read(HERE/'report/paired_differences.json');a=read(HERE/'report/audit.json');group=['overall_13','balance_fixed','aoi_fixed','quality_fixed','switch_10']
 lines=['# 新质量权重下重新训练联合 HAPPO：固定预算结果','',
 '三个原先封存的种子分别从随机初始化训练100万环境步，共300万步。SUT与三个UAV均参与联合更新，未加载旧actor、adapter、critic或优化器。三指令质量收益系数减半，AoI及资源系数未变，未归一化。',
 '', '以下数值为每槽共同奖励均值×100，越大越好。综合按13场景、20开发环境和3个固定训练种子等权。所有方法使用相同新权重；中间检查点只用于固定任务过程观察。','',
 '| 方法 | 综合 | 固定均衡 | 固定AoI | 固定质量 | 切换场景 |','|---|---:|---:|---:|---:|---:|']
 for method in r['methods']:lines.append('| '+method+' | '+' | '.join(f"{r['scores'][method]['conditional_mean'][g]['mean']:.6f}" for g in group)+' |')
 lines+=['','## 三个固定训练种子','', '| 种子 | 新训RL综合 | 旧RL新权重得分 | 配对分差 | 95%区间 |','|---|---:|---:|---:|---|']
 for seed in r['seeds']:
  new=r['scores']['retrained_rl_Qhalf']['by_parent'][str(seed)]['groups']['overall_13']['mean'];old=r['scores']['original_rl']['by_parent'][str(seed)]['groups']['overall_13']['mean'];x=d['retrained_rl_Qhalf_minus_original_rl']['by_parent'][str(seed)]['groups']['overall_13']
  lines.append(f"| {seed} | {new:.6f} | {old:.6f} | {x['mean']:+.6f} | [{x['ci95'][0]:+.6f}, {x['ci95'][1]:+.6f}] |")
 lines+=['','## 与全部参照的配对分差','', '| 新训RL减去参照 | 综合分差 | 95%区间 |','|---|---:|---|']
 for method in r['methods']:
  if method=='retrained_rl_Qhalf':continue
  x=d[f'retrained_rl_Qhalf_minus_{method}']['conditional_mean']['overall_13'];lines.append(f"| {method} | {x['mean']:+.6f} | [{x['ci95'][0]:+.6f}, {x['ci95'][1]:+.6f}] |")
 lines+=['','## 判断与边界','']
 for method,label in [('original_rl','旧权重训练的冻结RL'),('G_equal_local16_Qhalf_local','局部评分同步新权重的均分贪心'),('greedy_modes_16_Qhalf_local','局部评分同步新权重的同观测强贪心')]:
  x=d[f'retrained_rl_Qhalf_minus_{method}']['conditional_mean']['overall_13'];state='平均得分提高' if x['mean']>0 else '平均得分仍较低'
  lines.append(f"相对{label}，新训RL{state}，分差{x['mean']:+.6f}，区间[{x['ci95'][0]:+.6f}, {x['ci95'][1]:+.6f}]。"+('区间跨零，不能声称稳定胜出或统计等价。' if x['ci95'][0]<=0<=x['ci95'][1] else ''))
 lines+=['','是否减半质量权重有利于学习，应看新旧RL在相同新评价下的配对差；不能把本表与旧目标的历史分数直接相减。超过部分简单规则不等于解决RL与强基线的差距。全部固定预算结果均保留，没有更换种子、挑最佳检查点、调整奖励或继续延长训练。',
 '强贪心_Qhalf_local仅将UAV局部评分同步新目标，其SUT离线预测器仍为旧目标拟合版本；因此同时保留无拟合依赖的G_equal_local16_Qhalf_local作为清晰的新目标参照。C1/C3为旧模型手工组合，本轮未对其重新训练。',
 '区间按环境种子成簇配对bootstrap4000次，保留同环境13场景和所有固定模型关联。仅支持当前3个初始化和反复使用的开发环境，不是随机训练总体结论，也不是独立最终测试。质量指标为固定平均表预测PSNR，不是实测视频解码质量。',
 '', '## 核验与成本','',
 '正式新增RL环境步3000000；actor优化器更新30000次，critic更新7500次。阶段评估与最终评估共1500个完整回合、900000步。预检实际物理步'+str(a['preflight']['physical_steps'])+'，其训练更新未计入正式预算。旧参照5980回合仅认证复用，新增采集为0。',
 '训练全部300万物理步执行原逐槽物理/奖励检查；另保存并独立重建126000个训练物理步的完整轨迹（4.2%覆盖）。评估90万步同时逐槽检查和离线完整重建。精确断点恢复、保存加载、初始概率比、三种子随机初始哈希匹配均通过。',
 '实际训练派生种子及评估轨迹seed数组均与reserved_final_test不相交。所有'+str(a['protected_files'])+'个受保护输入结束时哈希一致。数值聚合重复执行两次，结果哈希在reproducibility.json中记录。',
 '', '## 数据索引','',
 '- results.json：每种方法/种子/场景的分数与过程检查点。',
 '- paired_differences.json、gap_decomposition.json：全部配对差、区间和奖励分项差。',
 '- physical_statistics.json：交付加权预测质量、各UAV AoI、模式、资源及未用预算。',
 '- audit.json：逐槽与离线核验、训练/评估成本和输入保护。',
 '- ../jobs/：新actor、critic、ValueNorm、完整恢复检查点与日志。',
 '- ../evaluation/：可重算的全部新评估轨迹。','',
 '本轮固定预算计算完成后停止；只依据实际差距给后续讨论提供证据，不自动追加训练、修改论文或推送GitHub。']
 textfile(HERE/'report/REPORT.md','\n'.join(lines)+'\n')

def main():
 guard();verify();numeric=['results.json','paired_differences.json','physical_statistics.json','gap_decomposition.json','audit.json','REPORT.md'];first=None
 for filename,h in read(HERE/'analysis_seal.json')['sources'].items():assert sha(HERE/filename)==h
 for repeat in range(2):
  subprocess.run([sys.executable,str(HERE/'aggregate.py')],check=True);report();hashes={f:sha(HERE/'report'/f) for f in numeric}
  if first is None:first=hashes
  else:assert hashes==first,'Deterministic aggregation/report mismatch'
 verify();before=read(HERE/'git_status_start.json');after={}
 for directory in before:
  after[directory]={}
  for key,args in [('head',['rev-parse','HEAD']),('status',['status','--porcelain=v1'])]:
   response=subprocess.run(['git','-C',directory,*args],capture_output=True,text=True,env={**os.environ,'GIT_OPTIONAL_LOCKS':'0'});after[directory][key]=dict(returncode=response.returncode,stdout=response.stdout,stderr=response.stderr)
 write(HERE/'git_status_end.json',after)
 write(HERE/'report/reproducibility.json',dict(state='PASS',complete_aggregation_runs=2,deterministic_output_sha256=first,protected_inputs_unchanged=True,git_status_unchanged=before==after))
 write(HERE/'status.json',dict(state='complete',updated_utc=stamp(),completed_training_steps=3000000,completed_evaluation_steps=900000,report=str(HERE/'report/REPORT.md'),reserved_final_test_used=False))
if __name__=='__main__':main()
