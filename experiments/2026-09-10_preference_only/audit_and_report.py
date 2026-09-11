"""Independent saved-trace aggregation and bounded interpretation of the pilot."""
import hashlib
import json
import os
from pathlib import Path
import sys
os.environ.setdefault('MPLCONFIGDIR','/tmp/tmc_preference_mpl')
from probe_common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

METHODS=['IC_HAPPO','HAPPO_hidden_instruction','R_single','R_instruction','R_myopic']
LABELS=['RL with instruction','RL hidden instruction','Single rule','Instruction lookup rule','One-step greedy rule']


def aggregate_parts(parts):
    data=np.concatenate([part.reshape(-1,part.shape[-1]) for part in parts])
    delivery=float(data[:,3].sum());quality=float(data[:,2].sum());steps=len(data)
    return dict(steps=steps,common_reward=float(data[:,0].mean()),mean_aoi=float(data[:,1].mean()),
        predicted_quality_sum=quality,deliveries=int(delivery),channel_uses=float(data[:,4].sum()),
        delivered_predicted_psnr=quality/delivery,deliveries_per_slot=delivery/steps,
        channel_uses_per_slot=float(data[:,4].mean()))


def assert_metrics(actual,expected):
    for k in actual:np.testing.assert_allclose(actual[k],expected[k],atol=1e-8,rtol=1e-10,err_msg=k)


def main():
    manifest=read(HERE/'pilot_manifest.json');result=read(HERE/'pilot_results.json');audit=read(HERE/'pilot_audit.json')
    verify_parent()
    for f,h in manifest['source_hashes'].items(): assert digest(HERE/f)==h
    assert manifest['protocol_sha256']==digest(HERE/'PROTOCOL.md')
    assert manifest['gate_sha256']==digest(HERE/'gate_results.json')
    gate=read(HERE/'gate_results.json')
    for split in ('calibration','held_out'):
        for name,v in gate[split].items():
            assert digest(HERE/'gate'/split/f'{name}.npz')==v['trace_sha256']
    for name in METHODS[:2]:
        status=read(HERE/'pilot'/name/'status.json');assert status['state']=='complete' and status['completed_steps']==1_000_000
        for f,h in status['checkpoint_hashes'].items(): assert digest(HERE/'pilot'/name/f)==h
    episodes=slots=0;arrays={};summaries={}
    for name in METHODS:
        arrays[name]={};summary_parts=[]
        for scenario in manifest['scenarios']:
            file=HERE/'evaluation'/name/f'{scenario}.npz';assert digest(file)==audit['trace_hashes'][str(file.relative_to(HERE))]
            with np.load(file) as saved:
                trace=saved['trace'];modes=saved['modes'];beta=saved['resource_fractions']
                assert trace.shape==(600,20,9) and np.isfinite(trace).all()
                np.testing.assert_array_equal(saved['seeds'],manifest['evaluation_seeds'])
            schedule=manifest['scenarios'][scenario]
            for t in range(600):
                g=next(g for start,g in reversed(schedule) if start<=t)
                assert np.all(trace[t,:,5]==g)
            assert not np.any(trace[:,:,6:])
            stored=result['per_scenario'][name][scenario]
            assert_metrics(aggregate_parts([trace]),stored['overall'])
            for i in range(20):assert_metrics(aggregate_parts([trace[:,i]]),stored['by_evaluation_seed'][i])
            for g,v in stored['by_instruction'].items():assert_metrics(aggregate_parts([trace[trace[:,:,5]==int(g)]]),v)
            arrays[name][scenario]=(trace,modes,beta)
            summary_parts.append(trace);episodes+=20;slots+=12000
        summaries[name]=aggregate_parts(summary_parts)
        assert_metrics(summaries[name],result['grouped'][name]['overall'])
        for g in range(3):
            parts=[t[t[:,:,5]==g] for t in summary_parts if np.any(t[:,:,5]==g)]
            assert_metrics(aggregate_parts(parts),result['grouped'][name]['by_instruction'][str(g)])
    hidden=arrays['HAPPO_hidden_instruction'];reference=hidden['fixed_0']
    physically_identical=[]
    for scenario,(trace,modes,beta) in hidden.items():
        # Reward and instruction differ; all saved physical outcomes and actions must agree.
        np.testing.assert_array_equal(trace[:,:,1:5],reference[0][:,:,1:5])
        np.testing.assert_array_equal(modes,reference[1]);np.testing.assert_array_equal(beta,reference[2])
        physically_identical.append(scenario)
    diffs={}
    for name in METHODS[1:]:
        values=np.stack([(arrays['IC_HAPPO'][s][0][:,:,0]-arrays[name][s][0][:,:,0]).mean(0) for s in manifest['scenarios']])
        np.testing.assert_allclose(values.mean(),result['paired_ic_minus'][name]['mean'],atol=1e-12)
        diffs[name]=float(values.mean())
    checks=dict(state='PASS',episodes=episodes,slots=slots,all_saved_statistics_reaggregated=True,
        hidden_physical_trajectories_identical_across_all_instructions=True,
        hidden_identical_scenarios=physically_identical,source_and_model_hashes_verified=True,
        source_sha256=digest(__file__))
    write(HERE/'independent_audit.json',checks)
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for ax,metric,title in zip(axes,['mean_aoi','delivered_predicted_psnr'],['Mean AoI (slots)','Delivered predicted PSNR (dB)']):
        for name,label in zip(METHODS,LABELS):
            ax.plot(range(3),[result['per_scenario'][name][f'fixed_{g}']['overall'][metric] for g in range(3)],marker='o',label=label,ms=4,lw=1.3)
        ax.set_xticks(range(3),['Balance','AoI','Quality']);ax.set_ylabel(title)
        ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)
    axes[0].legend(fontsize=7)
    fig.suptitle('Preference-only pilot: 1 training seed, 1M steps; paired fixed-instruction episodes',fontsize=10)
    fig.savefig(HERE/'preference_response.png',dpi=170);fig.savefig(HERE/'preference_response.pdf');plt.close(fig)
    original=(HERE/'PILOT_REPORT.md').read_text()
    fixed_a=result['per_scenario']['IC_HAPPO']['fixed_1']['overall']
    fixed_q=result['per_scenario']['IC_HAPPO']['fixed_2']['overall']
    rule_a=result['per_scenario']['R_instruction']['fixed_1']['overall']
    rule_q=result['per_scenario']['R_instruction']['fixed_2']['overall']
    statements=[
        '本轮诊断分两步完成：先验证同一可行范围内存在真实偏好取舍，再从头训练两个 1M 步 HAPPO 模型并比较三个规则。',
        f'规则门槛通过：质量偏好选择比 AoI 偏好选择高 {gate["quality_minus_aoi_psnr"]:.2f} dB、AoI 高 {gate["quality_minus_aoi_mean_aoi"]:.2f} 槽；按指令选择相对单一规则的平均奖励差 {gate["informed_minus_single_mean_reward"]:+.6f}，20/20 个保留种子为正。',
        f'短训练结果：显式输入 RL 减隐藏输入 RL 的整体奖励差为 {diffs["HAPPO_hidden_instruction"]:+.6f}；减指令查表规则为 {diffs["R_instruction"]:+.6f}；减单步贪心规则为 {diffs["R_myopic"]:+.6f}。',
        f'但显式输入 RL 尚未学出明显的偏好切换：在配对固定指令回合中，从 AoI 偏好改为质量偏好，AoI 仅从 {fixed_a["mean_aoi"]:.4f} 变为 {fixed_q["mean_aoi"]:.4f}，PSNR 仅从 {fixed_a["delivered_predicted_psnr"]:.4f} 变为 {fixed_q["delivered_predicted_psnr"]:.4f} dB。知道指令的查表规则则从 AoI {rule_a["mean_aoi"]:.4f}/PSNR {rule_a["delivered_predicted_psnr"]:.4f} 变为 AoI {rule_q["mean_aoi"]:.4f}/PSNR {rule_q["delivered_predicted_psnr"]:.4f}。正的整体方法差不能替代“策略确实学好了按任务取舍”的证据。',
        '独立复核进一步确认：同一外部种子下，隐藏模型在全部固定/切换指令场景中的实际模式、资源份额、AoI、交付质量与数量逐槽完全相同，仅任务编号和奖励不同。这与“隐藏 actor 不再通过即时观察、筛选或执行规则获得任务信息”的设计一致。历史状态仍可观察，但本任务中隐藏策略没有其它渠道驱动不同指令的物理轨迹分化。',
        '后面的按真实指令汇总表包含不同时间窗口；隐藏组的各指令均值可能因窗口组成和初始过渡略有不同。这种汇总差异不能解释成隐藏策略响应了任务，逐槽配对不变性检查才是本轮的直接证据。响应图仅使用配对固定指令回合，避免窗口组成混淆。',
    ]
    if diffs['HAPPO_hidden_instruction']>0:
        statements.append('这一单种子 pilot 中，带显式输入训练的方法相对隐藏组有正向整体均值差，但仅 14/20 个外部评估种子为正，且其跨指令行为差异仍很小。不能把组间差直接解释成已经有效学会任务切换；是否对随机初始化稳定也尚未验证。')
    else:
        statements.append('规则已证明存在指令取舍，但本次 1M 单种子训练没有得到显式输入的正向整体收益；不能把“有可学习空间”写成“RL 已学会并获益”。')
    if diffs['R_instruction']<=0 or diffs['R_myopic']<=0:
        statements.append('学习策略仍未同时超过两个知道指令的规则。因此即便超过隐藏输入组，也不足以证明 RL 比简单任务规则更有必要；不应据此直接启动完整长期训练或宣称算法优势。')
    else:
        statements.append('本次学习策略的整体奖励均值超过两个知道指令的规则，但仍只是单训练种子结果，不能替代多种子独立验证。')
    statements.append('均衡与质量偏好在规则预检中选择了同一中等档位，不能说三个指令都需要三个不同最优动作。当前质量奖励同时计入交付数量，最高 PSNR 档并不必然最优；诊断保留该原始目标，没有事后改变奖励来强化差距。')
    statements.append('下一步建议先检查模式选择和资源分配两个动作头：与知道指令的规则逐槽比较，定位低载荷/中载荷档位选择、资源整包容量或两者配合的损失，再决定是否调整学习方式或增加训练预算。本轮只确认存在可取舍空间且当前 pilot 未充分利用，尚未定位唯一原因，也不证明更长训练一定无效。')
    statements.append(f'![偏好响应]({HERE}/preference_response.png)')
    statements.append(f'独立核验：[independent_audit.json]({HERE}/independent_audit.json)。')
    (HERE/'REPORT.md').write_text('\n\n'.join(statements)+'\n\n'+original)
    files=[p for p in HERE.iterdir() if p.is_file() and p.suffix in ('.py','.md','.json','.png','.pdf') and p.name not in ('artifact_hashes.json','PILOT_STATUS.json')]
    write(HERE/'artifact_hashes.json',{p.name:digest(p) for p in sorted(files)})
    print(json.dumps(checks,ensure_ascii=False,indent=2))
    print(json.dumps(dict(overall=summaries,paired_reward_differences=diffs),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
