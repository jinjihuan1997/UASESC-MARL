"""Summarize fixed-state policy sensitivity, without treating it as performance."""
import csv
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys

for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[name]='1'
sys.dont_write_bytecode=True
os.environ.setdefault('MPLCONFIGDIR','/tmp/tmc_instruction_probe_mpl')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parent.parent
RUN=PROJECT/'experiments/2026-09-09_instruction_long_training/runs/three_seed_sc_20260909'
VARIANTS=['original']+[f'{k}_{g}' for k in ('full','id','limits') for g in range(3)]+['zero']
PAIRS={f'{kind}_{a}_to_{b}':(VARIANTS.index(f'{kind}_{a}'),VARIANTS.index(f'{kind}_{b}'))
       for kind in ('full','id','limits') for a,b in [(0,1),(0,2),(1,2)]}
PAIRS['original_to_zero']=(0,VARIANTS.index('zero'))


def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v): Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def metrics(z,i,j,selection):
    active=z['cache_active'][selection]
    first=z['mode'][selection,i];second=z['mode'][selection,j]
    q=z['mode_quality'][selection];load=z['mode_load'][selection]
    q1=np.take_along_axis(q,first[:,:,None],axis=-1).squeeze(-1)
    q2=np.take_along_axis(q,second[:,:,None],axis=-1).squeeze(-1)
    l1=np.take_along_axis(load,first[:,:,None],axis=-1).squeeze(-1)
    l2=np.take_along_axis(load,second[:,:,None],axis=-1).squeeze(-1)
    beta=np.abs(z['beta'][selection,j]-z['beta'][selection,i])
    raw_tv=.5*np.abs(z['raw_prob'][selection,j]-z['raw_prob'][selection,i]).sum(-1)
    masked_tv=.5*np.abs(z['masked_prob'][selection,j]-z['masked_prob'][selection,i]).sum(-1)
    physical=(np.abs(q2-q1)>1e-6)|(np.abs(l2-l1)>1e-6)
    return dict(states=int(len(beta)),cached_uav_decisions=int(active.sum()),
        nominal_mode_change_fraction=float(np.mean((first!=second)[active])),
        physical_mode_change_fraction=float(np.mean(physical[active])),
        raw_probability_tv_mean=float(raw_tv[active].mean()),masked_probability_tv_mean=float(masked_tv[active].mean()),
        raw_probability_tv_over_1e_5_fraction=float(np.mean(raw_tv[active]>1e-5)),
        masked_probability_tv_over_1e_5_fraction=float(np.mean(masked_tv[active]>1e-5)),
        mean_max_beta_change_pp=float(beta.max(-1).mean()*100),
        max_beta_change_pp=float(beta.max()*100),
        beta_at_least_1pp_state_fraction=float(np.mean(beta.max(-1)>=.01)),
        proposed_quality_delta_mean_db=float((q2-q1)[active].mean()),
        proposed_load_delta_mean=float((l2-l1)[active].mean()),
        higher_quality_choice_fraction=float(np.mean((q2-q1)[active]>1e-6)),
        lower_quality_choice_fraction=float(np.mean((q2-q1)[active]<-1e-6)))


def stats(v):
    return dict(mean=float(np.mean(v)),sd=float(np.std(v,ddof=1)),min=float(np.min(v)),max=float(np.max(v)),values=v)


def main():
    manifest=read(RUN/'manifest.json')
    changed=[k for k,v in manifest['input_hashes'].items() if sha(RUN/k)!=v]
    assert not changed,changed
    result={};audits={};all_fallbacks=[]
    mode_pairs=Counter();mode_decisions=0;max_load_difference=0.;max_quality_difference=0.
    for method in ('IC_HAPPO','IC_MAPPO'):
        for seed in manifest['seeds']:
            item=f'seed_{seed}/{method}';out=HERE/'runs'/item
            state=read(out/'status.json');identity=read(out/'identity.json')
            assert state['state']=='complete' and state['scenarios_completed']==13
            assert identity['script_sha256']==sha(HERE/'probe.py')
            assert identity['protocol_sha256']==sha(HERE/'PROTOCOL.md')
            assert identity['manifest_sha256']==sha(RUN/'manifest.json')
            hidden=f'seed_{seed}/{method[3:]}_hidden_instruction'
            for key,name in [('model_hashes',item),('hidden_model_hashes',hidden)]:
                for file,expected in identity[key].items(): assert sha(RUN/'jobs'/name/file)==expected
            values={};scenarios=[];maximum_errors={};hashes={}
            for scenario in manifest['scenarios']:
                meta=read(out/f'{scenario}.json');file=out/f'{scenario}.npz'
                assert sha(file)==meta['data_sha256']
                assert meta['native_forward_identical'] and meta['non_context_observation_unchanged'] and meta['negative_control_exactly_invariant']
                hashes[str(file.relative_to(HERE))]=meta['data_sha256']
                for path,expected in meta['trace_hashes'].items(): assert sha(RUN/path)==expected
                with np.load(file,allow_pickle=False) as z:
                    for key in z.files: values.setdefault(key,[]).append(z[key].copy())
                    scenarios.extend([scenario]*len(z['slot']))
                for key,value in meta['maximum_errors'].items(): maximum_errors[key]=max(maximum_errors.get(key,0),value)
                all_fallbacks.extend(dict(item=item,scenario=scenario,**x) for x in meta['single_state_fallbacks'])
            z={k:np.concatenate(v,axis=0) for k,v in values.items()}
            assert len(z['slot'])==state['probes']==16080
            assert np.allclose(z['masked_prob'].sum(-1),1.,atol=1e-6)
            assert np.all(z['masked_prob'][np.broadcast_to(z['available'][:,None]==0,z['masked_prob'].shape)]==0)
            assert np.allclose(z['beta'].sum(-1),1.,atol=1e-12)
            active=z['cache_active'];first=z['mode'][:,2];second=z['mode'][:,3]
            q=z['mode_quality'];l=z['mode_load']
            dq=np.take_along_axis(q,second[:,:,None],-1).squeeze(-1)-np.take_along_axis(q,first[:,:,None],-1).squeeze(-1)
            dl=np.take_along_axis(l,second[:,:,None],-1).squeeze(-1)-np.take_along_axis(l,first[:,:,None],-1).squeeze(-1)
            max_load_difference=max(max_load_difference,float(np.abs(dl[active]).max()))
            max_quality_difference=max(max_quality_difference,float(np.abs(dq[active]).max()))
            mode_decisions+=int(active.sum())
            effective=active & ((np.abs(dq)>1e-6)|(np.abs(dl)>1e-6))
            mode_pairs.update(zip(first[effective].tolist(),second[effective].tolist()))
            result[item]={}
            scopes={'all':np.ones(len(z['slot']),bool)}
            scopes.update({f'true_instruction_{g}':z['true_gid']==g for g in range(3)})
            scopes.update({s:np.asarray(scenarios)==s for s in manifest['scenarios']})
            for scope,sel in scopes.items():
                result[item][scope]={pair:metrics(z,i,j,sel) for pair,(i,j) in PAIRS.items()}
            audits[item]=dict(**state,maximum_errors=maximum_errors,data_hashes=hashes,
                hidden_method=hidden,single_state_inference_fallbacks=int(np.sum(z['inference_batch_size']==1)))
            print(item, result[item]['all']['full_1_to_2'],flush=True)
    grouped={}
    for method in ('IC_HAPPO','IC_MAPPO'):
        grouped[method]={}
        for scope in next(iter(result.values())):
            grouped[method][scope]={}
            for pair in PAIRS:
                rows=[result[f'seed_{s}/{method}'][scope][pair] for s in manifest['seeds']]
                grouped[method][scope][pair]={k:stats([r[k] for r in rows]) for k in rows[0]}
    audit=dict(state='PASS',source_manifest_sha256=sha(RUN/'manifest.json'),
        completed_state_carrier_models=6,completed_negative_control_models=6,
        replayed_episodes=1560,replayed_slots=sum(v['replay_slots'] for v in audits.values()),
        sampled_states=sum(v['probes'] for v in audits.values()),variants=VARIANTS,
        negative_controls_exactly_invariant=True,models_and_frozen_inputs_unchanged=True,
        all_original_actions_and_replay_metrics_verified=True,single_state_fallbacks=all_fallbacks,items=audits)
    write(HERE/'summary.json',dict(by_training_seed=result,methods=grouped))
    write(HERE/'audit.json',audit)
    write(HERE/'mode_pair_check.json',dict(context_pair='full_1_to_2',cached_uav_decisions=mode_decisions,
        max_absolute_load_difference=max_load_difference,max_absolute_quality_difference_db=max_quality_difference,
        nonduplicate_mode_pairs={f'{a}->{b}':count for (a,b),count in sorted(mode_pairs.items())}))
    csvrows=[]
    for item,scopes in result.items():
        for scope,pairs in scopes.items():
            for pair,value in pairs.items(): csvrows.append(dict(item=item,scope=scope,pair=pair,**value))
    with (HERE/'sensitivity.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(csvrows[0]));w.writeheader();w.writerows(csvrows)
    render(grouped,audit)


def render(grouped,audit):
    def val(method,pair,key): return grouped[method]['all'][pair][key]['mean']
    fig,axes=plt.subplots(1,2,figsize=(10.8,4.5),layout='constrained')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
    keys=['physical_mode_change_fraction','mean_max_beta_change_pp']
    labels=['Different quality/load mode choices (%)','Mean maximum resource-share change (pp)']
    for ax,key,label in zip(axes,keys,labels):
        for offset,method,color in [(-.18,'IC_HAPPO','#186ca7'),(.18,'IC_MAPPO','#b76627')]:
            pairs=['full_1_to_2','id_1_to_2','limits_1_to_2']
            scale=100 if key.endswith('fraction') else 1
            means=[val(method,p,key)*scale for p in pairs]
            sds=[grouped[method]['all'][p][key]['sd']*scale for p in pairs]
            bars=ax.bar(np.arange(3)+offset,means,width=.34,label=method,color=color,yerr=sds,capsize=3)
            ax.bar_label(bars,fmt='%.2f',fontsize=8,padding=4)
        ax.set_xticks(range(3),['ID + limits','ID only','Limits only'])
        ax.set_ylabel(label);ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
        ax.set_ylim(0,max(ax.get_ylim()[1]*1.15,.1))
    axes[0].legend(fontsize=8)
    fig.suptitle('Shown AoI vs quality context: physical state and feasible actions held fixed',fontsize=11)
    fig.supxlabel('Means ± SD across 3 training seeds. Actor sensitivity; no performance claim.',fontsize=9)
    fig.savefig(HERE/'sensitivity.png',dpi=170);fig.savefig(HERE/'sensitivity.pdf');plt.close(fig)
    lines=[
        '第一步已完成：冻结策略的同状态显式指令输入干预。该实验检查“输入是否影响策略输出”，不重训练、不测新的闭环回报。',
        '',
        '本次有直接证据表明：当前 policy 的输出会受到显式指令输入影响，即使真实任务和模式筛选完全固定，资源份额仍然会改变。因此不能把全部指令相关变化都归因于模式筛选。与此同时，固定候选范围后，真正不同质量/载荷的模式切换很少；“有反应”和“有性能收益”仍然是两个问题。',
        '',
        '共检查 IC_HAPPO、IC_MAPPO 各 3 个最终模型，以及对应的 6 个隐藏指令阴性对照。重放 1560 回合、936000 槽，在 96480 个决策前状态上逐一固定物理状态、可行动作掩码、当前资源预算、真实任务和其他观察。每个状态构造 11 种显示输入，记录筛选前/后概率、提出模式和资源份额。',
        '',
        '下表只比较同一状态下“显示 AoI 优先”与“显示质量优先”，按每个训练种子汇总后取三种子平均。只替换显式编号或门限的测试用于区分输入来源；门限指质量要求和 AoI 上限。',
        '',
        '| 模型 | 改哪些显式输入 | 模式编号变化 | 质量/载荷不同的模式变化 | 资源份额最大变化的平均值 | 至少一架 UAV 份额变化 ≥1 个百分点的状态 |',
        '|---|---|---:|---:|---:|---:|']
    for method in grouped:
        for pair,label in [('full_1_to_2','编号＋门限'),('id_1_to_2','只改编号'),('limits_1_to_2','只改门限')]:
            lines.append(f'| {method} | {label} | {100*val(method,pair,"nominal_mode_change_fraction"):.3f}% | {100*val(method,pair,"physical_mode_change_fraction"):.3f}% | {val(method,pair,"mean_max_beta_change_pp"):.4f} 个百分点 | {100*val(method,pair,"beta_at_least_1pp_state_fraction"):.3f}% |')
    lines += ['',f'![策略输入敏感性]({HERE}/sensitivity.png)','',
        '模式变化率以有待传缓存的 UAV 决策为分母。第二项模式变化率依据该状态下实际表查询的质量/载荷判断，排除模式编号不同但实际质量和载荷相同的情况。资源指标先在每个状态计算三个 UAV 中最大的份额绝对变化，再在状态内平均。一个百分点表示份额从 0.33 到 0.34。模式是策略提出的选择，不是新增仿真中已经执行或成功交付的动作。',
        '',
        '概率层面的完整任务输入变化：',
        '',
        '| 模型 | 筛选前概率总变差均值 | 筛选后概率总变差均值 | 提出模式的质量平均变化（质量输入−AoI 输入） |',
        '|---|---:|---:|---:|']
    for method in grouped:
        lines.append(f'| {method} | {val(method,"full_1_to_2","raw_probability_tv_mean"):.6f} | {val(method,"full_1_to_2","masked_probability_tv_mean"):.6f} | {val(method,"full_1_to_2","proposed_quality_delta_mean_db"):+.6f} dB |')
    lines += ['',
        '进一步逐项检查了 289440 个有缓存的 UAV 决策：完整任务输入从 AoI 换成质量时，没有一次改变所选模式的载荷。排除完全重复的模式后，只有 14→10（1475 次）和 10→14（1582 次）；这两个模式载荷相同，质量相差 0.44 dB。因此本轮固定筛选后的模式变化并没有形成低载荷与高载荷档之间的大幅切换，变化方向也不都是质量提高。该结论仅针对这些冻结模型和预定状态样本。',
        '',
        '概率总变差为两组概率差绝对值之和的一半，范围 0–1；大于零说明偏好分布变化，不要求最终最大概率模式一定改变。完整数值同时按真实任务、13 个场景和 3 个训练种子展开，见 summary.json。平均值不是对所有时隙均匀加权的系统指标：状态按每 10 槽及切换附近预定取样。',
        '',
        '隐藏指令组的 11 组名义干预保持输入为零，输出全部严格一致；固定状态和候选动作的断言全部通过。模型文件和冻结输入哈希未改变。原始输入的动作与历史轨迹核对通过，重放的执行模式、AoI、交付量、质量、资源和共同奖励与原记录一致。',
        '',
        f'数值核验采用相同批次逐组推理；{len(audit["single_state_fallbacks"])} 个状态因批量概率接近打平而恢复历史单环境推理，所有干预在该状态下都使用相同单环境布局，并验证恢复历史提出模式。早期开发检查与来源保留在 development/，仅正式 runs/ 的六组完整数据进入报告。',
        '',
        '解读边界：这里改变显示给 actor 的输入，真实环境任务、掩码和待传载荷观察都固定。所以它能隔离显式输入对当前策略函数的影响；某些组合会与真实任务矛盾，可能超出训练分布。输出敏感不等于正确理解任务、不等于更优的长期回报，也不能覆盖原有完整训练消融的结果。',
        '',
        '下一步若要检验性能增益，需要在同一实际任务和外生轨迹下比较正确/隐藏/错误显式输入的闭环结果，并把错误输入导致的分布外表现与重新训练的消融区分；或者开展已讨论的共同可行动作范围下的独立训练实验。本轮没有执行这些后续步骤。',
        '',
        f'核验：[audit.json]({HERE}/audit.json)；逐种子及分场景结果：[summary.json]({HERE}/summary.json)；[协议]({HERE}/PROTOCOL.md)；[原始概率与动作文件]({HERE}/runs)。',
        '',
        '复现命令（项目根目录；完成的场景经哈希验证后复用）：','',
        '```bash',
        'export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1',
        'TMC_PY=/home/king/miniconda3/envs/harl_sionna/bin/python',
        'TMC_PROBE=experiments/2026-09-10_policy_instruction_probe',
        '"$TMC_PY" "$TMC_PROBE/probe.py" --algorithm happo --core 0',
        '"$TMC_PY" "$TMC_PROBE/probe.py" --algorithm mappo --core 2',
        '"$TMC_PY" "$TMC_PROBE/summarize.py"','```']
    (HERE/'REPORT.md').write_text('\n'.join(lines)+'\n')
    files=['PROTOCOL.md','probe.py','summarize.py','summary.json','audit.json','mode_pair_check.json','sensitivity.csv',
           'REPORT.md','sensitivity.png','sensitivity.pdf']
    write(HERE/'artifact_hashes.json',{f:sha(HERE/f) for f in files})


if __name__=='__main__': main()
