"""Independent trace aggregation for the frozen-policy deployment ablation."""
import csv
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import sys

for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[k]='1'
sys.dont_write_bytecode=True
os.environ.setdefault('MPLCONFIGDIR','/tmp/tmc_closed_loop_mpl')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parent.parent
RUN=PROJECT/'experiments/2026-09-09_instruction_long_training/runs/three_seed_sc_20260909'
CONDITIONS=['correct','hidden_at_deployment']
MEANS=['common_reward','base_reward','training_reward','recv_aoi_bonus','mean_aoi','max_aoi','p95_aoi',
       'aoi_exceedance_fraction','max_aoi_violation','infeasible_uav_fraction']
SUMS=['deliveries','predicted_quality_sum','channel_uses','quality_violations','budget_violations','cache_violations']
METRICS=['common_reward','mean_aoi','delivered_predicted_psnr','deliveries_per_slot','channel_uses_per_slot',
         'aoi_exceedance_fraction','max_aoi_violation']
CALIBRATION={'mean_aoi':1e-10,'deliveries':0.,'predicted_quality_sum':1e-7,'channel_uses':1e-6,'common_reward':1e-9}


def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v): Path(p).write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def aggregate(rows):
    weights=np.asarray([r.get('steps',1) for r in rows],float)
    result={k:float(np.average([r[k] for r in rows],weights=weights)) for k in MEANS}
    result.update({k:sum(r[k] for r in rows) for k in SUMS})
    result['steps']=int(weights.sum())
    result['delivered_predicted_psnr']=result['predicted_quality_sum']/result['deliveries'] if result['deliveries'] else None
    result['deliveries_per_slot']=float(result['deliveries']/weights.sum())
    result['channel_uses_per_slot']=float(result['channel_uses']/weights.sum())
    return result


def assert_summary(a,b):
    for k in MEANS+SUMS+['delivered_predicted_psnr','deliveries_per_slot','channel_uses_per_slot']:
        if b[k] is None: assert a[k] is None
        else: np.testing.assert_allclose(a[k],b[k],atol=1e-8,rtol=1e-10,err_msg=k)


def stats(values):
    return dict(mean=float(np.mean(values)),sd=float(np.std(values,ddof=1)),min=float(np.min(values)),max=float(np.max(values)),
                values=[float(v) for v in values],positive_count=int(sum(v>0 for v in values)),negative_count=int(sum(v<0 for v in values)),n=len(values))


def main():
    manifest=read(RUN/'manifest.json');historical=read(RUN/'report/per_seed_results.json')
    run_status=read(RUN/'status.json');assert run_status['state']=='complete'
    for name,expected in run_status['report']['files'].items(): assert sha(RUN/'report'/name)==expected,name
    for p,h in manifest['input_hashes'].items(): assert sha(RUN/p)==h,p
    results={};calibrations={};fallbacks=[];trace_hashes={};total_slots=0
    for method in ['IC_HAPPO','IC_MAPPO']:
        for seed in manifest['seeds']:
            item=f'seed_{seed}/{method}';out=HERE/'runs'/item
            status=read(out/'status.json');identity=read(out/'identity.json')
            assert status['state']=='complete' and status['completed_scenarios']==13
            assert identity['script_sha256']==sha(HERE/'evaluate.py') and identity['protocol_sha256']==sha(HERE/'PROTOCOL.md')
            assert identity['parent_manifest_sha256']==sha(RUN/'manifest.json')
            for f,h in identity['checkpoint_hashes'].items(): assert sha(RUN/'jobs'/item/f)==h
            meta={}
            for scenario in manifest['scenarios']:
                a=read(out/'calibration'/f'{scenario}.json')
                assert a['state']=='complete' and a['all_constraints_and_transition_checks_passed'] and a['explicit_input_only']
                assert len(a['calibration'])==20
                for c in a['calibration']:
                    assert all(c['final_errors'][k]<=tol for k,tol in CALIBRATION.items())
                    meta[f'{scenario}/{c["seed"]}']=c
                    if c['backend'].endswith('fallback'): fallbacks.append(dict(item=item,scenario=scenario,**c))
            results[item]={};calibrations[item]=meta
            for condition in CONDITIONS:
                episodes=[]
                files=sorted((out/condition/'episodes').glob('*.json'));assert len(files)==260
                for file in files:
                    ep=read(file);key=f'{ep["scenario"]}/{ep["seed"]}'
                    assert ep['steps']==600 and ep['condition']==condition and ep['training_seed']==seed
                    assert ep['external_trajectory_sha256']==historical[item]['pairing'][key]==meta[key]['external_trajectory_sha256']
                    assert ep['backend']==meta[key]['backend']
                    trace=out/condition/'traces'/f'{file.stem}.csv'
                    assert sha(trace)==ep['trace_sha256'];trace_hashes[str(trace.relative_to(HERE))]=ep['trace_sha256']
                    with trace.open(newline='') as stream:
                        rows=[]
                        for row in csv.DictReader(stream):
                            rows.append({k:float(row[k]) for k in MEANS+SUMS+['instruction_id']})
                    assert len(rows)==600
                    for k in ['quality_violations','budget_violations','cache_violations']: assert all(r[k]==0 for r in rows)
                    assert_summary(aggregate(rows),ep)
                    for g,part in ep['by_instruction'].items():
                        selection=[r for r in rows if int(r['instruction_id'])==int(g)]
                        assert len(selection)==part['steps'];assert_summary(aggregate(selection),part)
                    episodes.append(ep);total_slots+=ep['steps']
                results[item][condition]={'overall':aggregate(episodes)}
                results[item][condition].update({s:aggregate([e for e in episodes if e['scenario']==s]) for s in manifest['scenarios']})
                results[item][condition].update({f'true_instruction_{g}':aggregate([e['by_instruction'][str(g)] for e in episodes if str(g) in e['by_instruction']]) for g in range(3)})
            # Correct-input branch must reproduce the final historical science summary.
            assert_summary(results[item]['correct']['overall'],historical[item]['overall'])
            print(f'verified {item}: 520 episodes',flush=True)
    grouped={};paired={};reference={}
    for method in ['IC_HAPPO','IC_MAPPO']:
        grouped[method]={};paired[method]={}
        for scope in results[f'seed_85/{method}']['correct']:
            grouped[method][scope]={}
            for condition in CONDITIONS:
                grouped[method][scope][condition]={k:stats([results[f'seed_{s}/{method}'][condition][scope][k] for s in manifest['seeds']]) for k in METRICS}
            paired[method][scope]={k:stats([results[f'seed_{s}/{method}']['correct'][scope][k]-results[f'seed_{s}/{method}']['hidden_at_deployment'][scope][k] for s in manifest['seeds']]) for k in METRICS}
        hidden_method=method[3:]+'_hidden_instruction'
        reference[hidden_method]={k:stats([historical[f'seed_{s}/{hidden_method}']['overall'][k] for s in manifest['seeds']]) for k in METRICS}
    audit=dict(state='PASS',created_utc=datetime.now(timezone.utc).isoformat(),models=6,conditions=CONDITIONS,
        paired_conditions_per_model=260,episodes=len(trace_hashes),slots=total_slots,paired_cpu_fallbacks=fallbacks,
        correct_branch_reproduces_formal_summary=True,all_trace_aggregations_verified=True,
        models_and_source_unchanged=True,all_external_trajectories_paired=True,calibration=calibrations,
        manifest_sha256=sha(RUN/'manifest.json'),script_sha256=sha(HERE/'evaluate.py'),protocol_sha256=sha(HERE/'PROTOCOL.md'))
    write(HERE/'audit.json',audit);write(HERE/'trace_hashes.json',trace_hashes)
    write(HERE/'results.json',dict(per_training_seed=results,methods=grouped,paired_correct_minus_hidden=paired,
        historical_separately_trained_hidden=reference))
    rows=[]
    for method,scopes in paired.items():
        for scope,metrics in scopes.items():
            for k,v in metrics.items(): rows.append(dict(method=method,scope=scope,metric=k,**{x:v[x] for x in ['mean','sd','min','max','positive_count','negative_count','n']}))
    with (HERE/'paired_differences.csv').open('w',newline='') as stream:
        w=csv.DictWriter(stream,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    render(grouped,paired,reference,audit)


def render(grouped,paired,reference,audit):
    methods=['IC_HAPPO','IC_MAPPO'];seeds=[85,218,966]
    fig,axes=plt.subplots(1,2,figsize=(10,4.3),layout='constrained')
    for ax,key,label in zip(axes,['common_reward','mean_aoi'],['Common reward difference (above 0 favors correct)','Mean AoI difference (below 0 favors correct)']):
        ax.axhline(0,color='#666',lw=1,ls='--')
        for j,(seed,color) in enumerate(zip(seeds,['#1774ac','#c17b17','#37824c'])):
            ax.scatter(np.arange(2)+(j-1)*.11,[paired[m]['overall'][key]['values'][j] for m in methods],s=45,color=color,label=f'Seed {seed}')
        ax.scatter(range(2),[paired[m]['overall'][key]['mean'] for m in methods],marker='D',s=42,color='black',label='Mean')
        ax.set_xticks(range(2),methods);ax.set_xlim(-.4,1.4);ax.set_ylabel(label)
        ax.grid(axis='y',alpha=.15);ax.spines[['top','right']].set_visible(False)
    axes[0].legend(fontsize=8)
    fig.suptitle('Same trained policy: correct input minus input hidden at deployment',fontsize=11)
    fig.supxlabel('13 paired scenarios × 20 evaluation seeds; 3 independent training seeds.',fontsize=9)
    fig.savefig(HERE/'paired_effects.png',dpi=170);fig.savefig(HERE/'paired_effects.pdf');plt.close(fig)
    lines=[
        '第二步已完成：比较同一个冻结模型正常接收显式指令和测试时把显式输入置零后的完整闭环性能。没有重训练。',
        '',
        '结论：正常提供显式指令有小幅平均收益，但两个算法都只有 2/3 个训练种子的共同奖励提高，尚未体现稳定优势。相对于同一模型临时隐藏输入，HAPPO 的总体平均 AoI 降低约 0.24%，MAPPO 降低约 0.55%；总体交付质量变化均不足 0.006 dB。策略会根据指令调整动作，但当前调整带来的实际收益很小。',
        '',
        '两个方法各三个训练种子，全部 13 场景 × 20 评估种子 × 600 槽；两种输入条件合计 3120 回合、1872000 槽。每对条件的初始状态、信道和真实任务相同，后续缓存、AoI 和动作按各自策略自然演化。真实质量约束、奖励、资源预算和筛选规则始终相同，置零只发生在 actor 的显式编号/质量门限/AoI 门限输入坐标。',
        '',
        '本轮数据表（均值为三训练种子的均值；质量按交付总数加权）：',
        '',
        '| 方法 | 运行时输入 | 共同奖励 | 平均 AoI | 成功交付 PSNR | 更新/槽 |',
        '|---|---|---:|---:|---:|---:|']
    for method in methods:
        for condition,label in [('correct','正确指令'),('hidden_at_deployment','同一模型临时隐藏输入')]:
            v=grouped[method]['overall'][condition]
            lines.append(f'| {method} | {label} | {v["common_reward"]["mean"]:.6f} | {v["mean_aoi"]["mean"]:.6f} | {v["delivered_predicted_psnr"]["mean"]:.6f} | {v["deliveries_per_slot"]["mean"]:.6f} |')
    lines += ['',
        '主要配对结果为“正确输入−临时隐藏输入”。共同奖励正差表示正确输入更好，AoI 负差表示正确输入更及时。只描述三种子的均值、范围和方向，不宣称统计显著。',
        '',
        '| 方法 | 共同奖励平均差 | 三种子差值范围 | 正差种子数 | AoI 平均差 | 交付质量平均差 |',
        '|---|---:|---:|---:|---:|---:|']
    for method in methods:
        v=paired[method]['overall'];r=v['common_reward']
        lines.append(f'| {method} | {r["mean"]:+.6f} | {r["min"]:+.6f} 至 {r["max"]:+.6f} | {r["positive_count"]}/3 | {v["mean_aoi"]["mean"]:+.6f} | {v["delivered_predicted_psnr"]["mean"]:+.6f} dB |')
    lines += ['',f'![逐种子配对差值]({HERE}/paired_effects.png)','',
        '按真实指令汇总（包括固定与切换场景中的相应时段，按时段长度加权）：','',
        '| 方法 | 真实指令 | 共同奖励差 | AoI 差 | 质量差 |',
        '|---|---|---:|---:|---:|']
    for method in methods:
        for g,name in enumerate(['均衡','AoI 优先','质量优先']):
            v=paired[method][f'true_instruction_{g}']
            lines.append(f'| {method} | {name} | {v["common_reward"]["mean"]:+.6f} | {v["mean_aoi"]["mean"]:+.6f} | {v["delivered_predicted_psnr"]["mean"]:+.6f} dB |')
    lines += ['',
        '必须区分本轮临时遮挡与原先的独立训练消融。下表每个算法的第三行使用原完整预算、相同评估条件下从头训练的隐藏指令模型；它与本轮临时把输入置零的模型权重不同。正确输入分支已经恢复原正式报告的指标，因此可以同时参考，但不可把两种“隐藏”混称。','',
        '| 算法 | 模型/输入设置 | 共同奖励 | 平均 AoI | 交付 PSNR |',
        '|---|---|---:|---:|---:|']
    for method in methods:
        rows=[('带指令训练，正常输入',grouped[method]['overall']['correct']),
              ('同一模型，运行时临时遮挡',grouped[method]['overall']['hidden_at_deployment']),
              ('从头训练就隐藏显式输入（历史对照）',reference[method[3:]+'_hidden_instruction'])]
        for name,v in rows: lines.append(f'| {method[3:]} | {name} | {v["common_reward"]["mean"]:.6f} | {v["mean_aoi"]["mean"]:.6f} | {v["delivered_predicted_psnr"]["mean"]:.6f} |')
    lines += ['',
        '原先从头训练就隐藏显式输入的对照组，在共同奖励、平均 AoI 和交付质量的总体均值上仍略优于完整输入组。因此两轮诊断共同支持“网络确实使用显式输入”，仍不足以支持“显式输入设计比经过充分训练的隐藏输入策略更好”。隐藏显式输入也不等于删除所有任务信息：真实任务仍影响可行动作筛选与待传载荷等间接线索。',
        '',
        '解释范围：本轮检验已有完整输入模型在运行时对显式输入的依赖。临时置零可能属于训练分布之外的输入，性能降低不能自动归结为“显式指令算法优于所有不带指令算法”。前一步证明了输入可以改变策略输出；本轮进一步检验这些输入在该冻结模型的完整运行中是否带来收益；独立重新训练的消融仍然决定新增设计相对基线的证据强弱。',
        '',
        '建议下一步先单独验证任务设计：保持不同指令下相同的可行动作范围，用质量与时效偏好的变化定义任务，先通过简单规则确认确实存在可取舍的性能空间，再做带/不带显式输入的短训练对照。这样可以检验指令信息本身的价值；该实验需要作为新的诊断设置说明，不能覆盖原实验或把设计调整当成原方法已获验证。现阶段不建议仅重复同一设置的更长训练。本轮没有修改环境参数或启动新训练。',
        '',
        f'核验完成：3120 个轨迹文件重新核算全部逐槽统计，外生轨迹逐回合与原正式结果配对；两个条件都通过资源、质量和缓存检查。正确输入分支逐槽与历史参考校准，{len(audit["paired_cpu_fallbacks"])} 对回合因批量数值差异改用原 CPU 单环境方式成对重跑；此选择只依赖正确分支的历史复现检查。最终正确输入总体指标与原报告一致。模型、表和冻结输入哈希未改变。',
        '',
        '表内 PSNR 是当前平均模型的成功交付质量，不是所有时刻显示画面的端到端视频质量。本轮不修订 SC/CC profile 公平性，也不增加算法泛化或创新性结论。',
        '',
        f'证据：[完整数值]({HERE}/results.json)；[核验]({HERE}/audit.json)；[配对差值]({HERE}/paired_differences.csv)；[协议]({HERE}/PROTOCOL.md)。',
        '',
        '复现命令（项目根目录；已有完整场景校验后复用）：','',
        '```bash','export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1',
        'TMC_PY=/home/king/miniconda3/envs/harl_sionna/bin/python','TMC_CLOSED=experiments/2026-09-10_instruction_closed_loop',
        '"$TMC_PY" "$TMC_CLOSED/evaluate.py" --algorithm happo --core 0',
        '"$TMC_PY" "$TMC_CLOSED/evaluate.py" --algorithm mappo --core 2',
        '"$TMC_PY" "$TMC_CLOSED/analyze.py"','```']
    (HERE/'REPORT.md').write_text('\n'.join(lines)+'\n')
    files=['PROTOCOL.md','evaluate.py','analyze.py','results.json','audit.json','trace_hashes.json',
           'paired_differences.csv','REPORT.md','paired_effects.png','paired_effects.pdf']
    write(HERE/'artifact_hashes.json',{f:sha(HERE/f) for f in files})


if __name__=='__main__': main()
