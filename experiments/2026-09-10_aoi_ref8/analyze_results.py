"""Post-run interpretation only; never changes frozen training inputs."""
from helpers import *


def main():
    verify();assert read(HERE/'report/audit.json')['state']=='PASS'
    artifacts=read(HERE/'report/artifact_hashes.json')
    for f,h in artifacts.items():assert digest(HERE/f)==h
    r=read(HERE/'report/results.json');g=r['grouped'];out={}
    for method in METHODS:
        a=g['ref10'][method]['overall'];b=g['ref8'][method]['overall']
        get=lambda v,k:v[k]['mean']
        age=lambda v:sum(get(v,k) for k in ['age_mean_cost','age_max_cost','age_tail_cost'])
        diff=dict(quality_credit=get(b,'quality_credit')-get(a,'quality_credit'),
            aoi_savings=1.25*age(a)-age(b),resource_savings=get(a,'resource_cost')-get(b,'resource_cost'))
        expected=get(b,'objective_ref8')-get(a,'objective_ref8')
        np.testing.assert_allclose(sum(diff.values()),expected,atol=1e-12,rtol=0)
        response={}
        for o in OBJECTIVES:
            age_task=g[o][method]['fixed_1'];quality_task=g[o][method]['fixed_2']
            response[o]=dict(quality_minus_aoi_psnr=get(quality_task,'delivered_predicted_psnr')-get(age_task,'delivered_predicted_psnr'),
                quality_minus_aoi_age=get(quality_task,'mean_aoi')-get(age_task,'mean_aoi'))
        out[method]=dict(delta_components_under_ref8=diff,total_delta_ref8=expected,
            paired_overall=r['paired_ref8_minus_ref10'][method]['overall'],fixed_instruction_response=response)
    modes={}
    for o in OBJECTIVES:
        modes[o]={}
        for method in METHODS:
            modes[o][method]={}
            for scenario in ['fixed_0','fixed_1','fixed_2']:
                values=[]
                for seed in SEEDS:
                    f=HERE/'evaluation'/o/f'seed_{seed}'/method/f'{scenario}.npz'
                    assert digest(f)==read(f.with_suffix('.json'))['trace_sha256']
                    with np.load(f) as z:m=z['modes']
                    values.append(dict(low=float(np.isin(m,[0,4,8,12]).mean()),middle=float(np.isin(m,[5,9,13]).mean()),
                        no_send=float((m<0).mean()),other=float((~np.isin(m,[-1,0,4,8,12,5,9,13])).mean())))
                modes[o][method][scenario]={k:float(np.mean([v[k] for v in values])) for k in values[0]}
    for seed in SEEDS:
        for method in METHODS:
            a=read(HERE/f'jobs/ref10/seed_{seed}/{method}/manifest.json')
            b=read(HERE/f'jobs/ref8/seed_{seed}/{method}/manifest.json')
            assert a['initial_actor_hashes']==b['initial_actor_hashes'] and a['initial_critic_hash']==b['initial_critic_hash']
    write(HERE/'analysis_results.json',dict(methods=out,executed_modes=modes,initial_models_paired=True,
        scientific_status='exploratory_three_seed_pilot',report_hashes=artifacts))
    lines=['# 参考值8：新训练效果解释','',
        '下列分数全部采用参考值8，均为同一组新测试条件。旧模型为参考10训练，新模型为参考8训练，各100万步、3个训练种子。','',
        '| 方法 | 旧模型分数×100 | 新模型分数×100 | 新减旧×100 | 正差种子 |','|---|---:|---:|---:|---:|']
    for method in METHODS:
        d=r['paired_ref8_minus_ref10'][method]['overall']['objective_ref8']
        lines.append(f'| {method} | {100*g["ref10"][method]["overall"]["objective_ref8"]["mean"]:.4f} | {100*g["ref8"][method]["overall"]["objective_ref8"]["mean"]:.4f} | {100*d["mean"]:+.4f} | {d["positive_count"]}/3 |')
    lines+=['','## 改善或损失来自哪一项','',
        '| 方法 | 质量收益变化×100 | AoI节省变化×100 | 资源节省变化×100 |','|---|---:|---:|---:|']
    for method in METHODS:
        d=out[method]['delta_components_under_ref8']
        lines.append(f'| {method} | {100*d["quality_credit"]:+.4f} | {100*d["aoi_savings"]:+.4f} | {100*d["resource_savings"]:+.4f} |')
    lines+=['','## 指令区分度','',
        '| 方法 | 训练参考 | 质量指令减AoI指令：PSNR差 | 对应AoI差 |','|---|---|---:|---:|']
    for method in METHODS:
        for o,v in out[method]['fixed_instruction_response'].items():
            lines.append(f'| {method} | {o} | {v["quality_minus_aoi_psnr"]:+.4f} dB | {v["quality_minus_aoi_age"]:+.4f} |')
    lines+=['','更大的指令响应不自动等于更高绝对性能。规则也按新目标重新校准；单一规则在新目标下改为m0_urgency。所有结论限于当前三种子100万步，不能替代长期训练或统计显著性。',
        '',f'[正式完整表]({HERE}/REPORT.md)；[独立审计]({HERE}/report/audit.json)；[完整解释数据]({HERE}/analysis_results.json)。']
    (HERE/'ANALYSIS.md').write_text('\n'.join(lines)+'\n')
    write(HERE/'analysis_hashes.json',{str(p.relative_to(HERE)):digest(p) for p in [HERE/'analyze_results.py',HERE/'analysis_results.json',HERE/'ANALYSIS.md']})
    print(json.dumps(out,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
