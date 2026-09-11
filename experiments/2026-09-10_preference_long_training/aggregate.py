"""Independently aggregate saved traces and paired training-seed comparisons."""
from helpers import *


def aggregate(parts):
    data=np.concatenate([v.reshape(-1,9) for v in parts]);steps=len(data);q=data[:,2].sum();d=data[:,3].sum()
    return dict(steps=steps,common_reward=float(data[:,0].mean()),mean_aoi=float(data[:,1].mean()),
        predicted_quality_sum=float(q),deliveries=int(d),channel_uses=float(data[:,4].sum()),
        delivered_predicted_psnr=float(q/d),deliveries_per_slot=float(d/steps),channel_uses_per_slot=float(data[:,4].mean()))


def check(a,b):
    for k in a:np.testing.assert_allclose(a[k],b[k],atol=1e-8,rtol=1e-10,err_msg=k)


def stats(values):
    return dict(mean=float(np.mean(values)),sd=float(np.std(values,ddof=1)) if len(values)>1 else None,values=[float(v) for v in values],positive_count=sum(float(v)>0 for v in values))


def main():
    manifest=read(HERE/'manifest.json')
    for p,h in manifest['input_hashes'].items():assert digest(HERE/p)==h,p
    items=[j['id'] for j in manifest['jobs']]+['rules/'+r for r in manifest['rules']]
    all_results={};pairing=None;episodes=slots=0;hidden_checks=0
    metrics=['common_reward','mean_aoi','delivered_predicted_psnr','deliveries_per_slot','channel_uses_per_slot']
    for item in items:
        folder=HERE/'evaluation'/item;status=read(folder/'status.json');summary=read(folder/'summary.json')
        assert status['state']=='complete' and digest(folder/'summary.json')==status['summary_sha256']
        if pairing is None:pairing=summary['pairing']
        assert pairing==summary['pairing']
        all_results[item]={};parts=[];fixed_reference=None
        for scenario,schedule in manifest['scenarios'].items():
            path=folder/f'{scenario}.npz';assert digest(path)==summary['traces'][scenario]
            with np.load(path) as stored:
                data=stored['trace'];modes=stored['modes'];beta=stored['resource_fractions']
                np.testing.assert_array_equal(stored['seeds'],manifest['evaluation_seeds'])
            assert data.shape==(600,20,9) and np.isfinite(data).all() and not np.any(data[:,:,6:])
            for slot in range(600):assert np.all(data[slot,:,5]==next(g for t,g in reversed(schedule) if t<=slot))
            actual=aggregate([data]);check(actual,summary['scenarios'][scenario]['overall'])
            for i in range(20):check(aggregate([data[:,i]]),summary['scenarios'][scenario]['by_evaluation_seed'][i])
            for g,v in summary['scenarios'][scenario]['by_instruction'].items():check(aggregate([data[data[:,:,5]==int(g)]]),v)
            all_results[item][scenario]=actual;parts.append(data);episodes+=20;slots+=12000
            if item.endswith('HAPPO_hidden_instruction'):
                if fixed_reference is None:fixed_reference=(data.copy(),modes.copy(),beta.copy())
                np.testing.assert_array_equal(data[:,:,1:5],fixed_reference[0][:,:,1:5]);np.testing.assert_array_equal(modes,fixed_reference[1]);np.testing.assert_array_equal(beta,fixed_reference[2]);hidden_checks+=20
        all_results[item]['overall']=aggregate(parts)
        for g in range(3):all_results[item][f'true_instruction_{g}']=aggregate([v[v[:,:,5]==g] for v in parts if np.any(v[:,:,5]==g)])
    grouped={}
    for method in manifest['methods']+manifest['rules']:
        ids=[f'seed_{s}/{method}' for s in manifest['seeds']] if method in manifest['methods'] else ['rules/'+method]
        grouped[method]={scope:{m:stats([all_results[j][scope][m] for j in ids]) for m in metrics} for scope in all_results[ids[0]]}
    paired={}
    for other in ['HAPPO_hidden_instruction']+manifest['rules']:
        paired[other]={scope:{m:stats([all_results[f'seed_{s}/IC_HAPPO'][scope][m]-all_results[f'seed_{s}/{other}' if other=='HAPPO_hidden_instruction' else 'rules/'+other][scope][m] for s in manifest['seeds']]) for m in metrics} for scope in all_results['seed_85/IC_HAPPO']}
    out=HERE/'report';write(out/'results.json',dict(per_item=all_results,grouped=grouped,paired_ic_minus=paired))
    audit=dict(state='PASS',episodes=episodes,slots=slots,all_traces_reaggregated=True,external_pairing=True,hidden_invariant_episode_checks=hidden_checks,input_hashes_verified=True)
    write(out/'audit.json',audit)
    lines=['三种子长训练与评估完成。两方法各3个模型，每模型从头训练1000万步；固定末次模型评估。','',
        '本任务共享质量底线、动作范围与执行规则，仅改变奖励偏好。它与原来的指令硬门槛任务不同，不能混为同一设置的排名。','',
        '| 方法 | 平均奖励 | 平均 AoI | 交付预测 PSNR | 更新/槽 |','|---|---:|---:|---:|---:|']
    for method in manifest['methods']+manifest['rules']:
        v=grouped[method]['overall'];lines.append(f'| {method} | {v["common_reward"]["mean"]:.6f} | {v["mean_aoi"]["mean"]:.4f} | {v["delivered_predicted_psnr"]["mean"]:.4f} | {v["deliveries_per_slot"]["mean"]:.4f} |')
    lines+=['','| 显式输入 RL 减对照 | 三种子奖励平均差 | 样本标准差 | 正差训练种子 |','|---|---:|---:|---:|']
    for other,v in paired.items():
        r=v['overall']['common_reward'];lines.append(f'| {other} | {r["mean"]:+.6f} | {r["sd"]:.6f} | {r["positive_count"]}/3 |')
    lines+=['','| 方法 | 固定偏好 | 平均 AoI | 交付预测 PSNR |','|---|---|---:|---:|']
    for method in manifest['methods']+manifest['rules']:
        for g,label in enumerate(['均衡','AoI','质量']):
            v=grouped[method][f'fixed_{g}'];lines.append(f'| {method} | {label} | {v["mean_aoi"]["mean"]:.4f} | {v["delivered_predicted_psnr"]["mean"]:.4f} |')
    lines+=['','三训练种子是独立训练的重复；规则只有一套评估，20个外部评估种子不当作20个训练种子。不据描述性均值自动宣称显著性、创新性或跨设置泛化。',
        '',f'训练布局：{manifest["resources"]["layout"]}。同种子内的显式/隐藏模型使用相同训练后端。',
        '',f'核验完成：{episodes}回合，{slots}槽；全部逐槽统计重新聚合、外生条件配对、输入哈希通过。',
        '',f'完整数值：[results.json]({out}/results.json)；核验：[audit.json]({out}/audit.json)。']
    (HERE/'REPORT.md').write_text('\n'.join(lines)+'\n')
    write(out/'artifact_hashes.json',{str(p.relative_to(HERE)):digest(p) for p in [out/'results.json',out/'audit.json',HERE/'REPORT.md']})


if __name__=='__main__':main()
