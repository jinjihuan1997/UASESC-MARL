"""Aggregate all intervention traces without changing frozen science inputs."""
from diagnose import *


def stats(values):
    return dict(mean=float(np.mean(values)),sd=float(np.std(values,ddof=1)),values=[float(x) for x in values],positive_count=int(np.sum(np.asarray(values)>0)))


def main():
    manifest=verify_parent();seeds=[85,218,966];results={};all_hashes={};episodes=slots=0
    for seed in seeds:
        folder=OUT/f'seed_{seed}';identity=read(folder/'identity.json');audit=read(folder/'audit.json')
        assert audit['state']=='PASS' and identity['script_sha256']==digest(OUT/'diagnose.py')
        assert identity['protocol_sha256']==digest(OUT/'PROTOCOL.md') and identity['parent_manifest_sha256']==digest(LONG/'manifest.json')
        for f,h in identity['checkpoint_hashes'].items():assert digest(LONG/'jobs'/f'seed_{seed}/IC_HAPPO'/f)==h
        saved=read(folder/'results.json');results[seed]={}
        for condition in CONDITIONS:
            results[seed][condition]={};parts=[]
            for scenario in manifest['scenarios']:
                path=folder/condition/f'{scenario}.npz';key=str(path.relative_to(OUT));assert digest(path)==audit['trace_hashes'][key]
                with np.load(path) as z:data=z['trace'];assert data.shape==(600,20,9) and np.isfinite(data).all()
                actual=aggregate(data)
                for k,v in actual.items():np.testing.assert_allclose(v,saved[condition][scenario][k],rtol=1e-10,atol=1e-8)
                parts.append(data);results[seed][condition][scenario]=actual;episodes+=20;slots+=12000;all_hashes[key]=digest(path)
            merged=np.concatenate(parts);results[seed][condition]['overall']=aggregate(merged)
            for g in range(3):results[seed][condition][f'true_instruction_{g}']=aggregate(merged[merged[:,:,5]==g])
            for scope in ['overall']+[f'true_instruction_{g}' for g in range(3)]:
                for k,v in results[seed][condition][scope].items():np.testing.assert_allclose(v,saved[condition][scope][k],rtol=1e-10,atol=1e-8)
    grouped={condition:{scope:{k:stats([results[s][condition][scope][k] for s in seeds]) for k in results[seeds[0]][condition][scope]} for scope in results[seeds[0]][condition]} for condition in CONDITIONS}
    paired={condition:{scope:{k:stats([results[s][condition][scope][k]-results[s]['original'][scope][k] for s in seeds]) for k in results[seeds[0]][condition][scope]} for scope in results[seeds[0]][condition]} for condition in CONDITIONS[1:]}
    formal=read(LONG/'report/results.json');weights=np.asarray(read(LONG/'configs/seed_85/IC_HAPPO.json')['env_args']['reward_weights_by_instruction'])
    components={};histograms={}
    for g in range(3):
        scope=f'fixed_{g}'
        components[str(g)]={condition:dict(quality=float(weights[g,0]*grouped[condition][scope]['quality_term']['mean']),
            aoi=float(-weights[g,1]*grouped[condition][scope]['aoi_term']['mean']),load=float(-weights[g,2]*grouped[condition][scope]['load_term']['mean'])) for condition in CONDITIONS}
        for condition in CONDITIONS:np.testing.assert_allclose(sum(components[str(g)][condition].values()),grouped[condition][scope]['common_reward']['mean'],atol=1e-9,rtol=0)
        hist=np.zeros(16,dtype=int);deviations=[]
        for seed in seeds:
            with np.load(OUT/f'seed_{seed}/original/{scope}.npz') as z:
                mode=z['modes'];hist+=np.bincount(mode[mode>=0],minlength=16);deviations.append(np.abs(z['resource_fractions']-1/3).mean())
        histograms[str(g)]=dict(mode_fractions=(hist/hist.sum()).tolist(),mean_abs_resource_deviation_percentage_points=float(np.mean(deviations)*100))
    report=dict(grouped=grouped,paired_intervention_minus_original=paired,reward_components_fixed=components,original_mode_statistics=histograms)
    write(OUT/'results.json',report)
    write(OUT/'audit.json',dict(state='PASS',episodes=episodes,slots=slots,trace_hashes=all_hashes,all_saved_statistics_reaggregated=True,
        original_branch_reproduces_formal=True,models_unchanged=True,interventions='deployment_only_no_training'))
    lines=[
        '本轮把“RL整体奖励低于规则”的差距定位到具体任务和控制环节。没有训练模型，也没有修改旧实验。',
        '',
        '最明确的结果在质量优先任务：只把RL的资源分配改成均分，保留RL模式策略，三个训练种子的奖励都改善，整体均值几乎追平R_instruction；只把模式改成规则，奖励基本不变。这支持质量任务中的主要可避免损失来自当前部署资源分配及其与模式的配合。',
        '',
        '| 质量优先的控制方式 | 共同奖励 | AoI | 交付预测PSNR | 更新/槽 |',
        '|---|---:|---:|---:|---:|']
    for condition,label in [('original','原RL'),('equal_resources','仅资源改均分'),('rule_mode','仅模式改规则')]:
        v=grouped[condition]['fixed_2'];lines.append(f'| {label} | {v["common_reward"]["mean"]:.6f} | {v["mean_aoi"]["mean"]:.4f} | {v["delivered_predicted_psnr"]["mean"]:.4f} | {v["deliveries_per_slot"]["mean"]:.4f} |')
    for method in ['R_instruction','R_myopic']:
        v=formal['grouped'][method]['fixed_2'];lines.append(f'| {method} | {v["common_reward"]["mean"]:.6f} | {v["mean_aoi"]["mean"]:.4f} | {v["delivered_predicted_psnr"]["mean"]:.4f} | {v["deliveries_per_slot"]["mean"]:.4f} |')
    lines+=['','质量任务中RL有约99.95%的执行模式落在5/9/13这组完全相同的中等档位。R_instruction使用5，故不能把不同编号误判为不同编码质量。RL资源份额偏离均分的平均绝对值约2个百分点。在整块传输中，某UAV少分一点资源可能少发一块，而别处多出的资源未必足以多发一块；环境逐块扣预算而非连续传输比例。均分干预的实际结果与这一容量机制相符，但尚未单独分解每个未发块的唯一原因。',
        '', '| 干预减原RL：奖励差 | 全部场景 | 固定均衡 | 固定AoI | 固定质量 |','|---|---:|---:|---:|---:|']
    for condition in CONDITIONS[1:]:
        values=[paired[condition][scope]['common_reward']['mean'] for scope in ['overall','fixed_0','fixed_1','fixed_2']]
        lines.append('| '+condition+' | '+' | '.join(f'{v:+.6f}' for v in values)+' |')
    lines+=['',
        '不能把全部任务统一改为均分：AoI任务中该替换的奖励在3个种子上均下降；总体只在1/3种子上改善。只替换模式的总体奖励在3个种子上均下降。这说明学习到的两个动作部分存在配合，不能将两个干预效果相加。',
        '',
        'RL也并非在所有任务上输给所有规则。正式固定AoI任务中，RL奖励-0.112749高于R_instruction的-0.114496；按全部场景的真实任务时段分解，相对R_instruction的主要负贡献来自质量优先，而AoI时段是正贡献。',
        '',
        '规则较强的结构性原因：当前平均表只有三类不同的非支配质量—载荷选择；缓存服务顺序由共同AoI调度器确定；均衡/质量下简单规则使用中档和均分，AoI下使用低档。R_myopic更进一步，用当前全局缓存/AoI及精确一步环境模型比较9个候选动作。分散RL actor只接收各自局部或汇总观察，因此R_myopic比较包含信息与模型计算条件差异；R_instruction仍胜出说明差距也不能全部归因于这项优势。',
        '',
        '当前UAV观察中的预算由已有资源份额计算，而同槽SUT动作会再次改变份额，之后执行器才按新预算选模式并发送。这是潜在的协调难点。训练随机动作与确定性评估使用均值的差异也可能影响整块容量，但本轮没有验证它们各自造成多少损失，不将其认定为唯一训练原因。',
        '',
        '结论范围：已定位质量任务中的资源分配损失，并看到模式与资源的任务相关配合；尚未证明具体学习率、训练预算、梯度更新或动作分布是唯一原因，也未证明规则最优。无需为解释这次结果立即追加训练。',
        '',
        f'核验：{episodes}回合、{slots}槽，原分支逐槽复现正式轨迹；所有干预的转移、预算、质量与奖励检查通过，重新读取轨迹聚合，模型和冻结输入哈希未改变。',
        '',f'证据：[完整数值]({OUT}/results.json)、[核验]({OUT}/audit.json)、[诊断协议]({OUT}/PROTOCOL.md)。']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    write(OUT/'artifact_hashes.json',{p.name:digest(p) for p in OUT.iterdir() if p.is_file() and p.suffix in ('.py','.md','.json') and p.name!='artifact_hashes.json'})
    print('PASS',episodes,slots)


if __name__=='__main__':main()
