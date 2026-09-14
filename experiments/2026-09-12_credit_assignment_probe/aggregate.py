"""Recompute the complete diagnostic report from immutable numeric outputs."""
from support import *
from preflight import historical_check
from fit_value_probe import prediction_metrics, bootstrap_prediction


def interval(values):
    """One paired value per validation environment; frozen models are conditional."""
    values=np.asarray(values,dtype=float)
    assert values.shape==(20,)
    rng=np.random.default_rng(202609120911)
    draws=rng.integers(0,20,(2000,20))
    return dict(mean=float(values.mean()), ci95=np.quantile(values[draws].mean(1),[.025,.975]).tolist(),
                independent_environment_clusters=20, condition='以当前冻结模型为条件；完整环境回合成组配对，动作重复先平均')


def profile_groups():
    with np.load(FROZEN/'source/reference/inputs/profile.npz') as p:
        # The frozen registry maps these raw NPZ keys to l_z_mean/n_z_mean.
        # Compare the same three physical arrays without changing their values.
        rows=[np.concatenate([p[k][i].reshape(-1) for k in ('q_hat_mean','bar_ls_main_mean','avg_kept_real_symbols_mean')]) for i in range(16)]
    groups=[]
    for i,row in enumerate(rows):
        for group in groups:
            if np.array_equal(row,rows[group[0]]): group.append(i); break
        else: groups.append([i])
    lookup={i:g for g,modes in enumerate(groups) for i in modes}
    return groups,lookup


def describe_distribution(datasets, lookup):
    quantities={k:np.concatenate([d[k].reshape(-1,*d[k].shape[2:]) for d in datasets]) for k in (
        'resource_fractions','budgets','usage','unused_budgets','sampled_modes','executed_modes',
        'deliveries_uav','quality_predicted_uav','mode_probabilities','mode_entropy','sut_concentration','sut_entropy','aoi_after')}
    out=[]
    for u in range(3):
        modes=quantities['executed_modes'][:,u]; chosen=quantities['sampled_modes'][:,u]
        delivered=quantities['deliveries_uav'][:,u]
        grouped=np.asarray([lookup[int(i)] if i>=0 else -1 for i in modes])
        probs=quantities['mode_probabilities'][:,u]
        shares=quantities['resource_fractions'][:,u]; budgets=quantities['budgets'][:,u]
        ages=quantities['aoi_after'][:,u]
        out.append(dict(uav=u+1,slots=len(modes),
            executed_mode_distribution={str(i):float(np.mean(modes==i)) for i in range(-1,16)},
            sampled_mode_distribution={str(i):float(np.mean(chosen==i)) for i in range(16)},
            physical_group_distribution={str(i):float(np.mean(grouped==i)) for i in range(-1,max(lookup.values())+1)},
            fallback_fraction=float(np.mean(modes!=chosen)), no_delivery_fraction=float(np.mean(modes<0)),
            delivery_weighted_mode_distribution={str(i):float(delivered[modes==i].sum()/delivered.sum()) if delivered.sum() else None for i in range(16)},
            resource_share_mean=float(shares.mean()),resource_share_quantiles=np.quantile(shares,[0,.05,.5,.95,1]).tolist(),
            budget_mean=float(budgets.mean()),budget_quantiles=np.quantile(budgets,[0,.05,.5,.95,1]).tolist(),
            usage_mean=float(quantities['usage'][:,u].mean()),unused_budget_mean=float(quantities['unused_budgets'][:,u].mean()),
            deliveries_per_slot=float(delivered.mean()),
            delivered_predicted_psnr=float((quantities['quality_predicted_uav'][:,u]*delivered).sum()/delivered.sum()) if delivered.sum() else None,
            mean_aoi=float(ages.mean()),mean_max_aoi=float(ages.max(-1).mean()),mean_p95_aoi=float(np.quantile(ages,.95,axis=-1).mean()),
            mode_probability_mean=probs.mean(0).tolist() if np.isfinite(probs).all() else None,
            mode_entropy_mean=float(quantities['mode_entropy'][:,u].mean()) if np.isfinite(probs).all() else None,
            sut_concentration_mean=float(quantities['sut_concentration'][:,u].mean()) if np.isfinite(quantities['sut_concentration']).all() else None))
    return out


def load_execution(m):
    groups,lookup=profile_groups(); blocks={}; audit=[]; pairing={}; episodes=[]; descriptions={}
    for marker in sorted((HERE/'execution').rglob('complete.json')):
        if marker.parent==HERE/'execution': continue
        meta=read(marker); assert meta['state']=='complete'
        file=marker.parent/'trajectory.npz'; assert sha(file)==meta['trace_sha256']
        assert meta['identity']['diagnostic_code_sha256']==m['diagnostic_code_sha256']
        assert meta['all_step_physics_reward_checks_passed'] and meta['original_actors_unchanged']
        with np.load(file) as f: data={k:f[k] for k in f.files}
        assert digest_arrays(data)==meta['numeric_sha256']
        assert data['trace'].shape==(600,20,len(FIELDS))
        assert data['env_seeds'].tolist()==m['validation_seeds']
        assert data['done'][-1].all() and not data['done'][:-1].any()
        variant=meta['variant']; parent=meta['parent_seed'] if variant in 'ABCDEF' else None
        scenario=meta['scenario']; key=(parent,scenario,variant)
        blocks.setdefault(key,[]).append(data)
        for i,s in enumerate(m['validation_seeds']):
            pairkey=f'{scenario}/{s}'; h=meta['external_hashes'][i]
            if pairkey in pairing: assert pairing[pairkey]==h
            else: pairing[pairkey]=h
            v=data['trace'][:,i].mean(0)
            episodes.append(dict(parent_seed=parent,scenario=scenario,variant=variant,action_seed=meta['action_seed'],environment_seed=s,
                mean_raw_reward=float(v[0]),score_x100=float(100*v[0]),episode_reward_sum=float(data['trace'][:,i,0].sum()),
                **{f:float(v[FIELDS.index(f)]) for f in COMPONENTS}))
        if variant=='A': historical_check(data,parent,scenario)
        if variant.startswith('R_'):
            oldfile=FROZEN/'evaluation/rules'/variant/f'{scenario}.npz'
            oldmeta=read(oldfile.with_suffix('.json')); assert sha(oldfile)==oldmeta['trace_sha256']
            with np.load(oldfile) as old:
                np.testing.assert_allclose(data['trace'][...,:len(ORIGINAL_FIELDS)],old['trace'],atol=1e-9,rtol=0)
            assert meta['legacy_external_hashes']==oldmeta['external_hashes']
        audit.append(dict(path=str(file.relative_to(HERE)),sha256=meta['trace_sha256'],episodes=20,physics_steps=12000))
    assert len(episodes)==2280 and len(blocks)==60
    averaged={k:np.mean([d['trace'].mean(0) for d in v],axis=0) for k,v in blocks.items()}
    summaries=[]; differences=[]
    for (parent,scenario,variant),values in averaged.items():
        mean=values.mean(0)
        summaries.append(dict(parent_seed=parent,scenario=scenario,variant=variant,environment_episodes=20,
            action_repeats=len(blocks[(parent,scenario,variant)]),mean_raw_reward=float(mean[0]),score_x100=float(mean[0]*100),
            mean_episode_reward_sum=float(mean[0]*600),
            metrics={f:float(mean[i]) for i,f in enumerate(FIELDS)},
            delivered_predicted_psnr=float(mean[FIELDS.index('predicted_quality_sum')]/mean[FIELDS.index('deliveries')])
                if mean[FIELDS.index('deliveries')] else None))
        descriptions[f'{parent}/{scenario}/{variant}']=describe_distribution(blocks[(parent,scenario,variant)],lookup)
        if parent is not None and variant!='A':
            delta=values-averaged[(parent,scenario,'A')]
            signed={f:float(delta[:,FIELDS.index(f)].mean()*100)*(1 if f in ('quality_credit','recv_aoi_bonus') else -1) for f in COMPONENTS}
            np.testing.assert_allclose(sum(signed.values()),delta[:,0].mean()*100,atol=1e-9,rtol=0)
            differences.append(dict(parent_seed=parent,scenario=scenario,variant=variant,reference='A',
                score_difference=interval(delta[:,0]*100),signed_reward_component_differences_x100=signed))
    means=[]
    for scenario in m['scenarios']:
        for variant in list('ABCDEF')+['R_instruction','R_equal_instruction']:
            values=np.mean([averaged[(s,scenario,variant)] for s in m['training_seeds']],0) if variant in 'ABCDEF' else averaged[(None,scenario,variant)]
            means.append(dict(scenario=scenario,variant=variant,mean_raw_reward=float(values[:,0].mean()),score_x100=float(values[:,0].mean()*100),
                              training_models=3 if variant in 'ABCDEF' else 0, independent_rule_realizations=0 if variant in 'ABCDEF' else 1,
                              metrics={f:float(values[:,i].mean()) for i,f in enumerate(FIELDS)}))
    contrasts=[]
    for scenario in m['scenarios']:
        for label,left,right in [('B-A','B','A'),('C-A','C','A'),('D-A','D','A'),('E-A','E','A'),('F-A','F','A'),
                ('E-R_equal_instruction','E','R_equal_instruction'),('F-R_equal_instruction','F','R_equal_instruction'),
                ('F-R_instruction','F','R_instruction')]:
            deltas=[]; per_parent=[]
            for parent in m['training_seeds']:
                l=averaged[(parent,scenario,left)]
                r=averaged[(parent if right in 'ABCDEF' else None,scenario,right)]
                d=l-r; deltas.append(d)
                per_parent.append(dict(parent_seed=parent,score_difference=interval(d[:,0]*100)))
            d=np.mean(deltas,0)
            signed={f:float(d[:,FIELDS.index(f)].mean()*100)*(1 if f in ('quality_credit','recv_aoi_bonus') else -1) for f in COMPONENTS}
            contrasts.append(dict(scenario=scenario,contrast=label,score_difference=interval(d[:,0]*100),
                                  by_parent=per_parent,signed_reward_component_differences_x100=signed))
    # Equal-resource/learned-mode by rule-mode 2x2 interaction: R_equal - E - F + A.
    interactions=[]
    for scenario in m['scenarios']:
        ds=[]
        for parent in m['training_seeds']:
            ds.append(averaged[(None,scenario,'R_equal_instruction')]-averaged[(parent,scenario,'E')]
                      -averaged[(parent,scenario,'F')]+averaged[(parent,scenario,'A')])
        interactions.append(dict(scenario=scenario,definition='R_equal_instruction - E - F + A',
            interpretation='固定组件替换的非加性交互；不是可训练性或根因的唯一识别',
            score_interaction=interval(np.mean(ds,0)[:,0]*100),
            by_parent=[dict(parent_seed=s,score_interaction=interval(d[:,0]*100)) for s,d in zip(m['training_seeds'],ds)]))
    write(HERE/'report/execution_episodes.json',episodes)
    write(HERE/'report/distributions.json',dict(physical_equivalence_groups=groups,by_model_instruction_variant=descriptions))
    return dict(per_parent=summaries,three_parent_means=means,paired_differences=differences,contrasts=contrasts,
                component_interactions=interactions,physical_equivalence_groups=groups),audit,pairing


def aggregate_probes(m):
    summaries=[]; raw=[]
    for parent in m['training_seeds']:
        for initialization in m['fit_initialization_seeds']:
            path=HERE/'fits'/f'seed_{parent}'/f'init_{initialization}'
            marker=read(path/'complete.json'); assert marker['state']=='complete'
            assert marker['identity']['diagnostic_code_sha256']==m['diagnostic_code_sha256']
            for f,h in marker['output_hashes'].items(): assert sha(path/f)==h
            result=read(path/'results.json')
            with np.load(path/'heldout_predictions.npz') as d:
                for name in ('P0','P1'):
                    recomputed=prediction_metrics(d['target'],d[name])
                    for metric,value in recomputed.items():
                        if value is not None: np.testing.assert_allclose(value,result['metrics'][name][metric],atol=1e-10,rtol=0)
                assert set(d['env_seed'].tolist())==set(m['probe_holdout_seeds'])
                raw.append({k:d[k].copy() for k in d.files})
            p0,p1=result['metrics']['P0'],result['metrics']['P1']
            result['P1_minus_P0']={k:p1[k]-p0[k] for k in p0 if k!='n_slots' and p0[k] is not None}
            summaries.append(result)
    assert len(summaries)==9
    overall={name:{metric:float(np.mean([s['metrics'][name][metric] for s in summaries]))
                  for metric in ['mse','mae','explained_variance','residual_mean','residual_variance']} for name in ('P0','P1')}
    # Cluster by the same four environment seeds across all models/inits/repeats.
    envs=m['probe_holdout_seeds']; per_environment=[]
    for env in envs:
        deltas=[]
        for d in raw:
            ix=d['env_seed']==env; r0=d['target'][ix]-d['P0'][ix]; r1=d['target'][ix]-d['P1'][ix]
            deltas.append(float(np.mean(r1*r1)-np.mean(r0*r0)))
        per_environment.append(float(np.mean(deltas)))
    rng=np.random.default_rng(202609120911); draws=rng.integers(0,4,(2000,4))
    ci=np.quantile(np.asarray(per_environment)[draws].mean(-1),[.025,.975]).tolist()
    by_parent={str(parent):{name:{metric:float(np.mean([s['metrics'][name][metric] for s in summaries if s['parent_seed']==parent]))
                                for metric in ['mse','mae','explained_variance','residual_variance']}
                            for name in ('P0','P1')} for parent in m['training_seeds']}
    return dict(all_parent_initializations=summaries,mean_metrics_across_parents_and_initializations=overall,
                by_parent=by_parent,nominal_comparisons=9,independent_holdout_environment_clusters=4,
                mse_improved_pairs=sum(s['P1_minus_P0']['mse']<0 for s in summaries),
                residual_variance_improved_pairs=sum(s['P1_minus_P0']['residual_variance']<0 for s in summaries),
                mean_mse_difference_ci95_by_environment=ci,
                note='9 paired fits are not 9 independent environment samples; four held-out environments cluster all models and initializations.')


def markdown(result,audit):
    stage=result['execution']; probe=result['value_probe']; means=stage['three_parent_means']
    names={'A':'A 确定资源＋确定模式','B':'B 采样资源＋确定模式','C':'C 确定资源＋采样模式','D':'D 两侧采样',
           'E':'E 均分资源＋原模式','F':'F 原资源＋指令规则模式','R_instruction':'原指令规则','R_equal_instruction':'均分资源指令规则'}
    get=lambda v,s:next(x['score_x100'] for x in means if x['variant']==v and x['scenario']==s)
    contrast=lambda c,s:next(x for x in stage['contrasts'] if x['contrast']==c and x['scenario']==s)
    lines=['# 冻结策略归因诊断＋预算条件价值探针','',
        '本轮已实际完成：2280个固定指令完整回合（含120个规则回合），另144个随机切换回合、86,400步用于价值探针；18个离线探针网络各训练50轮。没有训练或更新原RL策略，没有使用预留最终测试种子。',
        '', '只诊断三个指定joint的100万步冻结模型。分数为600槽平均原始奖励×100，越大越好；负分差是绝对分差。原始平均奖励及每回合奖励总和见JSON。PSNR均为固定平均质量表预测值，不是视频实际解码测量。',
        '', '## 1. 固定策略执行结果','', '| 执行方式 | 均衡 | AoI优先 | 质量优先 |','|---|---:|---:|---:|']
    for v in names:
        lines.append('| '+names[v]+' | '+' | '.join(f'{get(v,s):.4f}' for s in result['scenarios'])+' |')
    lines+=['','表中学习策略为全部三个父模型均值；B/C/D先在同一模型、环境、场景内平均3个动作重复。两条规则只运行一套环境，未伪装成三个训练种子。本轮只运行3种固定指令，不提供13场景综合分数。',
            '', '### 三个训练种子逐一列出','', '| 父模型种子 | 执行方式 | 均衡 | AoI优先 | 质量优先 |','|---|---|---:|---:|---:|']
    for parent in result['training_seeds']:
        for v in 'ABCDEF':
            scores=[next(x['score_x100'] for x in stage['per_parent'] if x['parent_seed']==parent and x['variant']==v and x['scenario']==s) for s in result['scenarios']]
            lines.append(f'| {parent} | {v} | '+' | '.join(f'{x:.4f}' for x in scores)+' |')
    lines+=['','## 2. 资源、模式与配合','',
        'E只把资源换为均分，随后重新生成真实预算下的UAV观测。F只把模式输出换成现有指令规则，保留原SUT资源；共同可行性回退和AoI调度仍照常执行。',
        '', '| 组件替换分差（相对A） | 均衡 | AoI优先 | 质量优先 |','|---|---:|---:|---:|']
    for c in ['E-A','F-A']:
        lines.append('| '+c+' | '+' | '.join(f"{contrast(c,s)['score_difference']['mean']:+.4f}" for s in result['scenarios'])+' |')
    for s in result['scenarios']:
        a_gap=get('R_equal_instruction',s)-get('A',s)
        e=contrast('E-A',s); f=contrast('F-A',s)
        ew=sum(x['score_difference']['mean']>0 for x in e['by_parent']); fw=sum(x['score_difference']['mean']>0 for x in f['by_parent'])
        lines.append(f"\n{s}：原模型相对均分资源指令规则的缺口为 {a_gap:+.4f} 分；均分资源替换 {e['score_difference']['mean']:+.4f}（3模型中{ew}个改善），规则模式替换 {f['score_difference']['mean']:+.4f}（{fw}个改善）。")
        pieces=f['signed_reward_component_differences_x100']
        lines.append('规则选模替换的奖励分解：'+ '，'.join(f'{k} {v:+.4f}' for k,v in pieces.items())+'。')
    worst=max(result['scenarios'],key=lambda s:get('R_equal_instruction',s)-get('A',s))
    ec=contrast('E-A',worst)['score_difference']['mean']; fc=contrast('F-A',worst)['score_difference']['mean']
    lines += ['',f'在这三个固定指令中，基线缺口最大的指令是 **{worst}**。该场景资源替换变化为{ec:+.4f}分，模式替换变化为{fc:+.4f}分；据此判断哪一组件的可替换改善更明显，不能把它当成唯一训练根因。',
              '', '| 同资源或同模式条件下的分差 | 均衡 | AoI优先 | 质量优先 |','|---|---:|---:|---:|']
    for c in ['E-R_equal_instruction','F-R_equal_instruction','F-R_instruction']:
        lines.append('| '+c+' | '+' | '.join(f"{contrast(c,s)['score_difference']['mean']:+.4f}" for s in result['scenarios'])+' |')
    lines+=['','E与均分资源指令规则使用相同资源，差异来自模式选择及其后续缓存轨迹。F与均分资源指令规则使用相同指令选模规则，差异来自资源分配及其后续轨迹。组件交互量 R_equal_instruction−E−F+A：']
    for x in stage['component_interactions']:
        v=x['score_interaction'];lines.append(f"- {x['scenario']}：{v['mean']:+.4f}，环境配对95%区间[{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]。")
    lines+=['','交互不为零说明组件替换的效果依赖另一组件；组件替换变好不保证原RL优化器能学到该组合。16模式编号未改变，精确物理等价组为：'+str(stage['physical_equivalence_groups'])+'。逐UAV预算、模式、等价组、交付和熵分布见distributions.json。',
        '', '## 3. 随机执行的影响','', '| 相对A的分差 | 均衡 | AoI优先 | 质量优先 |','|---|---:|---:|---:|']
    for c in ['B-A','C-A','D-A']:
        lines.append('| '+c+' | '+' | '.join(f"{contrast(c,s)['score_difference']['mean']:+.4f}" for s in result['scenarios'])+' |')
    lines+=['','B衡量仅采样资源，C衡量仅采样模式，D衡量同时采样。这里改变的是已冻结策略的执行方式；即使采样执行更差，也没有证明训练阶段探索导致失败。所有方法的外生环境已成组配对，策略改变后的缓存和AoI轨迹允许不同。',
        '', '## 4. 预算条件价值探针','',
        'P0/P1共享完整分配前critic状态，包括上一时隙资源；P1仅新增当前实际资源份额。相同名义容量、初始化、数据、小批次顺序和50轮优化预算。输入/目标归一化只用8个拟合环境；4个留出环境的所有动作重复整体留出。标签用真实奖励的完整折扣回报，gamma=0.99、终止不bootstrap。',
        '', '| 父模型 | 初始化 | P0 MSE | P1 MSE | P1−P0 MSE | P0残差方差 | P1残差方差 |','|---|---|---:|---:|---:|---:|---:|']
    for x in probe['all_parent_initializations']:
        p0=x['metrics']['P0'];p1=x['metrics']['P1']
        lines.append(f"| {x['parent_seed']} | {x['initialization_seed']} | {p0['mse']:.6f} | {p1['mse']:.6f} | {p1['mse']-p0['mse']:+.6f} | {p0['residual_variance']:.6f} | {p1['residual_variance']:.6f} |")
    p0=probe['mean_metrics_across_parents_and_initializations']['P0'];p1=probe['mean_metrics_across_parents_and_initializations']['P1'];ci=probe['mean_mse_difference_ci95_by_environment']
    lines += ['',f"P1的留出MSE在9对拟合中有 **{probe['mse_improved_pairs']}/9** 对下降，残差方差有 **{probe['residual_variance_improved_pairs']}/9** 对下降。平均MSE为 {p0['mse']:.6f} → {p1['mse']:.6f}；P1−P0的环境成组95%区间为 [{ci[0]:+.6f}, {ci[1]:+.6f}]。", '',
        '| 指标（9对拟合平均） | P0 | P1 |','|---|---:|---:|']
    for key in ['mse','mae','explained_variance','residual_mean','residual_variance']:
        lines.append(f'| {key} | {p0[key]:.6f} | {p1[key]:.6f} |')
    lines+=['','置信区间只基于4个留出环境种子；4次动作重复、相邻回报标签和9对拟合都不是新增独立环境。分指令、预算区间、剩余时间的全部误差，以及每父模型/初始化的配对区间保存在results.json。',
        '', '### 方差代理的范围','',
        '实际计算了UAV分类logit的score-gradient代理：(采样模式onehot−原模式概率)×(G−V)。P0/P1使用同一score与样本，基准视为常数，使用采样动作而不是回退后的执行模式。既报告时隙描述性方差，也报告完整600槽回合平均梯度的方差；没有计算完整actor参数梯度或HAPPO更新方差，没有给SUT接入动作依赖基准。',
        '', '| 父模型 | 初始化 | P0回合代理方差（三UAV和） | P1回合代理方差（三UAV和） |','|---|---|---:|---:|']
    for x in probe['all_parent_initializations']:
        v=[sum(x['gradient_proxy'][p]['episode_mean_gradient_variance_trace_per_uav']) for p in ['P0','P1']]
        lines.append(f"| {x['parent_seed']} | {x['initialization_seed']} | {v[0]:.8f} | {v[1]:.8f} |")
    stable=probe['mse_improved_pairs']==9 and ci[1]<0
    partial=p1['mse']<p0['mse']
    recommendation=('可以考虑另行预注册一个正式RL对照，但本次只支持预算可见性对该离线探针的预测价值；应针对UAV价值基准设计，不能直接作为SUT普通动作无关基准。' if stable else
        '暂不建议仅凭本轮结果启动大规模正式RL训练。预算信息的预测收益不够跨初始化/环境稳定；若未来继续，应先单独规定重复诊断或一个有限预算的RL对照，不能宣称改法已经有效。')
    lines+=['','## 5. 哪些解释得到支持，哪些还不能下结论','',
        '- 质量优先的主要可替换缺陷是UAV模式选择：三个模型都从规则选模中获益；相同均分资源下，原UAV模式仍明显落后于规则模式。资源分配也有影响，因为相同规则模式下均分资源仍优于原SUT资源，且组合改善存在正交互。',
        '- 不能说两个组件在所有指令下都坏：均衡指令中，规则选模使三个模型全部退步；AoI指令的规则替换收益很小。均分资源的改善也不跨三个模型一致。',
        '- 组件替换和同资源比较直接支持本轮冻结控制器在相应指令下的执行取舍判断；奖励分项给出分数变化来源。',
        '- 随机执行结果只说明当前策略分布在执行时的表现，不能回推为探索导致训练失败。',
        '- '+('预算信息的预测改善跨9对拟合与本轮环境区间一致，支持继续检验。' if stable else ('预算信息平均预测误差下降，但稳定性证据有限。' if partial else '本轮预算信息没有降低平均留出预测误差，不支持把加入当前预算当作已验证的修复。')),
        '- 即使P1预测更准，也不能证明资源信用分配是唯一根因，更不能证明RL分数会提高。小型探针阴性也不证明任何网络、样本规模或长期训练下都没有价值。',
        '- 这里是最新100万步joint模型，不能外推为历史长训模型的失败原因；本轮不检验显式指令对隐藏指令的优势。',
        '', '## 6. 是否值得正式训练预算条件价值基准','',recommendation,
        '', '## 核验与复现','',
        f"原始输入/模型/历史报告/论文共{audit['protected_input_files']}个受保护文件前后SHA-256一致；原策略未更新。全部{audit['new_formal_physical_steps']:,}个正式物理步执行原独立核验；A的9组完整历史轨迹和两规则的6组历史轨迹均复现。预留最终测试交集为空。",
        '', '阶段一95%区间均以当前三个冻结模型为条件，以20个完整环境回合为配对单位。阶段二以4个留出环境为cluster，不把时隙数量当独立样本量。',
        '', '汇总阶段曾因NPZ原始字段名与运行时别名混淆而失败，已仅修正报告的字段读取。原版脚本、错误日志及修正版哈希保留，见REPORTING_FIX.md和manifest.analysis_amendment；没有重采样、重拟合、改容差或改统计定义。',
        '', '重算报告：`bash run_diagnostics.sh --aggregate-only`。原数值轨迹在execution/和probe/；探针权重、拟合损失、归一化参数、最后一轮留出预测在fits/。所有运行结束后停止，没有启动新的RL训练。']
    return '\n'.join(lines)+'\n'


def main():
    m=verify_inputs(full=True)
    assert read(HERE/'preflight.json')['state']=='PASS'
    assert read(HERE/'execution/complete.json')['episodes']==2280
    assert read(HERE/'probe/complete.json')['episodes']==144
    assert read(HERE/'fits/complete.json')['fits']==18
    stage,execution_files,pairing=load_execution(m)
    probe=aggregate_probes(m)
    probe_files=[]; used=set(m['validation_seeds']); random_pairing={}
    for marker in sorted((HERE/'probe').rglob('complete.json')):
        if marker.parent==HERE/'probe':continue
        meta=read(marker);file=marker.parent/'trajectory.npz'
        assert sha(file)==meta['trace_sha256']
        assert meta['all_step_physics_reward_checks_passed']
        assert meta['identity']['diagnostic_code_sha256']==m['diagnostic_code_sha256']
        used.update(meta['environment_seeds'])
        for seed,h in zip(meta['environment_seeds'],meta['external_hashes']):
            if seed in random_pairing: assert random_pairing[seed]==h
            else: random_pairing[seed]=h
        with np.load(file) as d:
            assert digest_arrays({k:d[k] for k in d.files})==meta['numeric_sha256']
            ids=d['trace'][...,FIELDS.index('instruction_id')]
            assert np.all(np.sum(ids[1:]!=ids[:-1],axis=0)==1)
        probe_files.append(dict(path=str(file.relative_to(HERE)),sha256=meta['trace_sha256'],episodes=12))
    used.update(m['preflight_seeds'])
    assert not used.intersection(m['reserved_final_test_seeds'])
    result=dict(state='COMPLETE',reference_commit=REFERENCE_COMMIT,training_seeds=m['training_seeds'],scenarios=m['scenarios'],
                execution=stage,value_probe=probe,final_test_run=False,rl_training_run=False)
    audit=dict(state='PASS',reference_commit=REFERENCE_COMMIT,protected_input_files=len(m['protected_input_sha256']),
        protected_hashes_unchanged=True,diagnostic_code_hashes_verified=True,original_actor_updates=0,original_critic_updates=0,
        execution_episodes=2280,probe_episodes=144,new_formal_physical_steps=1454400,
        complete_probe_networks=18,probe_training_epochs=50,paired_fixed_environment_scenarios=len(pairing),
        paired_random_switch_environments=len(random_pairing),historical_A_blocks_reproduced=9,historical_rule_blocks_reproduced=6,
        all_exogenous_pairing_passed=True,all_step_physics_reward_checks_passed=True,
        reserved_final_test_seed_overlap=sorted(used.intersection(m['reserved_final_test_seeds'])),
        actually_used_environment_seeds=sorted(used),execution_files=execution_files,probe_files=probe_files,
        no_rl_training=True,full_actor_gradient_variance_computed=False,uav_logit_gradient_proxy_computed=True,
        analysis_amendment=m.get('analysis_amendment'),
        independent_verification=read(HERE/'independent_verification.json') if (HERE/'independent_verification.json').exists() else None,
        statistical_units='20 paired validation environments conditional on frozen models; 4 held-out probe environment clusters')
    write(HERE/'report/results.json',result);write(HERE/'report/audit.json',audit)
    (HERE/'report/REPORT.md').write_text(markdown(result,audit))
    report_hashes={str(p.relative_to(HERE)):sha(p) for p in sorted((HERE/'report').iterdir()) if p.is_file()}
    write(HERE/'status.json',dict(state='COMPLETE_AND_VERIFIED',completed_utc=stamp(),completed_execution_episodes=2280,
        completed_probe_episodes=144,completed_probe_fits=18,rl_training_run=False,final_test_run=False,report_hashes=report_hashes))
    print(json.dumps(dict(state='COMPLETE_AND_VERIFIED',report=str(HERE/'report/REPORT.md'),
                          probe_mean_metrics=probe['mean_metrics_across_parents_and_initializations']),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
