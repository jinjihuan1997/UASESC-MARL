"""Audit frozen evaluations and report every preregistered matched-info variant."""
from runtime import OUT, REFERENCE, h, FIELDS, summarize, audit_trace, np, torch, verify
from online_policy import UAVGreedy, TreeScorePredictor, score_features


def main():
    torch.set_num_threads(1)
    protocol = verify()
    old_folder = OUT.parent / '2026-09-11_selected_seed_comparison'
    for file, digest in h.read(old_folder/'artifact_hashes.json').items():
        assert h.digest(old_folder/file) == digest
    old = h.read(old_folder/'comparison.json')
    expected_pairing = h.read(REFERENCE/'evaluation/rules/R_myopic/summary.json')['pairing']
    results, hashes, mode_counts = {}, {}, {}
    heldout = {}
    episodes = 0
    for count in (3, 16):
        branch = OUT / f'modes_{count}'
        assert h.read(branch/'status.json')['state'] == 'complete'
        assert h.read(branch/'interface_audit.json')['state'] == 'PASS'
        calibration = h.read(branch/'calibration.json')
        assert h.digest(branch/'calibration_data.npz') == calibration['dataset_sha256']
        fit = h.read(branch/'fit.json')
        assert h.digest(branch/'predictor.npz') == fit['predictor_sha256']
        # Resource-choice prediction quality is assessed on held-out calibration
        # seeds, not on the final evaluation used to compare controllers.
        model = TreeScorePredictor(branch/'predictor.npz')
        with np.load(branch/'calibration_data.npz') as z:
            selected = np.isin(z['seeds'], protocol['validation_seeds'])
            x, y = z['X'][selected], z['y'][selected]
        predictions = np.concatenate([h.arr(model.predict(torch.as_tensor(x[start:start+512]))) for start in range(0,len(x),512)])
        predictions, y = predictions.reshape(-1,3), y.reshape(-1,3)
        choice = predictions.argmax(-1)
        regrets = y.max(-1)-y[np.arange(len(y)),choice]
        heldout[f'modes_{count}'] = dict(validation_states=len(y),
            resource_choice_regret_score_x100=float(regrets.mean()),
            optimal_or_tied_fraction=float((regrets<=1e-9).mean()),
            rmse_score_x100=fit['validation_rmse_score_x100'],
            note='Against all-current-state best of three resources, with the same local UAV rule; not test-set tuning.')
        for label in ('fitted_sut','calibrated_sut'):
            item = f'modes_{count}/{label}'
            folder = branch/'evaluation'/label
            status = h.read(folder/'status.json')
            assert status['state']=='complete'
            assert h.digest(folder/'results.json') == status['results_sha256']
            record = h.read(folder/'results.json')
            assert record['pairing']==expected_pairing
            per, parts, counts = {}, [], np.zeros(17,dtype=np.int64)
            for scene,schedule in protocol['scenarios'].items():
                file=folder/f'{scene}.npz'
                digest=record['trace_hashes'][str(file.relative_to(OUT))]
                assert h.digest(file)==digest
                hashes[str(file.relative_to(OUT))]=digest
                with np.load(file) as z:
                    trace=z['trace'];ages=z['aoi_after']
                    np.testing.assert_array_equal(z['fields'],FIELDS)
                    np.testing.assert_array_equal(z['seeds'],protocol['evaluation_seeds'])
                    counts+=np.bincount(z['modes'].ravel()+1,minlength=17)
                assert trace.shape==(600,20,len(FIELDS))
                audit_trace(trace,ages)
                gids=[next(g for t,g in reversed(schedule) if t<=slot) for slot in range(600)]
                np.testing.assert_array_equal(trace[...,5],np.broadcast_to(np.asarray(gids)[:,None],(600,20)))
                per[scene]=summarize(trace);parts.append(trace);episodes+=20
                for j,(start,g) in enumerate(schedule):
                    end=schedule[j+1][0] if j+1<len(schedule) else 600
                    per[f'{scene}/segment_{j}_g{g}']=summarize(trace[start:end])
                    if j:per[f'{scene}/first20_after_{start}']=summarize(trace[start:min(start+20,end)])
            merged=np.concatenate(parts);per['overall']=summarize(merged)
            for gid in range(3):per[f'true_instruction_{gid}']=summarize(merged[merged[...,5]==gid])
            for scope,metrics in per.items():
                for key,value in metrics.items():
                    np.testing.assert_allclose(value,record['per_scope'][scope][key],atol=1e-9,rtol=0)
            results[item]=per
            mode_counts[item]={str(i-1):int(n) for i,n in enumerate(counts) if n}
        for file in ['fit.json','predictor.npz','calibration.json','calibration_data.npz','interface_audit.json','status.json']:
            hashes[str((branch/file).relative_to(OUT))]=h.digest(branch/file)
    equality={}
    for label in ('fitted_sut','calibrated_sut'):
        for scene in protocol['scenarios']:
            with np.load(OUT/f'modes_3/evaluation/{label}/{scene}.npz') as a,np.load(OUT/f'modes_16/evaluation/{label}/{scene}.npz') as b:
                for key in a.files:np.testing.assert_array_equal(a[key],b[key])
        equality[label]='all 13 scenario traces, modes, resource allocations and AoI exactly equal'
    # Guard against accidentally ignoring modes outside {0,5,10}.
    obs=torch.zeros(1,76);obs[:,0]=.2;obs[:,1:11]=1;obs[:,11:21]=.125;obs[:,21:31]=.375
    obs[:,31:47]=.01;obs[:,47:63]=25/33;obs[:,54]=32/33;obs[:,69]=1;obs[:,74]=21.5/33
    mask=torch.ones(1,16)
    assert UAVGreedy([0,5,10]).act(obs,mask).argmax(-1).item() in [0,5,10]
    assert UAVGreedy(list(range(16))).act(obs,mask).argmax(-1).item()==7
    reference_methods=['selector','joint_continue','joint','R_myopic','R_instruction','R_single']
    references={method:old['methods'][method]['mean_all_three'] for method in reference_methods}
    paired={method:{str(seed):100*(old['methods']['selector']['by_training_seed'][str(seed)]['per_scope']['overall']['common_reward']-per['overall']['common_reward'])
                    for seed in old['selected_training_seeds']} for method,per in results.items()}
    chosen=h.read(old_folder/'random_instruction_examples/selection.json')
    chosen_index=protocol['evaluation_seeds'].index(chosen['evaluation_seed'])
    examples={}
    for scene in chosen['scenarios']:
        examples[scene]={}
        for method in results:
            mode,label=method.split('/')
            with np.load(OUT/mode/'evaluation'/label/f'{scene}.npz') as z:
                examples[scene][method]=summarize(z['trace'][:,chosen_index])
    audit=dict(state='PASS',new_evaluation_episodes=episodes,new_evaluation_slots=episodes*600,
        new_calibration_episodes=260,new_calibration_physical_steps=156000,new_calibration_candidate_labels=468000,
        no_rl_retraining=True,per_actor_information_interface_verified=True,paired_exogenous_sequences=True,
        independent_reward_and_physics_verified=True,all_preregistered_variants_reported=True,
        expanded_modes_reachable_unit_case=True,three_vs_sixteen=equality,source_files_verified=len(hashes))
    artifact=dict(protocol=protocol,results=results,references=references,
        selected_rl_minus_new_baselines_score_x100=paired,heldout_calibration_predictor_quality=heldout,
        mode_counts=mode_counts,random_example_selection=chosen,random_examples=examples,audit=audit)
    h.write(OUT/'comparison.json',artifact)
    h.write(OUT/'audit.json',audit)
    h.write(OUT/'trace_and_model_hashes.json',hashes)
    report(artifact,old)
    h.write(OUT/'artifact_hashes.json',{p.name:h.digest(p) for p in [OUT/'comparison.json',OUT/'REPORT.md',OUT/'audit.json',
        OUT/'protocol.json',OUT/'trace_and_model_hashes.json',OUT/'aggregate.py']})
    print('AGGREGATION PASS',audit,flush=True)
    print((OUT/'REPORT.md').read_text().split('## 全部13场景')[0],flush=True)


def report(data,old):
    names={'modes_3/fitted_sut':'同观测一步贪心：3模式、拟合SUT得分',
           'modes_16/fitted_sut':'同观测一步贪心：16模式、拟合SUT得分',
           'modes_3/calibrated_sut':'同观测局部贪心：3模式、固定校准分配规则',
           'modes_16/calibrated_sut':'同观测局部贪心：16模式、固定校准分配规则'}
    rows={**{names[k]:v for k,v in data['results'].items()},
          **{old['methods'][k]['label']:v for k,v in data['references'].items()}}
    lines=['# 与RL部署观测对齐的一步贪心对比','',
        '已按SUT和每个UAV各自的观测接口实现、运行并核验。原始全局一步择优保留。没有修改环境、平均质量载荷表、真实奖励、RL模型、指令过程或评估种子。','',
        '核心结果：严格观测接口下的拟合版一步贪心总体分数为−2.8035，优于原全局九候选规则的−2.9291，也优于此前所选三个RL的平均−2.8876。这证明额外在线信息不是超过当前RL的必要条件；由于决策方式也改变，本次没有单独量化原差距中信息因素的贡献。','',
        '## 新方法具体是什么','',
        '- SUT只接收原RL的SUT观测（76维，实际非零信息位在前27维）。从现有缓存数量、缓存AoI汇总构造均分、按缓存数量、按紧急度3种资源候选；输入不能包含各UAV的局部观测或集中critic状态。',
        '- 拟合版SUT使用离线校准的单步得分预测器，对3种候选估计当前奖励并择优。预测器为100棵数值回归树；使用187200个拟合样本，46800个独立校准验证样本，来自8个拟合环境种子与2个验证种子。它不是新的RL训练，也不是无需拟合的纯手写规则。',
        '- UAV先收到与RL相同的实际资源预算，再只用自己的76维观测和动作掩码评估模式。各UAV独立决定，不向SUT回传额外候选得分、不相互读取状态。',
        '- UAV的局部评分保留质量、平均AoI、尾部AoI和资源项的全局加和归一化；无法观察其他UAV的最大AoI，因此采用本地最大AoI除以3的代理项。环境评估仍使用原始全局最大AoI和完整奖励，未改变评分标准。',
        '- 固定校准分配版不拟合回归树：SUT使用校准数据冻结的指令→资源规则（均衡/质量均分，AoI按紧急度），UAV仍执行上述局部一步选模。它是单独的较简单对照。','',
        '所有参数、候选、拟合种子、验证种子和方法均在评估前写入protocol.json。没有根据评估结果挑选版本或调整奖励。','',
        '## 当前方法总体结果','',
        '统一13场景×20评估种子×600时隙；每种方法260回合。分数=原始奖励×100，越大越好。RL行为使用原有冻结模型。PSNR是平均表给出的已交付块预测质量，不是实测解码视频PSNR。','',
        '| 方法 | 综合分数 ↑ | 平均AoI ↓ | 交付预测PSNR/dB ↑ | 交付数/时隙 | 资源/时隙 |',
        '|---|---:|---:|---:|---:|---:|']
    for label,per in rows.items():
        x=per['overall']
        lines.append(f'| {label} | {100*x["common_reward"]:.4f} | {x["mean_aoi"]:.4f} | {x["delivered_predicted_psnr"]:.4f} | {x["deliveries_per_slot"]:.4f} | {x["channel_uses_per_slot"]:.2f} |')
    lines+=['','## 对应三个所选RL种子','',
        '| 训练种子 | 选择器RL分数 | 同观测拟合贪心 | RL减同观测拟合贪心 | 固定校准局部贪心 | RL减固定校准局部贪心 |',
        '|---|---:|---:|---:|---:|---:|']
    fitted=100*data['results']['modes_3/fitted_sut']['overall']['common_reward']
    calibrated=100*data['results']['modes_3/calibrated_sut']['overall']['common_reward']
    for seed in old['selected_training_seeds']:
        rl=100*old['methods']['selector']['by_training_seed'][str(seed)]['per_scope']['overall']['common_reward']
        lines.append(f'| {seed} | {rl:.4f} | {fitted:.4f} | {rl-fitted:+.4f} | {calibrated:.4f} | {rl-calibrated:+.4f} |')
    lines+=['','新控制器只用一套固定校准过程，并不需要对应的三个RL训练种子，因此基线在三列重复的是同一组配对环境评估，不是三个独立训练样本。','',
        '## 各指令得分','', '| 方法 | 全程均衡 | 全程AoI | 全程质量 |', '|---|---:|---:|---:|']
    for label,per in rows.items():
        lines.append('| '+label+' | '+' | '.join(f'{100*per[f"fixed_{g}"]["common_reward"]:.4f}' for g in range(3))+' |')
    lines+=['','## 如何理解这次变化','',
        '1. 原全局规则的信息更多，但仅枚举九个集中式候选。新版本允许三个UAV各自选择模式，名义组合数量从资源3×共同模式3=9变成3×3³=81；16模式版本为3×16³=12288，但执行时按节点分别选择，不枚举整个联合空间。因此不能把改善解释为“仅仅减少信息就更好”，也不能把差值全部解释成信息贡献。',
        '2. 新版本与RL严格对齐的是部署时各actor的信息接口和资源→选模消息流程；它还保留已知物理/奖励模型，并使用离线单步拟合与局部代理评分。模型知识、学习方法和动作搜索方式并未被强行做成与RL完全相同。',
        '3. 3模式与16模式版本在此次校准及全部评估轨迹上相同。代码另用合成观测验证过16模式版本能够选择新增的模式7，排除了代码实际上仍只搜索3模式的错误；实际模式计数见comparison.json。不能推论所有环境下新增模式都无价值。',
        '4. 结果给出了一个不读取额外在线状态的更强可执行对照。即使不取得更多在线信息，也可以超过当前RL；这支持继续改进RL优化与策略结构。但本次并未隔离原差距的信息贡献，也不能声称求得相同信息条件下的最优策略。','',
        '## 校准预测器的不确定性','']
    x=data['heldout_calibration_predictor_quality']['modes_3']
    lines += [f'独立校准验证的单步分数RMSE为{x["rmse_score_x100"]:.4f}。按同一局部选模规则、比较三种资源候选，拟合选择相对完整当前状态最佳候选的平均单步损失为{x["resource_choice_regret_score_x100"]:.4f}分，最优或并列最优选择比例{100*x["optimal_or_tied_fraction"]:.2f}%。因此这仍是观测条件下的近似得分预测，绝不是精确全局评分。没有用主评估集来重新拟合。','',
        '## 延续上次随机实例','',
        f'沿用训练种子218、评估种子{data["random_example_selection"]["evaluation_seed"]}；各场景完整600时隙，不重新挑选有利案例。','']
    prior=h.read(old_folder_path()/'random_instruction_examples/results.json')
    for scene in data['random_examples']:
        lines += [f'### {scene}', '', '| 方法 | 分数×100 | 平均AoI | 预测PSNR | 交付数/时隙 | 资源/时隙 |', '|---|---:|---:|---:|---:|---:|']
        exrows={names[k]:v for k,v in data['random_examples'][scene].items()}
        for k in ('selector','R_myopic','R_instruction'):
            exrows[old['methods'][k]['label']]=prior['per_scene'][scene][k]['metrics']
        for label,x in exrows.items():
            lines.append(f'| {label} | {100*x["common_reward"]:.4f} | {x["mean_aoi"]:.4f} | {x["delivered_predicted_psnr"]:.4f} | {x["deliveries_per_slot"]:.4f} | {x["channel_uses_per_slot"]:.2f} |')
    lines+=['','## 全部13场景','']
    for scope in [*data['protocol']['scenarios'],'true_instruction_0','true_instruction_1','true_instruction_2']:
        lines += [f'### {scope}', '', '| 方法 | 分数×100 | 平均AoI | 交付预测PSNR |', '|---|---:|---:|---:|']
        for label,per in rows.items():
            x=per[scope]
            lines.append(f'| {label} | {100*x["common_reward"]:.4f} | {x["mean_aoi"]:.4f} | {x["delivered_predicted_psnr"]:.4f} |')
    lines+=['','## 核验与范围','',
        '新增4组方法共1040个评估回合、624000物理时隙；校准共260回合、156000物理时隙、468000个当步候选标签。逐时隙独立重算缓存、AoI、载荷、质量和奖励；与旧评估外生序列逐项配对；冻结代码、profile、拟合模型及结果哈希均验证。SUT/各UAV隔离输入测试、环境批次顺序不干扰决策测试和数值树导出一致性检查均通过。','',
        '这些是沿用已有开发评估集的结果，RL三个种子也是此前筛选所得；不冒充新的独立泛化确认。原强规则和所有旧训练结果保留，本次未重新训练RL或修改论文。','']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')


def old_folder_path():
    return OUT.parent/'2026-09-11_selected_seed_comparison'


if __name__=='__main__':
    main()
