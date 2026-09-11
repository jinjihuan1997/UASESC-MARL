"""First prespecified test episode and frozen-policy component interventions."""
from helpers import *
from evaluation import FIELDS,summarize


SCENARIO='switch300_2_to_1'
VARIANTS=['original_RL','rule_mode','rule_resources','rule_both']
WINDOWS={'quality':(0,300),'aoi':(300,600),'first20_after':(300,320),'later_after':(320,600),'whole':(0,600)}


def describe(data,modes,beta,ages):
    result=summarize(data)
    age=ages.reshape(*data.shape[:-1],30).astype(float)
    result['age_total_cost']=sum(result[k] for k in ['age_mean_cost','age_max_cost','age_tail_cost'])
    result['low_mode_fraction']=float(np.isin(modes,[0,4,8,12]).mean())
    result['middle_mode_fraction']=float(np.isin(modes,[5,9,13]).mean())
    result['no_send_fraction']=float((modes<0).mean())
    result['mean_excess_above4']=float(np.maximum(age-4,0).mean())
    result['mean_resource_fractions']=beta.reshape(-1,3).mean(0).tolist()
    unique,counts=np.unique(age.max(-1),return_counts=True)
    result['max_age_counts']={str(int(a)):int(n) for a,n in zip(unique,counts)}
    return result


def main():
    manifest=verify();cfg=config('ref8',seed=85);selection=read(HERE/'rule_selection.json')['ref8']
    assert manifest['evaluation_seeds'][0]==20262501
    assert manifest['scenarios'][SCENARIO]==[[0,2],[300,1]]
    sources={};source_hashes={};originals={}
    model=HERE/'jobs/ref8/seed_85/IC_HAPPO'
    status=read(model/'status.json')
    assert status['state']=='complete' and status['completed_steps']==1000000
    for name,h in status['checkpoint_hashes'].items():
        assert digest(model/name)==h
        source_hashes[str((model/name).relative_to(HERE))]=h
    for label,item in [('RL','ref8/seed_85/IC_HAPPO'),('rule','ref8/rules/R_instruction')]:
        f=HERE/'evaluation'/item/f'{SCENARIO}.npz';meta=read(f.with_suffix('.json'))
        assert digest(f)==meta['trace_sha256'];source_hashes[str(f.relative_to(HERE))]=digest(f)
        with np.load(f) as z:originals[label]={k:z[k].copy() for k in z.files}
        sources[label]=meta
    assert sources['RL']['external_hashes']==sources['rule']['external_hashes']
    summaries={};trace_list=[];mode_list=[];beta_list=[];age_list=[];actors=None
    for name in VARIANTS:
        env,obs,_,mask=make_env(cfg,EVAL_SEEDS,manifest['scenarios'][SCENARIO])
        assert external_hashes(env)==sources['RL']['external_hashes']
        if actors is None:actors=load_actors(cfg,env,HERE/'jobs/ref8/seed_85/IC_HAPPO')
        rows=[];modes=[];betas=[];ages=[]
        for slot in range(600):
            acts=actions_for(actors,obs,mask)
            gid=int(env.context()[0][0]);rule=rule_actions(env,selection['by_instruction'][gid])
            if name in ['rule_resources','rule_both']:acts[0]=rule[0]
            if name in ['rule_mode','rule_both']:acts[1:]=rule[1:]
            obs,_,mask,info,v=checked_step(env,acts);v['instruction_id']=arr(info['gid'])
            row=np.column_stack([v[k] for k in FIELDS]);rows.append(row)
            modes.append(arr(info['mode']));betas.append(arr(env.beta));ages.append(arr(env.aoi).astype(np.uint16))
            if name in ['original_RL','rule_both']:
                original=originals['RL' if name=='original_RL' else 'rule']
                np.testing.assert_allclose(row,original['trace'][slot],atol=1e-8,rtol=1e-10)
                np.testing.assert_array_equal(modes[-1],original['modes'][slot])
                np.testing.assert_array_equal(ages[-1],original['aoi_after'][slot])
                np.testing.assert_allclose(betas[-1],original['resource_fractions'][slot],atol=1e-12,rtol=0)
        data=np.stack(rows);modes=np.stack(modes);beta=np.stack(betas);ages=np.stack(ages)
        trace_list.append(data);mode_list.append(modes);beta_list.append(beta);age_list.append(ages)
        summaries[name]={}
        for window,(start,end) in WINDOWS.items():
            sl=slice(start,end)
            summaries[name][window]=dict(first_episode=describe(data[sl,0],modes[sl,0],beta[sl,0],ages[sl,0]),
                all20_episodes=describe(data[sl],modes[sl],beta[sl],ages[sl]))
        print('Completed frozen-policy rollout',name,flush=True)
    output=HERE/'worked_example_replay.npz'
    assert not output.exists()
    np.savez_compressed(output,trace=np.stack(trace_list),modes=np.stack(mode_list),resource_fractions=np.stack(beta_list),
        aoi_after=np.stack(age_list),variants=np.asarray(VARIANTS),fields=np.asarray(FIELDS),seeds=np.asarray(EVAL_SEEDS))
    first={}
    for label,z in originals.items():
        first[label]={k:describe(z['trace'][lo:hi,0],z['modes'][lo:hi,0],z['resource_fractions'][lo:hi,0],z['aoi_after'][lo:hi,0]) for k,(lo,hi) in WINDOWS.items()}
    differences={}
    for window in WINDOWS:
        a=first['RL'][window];b=first['rule'][window]
        d=dict(quality_difference=a['quality_credit']-b['quality_credit'],
            aoi_savings=b['age_total_cost']-a['age_total_cost'],resource_savings=b['resource_cost']-a['resource_cost'])
        d['reward_difference']=a['common_reward']-b['common_reward']
        np.testing.assert_allclose(d['quality_difference']+d['aoi_savings']+d['resource_savings'],d['reward_difference'],atol=1e-12,rtol=0)
        differences[window]=d
    quality_gap=-differences['quality']['reward_difference']/2
    total_gap=-differences['whole']['reward_difference']
    instruction_ids=np.asarray(originals['RL']['trace'][:,0,FIELDS.index('instruction_id')],int)
    assert np.all(instruction_ids[:300]==2) and np.all(instruction_ids[300:]==1)
    # A few chronological checkpoints, selected before viewing replay results.
    checkpoints={}
    for label,z in originals.items():
        checkpoints[label]={}
        for t in [0,1,150,299,300,301,305,319,450,599]:
            d=describe(z['trace'][t:t+1,0],z['modes'][t:t+1,0],z['resource_fractions'][t:t+1,0],z['aoi_after'][t:t+1,0])
            d.update(slot_number=t+1,slot_index=t,instruction_id=int(z['trace'][t,0,FIELDS.index('instruction_id')]),
                     executed_modes=z['modes'][t,0].tolist())
            checkpoints[label][str(t)]=d
    out=dict(state='PASS',selection='first_training_seed_and_first_evaluation_seed_not_selected_by_reward',
        training_seed=85,evaluation_seed=20262501,scenario=SCENARIO,slots_per_episode=600,
        model_training_steps=1000000,aoi_reward_ref=8,first_episode=first,differences=differences,
        fraction_of_whole_reward_gap_from_quality_segment=quality_gap/total_gap,
        chronological_checkpoints=checkpoints,interventions=summaries,
        intervention_scope='same_initial_conditions; frozen policy recomputes its remaining decisions on the intervened trajectory',
        intervention_slots=48000,original_and_rule_replay_match=True,input_hashes=source_hashes,
        replay_hash=digest(output),no_training=True,physical_and_reward_checks=True)
    write(HERE/'worked_example.json',out)
    lines=['# 参考值8：一个完整指令切换回合','',
        '按文件顺序选择首个训练种子85、首个测试种子20262501；不是按结果挑选。新模型100万步。前300槽质量指令，后300槽AoI指令。以下得分均乘100。',
        '', '| 阶段 | 方法 | 质量加分 | AoI扣分 | 资源扣分 | 净分 | AoI | PSNR | 更新/槽 | 低档占比 |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for window in ['quality','aoi','whole']:
        for method in ['RL','rule']:
            v=first[method][window]
            lines.append(f'| {window} | {method} | {100*v["quality_credit"]:.4f} | {100*v["age_total_cost"]:.4f} | {100*v["resource_cost"]:.4f} | {100*v["common_reward"]:.4f} | {v["mean_aoi"]:.4f} | {v["delivered_predicted_psnr"]:.4f} | {v["deliveries_per_slot"]:.4f} | {100*v["low_mode_fraction"]:.2f}% |')
    lines+=['',f'完整回合差距中，质量阶段贡献约{100*quality_gap/total_gap:.2f}%。全程平均AoI优势主要出现在质量阶段；AoI阶段自身的平均和最大AoI反而略高。',
        '', '## 单独替换策略组成部分的完整回合回放','',
        '保留训练好的网络，分别只替换模式、只替换资源或全部使用规则。每个分支从相同初始条件运行600槽；RL根据各分支实际状态继续决策。改变的动作会影响后续状态，因此这不是单步重评分，也不是对最优长期策略的证明。',
        '', '| 分支 | 首回合质量段分数 | 首回合AoI段分数 | 首回合总分 | 20回合平均总分 |',
        '|---|---:|---:|---:|---:|']
    for name in VARIANTS:
        v=summaries[name]
        lines.append(f'| {name} | {100*v["quality"]["first_episode"]["common_reward"]:.4f} | {100*v["aoi"]["first_episode"]["common_reward"]:.4f} | {100*v["whole"]["first_episode"]["common_reward"]:.4f} | {100*v["whole"]["all20_episodes"]["common_reward"]:.4f} |')
    lines+=['','这里只诊断一个训练种子的固定模型，不能替代三种子总体结论，也不能从回放唯一确定训练未充分、表示限制或奖励偏好中的哪一项是根因。',
        '', '执行模式按等价低档0/4/8/12、中档5/9/13分类。mode是共同执行器实际执行的模式，不一定等于原始最高偏好；AoI仍由共同的按AoI优先服务调度器更新。',
        '',f'原RL和完整规则均逐时隙重现保存结果，合计48000时隙物理/奖励核算通过。完整数字：[worked_example.json]({HERE}/worked_example.json)。']
    (HERE/'WORKED_EXAMPLE.md').write_text('\n'.join(lines)+'\n')
    verify()
    write(HERE/'worked_example_hashes.json',{str(p.relative_to(HERE)):digest(p) for p in [HERE/'worked_example.py',HERE/'worked_example.json',HERE/'WORKED_EXAMPLE.md',output]})
    print(json.dumps(dict(first_episode=first,differences=differences,quality_gap_fraction=quality_gap/total_gap,
        interventions={k:{w:summaries[k][w] for w in ['quality','aoi','whole']} for k in VARIANTS}),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
