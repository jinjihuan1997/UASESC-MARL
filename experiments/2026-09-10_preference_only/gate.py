"""Predeclared rule-family tradeoff gate; no policy training."""
import csv
import json
from pathlib import Path
import sys
import time
from probe_common import *


def validate_interface():
    env,obs,state,masks=make_env([20261499,20261498],hidden=True)
    original=env.instructions.clone()
    checks=0
    for slot in range(600):
        observations=[];available=[]
        for g in range(3):
            env.instructions[slot]=g
            o,s,m=env.observe();observations.append(o);available.append(m)
        assert all(torch.equal(observations[0],o) for o in observations)
        assert all(torch.equal(available[0],m) for m in available)
        actions=rule_actions(env,RULES[(slot//31)%len(RULES)])
        dynamic={k:v.clone() for k,v in vars(env).items() if isinstance(v,torch.Tensor)}
        next_states=[];reward_values=[]
        for g in range(3):
            for k,v in dynamic.items(): setattr(env,k,v.clone())
            env.step_index=slot;env.instructions[slot]=g
            _,_,_,info,metrics=checked_step(env,actions)
            next_states.append({k:getattr(env,k).clone() for k in ('aoi','q','tau','beta','psi','gamma_us','gamma_sat')})
            reward_values.append(metrics['common_reward'])
        for other in next_states[1:]:
            for k in other: assert torch.equal(next_states[0][k],other[k]),k
        checks+=env.count
    # Correct and hidden actors differ only in explicit input fields; critics identical.
    full,fo,fs,fm=make_env([20261499,20261498],schedule=[[0,2]])
    hidden,ho,hs,hm=make_env([20261499,20261498],schedule=[[0,2]],hidden=True)
    allowed=torch.as_tensor(explicit_columns(full.p))
    assert torch.equal(fo[:,~allowed],ho[:,~allowed]) and torch.equal(fs,hs) and torch.equal(fm,hm)
    assert torch.count_nonzero(ho[:,allowed])==0
    # Independent CPU simulator comparison, 2 full episodes, varied rules and tasks.
    schedule=[[0,0],[200,1],[400,2]]
    tensor,_,_,_=make_env([20261499,20261498],schedule=schedule)
    refs=[]
    for seed in (20261499,20261498):
        args=config()['env_args'];args.update(instruction_mode_strategy='explicit_evaluation',explicit_instruction_schedule=schedule)
        ref=SCUAVEnv(args);ref.seed(seed);ref.reset();refs.append(ref)
    max_reward_error=0.
    for slot in range(600):
        actions=rule_actions(tensor,RULES[(slot//31)%len(RULES)])
        _,_,_,info,metrics=checked_step(tensor,actions)
        for i,ref in enumerate(refs):
            _,_,_,_,infos,_=ref.step([arr(a)[i] for a in actions])
            ext=closed.external_metrics(ref,infos[0])
            for key in ('mean_aoi','deliveries','predicted_quality_sum','channel_uses','common_reward'):
                np.testing.assert_allclose(metrics[key][i],ext[key],atol=1e-7,rtol=1e-10,err_msg=key)
            max_reward_error=max(max_reward_error,abs(metrics['common_reward'][i]-ext['common_reward']))
    return dict(state='PASS',same_state_hidden_input_mask_checks=checks,
        same_action_physical_transition_checks=checks,reference_slots=1200,max_reward_error=max_reward_error,
        explicit_fields_only=True,critic_equal_between_groups=True)


def run_rules(seeds,split):
    all_results={};expected_hashes=None
    for name in RULES:
        env,_,_,_=make_env(seeds)
        hashes=external_hashes(env)
        if expected_hashes is None: expected_hashes=hashes
        assert hashes==expected_hashes
        folder=HERE/'gate'/split;folder.mkdir(parents=True,exist_ok=True)
        file=folder/f'{name}.npz'
        records=[]
        for slot in range(600):
            _,_,_,info,metrics=checked_step(env,rule_actions(env,name))
            records.append(np.column_stack([arr(info[k]) for k in ('quality_term','aoi_term','load_term')]+
                [metrics[k] for k in ('mean_aoi','deliveries','predicted_quality_sum','channel_uses')]))
        trace=np.stack(records)
        np.savez_compressed(file,trace=trace,seeds=np.asarray(seeds),fields=np.asarray(['quality_term','aoi_term','load_term','mean_aoi','deliveries','predicted_quality_sum','channel_uses']))
        # Separate disk read/reaggregation detects serialization and aggregation issues.
        with np.load(file) as saved:
            np.testing.assert_array_equal(trace,saved['trace'])
            values=saved['trace'].mean(0)
        w=np.asarray(env.p.reward_weights_by_instruction)
        rewards=values[:,0,None]*w[:,0]-values[:,1,None]*w[:,1]-values[:,2,None]*w[:,2]
        all_results[name]=dict(rewards=rewards.tolist(),mean_aoi=values[:,3].tolist(),
            psnr=(values[:,5]/values[:,4]).tolist(),deliveries_per_slot=values[:,4].tolist(),
            channel_uses_per_slot=values[:,6].tolist(),external_hashes=hashes,trace_sha256=digest(file))
        print(f'{split} {name}: complete ({len(seeds)} episodes)',flush=True)
    return all_results


def choose(calibration):
    scores=np.asarray([np.mean(calibration[name]['rewards'],axis=0) for name in RULES])
    return dict(by_instruction=[RULES[i] for i in scores.argmax(0)],single=RULES[int(scores.mean(-1).argmax())],
        calibration_mean_rewards=scores.tolist(),selection='calibration_only')


def analyze(calibration,held,selection,validation):
    names=selection['by_instruction'];single=selection['single']
    a,q=names[1],names[2]
    differences=np.column_stack([np.asarray(held[names[g]]['rewards'])[:,g]-np.asarray(held[single]['rewards'])[:,g] for g in range(3)]).mean(-1)
    delta_q=float(np.mean(held[q]['psnr'])-np.mean(held[a]['psnr']))
    delta_a=float(np.mean(held[q]['mean_aoi'])-np.mean(held[a]['mean_aoi']))
    own_a=float(np.mean(np.asarray(held[a]['rewards'])[:,1]-np.asarray(held[q]['rewards'])[:,1]))
    own_q=float(np.mean(np.asarray(held[q]['rewards'])[:,2]-np.asarray(held[a]['rewards'])[:,2]))
    conditions=dict(different_selected_rules=a!=q,quality_tradeoff=delta_q>=.5,aoi_tradeoff=delta_a>=.1,
        each_prefers_own_rule=own_a>0 and own_q>0,mean_gain=float(differences.mean())>=.001,paired_direction=int((differences>0).sum())>=16)
    result=dict(state='PASS' if all(conditions.values()) else 'FAIL',conditions=conditions,selection=selection,
        quality_minus_aoi_psnr=delta_q,quality_minus_aoi_mean_aoi=delta_a,
        aoi_own_rule_reward_gain=own_a,quality_own_rule_reward_gain=own_q,
        informed_minus_single_mean_reward=float(differences.mean()),positive_pairs=int((differences>0).sum()),
        paired_reward_differences=differences.tolist(),calibration=calibration,held_out=held,validation=validation,
        episodes=len(RULES)*(len(CAL_SEEDS)+len(GATE_SEEDS)),slots=len(RULES)*(len(CAL_SEEDS)+len(GATE_SEEDS))*600,
        note='Finite rule-family feasibility gate, not proof of RL benefit or an upper bound on hidden policies.')
    write(HERE/'gate_results.json',result)
    lines=['规则预检完成。'+('达到预先固定的短训练启动条件。' if result['state']=='PASS' else '未达到预先固定的短训练启动条件，本轮不启动训练。'),'',
        '本实验只改变质量、AoI 和载荷的奖励偏好；三个指令使用完全相同的质量底线、动作筛选和执行规则。原正式实验保持不变。',
        '',f'校准集为各指令选出的规则：{names}；单一通用规则：{single}。选择后未根据保留集更改。',
        '', '| 控制器 | AoI | 交付 PSNR | 更新/槽 | 均衡奖励 | AoI 奖励 | 质量奖励 |','|---|---:|---:|---:|---:|---:|---:|']
    for name in RULES:
        v=held[name];r=np.mean(v['rewards'],axis=0)
        lines.append(f'| {name} | {np.mean(v["mean_aoi"]):.4f} | {np.mean(v["psnr"]):.4f} | {np.mean(v["deliveries_per_slot"]):.4f} | {r[0]:.6f} | {r[1]:.6f} | {r[2]:.6f} |')
    lines += ['',f'质量偏好选择相对 AoI 偏好选择：质量 {delta_q:+.4f} dB，AoI {delta_a:+.4f} 槽。根据指令选择规则相对单一规则，平均奖励 {differences.mean():+.6f}，20 对中 {int((differences>0).sum())} 对为正。',
        '',f'各门槛：`{conditions}`。',
        '', '这里的奖励质量项是每槽成功交付质量收益之和，并非只奖励单个视频的清晰度，因此“质量偏好”不必选择最高 PSNR 档。表内 PSNR 仅表示固定平均模型的交付质量。',
        '', '验证通过：同状态隐藏输入与可选动作对指令不变，同动作跨指令物理转移一致；两组 critic 一致；1200 槽与独立 CPU 环境逐槽对照，资源/质量/缓存与奖励核算通过。',
        '', '门槛仅验证这组规则中存在偏好取舍，不证明 RL 优于规则，也不证明隐藏策略无法根据历史状态获得好表现。',
        '',f'证据：[协议]({HERE}/PROTOCOL.md)、[完整结果]({HERE}/gate_results.json)、[校准选择]({HERE}/gate_selection.json)。']
    (HERE/'GATE_REPORT.md').write_text('\n'.join(lines)+'\n')
    return result


if __name__=='__main__':
    torch.set_num_threads(1)
    verify_parent()
    sources={p.name:digest(p) for p in HERE.glob('*.py')}
    identity=dict(created_utc=stamp(),purpose='preference_only_diagnostic',sources=sources,
        protocol_sha256=digest(HERE/'PROTOCOL.md'),parent_manifest_sha256=digest(PARENT/'manifest.json'),
        calibration_seeds=CAL_SEEDS,held_out_seeds=GATE_SEEDS)
    assert not (HERE/'gate_results.json').exists(),'Preserve existing results'
    write(HERE/'gate_identity.json',identity)
    for hidden in (False,True):
        c=config(hidden);write(HERE/'configs'/f'{c["main_args"]["exp_name"]}.json',c)
    validation=validate_interface();write(HERE/'interface_validation.json',validation)
    print('Interface and independent CPU transition validation PASS',flush=True)
    calibration=run_rules(CAL_SEEDS,'calibration')
    selection=choose(calibration);write(HERE/'gate_selection.json',selection)
    held=run_rules(GATE_SEEDS,'held_out')
    result=analyze(calibration,held,selection,validation)
    verify_parent()
    print(json.dumps({k:v for k,v in result.items() if k not in ('calibration','held_out')},ensure_ascii=False,indent=2),flush=True)
