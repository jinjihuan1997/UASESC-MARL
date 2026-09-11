"""Frozen 10M controller swaps; diagnostic evidence, never substitute for learned results."""
import argparse,concurrent.futures,subprocess
from pathlib import Path
import sys
PARENT=Path(__file__).resolve().parent.parent/'2026-09-10_sequential_long_training'
sys.path.insert(0,str(PARENT))
from helpers import (torch,np,read,config,digest,make_env,external_hashes,
                     load_actors,checked_step,arr,write,verify,SEEDS,EVAL_SEEDS,
                     stamp,rule_actions)
from evaluation import FIELDS,summarize
ROOT=Path(__file__).resolve().parent
VARIANTS=['rule_modes_rl_resources','rl_modes_rule_resources','quality_rule_modes_only']

@torch.no_grad()
def actions(env,actors,obs,masks,variant):
    rnn=torch.zeros((env.count,1,256));active=torch.ones((env.count,1))
    selection=read(PARENT/'rule_selection.json');gid=int(env.context()[0][0])
    rule=rule_actions(env,selection['by_instruction'][gid])
    resource=rule[0] if variant=='rl_modes_rule_resources' else actors[0].act(obs[:,0],rnn,active,masks[:,0],deterministic=True)[0]
    post,_,available=env.allocate_resources(resource)
    if variant=='rule_modes_rl_resources' or (variant=='quality_rule_modes_only' and gid==2):
        modes=rule[1:]
    else:
        modes=[actors[i].act(post[:,i],rnn,active,available[:,i],deterministic=True)[0] for i in range(1,env.n_agents)]
    return [resource]+modes

def worker(seed,variant):
    m=read(ROOT/'protocol.json');cfg=config('joint',seed);selection=read(PARENT/'rule_selection.json')
    model=PARENT/f'jobs/seed_{seed}/joint/milestones/steps_10000000';status=read(model/'status.json')
    for f,h in status['checkpoint_hashes'].items(): assert digest(model/f)==h
    folder=ROOT/'evaluation'/f'seed_{seed}'/variant;folder.mkdir(parents=True,exist_ok=False)
    actors=None;results={};parts=[]
    for scenario,schedule in m['scenarios'].items():
        env,obs,_,masks=make_env(cfg,m['evaluation_seeds'],schedule)
        expected=read(PARENT/f'evaluation/seed_{seed}/joint_at_10000000/{scenario}.json')['external_hashes']
        assert external_hashes(env)==expected
        if actors is None: actors=load_actors(cfg,env,model)
        rows=[];modes=[];betas=[];ages=[]
        for slot in range(600):
            act=actions(env,actors,obs,masks,variant)
            obs,_,masks,info,values=checked_step(env,act);values['instruction_id']=arr(info['gid'])
            rows.append(np.column_stack([values[f] for f in FIELDS]));modes.append(arr(info['mode']));betas.append(arr(env.beta));ages.append(arr(env.aoi))
        data=np.stack(rows);parts.append(data)
        file=folder/f'{scenario}.npz'
        np.savez_compressed(file,trace=data,fields=FIELDS,seeds=m['evaluation_seeds'],modes=np.stack(modes),resource_fractions=np.stack(betas),aoi_after=np.stack(ages))
        results[scenario]=dict(**summarize(data),trace_sha256=digest(file),external_hashes=expected)
        print(seed,variant,scenario,results[scenario]['common_reward'],flush=True)
    results['overall']=summarize(np.concatenate(parts))
    write(folder/'summary.json',results)
    write(folder/'status.json',dict(state='complete',seed=seed,variant=variant,episodes=13*20,summary_sha256=digest(folder/'summary.json')))

def main():
    assert not (ROOT/'protocol.json').exists()
    m=verify()
    jobs=[(seed,variant) for seed in SEEDS for variant in VARIANTS]
    source_hashes={f:h for f,h in m['input_hashes'].items()}
    write(ROOT/'protocol.json',dict(created_utc=stamp(),purpose='post_long_controller_swap_diagnosis',
        seeds=SEEDS,variants=VARIANTS,scenarios=m['scenarios'],evaluation_seeds=EVAL_SEEDS,
        learned_weights='joint 10M frozen',original_inputs=source_hashes,script_sha256=digest(__file__),
        limitation='Post-hoc component interventions on existing development seeds; hybrids are not pure RL and do not establish instruction benefit.'))
    (ROOT/'logs').mkdir()
    def launch(index):
        seed,variant=jobs[index]
        with (ROOT/'logs'/f'{seed}_{variant}.log').open('w') as log:
            result=subprocess.run(['taskset','-c',str(index%6),sys.executable,'-u',__file__,'--seed',str(seed),'--variant',variant],stdout=log,stderr=subprocess.STDOUT)
        assert result.returncode==0,(seed,variant)
        print('Completed',seed,variant,flush=True)
    # Six simultaneous workers, one core each; subsequent jobs form a second wave.
    for start in range(0,len(jobs),6):
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:list(pool.map(launch,range(start,min(start+6,len(jobs)))))
    from aggregate import audit_trace
    parent=read(PARENT/'report/results.json');original=parent['grouped_by_steps']['10000000'];combined={};episodes=0
    for variant in VARIANTS:
        summaries=[read(ROOT/'evaluation'/f'seed_{s}'/variant/'summary.json') for s in SEEDS]
        combined[variant]={scope:{k:float(np.mean([x[scope][k] for x in summaries])) for k in ['common_reward','mean_aoi','delivered_predicted_psnr','quality_credit','age_mean_cost','age_max_cost','age_tail_cost','resource_cost']} for scope in list(m['scenarios'])+['overall']}
        combined[variant]['paired_overall_minus_rl']=[x['overall']['common_reward']-parent['per_item'][f'seed_{seed}/joint_at_10000000']['overall']['common_reward'] for seed,x in zip(SEEDS,summaries)]
        for seed,summary in zip(SEEDS,summaries):
            for scene in m['scenarios']:
                f=ROOT/'evaluation'/f'seed_{seed}'/variant/f'{scene}.npz';assert digest(f)==summary[scene]['trace_sha256']
                with np.load(f) as z:audit_trace(z['trace'],z['aoi_after'])
                episodes+=20
    verify()
    write(ROOT/'summary.json',combined)
    write(ROOT/'audit.json',dict(state='PASS',episodes=episodes,slots=episodes*600,independent_reward_and_physics=True,paired_exogenous_tapes=True,original_inputs_unchanged=True,
        controller_swaps_only=True,no_optimizer_steps=True))
    lines=['# 1000万步模型的控制器交叉诊断','',
        '复用joint的三个1000万步模型。交叉替换只是定位工具，混合方法不作为新的纯RL结果。所有分数乘100。','',
        '| 方法 | 全部13场景 | 均衡 | AoI | 质量 |','|---|---:|---:|---:|---:|']
    for name in ['joint','R_instruction','R_myopic']:
        g=original[name];lines.append('| '+name+' | '+' | '.join(f'{100*g[s]["common_reward"]["mean"]:.4f}' for s in ['overall','fixed_0','fixed_1','fixed_2'])+' |')
    for name in VARIANTS:
        g=combined[name];lines.append('| '+name+' | '+' | '.join(f'{100*g[s]["common_reward"]:.4f}' for s in ['overall','fixed_0','fixed_1','fixed_2'])+' |')
    lines+=['','rule_modes_rl_resources：只用指令规则的模式，资源仍由RL决定。',
        'rl_modes_rule_resources：只用指令规则的资源，UAV看到这些当前预算后由RL选模式。',
        'quality_rule_modes_only：仅质量指令时用规则模式，其余指令保留RL；资源一直由RL决定。这是事后指令条件干预，不是学习出来的新方法。',
        '',f'独立审计{episodes}回合、{episodes*600}时隙。未改奖励、环境、原始权重。']
    (ROOT/'DIAGNOSIS.md').write_text('\n'.join(lines)+'\n')
    write(ROOT/'artifact_hashes.json',{str(f.relative_to(ROOT)):digest(f) for f in [ROOT/'summary.json',ROOT/'audit.json',ROOT/'DIAGNOSIS.md',ROOT/'protocol.json']})
    print('Diagnosis PASS',episodes,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int);p.add_argument('--variant',choices=VARIANTS);a=p.parse_args()
    if a.seed is not None:worker(a.seed,a.variant)
    else:main()
