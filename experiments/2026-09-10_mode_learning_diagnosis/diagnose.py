"""Frozen-model mode-learning diagnosis; source experiments are read-only."""
import sys
from pathlib import Path
OUT=Path(__file__).resolve().parent
REF=OUT.parent/'2026-09-10_aoi_ref8'
sys.path.insert(0,str(REF))
from helpers import *
from rule_tools import predict_reward
from evaluation import FIELDS
import time

SCENARIOS=['fixed_2','fixed_1','switch300_2_to_1']
VARIANTS=['deterministic','mode_sample','resource_sample','all_sample',
          'group_argmax','rule_mode_det_resources','rule_mode_sample_resources','rule_both']

def groups_for(env):
    groups=[]
    for i in range(env.M):
        for group in groups:
            if torch.equal(env.profiles[:,i],env.profiles[:,group[0]]):
                group.append(i);break
        else:groups.append([i])
    return groups

@torch.no_grad()
def probabilities(actors,obs,available):
    values=[]
    for i,a in enumerate(actors[1:],1):
        assert a.actor.act.hybrid_box_action
        head=a.actor.act.action_out
        assert len(head.categorical_heads)==1 and not head.continuous_heads
        logits=a.actor.act.get_logits(a.actor.base(obs[:,i]),available[:,i])
        values.append(torch.softmax(logits[:,:16],-1))
    return torch.stack(values,1)

@torch.no_grad()
def get_actions(actors,obs,available,variant,env,groups):
    rnn=torch.zeros((env.count,1,256));mask=torch.ones((env.count,1))
    actions=[]
    for i,a in enumerate(actors):
        stochastic=(i==0 and variant in ['resource_sample','all_sample','rule_mode_sample_resources']) or (i>0 and variant in ['mode_sample','all_sample'])
        actions.append(a.act(obs[:,i],rnn,mask,available[:,i],deterministic=not stochastic)[0])
    if variant=='group_argmax':
        ps=probabilities(actors,obs,available)
        mass=torch.stack([ps[:,:,g].sum(-1) for g in groups],-1)
        chosen=mass.argmax(-1)
        for u in range(env.U):
            mode=torch.tensor([g[0] for g in groups])[chosen[:,u]]
            actions[u+1]=2*torch.nn.functional.one_hot(mode,env.M).float()-1
    if variant.startswith('rule_mode') or variant=='rule_both':
        gid=int(env.context()[0][0]);rule=rule_actions(env,'m0_urgency' if gid==1 else 'm5_equal')
        actions[1:]=rule[1:]
        if variant=='rule_both':actions[0]=rule[0]
    return actions

def summary(data,modes):
    out={k:float(data[...,j].mean()) for j,k in enumerate(FIELDS)}
    out['psnr']=float(data[...,FIELDS.index('predicted_quality_sum')].sum()/data[...,FIELDS.index('deliveries')].sum())
    out['low_fraction']=float(np.isin(modes,[0,4,8,12]).mean())
    out['middle_fraction']=float(np.isin(modes,[5,9,13]).mean())
    out['episode_rewards']=data[...,FIELDS.index('common_reward')].mean(0).tolist()
    return out

def main():
    torch.set_num_threads(1)
    manifest=verify()
    assert not (OUT/'results.json').exists()
    selection=read(REF/'calibration/selection.json') if (REF/'calibration/selection.json').exists() else None
    hashes={str(p.relative_to(REF)):digest(p) for p in REF.glob('jobs/ref8/seed_*/IC_HAPPO/*.pt')}
    proto=dict(created_utc=stamp(),seeds=SEEDS,eval_seeds=EVAL_SEEDS,scenarios=SCENARIOS,variants=VARIANTS,
        training=False,physics_and_reward='unchanged frozen ref8',mode_groups='exact equality of all frozen profile quantities and SNR columns',
        probe_slots=list(range(0,600,20)),rng_seed_rule='20262600 + training_seed; reset each branch',
        expected_slots=3*3*8*20*600,model_hashes=hashes,source_manifest_sha256=digest(REF/'manifest.json'),script_sha256=digest(__file__))
    write(OUT/'protocol.json',proto)
    results={};probes=[];groups=None;paired_hashes={}
    started=time.monotonic()
    for seed in SEEDS:
        cfg=config('ref8',seed=seed)
        for scenario in SCENARIOS:
            for variant in VARIANTS:
                env,obs,_,available=make_env(cfg,EVAL_SEEDS,manifest['scenarios'][scenario])
                ex=external_hashes(env);key=f'{seed}/{scenario}'
                if key in paired_hashes:assert paired_hashes[key]==ex
                else:paired_hashes[key]=ex
                actors=load_actors(cfg,env,REF/f'jobs/ref8/seed_{seed}/IC_HAPPO')
                if groups is None:groups=groups_for(env)
                assert groups==groups_for(env)
                torch.manual_seed(20262600+seed)
                rows=[];mode_rows=[]
                for slot in range(600):
                    actions=get_actions(actors,obs,available,variant,env,groups)
                    if variant=='deterministic' and slot%20==0:
                        ps=probabilities(actors,obs,available)
                        cf=[]
                        for gid in range(3):
                            o=obs.clone();start=env.p.uav_obs_dim-9
                            o[:,1:,start:start+3]=0;o[:,1:,start+gid]=1
                            cf.append(arr(probabilities(actors,o,available)))
                        rewards=[]
                        for u in range(env.U):
                            by_mode=[]
                            for g in groups:
                                a=list(actions);a[u+1]=torch.zeros_like(a[u+1]);a[u+1][:,g[0]]=1
                                by_mode.append(arr(predict_reward(env,a)))
                            rewards.append(np.stack(by_mode,-1))
                        probes.append(dict(seed=seed,scenario=scenario,slot=slot,
                            gid=int(env.context()[0][0]),probabilities=arr(ps),counterfactual_probabilities=np.stack(cf),
                            one_step_reward_by_agent_group=np.stack(rewards,1),
                            one_step_original_reward=arr(predict_reward(env,actions))))
                    obs,_,available,info,values=checked_step(env,actions)
                    values['instruction_id']=arr(info['gid'])
                    rows.append(np.column_stack([values[k] for k in FIELDS]));mode_rows.append(arr(info['mode']))
                data=np.stack(rows);modes=np.stack(mode_rows)
                name=f'{seed}/{scenario}/{variant}'
                results[name]={'whole':summary(data,modes),'first300':summary(data[:300],modes[:300]),'last300':summary(data[300:],modes[300:])}
                if variant=='deterministic':
                    path=REF/f'evaluation/ref8/seed_{seed}/IC_HAPPO/{scenario}.npz'
                    original=np.load(path)
                    np.testing.assert_allclose(data,original['trace'],atol=1e-8,rtol=1e-10)
                    np.testing.assert_array_equal(modes,original['modes'])
                write(OUT/'progress.json',dict(completed=name,branches=len(results),total_branches=72,seconds=time.monotonic()-started))
                print(name,'reward',results[name]['whole']['common_reward'],'Q',results[name]['whole']['psnr'],flush=True)
    # Numeric arrays, with row metadata retained in JSON; never pickle.
    fields=['probabilities','counterfactual_probabilities','one_step_reward_by_agent_group','one_step_original_reward']
    np.savez_compressed(OUT/'probes.npz',**{k:np.stack([p[k] for p in probes]) for k in fields})
    write(OUT/'probe_index.json',[{k:p[k] for k in ['seed','scenario','slot','gid']} for p in probes])
    for f,h in hashes.items():assert digest(REF/f)==h
    verify()
    write(OUT/'results.json',dict(state='PASS',protocol=proto,groups=groups,results=results,
        external_hashes=paired_hashes,original_replay_match=True,slots=proto['expected_slots'],wall_seconds=time.monotonic()-started))
    print('COMPLETE',flush=True)

if __name__=='__main__':main()
