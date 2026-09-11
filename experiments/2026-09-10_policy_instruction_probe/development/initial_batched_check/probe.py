"""Read-only fixed-state actor-input intervention with original-trace replay."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='1'
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
PROJECT=HERE.parent.parent
RUN=PROJECT/'experiments/2026-09-09_instruction_long_training/runs/three_seed_sc_20260909'
sys.path.insert(0,str(RUN/'source'))
import numpy as np
import torch
from common import write,stamp
from multiseed_protocol import read,verify_run,verify_model
from training_checkpoint import digest
from evaluate import load_actors
from tensor_env import TensorSCEnv

VARIANTS=['original']+[f'{kind}_{g}' for kind in ('full','id','limits') for g in range(3)]+['zero']
V=len(VARIANTS)


def array(t): return t.detach().cpu().numpy()


def context_columns(env):
    p=env.p
    assert p.include_instruction_id_in_obs and p.include_instruction_constraints_in_obs
    sut_start=3*env.U+1
    uav_start=1+3*env.K+env.M+env.K*env.M+env.U
    assert p.sut_obs_dim==sut_start+4
    assert p.uav_obs_dim==uav_start+9
    allowed=torch.zeros((env.n_agents,p.obs_dim_common),dtype=torch.bool)
    allowed[0,sut_start:sut_start+4]=True
    allowed[1:,uav_start:uav_start+3]=True
    allowed[1:,p.uav_obs_dim-2:p.uav_obs_dim]=True
    return sut_start,uav_start,allowed


def interventions(env,obs,visible):
    p=env.p
    sut,uav,allowed=context_columns(env)
    base=obs.clone()
    if not visible: base[:,allowed]=0
    result=base[None].expand(V,-1,-1,-1).clone()
    if visible:
        for idx,name in enumerate(VARIANTS):
            if name=='original': continue
            if name=='zero':
                result[idx,:,allowed]=0
                continue
            kind,g=name.split('_');g=int(g)
            if kind in ('full','id'):
                result[idx,:,0,sut:sut+3]=0
                result[idx,:,1:,uav:uav+3]=0
                result[idx,:,0,sut+g]=1
                result[idx,:,1:,uav+g]=1
            if kind in ('full','limits'):
                result[idx,:,0,sut+3]=p.A_limit_by_instruction[g]/p.A_limit_context_ref
                result[idx,:,1:,p.uav_obs_dim-2]=(env.requirements[g,env.buckets]/p.Q_max).float()
                result[idx,:,1:,p.uav_obs_dim-1]=p.A_limit_by_instruction[g]/p.A_limit_context_ref
        # Presenting the true task must reconstruct exactly the original observation.
        true_g=env.instructions[env.step_index]
        for g in range(3):
            sel=true_g==g
            if bool(sel.any()):
                assert torch.equal(result[VARIANTS.index(f'full_{g}'),sel],base[sel])
    assert torch.equal(result[:,:,:,~allowed[0]] if False else result[:,:,~allowed],
                       base[None].expand_as(result)[:,:,~allowed])
    if not visible: assert torch.equal(result,base[None].expand_as(result))
    return result


@torch.no_grad()
def forward(actors,variants,available,env,validate_native=False):
    nvariants,E,A,D=variants.shape
    x=variants.reshape(nvariants*E,A,D)
    masks=available[None].expand(nvariants,-1,-1,-1).reshape(nvariants*E,A,-1)
    actions=[];raw=[];masked=[]
    for i,actor in enumerate(actors):
        policy=actor.actor
        assert not policy.use_recurrent_policy and not policy.use_naive_recurrent_policy
        features=policy.base(x[:,i])
        action,_=policy.act(features,masks[:,i],deterministic=True)
        if validate_native:
            native,_=actor.act(x[:,i],np.zeros((len(x),1,256),np.float32),
                np.ones((len(x),1),np.float32),masks[:,i],deterministic=True)
            assert torch.equal(action,native)
        actions.append(action)
        if i:
            layer=policy.act.action_out
            assert len(layer.categorical_heads)==1
            head=layer.categorical_heads[0]
            logits=head.get_logits(features)
            raw.append(torch.softmax(logits,-1).squeeze(1))
            prob=head(features,masks[:,i,:env.M].reshape(-1,1,env.M)).probs.squeeze(1)
            masked.append(prob)
            assert torch.equal(prob.argmax(-1),action.argmax(-1))
    sut=actions[0].double()
    b=((sut+1)/2).clamp_min(0)
    total=b.sum(-1,keepdim=True)
    b=torch.where(total>0,b/total.clamp_min(1e-12),torch.full_like(b,1/env.U))
    b=(env.p.beta_sat_lower_bound+(1-env.p.beta_sat_lower_bound*env.U)*b).clamp(env.p.beta_sat_lower_bound,1.)
    b=b/b.sum(-1,keepdim=True).clamp_min(1e-12)
    return dict(beta=array(b).reshape(nvariants,E,env.U).transpose(1,0,2),
        raw_prob=array(torch.stack(raw,1)).reshape(nvariants,E,env.U,env.M).transpose(1,0,2,3),
        masked_prob=array(torch.stack(masked,1)).reshape(nvariants,E,env.U,env.M).transpose(1,0,2,3),
        mode=array(torch.stack([a.argmax(-1) for a in actions[1:]],1)).reshape(nvariants,E,env.U).transpose(1,0,2))


def load_traces(item,scenario,seeds):
    keys=['common_reward','mean_aoi','deliveries','predicted_quality_sum','channel_uses']
    fields={k:[] for k in keys+['sut','mode','executed','beta','gid']}
    hashes={}
    for seed in seeds:
        stem=f'{scenario}_seed{seed}'
        folder=RUN/'evaluation'/item
        summary=read(folder/'episodes'/f'{stem}.json')
        file=folder/'traces'/f'{stem}.csv'
        assert digest(file)==summary['trace_sha256']
        hashes[str(file.relative_to(RUN))]=summary['trace_sha256']
        with file.open(newline='') as stream: rows=list(csv.DictReader(stream))
        assert len(rows)==600
        for key in keys: fields[key].append([float(row[key]) for row in rows])
        for key,col in [('sut','proposed_sut_logits'),('mode','proposed_modes'),('executed','executed_modes'),('beta','resource_fractions')]:
            fields[key].append([json.loads(row[col]) for row in rows])
        fields['gid'].append([int(row['instruction_id']) for row in rows])
    return {k:np.asarray(v) for k,v in fields.items()},hashes


def run_one(algorithm,seed,manifest):
    method='IC_'+algorithm.upper()
    hidden_method=algorithm.upper()+'_hidden_instruction'
    item=f'seed_{seed}/{method}';hidden_item=f'seed_{seed}/{hidden_method}'
    config=read(RUN/'configs'/f'{item}.json')
    hidden_config=read(RUN/'configs'/f'{hidden_item}.json')
    model=verify_model(RUN/'jobs'/item,10000000)
    hidden_model=verify_model(RUN/'jobs'/hidden_item,10000000)
    out=HERE/'runs'/item;out.mkdir(parents=True,exist_ok=True)
    identity=dict(manifest_sha256=digest(RUN/'manifest.json'),script_sha256=digest(__file__),
        protocol_sha256=digest(HERE/'PROTOCOL.md'),model_hashes=model['checkpoint_hashes'],
        hidden_model_hashes=hidden_model['checkpoint_hashes'],variants=VARIANTS)
    if (out/'identity.json').exists(): assert read(out/'identity.json')==identity
    write(out/'identity.json',identity)
    seeds=manifest['evaluation_seeds'];E=len(seeds)
    args=dict(config['env_args'],instruction_mode_strategy='explicit_evaluation')
    actors=hidden_actors=None
    started=time.monotonic();completed=0;total_probes=0;replay_slots=0
    for scenario,schedule in manifest['scenarios'].items():
        resultfile=out/f'{scenario}.npz';statusfile=out/f'{scenario}.json'
        if statusfile.exists():
            old=read(statusfile);assert digest(resultfile)==old['data_sha256']
            completed+=1;total_probes+=old['probes'];replay_slots+=old['replay_slots'];continue
        args['explicit_instruction_schedule']=schedule
        env=TensorSCEnv(args,count=E,seed=seeds[0],device='cpu')
        for source,s in zip(env.source.envs,seeds): source.seed(s)
        obs,shared,available=env.reset()
        if actors is None:
            actors=load_actors(config,env,RUN/'jobs'/item)
            hidden_actors=load_actors(hidden_config,env,RUN/'jobs'/hidden_item)
        trace,hashes=load_traces(item,scenario,seeds)
        points=set(range(0,600,10))
        for t,g in schedule[1:]: points.update(t+d for d in (-1,0,1) if 0<=t+d<600)
        store={k:[] for k in ['slot','seed','true_gid','available','observation_sha256','hidden_beta','hidden_prob']}
        store.update({k:[] for k in ['beta','raw_prob','masked_prob','mode']})
        max_errors={k:0. for k in ['mean_aoi','common_reward','channel_uses','predicted_quality_sum','policy_beta']}
        for slot in range(600):
            assert np.array_equal(array(env.instructions[slot]),trace['gid'][:,slot])
            if slot in points:
                before_obs=obs.clone();before_available=available.clone();before_state=shared.clone()
                variant=interventions(env,obs,True)
                pred=forward(actors,variant,available,env,validate_native=slot==0)
                pred_hidden=forward(hidden_actors,interventions(env,obs,False),available,env,validate_native=slot==0)
                for key in pred_hidden:
                    assert np.array_equal(pred_hidden[key],np.repeat(pred_hidden[key][:,:1],V,axis=1)),f'Negative control changed: {key}'
                assert torch.equal(obs,before_obs) and torch.equal(shared,before_state) and torch.equal(available,before_available)
                max_errors['policy_beta']=max(max_errors['policy_beta'],float(np.max(np.abs(pred['beta'][:,0]-trace['beta'][:,slot]))))
                np.testing.assert_allclose(pred['beta'][:,0],trace['beta'][:,slot],rtol=0,atol=2e-6)
                np.testing.assert_array_equal(pred['mode'][:,0],trace['mode'][:,slot])
                for key in pred: store[key].append(pred[key])
                store['hidden_beta'].append(pred_hidden['beta'][:,0])
                store['hidden_prob'].append(pred_hidden['masked_prob'][:,0])
                store['slot'].append(np.full(E,slot,dtype=np.int16))
                store['seed'].append(np.asarray(seeds,dtype=np.int64))
                store['true_gid'].append(array(env.instructions[slot]).copy())
                store['available'].append(array(available[:,1:,:env.M]).copy())
                store['observation_sha256'].append(np.asarray([hashlib.sha256(array(o).tobytes()).hexdigest() for o in obs]))
            # Continue only the actual historical trajectory, never a counterfactual one.
            actions=[torch.as_tensor(trace['sut'][:,slot],dtype=torch.float32)]
            for u in range(env.U):
                a=torch.full((E,env.M),-1.,dtype=torch.float32)
                a.scatter_(1,torch.as_tensor(trace['mode'][:,slot,u],dtype=torch.int64)[:,None],1.)
                actions.append(a)
            obs,shared,reward,done,info,available=env.step(actions,auto_reset=False)
            assert bool(done.all())==(slot==599)
            np.testing.assert_array_equal(array(info['mode']),trace['executed'][:,slot])
            np.testing.assert_allclose(array(env.beta),trace['beta'][:,slot],rtol=0,atol=1e-12)
            values={'mean_aoi':array(env.aoi.mean((-1,-2))),
                'common_reward':array(info['common_reward']),
                'channel_uses':array(info['usage'].sum(-1)),
                'predicted_quality_sum':array((info['quality']*info['served'].sum(-1)).sum(-1))}
            for key,value in values.items():
                error=float(np.max(np.abs(value-trace[key][:,slot])));max_errors[key]=max(max_errors[key],error)
                np.testing.assert_allclose(value,trace[key][:,slot],rtol=1e-9,atol=1e-7,err_msg=key)
            np.testing.assert_array_equal(array(info['served'].sum((-1,-2))),trace['deliveries'][:,slot])
        values={k:np.concatenate(v,axis=0) for k,v in store.items()}
        np.savez_compressed(resultfile,**values)
        count=len(values['slot']);total_probes+=count;completed+=1;replay_slots+=E*600
        write(statusfile,dict(state='complete',scenario=scenario,probes=count,replay_slots=E*600,
            data_sha256=digest(resultfile),trace_hashes=hashes,maximum_errors=max_errors,
            native_forward_identical=True,non_context_observation_unchanged=True,negative_control_exactly_invariant=True))
        write(out/'status.json',dict(state='running',updated_utc=stamp(),scenarios_completed=completed,
            total_scenarios=13,probes=total_probes,replay_slots=replay_slots,elapsed_seconds=time.monotonic()-started))
        print(f'{item} {scenario}: {completed}/13, probes={total_probes}, elapsed={time.monotonic()-started:.1f}s',flush=True)
    assert verify_model(RUN/'jobs'/item,10000000)['checkpoint_hashes']==model['checkpoint_hashes']
    assert verify_model(RUN/'jobs'/hidden_item,10000000)['checkpoint_hashes']==hidden_model['checkpoint_hashes']
    write(out/'status.json',dict(state='complete',updated_utc=stamp(),scenarios_completed=completed,
        total_scenarios=13,probes=total_probes,replay_slots=replay_slots,elapsed_seconds=time.monotonic()-started,
        models_unchanged=True,negative_control_exactly_invariant=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--algorithm',choices=['happo','mappo'],required=True)
    parser.add_argument('--core',type=int,required=True)
    parser.add_argument('--seed',type=int,choices=[85,218,966])
    args=parser.parse_args()
    os.sched_setaffinity(0,{args.core});torch.set_num_threads(1)
    manifest=verify_run(RUN)
    for seed in ([args.seed] if args.seed else manifest['seeds']): run_one(args.algorithm,seed,manifest)
