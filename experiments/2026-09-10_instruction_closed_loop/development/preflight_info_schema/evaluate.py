"""Paired closed-loop deployment ablation of frozen SC actors."""
import argparse
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'): os.environ[key]='1'
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
# common inserts the immutable reference directory ahead of this script.
from evaluate import aggregate,external_metrics,load_actors
from tensor_env import TensorSCEnv
from harl.envs.uav_escs.SC.uav_escs_env_sc import SCUAVEnv

CONDITIONS=['correct','hidden_at_deployment']
CALIBRATION={'mean_aoi':1e-10,'deliveries':0.,'predicted_quality_sum':1e-7,'channel_uses':1e-6,'common_reward':1e-9}


def arr(t): return t.detach().cpu().numpy()


def explicit_columns(p):
    allowed=np.zeros((1+p.n_uav,p.obs_dim_common),bool)
    allowed[0,p.sut_obs_dim-4:p.sut_obs_dim]=True
    allowed[1:,p.uav_obs_dim-9:p.uav_obs_dim-6]=True
    allowed[1:,p.uav_obs_dim-2:p.uav_obs_dim]=True
    assert p.num_instructions==3 and p.include_instruction_id_in_obs and p.include_instruction_constraints_in_obs
    return allowed


def tensor_input(obs,allowed,condition):
    value=obs.clone()
    if condition=='hidden_at_deployment': value[:,allowed]=0
    assert torch.equal(value[:,~allowed],obs[:,~allowed])
    if condition=='correct': assert torch.equal(value,obs)
    else: assert torch.count_nonzero(value[:,allowed])==0
    return value


@torch.no_grad()
def actions_for(actors,obs,available):
    E=len(obs);rnn=np.zeros((E,1,256),np.float32);mask=np.ones((E,1),np.float32)
    return [actor.act(obs[:,i],rnn,mask,available[:,i],deterministic=True)[0] for i,actor in enumerate(actors)]


def check_tensor(env,info,slot):
    p=env.p
    x={k:arr(v) for k,v in info.items()}
    served=x['served'];old=x['old_aoi'];after=x['next_aoi'];cache=x['cache_before'];tau=x['tau_before']
    expected=np.where(served,slot-np.where(tau<0,slot,tau)+1,old+1).clip(max=p.A_max)
    np.testing.assert_allclose(after,expected,rtol=0,atol=1e-12)
    np.testing.assert_array_equal(arr(env.q),(cache & ~served)|(~cache))
    np.testing.assert_array_equal(arr(env.tau),np.where(~cache,slot,np.where(cache & ~served,tau,-1)))
    assert not np.any(served & ~cache)
    mode=x['mode'];selected=np.take_along_axis(x['load_table'],mode.clip(min=0)[:,:,None],-1).squeeze(-1)
    usage=selected*served.sum(-1)
    np.testing.assert_allclose(usage,x['usage'],rtol=0,atol=1e-8)
    assert not np.any(usage>x['budget']+1e-8)
    assert not np.any(served & (x['quality']<x['req']-1e-8)[:,:,None])
    gid=x['gid'].astype(int);limits=np.asarray(p.A_limit_by_instruction)[gid]
    weights=np.asarray(p.reward_weights_by_instruction)[gid]
    qgain=(np.maximum((x['quality']-p.Q_min_eval)/(p.Q_max-p.Q_min_eval),0)*served.sum(-1)).sum(-1)/p.n_ds
    flat=after.reshape(env.count,p.n_ds)
    aoi=p.aoi_mean_weight*np.mean(flat/p.aoi_reward_ref,-1)+p.aoi_max_weight*np.max(flat/p.aoi_reward_ref,-1)+p.aoi_tail_weight*np.mean(np.maximum((flat-p.aoi_tail_threshold)/p.aoi_reward_ref,0),-1)
    base=weights[:,0]*qgain-weights[:,2]*usage.sum(-1)/p.Lambda_ref/p.n_uav-weights[:,1]*aoi
    violation=np.maximum((flat.max(-1)-limits)/limits,0)
    bonus=p.eta_recv_aoi_bonus*np.maximum(old-after,0).mean((-1,-2))/limits
    common=base-np.asarray(p.constraint_penalty_A_by_instruction)[gid]*violation+bonus
    np.testing.assert_allclose(common,x['common_reward'],atol=1e-9,rtol=0)
    zeros=np.zeros(env.count,dtype=int)
    return dict(common_reward=common,base_reward=base,training_reward=x['reward'],recv_aoi_bonus=bonus,
        mean_aoi=flat.mean(-1),max_aoi=flat.max(-1),p95_aoi=np.quantile(flat,.95,axis=-1),
        aoi_exceedance_fraction=(flat>limits[:,None]).mean(-1),max_aoi_violation=violation,
        deliveries=served.sum((-1,-2)),predicted_quality_sum=(x['quality']*served.sum(-1)).sum(-1),
        channel_uses=usage.sum(-1),quality_violations=zeros,budget_violations=zeros,cache_violations=zeros,
        infeasible_uav_fraction=(mode<0).mean(-1))


def start_hash(ref):
    h=hashlib.sha256()
    for value in (ref.pos_ds,ref.pos_uav,ref.owner_uav,ref.q_cache,ref.tau_cache): h.update(value.tobytes())
    return h


def historical(item,scenario,seeds):
    traces=[];summaries=[]
    for seed in seeds:
        folder=RUN/'evaluation'/item;key=f'{scenario}_seed{seed}'
        summary=read(folder/'episodes'/f'{key}.json');file=folder/'traces'/f'{key}.csv'
        assert digest(file)==summary['trace_sha256']
        with file.open(newline='') as stream: rows=list(csv.DictReader(stream))
        assert len(rows)==600
        traces.append(rows);summaries.append(summary)
    return traces,summaries


def differences(rows,reference):
    return {k:float(max(abs(float(a[k])-float(b[k])) for a,b in zip(rows,reference))) for k in CALIBRATION}


def cpu_pair(config,actors,scenario,schedule,seed,reference):
    result={};external={}
    for condition in CONDITIONS:
        args=dict(config['env_args'],instruction_mode_strategy='explicit_evaluation',explicit_instruction_schedule=schedule)
        env=SCUAVEnv(args);env.seed(seed);obs,_,available=env.reset()
        allowed=explicit_columns(env);h=start_hash(env);rows=[]
        for slot in range(env.max_steps):
            gid=int(env.current_instruction_id)
            h.update(env.gamma_uav_sut.tobytes());h.update(np.asarray([env.gamma_sut_sat,gid],dtype=np.float64).tobytes())
            feed=obs.copy()
            if condition=='hidden_at_deployment': feed[allowed]=0
            assert np.array_equal(feed[~allowed],obs[~allowed])
            actions=[a.cpu().numpy()[0] for a in actions_for(actors,torch.as_tensor(feed[None]),torch.as_tensor(available[None]))]
            obs,_,_,done,infos,available=env.step(actions)
            assert bool(np.all(done))==(slot==599)
            row=dict(slot=slot,instruction_id=gid,**external_metrics(env,infos[0]))
            row.update(proposed_sut_logits=json.dumps(env.last_sut_raw_action.tolist()),
                proposed_modes=json.dumps(np.argmax(env.last_uav_raw_actions,axis=1).tolist()),
                executed_modes=json.dumps(env.last_selected_modes.tolist()),
                resource_fractions=json.dumps(env.beta_sut_sat.tolist()))
            rows.append(row)
        result[condition]=rows;external[condition]=h.hexdigest()
    assert external['correct']==external['hidden_at_deployment']
    errors=differences(result['correct'],reference)
    assert all(errors[k]<=tol for k,tol in CALIBRATION.items()),errors
    return result,external['correct'],errors


def commit(out,condition,scenario,seed,method,training_seed,rows,external_hash,backend):
    key=f'{scenario}_seed{seed}';folder=out/condition
    trace=folder/'traces'/f'{key}.csv';trace.parent.mkdir(parents=True,exist_ok=True)
    tmp=trace.with_suffix('.tmp')
    with tmp.open('w',newline='') as stream:
        w=csv.DictWriter(stream,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    tmp.replace(trace)
    summary=dict(method=method,training_seed=training_seed,condition=condition,scenario=scenario,seed=seed,
        steps=len(rows),external_trajectory_sha256=external_hash,trace_sha256=digest(trace),backend=backend,**aggregate(rows))
    summary['by_instruction']={str(g):dict(steps=len(part),**aggregate(part)) for g in range(3)
        if (part:=[r for r in rows if r['instruction_id']==g])}
    write(folder/'episodes'/f'{key}.json',summary)
    return summary


def run_one(algorithm,seed,manifest,only_scenario=None):
    method='IC_'+algorithm.upper();item=f'seed_{seed}/{method}'
    config=read(RUN/'configs'/f'{item}.json');model=verify_model(RUN/'jobs'/item,10000000)
    out=HERE/'runs'/item;out.mkdir(parents=True,exist_ok=True)
    identity=dict(script_sha256=digest(__file__),protocol_sha256=digest(HERE/'PROTOCOL.md'),
        parent_manifest_sha256=digest(RUN/'manifest.json'),checkpoint_hashes=model['checkpoint_hashes'],conditions=CONDITIONS)
    if (out/'identity.json').exists(): assert read(out/'identity.json')==identity
    write(out/'identity.json',identity)
    seeds=manifest['evaluation_seeds'];E=len(seeds);actors=None;started=time.monotonic();completed=0
    scenarios=manifest['scenarios'] if only_scenario is None else {only_scenario:manifest['scenarios'][only_scenario]}
    for scenario,schedule in scenarios.items():
        auditfile=out/'calibration'/f'{scenario}.json'
        if auditfile.exists():
            old=read(auditfile)
            for condition in CONDITIONS:
                for s in seeds:
                    key=f'{scenario}_seed{s}';ep=read(out/condition/'episodes'/f'{key}.json')
                    assert digest(out/condition/'traces'/f'{key}.csv')==ep['trace_sha256']
            completed+=1;continue
        args=dict(config['env_args'],instruction_mode_strategy='explicit_evaluation',explicit_instruction_schedule=schedule)
        envs={};observations={};masks={}
        for condition in CONDITIONS:
            env=TensorSCEnv(args,count=E,seed=seeds[0],device='cpu')
            for source,s in zip(env.source.envs,seeds): source.seed(s)
            obs,_,available=env.reset();envs[condition]=env;observations[condition]=obs;masks[condition]=available
        a,b=envs.values()
        for name in ('pos_uav','pos_ds','q','tau','aoi','noise_us','noise_sat','noise_du','instructions','potential_content'):
            assert torch.equal(getattr(a,name),getattr(b,name)),name
        if actors is None: actors=load_actors(config,a,RUN/'jobs'/item)
        allowed=torch.as_tensor(explicit_columns(a.p))
        references=a.source.envs;hashes=[start_hash(ref) for ref in references]
        historical_rows,historical_summaries=historical(item,scenario,seeds)
        result={c:[[] for _ in seeds] for c in CONDITIONS}
        for slot in range(600):
            gid=next(g for t,g in reversed(schedule) if t<=slot)
            for idx,ref in enumerate(references):
                np.testing.assert_allclose(arr(a.gamma_us)[idx],ref.gamma_uav_sut,rtol=1e-12,atol=1e-10)
                np.testing.assert_allclose(float(a.gamma_sat[idx]),ref.gamma_sut_sat,rtol=1e-12,atol=1e-10)
                hashes[idx].update(ref.gamma_uav_sut.tobytes());hashes[idx].update(np.asarray([ref.gamma_sut_sat,gid],dtype=np.float64).tobytes())
            assert torch.equal(a.gamma_us,b.gamma_us) and torch.equal(a.gamma_sat,b.gamma_sat)
            for condition in CONDITIONS:
                env=envs[condition]
                assert bool((env.instructions[slot]==gid).all())
                feed=tensor_input(observations[condition],allowed,condition)
                actions=actions_for(actors,feed,masks[condition])
                obs,_,_,done,info,available=env.step(actions,auto_reset=False)
                observations[condition]=obs;masks[condition]=available
                assert bool(done.all())==(slot==599)
                values=check_tensor(env,info,slot)
                proposed=np.stack([arr(action).argmax(-1) for action in actions[1:]],-1)
                for idx in range(E):
                    row=dict(slot=slot,instruction_id=gid,**{k:v[idx].item() for k,v in values.items()})
                    row.update(proposed_sut_logits=json.dumps(arr(actions[0])[idx].tolist()),
                        proposed_modes=json.dumps(proposed[idx].tolist()),executed_modes=json.dumps(arr(info['mode'])[idx].tolist()),
                        resource_fractions=json.dumps(arr(env.beta)[idx].tolist()))
                    result[condition][idx].append(row)
            if slot<599:
                for ref in references: ref._update_channels()
        calibration=[]
        for idx,s in enumerate(seeds):
            external_hash=hashes[idx].hexdigest()
            assert external_hash==historical_summaries[idx]['external_trajectory_sha256']
            errors=differences(result['correct'][idx],historical_rows[idx]);backend='tensor_cpu_batch20'
            before_errors=errors.copy()
            if any(errors[k]>tol for k,tol in CALIBRATION.items()):
                solo,solo_hash,errors=cpu_pair(config,actors,scenario,schedule,s,historical_rows[idx])
                assert solo_hash==external_hash
                for condition in CONDITIONS: result[condition][idx]=solo[condition]
                backend='reference_cpu_batch1_paired_fallback'
            for condition in CONDITIONS: commit(out,condition,scenario,s,method,seed,result[condition][idx],external_hash,backend)
            calibration.append(dict(seed=s,backend=backend,initial_errors=before_errors,final_errors=errors,
                formal_trace_sha256=historical_summaries[idx]['trace_sha256'],external_trajectory_sha256=external_hash))
        write(auditfile,dict(state='complete',scenario=scenario,calibration=calibration,
            all_constraints_and_transition_checks_passed=True,explicit_input_only=True))
        completed+=1
        write(out/'status.json',dict(state='running',updated_utc=stamp(),completed_scenarios=completed,
            total_scenarios=13,elapsed_seconds=time.monotonic()-started))
        print(f'{item} {scenario}: {completed}/{len(scenarios)}, fallback={sum(x["backend"].endswith("fallback") for x in calibration)}, elapsed={time.monotonic()-started:.1f}s',flush=True)
    verify_model(RUN/'jobs'/item,10000000)
    all_done=all((out/'calibration'/f'{s}.json').exists() for s in manifest['scenarios'])
    write(out/'status.json',dict(state='complete' if all_done else 'partial',updated_utc=stamp(),
        completed_scenarios=sum((out/'calibration'/f'{s}.json').exists() for s in manifest['scenarios']),
        total_scenarios=13,elapsed_seconds=time.monotonic()-started,model_unchanged=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--algorithm',choices=['happo','mappo'],required=True)
    parser.add_argument('--core',type=int,required=True)
    parser.add_argument('--seed',type=int,choices=[85,218,966])
    parser.add_argument('--scenario')
    args=parser.parse_args();os.sched_setaffinity(0,{args.core});torch.set_num_threads(1)
    manifest=verify_run(RUN)
    for seed in ([args.seed] if args.seed else manifest['seeds']): run_one(args.algorithm,seed,manifest,args.scenario)
