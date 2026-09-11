"""Resumable CC evaluation with independent transition/reward and SC pairing checks."""
import argparse
import csv
import fcntl
import hashlib
import json
from pathlib import Path
import signal
import time
import numpy as np
import torch
from common import stamp,write
from multiseed_protocol import read,verify_model,verify_run
from training_checkpoint import digest
from evaluate import aggregate,load_actors
from tensor_env import TensorCCEnv
from cc_metrics import external_reference,metrics,array


def summarize(rows):
    value=aggregate(rows)
    value.update(attempts=sum(r['attempts'] for r in rows),failed_packets=sum(r['failed_packets'] for r in rows))
    return value


def evaluate_one(run,item_id):
    run=Path(run).resolve();manifest=verify_run(run);stopped=[]
    for sig in (signal.SIGTERM,signal.SIGINT): signal.signal(sig,lambda s,f:stopped.append(s))
    torch.set_num_threads(1)
    job=next(j for j in manifest['jobs'] if j['id']==item_id)
    config=read(run/job['config']);model_dir=run/job['output'];model=verify_model(model_dir,manifest['steps_per_method'])
    output=run/'evaluation'/item_id;output.mkdir(parents=True,exist_ok=True)
    lock=(output/'.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    identity=dict(run_sha256=digest(run/'manifest.json'),item_id=item_id,model_hashes=model['checkpoint_hashes'])
    if (output/'identity.json').exists() and read(output/'identity.json')!=identity: raise ValueError('Evaluation inputs changed')
    write(output/'identity.json',identity)
    args=dict(config['env_args'],instruction_mode_strategy='explicit_evaluation',explicit_instruction_schedule=[[0,0]])
    write(output/'config.json',args)
    summaries=[];total=len(manifest['scenarios'])*len(manifest['evaluation_seeds']);started=time.monotonic()
    try:
        for scenario,schedule in manifest['scenarios'].items():
            args['explicit_instruction_schedule']=schedule
            for seed in manifest['evaluation_seeds']:
                key=f'{scenario}_seed{seed}';summary_file=output/'episodes'/f'{key}.json';trace_file=output/'traces'/f'{key}.csv'
                if summary_file.exists():
                    summary=read(summary_file)
                    if digest(trace_file)!=summary['trace_sha256']: raise ValueError('Committed CC trace changed')
                    summaries.append(summary);continue
                if stopped:
                    write(output/'status.json',dict(state='paused',updated_utc=stamp(),completed_episodes=len(summaries),total_episodes=total));return 75
                env=TensorCCEnv(args,count=1,seed=seed,device='cpu');obs,shared,available=env.reset()
                actors=load_actors(config,env,model_dir)
                ref,trace=external_reference(config,seed,schedule)
                np.testing.assert_array_equal(array(env.pos_ds)[0],ref.pos_ds)
                np.testing.assert_array_equal(array(env.pos_uav)[0],ref.pos_uav)
                np.testing.assert_array_equal(array(env.q)[0],ref.q_cache[env.p.owner_mask].reshape(env.U,env.K))
                delivery_hash=hashlib.sha256(array(env.delivery_uniforms).tobytes()).hexdigest()
                rows=[]
                for slot in range(env.p.max_steps):
                    expected=next(g for t,g in reversed(schedule) if t<=slot)
                    assert int(env.instructions[slot,0])==expected
                    np.testing.assert_allclose(array(env.gamma_us)[0],ref.gamma_uav_sut,rtol=1e-12,atol=1e-10)
                    np.testing.assert_allclose(float(env.gamma_sat[0]),ref.gamma_sut_sat,rtol=1e-12,atol=1e-10)
                    trace.update(ref.gamma_uav_sut.tobytes())
                    trace.update(np.asarray([ref.gamma_sut_sat,expected],dtype=np.float64).tobytes())
                    actions=[]
                    with torch.no_grad():
                        for i,actor in enumerate(actors):
                            action,_=actor.act(obs[:,i],np.zeros((1,1,256),np.float32),np.ones((1,1),np.float32),available[:,i],deterministic=True)
                            actions.append(action)
                    obs,shared,reward,done,info,available=env.step(actions,auto_reset=False)
                    assert bool(done.all())==(slot==env.p.max_steps-1)
                    row=dict(slot=slot,instruction_id=expected,**metrics(env,info,slot))
                    row.update(proposed_sut_logits=json.dumps(array(actions[0])[0].tolist()),
                        proposed_modes=json.dumps([int(a[0].argmax()) for a in actions[1:]]),executed_modes=json.dumps(array(info['mode'])[0].tolist()),
                        selected_ds=json.dumps(np.flatnonzero(array(info['served'])[0]).tolist()),
                        attempted_ds=json.dumps(np.flatnonzero(array(info['attempted'])[0]).tolist()),
                        resource_fractions=json.dumps(array(env.beta)[0].tolist()))
                    rows.append(row)
                    if slot<env.p.max_steps-1: ref._update_channels()
                trace_file.parent.mkdir(parents=True,exist_ok=True);temp=trace_file.with_suffix('.csv.tmp')
                with temp.open('w',newline='') as stream:
                    writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
                temp.replace(trace_file)
                summary=dict(method=job['method'],training_seed=job['seed'],scenario=scenario,seed=seed,steps=len(rows),
                    external_trajectory_sha256=trace.hexdigest(),delivery_trajectory_sha256=delivery_hash,trace_sha256=digest(trace_file),**summarize(rows))
                summary['by_instruction']={str(g):dict(steps=len(part),**summarize(part)) for g in range(3) if (part:=[r for r in rows if r['instruction_id']==g])}
                write(summary_file,summary);summaries.append(summary)
                write(output/'status.json',dict(state='evaluating',updated_utc=stamp(),completed_episodes=len(summaries),total_episodes=total,elapsed_seconds=time.monotonic()-started))
        result=dict(item_id=item_id,method=job['method'],training_seed=job['seed'],overall=summarize(summaries),
            by_scenario={s:summarize([r for r in summaries if r['scenario']==s]) for s in manifest['scenarios']},
            pairing={f"{r['scenario']}/{r['seed']}":r['external_trajectory_sha256'] for r in summaries},
            delivery_pairing={f"{r['scenario']}/{r['seed']}":r['delivery_trajectory_sha256'] for r in summaries})
        write(output/'summary.json',result)
        write(output/'status.json',dict(state='complete',updated_utc=stamp(),completed_episodes=len(summaries),total_episodes=total,
            slots=sum(r['steps'] for r in summaries),summary_sha256=digest(output/'summary.json'),model_hashes=model['checkpoint_hashes'],all_executed_constraints_passed=True))
        return 0
    except BaseException as exc:
        write(output/'status.json',dict(state='failed',updated_utc=stamp(),error=repr(exc),completed_episodes=len(summaries)));raise
    finally: lock.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True);parser.add_argument('--item',required=True)
    args=parser.parse_args();raise SystemExit(evaluate_one(args.run,args.item))
