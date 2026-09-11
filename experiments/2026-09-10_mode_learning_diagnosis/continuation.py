"""Paired, predeclared 200k diagnostic continuation; does not replace formal models."""
from diagnose import *
from training_signal import TensorTrainer,load_checkpoint,restore,network_hash
import argparse

ARMS=['final_lr','restart_lr']
UPDATES=50

def temperature():
    values=[]
    for path in Path('/sys/class/hwmon').glob('hwmon*/temp*_input'):
        try:
            if (path.parent/'name').read_text().strip() in ['coretemp','k10temp']:
                values.append(float(path.read_text())/1000)
        except OSError:pass
    return max(values) if values else None

def guard():
    t=temperature()
    if t is not None and t>=88:
        print('Thermal pause',t,flush=True)
        while t is not None and t>78:
            time.sleep(5);t=temperature()
        print('Thermal resume',t,flush=True)

def run(seed):
    frozen=read(OUT/'continuation_protocol.json')
    assert frozen['script_sha256']==digest(__file__)
    state,entry=load_checkpoint(REF/f'jobs/ref8/seed_{seed}/IC_HAPPO/checkpoints')
    assert frozen['checkpoints'][str(seed)]==entry
    allresults={}
    for arm in ARMS:
        folder=OUT/'continuation'/f'seed_{seed}'/arm
        folder.mkdir(parents=True,exist_ok=False)
        trainer=TensorTrainer(state['config'],'cpu');restore(trainer,state,state['identity'])
        initial=[network_hash(a.actor) for a in trainer.actors]
        actor_lr=state['config']['algo_args']['model']['lr'] if arm=='restart_lr' else trainer.actors[0].actor_optimizer.param_groups[0]['lr']
        critic_lr=state['config']['algo_args']['model']['critic_lr'] if arm=='restart_lr' else trainer.critic.critic_optimizer.param_groups[0]['lr']
        write(folder/'identity.json',dict(source_checkpoint=entry,arm=arm,actor_lr=actor_lr,critic_lr=critic_lr,
            initial_actor_hashes=initial,additional_steps=UPDATES*trainer.batch,
            semantics='diagnostic transfer with same optimizer moments and RNG; fixed LR override; no original run resume'))
        started=time.monotonic()
        for i in range(UPDATES):
            guard()
            for a in trainer.actors:
                for g in a.actor_optimizer.param_groups:g['lr']=actor_lr
            for g in trainer.critic.critic_optimizer.param_groups:g['lr']=critic_lr
            trainer.collect();m=trainer.update();trainer.buffer.after_update()
            assert all(torch.isfinite(v).all() for v in m.values())
            with (folder/'metrics.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(update=i+1,reward=float(trainer.buffer.rewards.mean()),
                    actor=arr(m['actor']).tolist(),critic=arr(m['critic']).tolist(),temperature=temperature()))+'\n')
            write(folder/'status.json',dict(state='training',completed_steps=(i+1)*trainer.batch,target_steps=UPDATES*trainer.batch))
        for i,a in enumerate(trainer.actors):torch.save(a.actor.state_dict(),folder/f'actor_agent{i}.pt')
        scores={}
        for scenario in SCENARIOS:
            env,obs,_,available=make_env(state['config'],EVAL_SEEDS,read(REF/'manifest.json')['scenarios'][scenario])
            for a in trainer.actors:a.prep_rollout()
            trace=[];modes=[]
            for slot in range(600):
                actions=actions_for(trainer.actors,obs,available)
                obs,_,available,info,v=checked_step(env,actions);v['instruction_id']=arr(info['gid'])
                trace.append(np.column_stack([v[k] for k in FIELDS]));modes.append(arr(info['mode']))
            trace=np.stack(trace);modes=np.stack(modes)
            scores[scenario]=summary(trace,modes)
            np.savez_compressed(folder/f'{scenario}.npz',trace=trace,modes=modes,fields=np.array(FIELDS),seeds=np.array(EVAL_SEEDS))
        final=[network_hash(a.actor) for a in trainer.actors]
        assert all(a!=b for a,b in zip(initial,final))
        write(folder/'status.json',dict(state='complete',completed_steps=UPDATES*trainer.batch,seconds=time.monotonic()-started,
            initial_actor_hashes=initial,final_actor_hashes=final,actor_lr=actor_lr,critic_lr=critic_lr,scores=scores,
            output_hashes={str(p.name):digest(p) for p in folder.glob('*.pt')}))
        allresults[arm]=scores
        print('Complete',seed,arm,{k:100*v['common_reward'] for k,v in scores.items()},flush=True)
    verify()
    write(OUT/'continuation'/f'seed_{seed}'/'results.json',allresults)

def freeze():
    assert not (OUT/'continuation_protocol.json').exists()
    entries={str(seed):load_checkpoint(REF/f'jobs/ref8/seed_{seed}/IC_HAPPO/checkpoints')[1] for seed in SEEDS}
    write(OUT/'continuation_protocol.json',dict(created_utc=stamp(),script_sha256=digest(__file__),checkpoints=entries,
        arms=ARMS,updates=UPDATES,additional_steps_per_arm=200000,total_training_steps=1200000,
        seeds=SEEDS,eval_seeds=EVAL_SEEDS,scenarios=SCENARIOS,
        intervention='Only fixed actor/critic learning rate differs; final LR versus original initial LR. No reward, physics, entropy, sampling or network changes.',
        hypothesis='Recovery of instruction separation at restored LR would support premature LR decay / incomplete optimization; a null result is inconclusive.',
        role='diagnostic on existing evaluation seeds; not held-out paper result; no promotion or extension based on outcome'))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int);args=parser.parse_args()
    if args.seed is None:freeze()
    else:run(args.seed)
