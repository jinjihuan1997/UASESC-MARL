"""All eight low/middle mode combinations, with frozen learned resources."""
from diagnose import *
import itertools

def main():
    verify();patterns=list(itertools.product([0,5],repeat=3));results={}
    write(OUT/'coordination_protocol.json',dict(created_utc=stamp(),training_seeds=SEEDS,evaluation_seeds=EVAL_SEEDS,
        instruction=2,patterns=patterns,resources='frozen original deterministic SUT policy recomputed on each trajectory',
        slots=288000,training=False,role='posthoc exploratory diagnosis; reward unchanged',script_sha256=digest(__file__)))
    for seed in SEEDS:
        cfg=config('ref8',seed=seed)
        for pattern in patterns:
            env,obs,_,av=make_env(cfg,EVAL_SEEDS,[[0,2]])
            actors=load_actors(cfg,env,REF/f'jobs/ref8/seed_{seed}/IC_HAPPO');rows=[];modes=[]
            for t in range(600):
                acts=actions_for(actors,obs,av)
                for u,m in enumerate(pattern):
                    acts[u+1]=torch.zeros_like(acts[u+1]);acts[u+1][:,m]=1
                obs,_,av,info,v=checked_step(env,acts);v['instruction_id']=arr(info['gid'])
                rows.append(np.column_stack([v[k] for k in FIELDS]));modes.append(arr(info['mode']))
            data=np.stack(rows);mode=np.stack(modes);name=''.join(str(x) for x in pattern)
            results[f'{seed}/{name}']=summary(data,mode)
            np.savez_compressed(OUT/f'coordination_{seed}_{name}.npz',trace=data,modes=mode,fields=np.array(FIELDS),seeds=np.array(EVAL_SEEDS))
        print('Coordination complete',seed,flush=True)
    write(OUT/'coordination.json',dict(state='PASS',results=results,slots=288000))
    verify()

if __name__=='__main__':main()
