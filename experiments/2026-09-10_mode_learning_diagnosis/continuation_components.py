"""Separate changed mode and resource controllers after the LR intervention."""
from diagnose import *

def main():
    verify();out={}
    write(OUT/'component_protocol.json',dict(created_utc=stamp(),seeds=SEEDS,scenarios=SCENARIOS,
        variants=['old_resources_new_modes','new_resources_old_modes'],steps=216000,training=False,
        purpose='Posthoc diagnostic following adverse continuation; not an independent confirmatory experiment.',script_sha256=digest(__file__)))
    for seed in SEEDS:
        cfg=config('ref8',seed=seed)
        for sc in SCENARIOS:
            for variant in ['old_resources_new_modes','new_resources_old_modes']:
                env,obs,_,av=make_env(cfg,EVAL_SEEDS,read(REF/'manifest.json')['scenarios'][sc])
                old=load_actors(cfg,env,REF/f'jobs/ref8/seed_{seed}/IC_HAPPO')
                new=load_actors(cfg,env,OUT/f'continuation/seed_{seed}/restart_lr')
                actors=[old[0],*new[1:]] if variant=='old_resources_new_modes' else [new[0],*old[1:]]
                rows=[];modes=[];betas=[]
                for t in range(600):
                    actions=actions_for(actors,obs,av)
                    obs,_,av,info,v=checked_step(env,actions);v['instruction_id']=arr(info['gid'])
                    rows.append(np.column_stack([v[k] for k in FIELDS]));modes.append(arr(info['mode']));betas.append(arr(env.beta))
                data=np.stack(rows);mode=np.stack(modes)
                v=summary(data,mode);v['low_fraction_per_uav']=np.isin(mode,[0,4,8,12]).mean((0,1)).tolist()
                v['mean_beta_per_uav']=np.stack(betas).mean((0,1)).tolist()
                out[f'{seed}/{sc}/{variant}']=v
                np.savez_compressed(OUT/f'components_{seed}_{sc}_{variant}.npz',trace=data,modes=mode,resource_fractions=np.stack(betas),fields=np.array(FIELDS),seeds=np.array(EVAL_SEEDS))
        print('Components complete',seed,flush=True)
    write(OUT/'continuation_components.json',dict(state='PASS',results=out,steps=216000))
    verify()

if __name__=='__main__':main()
