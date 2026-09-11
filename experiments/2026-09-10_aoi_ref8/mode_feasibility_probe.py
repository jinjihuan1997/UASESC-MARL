"""Observe raw mode preferences and feasible alternatives on the exact RL replay."""
from helpers import *
from evaluation import FIELDS


def main():
    m=verify();scenario='switch300_2_to_1';cfg=config('ref8',seed=85)
    folder=HERE/'evaluation/ref8/seed_85/IC_HAPPO';path=folder/f'{scenario}.npz'
    assert digest(path)==read(path.with_suffix('.json'))['trace_sha256']
    with np.load(path) as z:trace=z['trace'];original_modes=z['modes']
    model=HERE/'jobs/ref8/seed_85/IC_HAPPO'
    for f,h in read(model/'status.json')['checkpoint_hashes'].items():assert digest(model/f)==h
    env,obs,_,mask=make_env(cfg,EVAL_SEEDS,m['scenarios'][scenario]);actors=load_actors(cfg,env,model)
    rows=[]
    for slot in range(600):
        before_cached=arr(env.q.any(-1)).copy();acts=actions_for(actors,obs,mask)
        raw_top=arr(torch.stack(acts[1:],1).argmax(-1))
        obs,_,mask,info,v=checked_step(env,acts)
        q,l,req,budget=[arr(info[k]) for k in ['quality_table','load_table','req','budget']]
        middle_feasible=(q[...,5]>=req-1e-9)&(l[...,5]<=budget+1e-9)&before_cached
        actual=arr(info['mode']);low=np.isin(actual,[0,4,8,12]);raw_low=np.isin(raw_top,[0,4,8,12])
        np.testing.assert_array_equal(actual,original_modes[slot])
        np.testing.assert_allclose(v['common_reward'],trace[slot,:,FIELDS.index('common_reward')],atol=1e-9,rtol=0)
        rows.append(np.stack([low,raw_low,middle_feasible,raw_top==actual],-1))
    rows=np.stack(rows);out={}
    for window,sl in [('quality',slice(0,300)),('aoi',slice(300,600))]:
        out[window]={}
        for scope,sub in [('first_episode',rows[sl,0]),('all20_episodes',rows[sl])]:
            low,raw_low,feasible,same=[sub[...,i] for i in range(4)]
            out[window][scope]=dict(decisions=int(low.size),executed_low_count=int(low.sum()),
                raw_top_low_count=int(raw_low.sum()),middle_feasible_count=int(feasible.sum()),
                executed_low_with_middle_feasible=int((low&feasible).sum()),
                raw_top_equals_executed_count=int(same.sum()))
    write(HERE/'mode_feasibility_probe.json',dict(state='PASS',training_seed=85,evaluation_seeds=EVAL_SEEDS,
        scenario=scenario,trace_sha256=digest(path),results=out,slots=12000,
        interpretation='mode5 feasibility uses actual same-slot RL budget and pre-step cache',
        no_training=True,source_hash=digest(Path(__file__))))
    print(json.dumps(out,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
