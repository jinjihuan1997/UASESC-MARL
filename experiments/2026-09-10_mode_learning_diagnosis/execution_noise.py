"""Audit proposed versus executed modes during genuine stochastic collection."""
from training_signal import *

def main():
    verify();out={};raws=[];executeds=[];gids=[];budgets=[];oldbudgets=[]
    for seed in SEEDS:
        state,entry=load_checkpoint(REF/f'jobs/ref8/seed_{seed}/IC_HAPPO/checkpoints')
        t=TensorTrainer(state['config'],'cpu');restore(t,state,state['identity'])
        env=t.env;step=env.step;counter=[0];rows=[]
        def observed(actions,auto_reset=True):
            n=counter[0];counter[0]+=1
            raw=torch.stack([x.argmax(-1) for x in actions[1:]],-1)
            previous_budget=t.buffer.obs[n,:,1:,0]*env.p.Lambda_ref
            result=step(actions,auto_reset)
            info=result[4];load=info['load_table'].gather(-1,raw[:,:,None]).squeeze(-1)
            rows.append(dict(raw=arr(raw),executed=arr(info['mode']),gid=arr(info['gid']),budget=arr(info['budget']),
                previous_budget=arr(previous_budget),raw_load=arr(load)))
            return result
        env.step=observed;t.collect()
        data={k:np.stack([r[k] for r in rows]) for k in rows[0]}
        raw=data['raw'];actual=data['executed'];has=actual>=0
        # Reasons are measured directly, not inferred from aliases.
        budget_fail=data['raw_load']>data['budget']+1e-9
        bytask={}
        for gid in range(3):
            keep=np.broadcast_to((data['gid']==gid)[...,None],raw.shape)
            middle=keep&np.isin(raw,[5,9,13])
            bytask[str(gid)]=dict(decisions=int(keep.sum()),raw_execution_mismatch=float((raw[keep]!=actual[keep]).mean()),
                budget_infeasible_proposal=float(budget_fail[keep].mean()),raw_middle_count=int(middle.sum()),
                raw_middle_replaced_fraction=float((raw[middle]!=actual[middle]).mean()),
                resource_fraction_change_mean=float(np.abs(data['budget'][keep]-data['previous_budget'][keep]).mean()/60000))
        out[str(seed)]=bytask
        np.savez_compressed(OUT/f'execution_noise_{seed}.npz',**data)
    write(OUT/'execution_noise.json',dict(state='PASS',results=out,slots=12000,training_updates=0,script_sha256=digest(__file__)))
    verify()

if __name__=='__main__':main()
