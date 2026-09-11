"""Fresh stochastic rollouts from final checkpoints, without optimizer updates."""
from diagnose import *
from tensor_train import TensorTrainer
from training_checkpoint import load_checkpoint,restore
from common import network_hash

def stats(x):
    x=arr(x)
    return dict(mean=float(x.mean()),std=float(x.std()),minimum=float(x.min()),maximum=float(x.max()))

def main():
    verify();torch.set_num_threads(1)
    results={};records=[]
    for seed in SEEDS:
        state,entry=load_checkpoint(REF/f'jobs/ref8/seed_{seed}/IC_HAPPO/checkpoints')
        trainer=TensorTrainer(state['config'],'cpu')
        restore(trainer,state,state['identity'])
        original_hashes=[network_hash(a.actor) for a in trainer.actors]
        by_batch=[]
        for batch in range(3):
            trainer.collect()
            b=trainer.buffer;env=trainer.env;args=trainer.algo['algo']
            returns=b.returns(trainer.normalizer,args['gamma'],args['gae_lambda'])
            values=trainer.normalizer.denormalize(b.values[:-1])
            advantage=returns-values
            norm=(advantage-advantage.mean())/(advantage.std(unbiased=False)+1e-5)
            ids=b.obs[:-1,:,1,env.p.uav_obs_dim-9:env.p.uav_obs_dim-6].argmax(-1)
            mode=torch.stack([a.argmax(-1) for a in b.actions[1:]],-1)
            ob=b.obs[:-1].flatten(0,1);av=b.available[:-1].flatten(0,1)
            ps=probabilities(trainer.actors,ob,av).reshape(b.length,b.count,env.U,env.M)
            onehot=torch.nn.functional.one_hot(mode,env.M).float()
            # Initial PPO ascent direction in action-logit space, before clipping or HAPPO sequential factors.
            policy_signal=(onehot-ps)*norm[:,:,:,None]
            logp=ps.clamp_min(1e-30).log()
            entropy_signal=args['entropy_coef']*ps*((ps*logp).sum(-1,keepdim=True)-logp)
            one={}
            for gid in range(3):
                keep=ids==gid
                if not keep.any():continue
                pol=policy_signal[keep].mean((0,1));ent=entropy_signal[keep].mean((0,1))
                one[str(gid)]=dict(steps=int(keep.sum()),reward=stats(b.rewards[keep]),advantage=stats(advantage[keep]),
                    normalized_advantage=stats(norm[keep]),value=stats(values[keep]),return_target=stats(returns[keep]),
                    mean_mode_probabilities=arr(ps[keep].mean((0,1))).tolist(),
                    selected_low_fraction=float(torch.isin(mode[keep],torch.tensor([0,4,8,12])).float().mean()),
                    policy_logit_ascent_by_mode=arr(pol).tolist(),entropy_logit_ascent_by_mode=arr(ent).tolist())
            with torch.no_grad():
                p=trainer.actors[0].actor
                dist=p.act.action_out.continuous_heads[0](p.base(ob[:,0]),av[:,0,:3])
                alpha=dist.concentration
                total=alpha.sum(-1,keepdim=True)
                sd=(alpha*(total-alpha)/(total**2*(total+1))).sqrt()*(1-env.p.beta_sat_lower_bound*env.U)
            # Exact categorical joint probability vs the per-coordinate stored representation.
            max_error=0.
            for u in range(env.U):
                direct=ps[:,:,u].gather(-1,mode[:,:,u,None]).squeeze(-1).log()
                max_error=max(max_error,float((b.log_probs[u+1].sum(-1)-direct).abs().max()))
            assert max_error<2e-5,max_error
            by_batch.append(dict(batch=batch,by_instruction=one,raw_advantage=stats(advantage),
                resource_concentration=stats(alpha),resource_fraction_sampling_sd=stats(sd),
                log_probability_sum_max_error=max_error,terminal_count=int((b.masks[1:]==0).sum())))
            records.append(dict(seed=seed,batch=batch,ids=arr(ids),advantage=arr(advantage),normalized_advantage=arr(norm),
                probs=arr(ps),policy_signal=arr(policy_signal),entropy_signal=arr(entropy_signal),modes=arr(mode)))
            b.after_update()
        assert original_hashes==[network_hash(a.actor) for a in trainer.actors]
        results[str(seed)]=dict(checkpoint=entry,learning_rates=[a.actor_optimizer.param_groups[0]['lr'] for a in trainer.actors],
            update=state['update'],total_updates=state['total_updates'],batches=by_batch,weights_unchanged=True)
        print('Signal complete',seed,flush=True)
    fields=['ids','advantage','normalized_advantage','probs','policy_signal','entropy_signal','modes']
    np.savez_compressed(OUT/'training_signal.npz',**{k:np.stack([r[k] for r in records]) for k in fields},
                        seeds=np.array([r['seed'] for r in records]),batches=np.array([r['batch'] for r in records]))
    write(OUT/'training_signal.json',dict(state='PASS',fresh_steps=36000,training_updates=0,results=results,
        limitation='Three consecutive frozen-policy stochastic rollout batches per seed; action-logit signal is not a full HAPPO optimizer trajectory.',script_sha256=digest(__file__)))
    verify()

if __name__=='__main__':main()
