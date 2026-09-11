"""One-step action effects at actual stochastic training states, fixed checkpoint."""
from training_signal import *

def main():
    verify();rows=[]
    for seed in SEEDS:
        state,entry=load_checkpoint(REF/f'jobs/ref8/seed_{seed}/IC_HAPPO/checkpoints')
        trainer=TensorTrainer(state['config'],'cpu');restore(trainer,state,state['identity'])
        env=trainer.env;original_step=env.step;groups=groups_for(env);indices=[];effects=[];probs=[];ids=[]
        counter=[0]
        def observed_step(actions,auto_reset=True):
            t=counter[0];counter[0]+=1
            if t%4==0:
                indices.append(t);ids.append(arr(env.context()[0]))
                ps=probabilities(trainer.actors,trainer.buffer.obs[t],trainer.buffer.available[t])
                by_agent=[]
                for u in range(env.U):
                    by_group=[]
                    for group in groups:
                        a=list(actions);a[u+1]=torch.zeros_like(a[u+1]);a[u+1][:,group[0]]=1
                        by_group.append(arr(predict_reward(env,a)))
                    by_agent.append(np.stack(by_group,-1))
                effects.append(np.stack(by_agent,1));probs.append(arr(ps))
                predicted=arr(predict_reward(env,actions))
            result=original_step(actions,auto_reset)
            if t%4==0:np.testing.assert_allclose(arr(result[2][:,0,0]),predicted,atol=1e-6,rtol=1e-6)
            return result
        env.step=observed_step
        trainer.collect()
        args=trainer.algo['algo'];b=trainer.buffer
        returns=b.returns(trainer.normalizer,args['gamma'],args['gae_lambda'])
        adv=arr(returns-trainer.normalizer.denormalize(b.values[:-1]))[indices,:,0]
        ps=np.stack(probs);values=np.stack(effects)
        mass=np.stack([ps[...,g].sum(-1) for g in groups],-1)
        expected=(mass*values).sum(-1,keepdims=True)
        # Immediate reward objective only; not asserted to be long-term Q or GAE truth.
        exact_local_direction=mass*(values-expected)
        row=dict(seed=seed,indices=indices,ids=np.stack(ids),probabilities=ps,group_probabilities=mass,
            one_step_rewards=values,immediate_gradient=exact_local_direction,gae=adv)
        rows.append(row)
        print('Training-state counterfactual complete',seed,flush=True)
    keys=['ids','probabilities','group_probabilities','one_step_rewards','immediate_gradient','gae']
    np.savez_compressed(OUT/'counterfactual_signal.npz',**{k:np.stack([r[k] for r in rows]) for k in keys},seeds=np.array(SEEDS))
    write(OUT/'counterfactual_signal.json',dict(state='PASS',groups=groups,training_updates=0,fresh_rollout_steps=12000,
        checked_states=3000,one_uav_mode_counterfactuals=99000,script_sha256=digest(__file__),
        limitation='Immediate one-step effects, conditional on sampled actions of the other agents; does not substitute for true long-term advantages.'))
    verify()

if __name__=='__main__':main()
