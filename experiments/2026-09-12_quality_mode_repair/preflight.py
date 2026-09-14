"""Fail closed preflight; failures are immutable attempt records."""
from repair_support import *
from repair_training import RepairTrainer
from repair_evaluation import actions_for, greedy_policy
import time
import traceback
import argparse
import subprocess

def equal_tree(a,b,path='root'):
    if isinstance(a,torch.Tensor): assert torch.equal(a,b),path
    elif isinstance(a,np.ndarray): assert np.array_equal(a,b),path
    elif isinstance(a,dict):
        assert a.keys()==b.keys(),path
        for k in a:equal_tree(a[k],b[k],path+'/'+str(k))
    elif isinstance(a,(tuple,list)):
        assert len(a)==len(b),path
        for i,(x,y) in enumerate(zip(a,b)):equal_tree(x,y,path+'/'+str(i))
    else: assert a==b,(path,a,b)

def advance(t):
    c=t.collect();u=t.update();t.buffer.after_update()
    return c,u

def semantic_checks():
    checks={};m=verify_inputs(False)
    checks['inputs_source_profile_git_model_config']='PASS'
    for seed in m['parents']:
        ta=RepairTrainer(seed,'residual_all',preflight=True,length=8)
        tq=RepairTrainer(seed,'residual_quality',preflight=True,length=8)
        assert ta.initial_adapters==tq.initial_adapters
        assert ta.initial_critic==tq.initial_critic and ta.initial_normalizer==tq.initial_normalizer
        equal_tree(ta.critic.critic.state_dict(),tq.critic.critic.state_dict())
        equal_tree(ta.normalizer.state_dict(),tq.normalizer.state_dict())
        for gid in range(3):
            env=make_env(cfg_for(seed),m['preflight_seeds']*2,'cpu',[[0,gid]],'preflight')
            obs,_,mask=env.observe();rnn=torch.zeros(2,1,256);active=torch.ones(2,1)
            resource=ta.actors[0].act(obs[:,0],rnn,active,mask[:,0],True)[0]
            step=env.step_index;post,_,available=env.allocate_resources(resource)
            assert env.step_index==step
            raw=((resource+1)/2).double().clamp_min(0);raw=raw/raw.sum(-1,keepdim=True)
            expected=.05+.85*raw;expected=expected/expected.sum(-1,keepdim=True)
            assert torch.equal(env.beta,expected)
            assert torch.equal(post[:,1:,0].double(),(env.budget/100000).float().double())
            for i in range(1,4):
                old=ta.actors[i].actor.base_policy
                obs_i=post[:,i];av=available[:,i]
                assert torch.equal(obs_i[:,67:70].argmax(-1),torch.full((2,),gid))
                with torch.no_grad():
                    features=old.base(obs_i)
                    olddist=old.act.action_out.categorical_heads[0](features,av[:,None,:16])
                    oldraw=old.act.action_out.categorical_heads[0].get_logits(features).squeeze(1)
                    oldact,oldlp,_=old(obs_i,rnn,active,av,True)
                    for t in (ta,tq):
                        p=t.actors[i].actor;raw,delta,logits=p.raw_logits(obs_i)
                        assert torch.equal(raw,oldraw) and torch.equal(logits,oldraw) and not delta.any()
                        assert torch.equal(p.distribution(obs_i,av).probs,olddist.probs)
                        act,lp,_=p(obs_i,rnn,active,av,True)
                        assert torch.equal(act,oldact) and torch.equal(lp,oldlp)
                        sampled,saved,_=p(obs_i,rnn,active,av,False)
                        eval_lp=p.evaluate_actions(obs_i,rnn,sampled,active,av,active)[0]
                        assert torch.equal(saved,eval_lp)
                        event=p.distribution(obs_i,av).log_probs(sampled.argmax(-1).reshape(-1,1,1)).reshape(-1)
                        # Divide by 16 is exact in float32; sum in float64 avoids
                        # introducing float32 reduction rounding into this identity.
                        assert torch.equal(saved.double().sum(-1),event.double())
                        assert torch.equal(aggregate_action_ratio(eval_lp-saved,'prod'),torch.ones(2,1))
            # Real executor independent physics/reward check after exactly one allocation.
            acts=[resource]+[ta.actors[i].act(post[:,i],rnn,active,available[:,i],True)[0] for i in range(1,4)]
            checked_step(env,acts);assert env.step_index==1
        checks[f'{seed}_zero_init_all_instructions_logits_probabilities_masks_actions_and_logp']='PASS'
        checks[f'{seed}_paired_critic_valuenorm_adapter_initialization']='PASS'
        for i in range(1,4):
            p=tq.actors[i].actor
            for parameter in p.adapter.parameters():
                with torch.no_grad():parameter.normal_(0,2)
            base=p.base_policy
            obs=tq.buffer.obs[0,:,i].clone();av=torch.ones(2,16);rnn=tq.buffer.rnn;active=torch.ones(2,1)
            for gid in (0,1):
                obs[:,67:70]=0;obs[:,67+gid]=1
                with torch.no_grad():
                    expected=base.act.action_out.categorical_heads[0](base.base(obs),av[:,None,:])
                    assert torch.equal(p.distribution(obs,av).probs,expected.probs)
                p.adapter.zero_grad(); lp,entropy,_=p.evaluate_actions(obs,rnn,torch.eye(16)[:2],active,av,active)
                (lp.sum()+entropy).backward()
                assert all(x.grad is not None and torch.count_nonzero(x.grad)==0 for x in p.adapter.parameters())
            # Small nonsaturating output makes the positive quality gradient meaningful.
            with torch.no_grad():p.adapter[4].weight.zero_();p.adapter[4].bias.zero_()
            obs[:,67:70]=0;obs[:,69]=1
            p.adapter.zero_grad();lp,entropy,_=p.evaluate_actions(obs,rnn,torch.eye(16)[:2],active,av,active)
            (lp.sum()+entropy).backward()
            assert sum(float(x.grad.abs().sum()) for x in p.adapter.parameters())>0
            tq.actors[i].actor_optimizer.step() # Establish nonzero Adam momentum.
            p.train();p.requires_grad_(True);assert not base.training and not any(x.requires_grad for x in base.parameters())
            p.eval()
        checks[f'{seed}_arbitrary_residual_gate_invariance_gradients_freeze_guard']='PASS'
        before=[copy.deepcopy(a.actor_optimizer.state_dict()) for a in tq.actors[1:]]
        hashes=[network_hash(a.actor.adapter) for a in tq.actors[1:]]
        critic_before=network_hash(tq.critic.critic)
        c,u=advance(tq)
        assert c['instruction_samples'][2]==0
        assert hashes==[network_hash(a.actor.adapter) for a in tq.actors[1:]]
        for b,a in zip(before,tq.actors[1:]):equal_tree(b,a.actor_optimizer.state_dict())
        assert network_hash(tq.critic.critic)!=critic_before
        assert tq.counters['actor_updates']==[0,0,0] and tq.counters['skipped_no_quality']==[10,10,10]
        tq.assert_frozen()
        checks[f'{seed}_no_quality_minibatch_skips_momentum_critic_updates_only']='PASS'
    checks['single_slot_floor_once_budget_reward_checker']='PASS'
    # Cross a real instruction switch and a 600-slot reset while checking exact resume.
    seed=m['parents'][0]
    for arm in m['arms']:
        t=RepairTrainer(seed,arm,preflight=True,length=400)
        advance(t)
        identity={'preflight':arm}
        state=t.state(1,identity)
        file=HERE/'preflight_artifacts'/f'{arm}_resume.pt';file.parent.mkdir(exist_ok=True)
        torch.save(state,file)
        second=advance(t);after=t.state(2,identity)
        r=RepairTrainer(seed,arm,preflight=True,length=400)
        r.restore(torch.load(file,weights_only=True),identity)
        equal_tree(state,r.state(1,identity))
        repeated=advance(r);after_repeat=r.state(2,identity)
        equal_tree(second,repeated);equal_tree(after,after_repeat)
        assert t.env.step_index==200 and t.counters['physical_steps']==1600
        checks[f'{arm}_full_checkpoint_save_load_resume_cross_termination']='PASS'
    # Extra draws on action streams cannot affect episode tapes now or after resets.
    ta=RepairTrainer(seed,'residual_all',preflight=True,length=8)
    tq=RepairTrainer(seed,'residual_quality',preflight=True,length=8)
    for _ in range(3):
        assert exogenous(ta.env)==exogenous(tq.env)
        with tq.streams.use(0):torch.rand(137)
        ta.env.reset();tq.env.reset()
    checks['exogenous_pairing_independent_of_extra_action_draws']='PASS'
    # Frozen strong predictor and deployed observation-only interface are loadable.
    for method in m['rules']:
        if not method.startswith('greedy_'):continue
        g=greedy_policy(method)
        env=make_env(cfg_for(seed),m['preflight_seeds'],'cpu',[[0,2]],'preflight')
        obs,_,mask=env.observe();acts,_,_,_=actions_for(env,obs,mask,method,greedy=g)
        checked_step(env,acts)
        checks[method+'_frozen_load_and_real_step']='PASS'
    verify_inputs(False)
    checks['original_critic_models_profile_configs_unchanged']='PASS'
    return checks

def benchmark(device):
    # Same small number of full 10x400 updates on each device; never formal artifacts.
    t=RepairTrainer(manifest()['parents'][0],'residual_all',device)
    timings=[]
    for _ in range(2):
        t.synchronize();start=time.monotonic();advance(t);t.synchronize();timings.append(time.monotonic()-start)
    return dict(device=device,update_seconds=timings,steps_per_second=8000/sum(timings),added_steps=8000,formal=False)

def main():
    p=argparse.ArgumentParser();p.add_argument('--benchmark-only',choices=['cpu','cuda:0']);args=p.parse_args()
    guard();folder=HERE/'preflight_attempts';folder.mkdir(exist_ok=True)
    path=folder/f'{datetime.datetime.now().strftime("%Y%m%dT%H%M%S%f")}.json'
    try:
        if args.benchmark_only:
            result=benchmark(args.benchmark_only);print(json.dumps(result));return
        checks=semantic_checks()
        result=dict(state='SEMANTICS_PASS',checks=checks,utc=stamp())
        write(path,result);write(HERE/'preflight_semantics.json',result);print(json.dumps(result))
    except Exception:
        write(path,dict(state='FAIL',error=traceback.format_exc(),utc=stamp()))
        raise

if __name__=='__main__':main()
