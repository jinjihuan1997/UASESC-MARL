"""Physical equivalence, causal observations, PPO replay and exact stage resume."""
import time
from types import SimpleNamespace
from helpers import *
from evaluate import external_metrics
from tensor_train import TensorTrainer,TensorRollout
from training_checkpoint import capture,restore

def equal(a,b):
    assert type(a) is type(b),(type(a),type(b))
    if torch.is_tensor(a): assert torch.equal(a,b),'Tensor mismatch'
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a: equal(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b): equal(x,y)
    else: assert a==b,(a,b)

def physics(device):
    cfg=config();seeds=[20262451,20262452];schedule=[[0,0],[200,1],[400,2]]
    env,_,_,_=make_env(cfg,seeds,schedule,device);refs=[]
    for seed in seeds:
        args=copy.deepcopy(cfg['env_args']);args.update(instruction_mode_strategy='explicit_evaluation',explicit_instruction_schedule=schedule)
        ref=SCUAVEnv(args);ref.seed(seed);ref.reset();refs.append(ref)
    for slot in range(600):
        actions=rule_actions(env,RULES[(slot//31)%len(RULES)])
        if slot%37==0:
            before={k:v.clone() for k,v in vars(env).items() if torch.is_tensor(v)}
            post,_,mask=env.allocate_resources(actions[0]);assert env.step_index==slot
            for k,v in before.items():
                if k not in ('beta','bandwidth','budget','allocated_action'): assert torch.equal(v,getattr(env,k)),k
            np.testing.assert_allclose(arr(post[:,1:,0])*env.p.Lambda_ref,arr(env.budget),atol=.004,rtol=1e-7)
            _,_,req=env.context();valid=(env.quality>=req[:,:,None]-1e-9)&(env.load<=env.budget[:,:,None]+1e-9)&env.q.any(-1)[:,:,None]
            assert torch.equal(mask[:,1:].bool(),valid|~valid.any(-1,keepdim=True))
        _,_,_,_,v=checked_step(env,actions)
        assert env.step_index==slot+1 and not env.allocation_pending
        assert env.switched.all().item()==(slot+1>=200)
        for i,ref in enumerate(refs):
            _,_,_,_,infos,_=ref.step([arr(a)[i] for a in actions]);out=external_metrics(ref,infos[0])
            for key in ['common_reward','mean_aoi','deliveries','predicted_quality_sum','channel_uses']:
                np.testing.assert_allclose(out[key],v[key][i],atol=1e-7,rtol=1e-10)
    print('Physics and single-slot sequencing PASS',device,flush=True)
    return 1200

def observations():
    cfg=config();env,obs,state,masks=make_env(cfg,[20262461])
    env.instructions[1:]=2
    assert torch.equal(obs,env.observe()[0]) and torch.equal(state,env.observe()[1])
    original_obs=obs.clone();original_state=state.clone();env.step_index=150;env.tau+=150
    obs,state,_=env.observe()
    assert not torch.equal(obs,original_obs) and not torch.equal(state,original_state)
    # High ages remain distinguishable beyond the observation reference of 8.
    env.aoi.fill_(16);a=env.observe()[0];env.aoi.fill_(24);b=env.observe()[0];assert not torch.equal(a,b)
    cfg['env_args']['actor_observe_instruction']=False
    h,_,_,_=make_env(cfg,[20262461]);feeds=[]
    for g in range(3):
        h.instructions[0]=g;h.switched.fill_(g>0);feeds.append(h.observe())
    for f in feeds[1:]: assert torch.equal(feeds[0][0],f[0]) and torch.equal(feeds[0][2],f[2])
    # The newly added cache summaries resolve a concrete old SUT ambiguity.
    h.q.zero_();h.q[:,:,0]=True;h.aoi.fill_(5);h.aoi[:,0,0]=2;a=h.observe()[0][:,0].clone()
    h.aoi[:,0,0]=4;b=h.observe()[0][:,0];assert not torch.equal(a,b)
    assert h.obs_dim_common==76 and h.share_obs_dim==202
    print('Observation and no-future/hidden-actor leakage PASS',flush=True)

def terminal_returns():
    x=SimpleNamespace(length=3,values=torch.tensor([10.,20.,30.,40.]).reshape(4,1,1),
        rewards=torch.tensor([1.,2.,3.]).reshape(3,1,1),masks=torch.tensor([1.,1.,0.,1.]).reshape(4,1,1))
    result=TensorRollout.returns(x,SimpleNamespace(denormalize=lambda v:v),.9,1.)
    torch.testing.assert_close(result.flatten(),torch.tensor([2.8,2.,39.]))

def one_update(t,u,check_replay=False):
    t.set_training_update(u)
    for actor in t.actors: actor.lr_decay(u,t.total_updates)
    t.critic.lr_decay(u,t.total_updates);t.collect()
    if check_replay:
        for i in t.trainable_agent_ids:
            actual=t.actors[i].evaluate_actions(*t.buffer.actor_inputs(i))[0]
            torch.testing.assert_close(actual,t.buffer.log_probs[i].flatten(0,1),atol=2e-6,rtol=1e-5)
        shares=((t.buffer.actions[0]+1)/2).clamp_min(0);shares/=shares.sum(-1,keepdim=True)
        budget=(.05+.85*shares)*60000
        torch.testing.assert_close(t.buffer.obs[:-1,:,1:,0]*100000,budget,atol=.01,rtol=1e-6)
    out=t.update();assert all(torch.isfinite(x).all() for x in out.values());t.buffer.after_update()
    if t.fixed_resources:
        assert network_hash(t.actors[0].actor)==t.initial_actors[0]
        assert not t.actors[0].actor_optimizer.state

def training(device,arm):
    cfg=config(arm);cfg['algo_args']['train']['num_env_steps']=16000
    if arm=='staged': cfg['training_design']['resource_warmup_steps']=4000
    t=TensorTrainer(cfg,device);initial=(t.initial_actors,t.initial_critic)
    one_update(t,1,True);saved=capture(t,1,{'test':arm});one_update(t,2);whole=capture(t,2,{'test':arm})
    resumed=TensorTrainer(cfg,device);assert restore(resumed,saved,{'test':arm})==1
    one_update(resumed,2);equal(whole,capture(resumed,2,{'test':arm}))
    assert t.env.step_index==200 and t.buffer.masks[-1].all()
    if arm=='staged':
        assert t.stage=='joint' and network_hash(t.actors[0].actor)!=t.initial_actors[0]
    print('Sampler replay, freeze/stage and exact resume PASS',device,arm,flush=True)
    return initial

def main():
    verify_reference();slots=physics('cpu')+physics('cuda:0');observations();terminal_returns()
    initial=[training('cpu',arm) for arm in METHODS];assert initial[0]==initial[1]==initial[2]
    gpu=training('cuda:0','staged');assert gpu==initial[0]
    for seed in SEEDS:
        c=config('staged',seed);t=TensorTrainer(c,'cpu');t.set_training_update(75);assert t.fixed_resources
        t.set_training_update(76);assert not t.fixed_resources
    for file,h in read(HERE/'provenance.json')['parent_input_hashes'].items(): assert digest(file)==h,file
    write(HERE/'validation.json',dict(state='PASS',physical_reference_slots=slots,one_slot_one_reward=True,
        current_budget_observed=True,recorded_actor_log_prob_replay=True,finite_terminal_gae=True,
        rollout_cutoff_bootstrap=True,no_future_tape_leakage=True,hidden_actor_invariance=True,
        observation_dimensions=[76,202],cpu_three_arm_initial_hashes_match=True,cpu_cuda_initial_hashes_match=True,
        full_state_exact_resume=['cpu_joint','cpu_mode_only','cpu_staged','cuda_staged'],
        frozen_sut_no_optimizer_steps=True,staged_boundary_steps=300000,parent_artifacts_unchanged=True))

if __name__=='__main__': main()
