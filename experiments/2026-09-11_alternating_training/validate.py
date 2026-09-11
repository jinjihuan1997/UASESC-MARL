"""Exercise all phase boundaries, freezing, exact resume, and physical parity."""
import argparse
from helpers import *
from prepare import make_config
from tensor_train import TensorTrainer
from training_checkpoint import capture,restore,cpu_tree

def equal_tree(a,b):
    if isinstance(a,torch.Tensor):return torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal_tree(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return len(a)==len(b) and all(equal_tree(x,y) for x,y in zip(a,b))
    return a==b

def step(trainer,update):
    trainer.set_training_update(update)
    before=[network_hash(a.actor) for a in trainer.actors]
    optimizers=[cpu_tree(a.actor_optimizer.state_dict()) for a in trainer.actors]
    trainer.collect();metrics=trainer.update()
    assert all(torch.isfinite(v).all() for v in metrics.values())
    after=[network_hash(a.actor) for a in trainer.actors]
    for i in range(4):
        if i in trainer.trainable_agent_ids:assert after[i]!=before[i],('untrained',update,i)
        else:
            assert after[i]==before[i],('frozen weight changed',update,i)
            assert equal_tree(optimizers[i],cpu_tree(trainer.actors[i].actor_optimizer.state_dict()))
    assert trainer.mode_budget_counts.sum().item()==trainer.batch*3
    trainer.buffer.after_update()
    return dict(update=update,stage=trainer.stage,trained_actor_ids=trainer.trainable_agent_ids)

def validate(device):
    verify_reference()
    cfg=make_config(9911123,'alternating')
    cfg['algo_args']['train'].update(n_rollout_threads=2,episode_length=12,num_env_steps=168)
    cfg['training_design']['evaluation_steps']=[]
    for phase,end in zip(cfg['training_design']['phases'],[48,72,96,120,144,168]):phase['end_step']=end
    a=TensorTrainer(cfg,device)
    joint=copy.deepcopy(cfg);joint['training_design'].update(arm='joint',phases=[])
    b=TensorTrainer(joint,device)
    assert a.initial_actors==b.initial_actors and a.initial_critic==b.initial_critic
    assert external_hashes(a.env)==external_hashes(b.env)
    # Distribution test is separate from training and uses the saved global RNG.
    initial=capture(a,0,{'test':'distribution'})
    equal_count=0;shares=[]
    for _ in range(2000):
        action,equal=a.sample_warmup_resources();equal_count+=int(equal.sum())
        beta=.05+.85*(action+1)/2
        assert (beta>=.05-1e-6).all() and (beta<=.90+1e-6).all()
        torch.testing.assert_close(beta.sum(-1),torch.ones(2,device=device))
        shares.append(beta.cpu())
    assert .46<equal_count/4000<.54
    assert torch.cat(shares).amin()<.08 and torch.cat(shares).amax()>.8
    restore(a,initial,{'test':'distribution'})
    rows=[step(a,1),step(a,2)]
    identity={'test':'phase_boundary_resume'}
    saved=capture(a,2,identity)
    rows.append(step(a,3));expected=capture(a,3,identity)
    c=TensorTrainer(cfg,device)
    assert restore(c,saved,identity)==2
    step(c,3);actual=capture(c,3,identity)
    assert equal_tree(expected,actual),'Resume changed state at warmup to SUT transition'
    for update in range(4,8):rows.append(step(c,update))
    assert [r['trained_actor_ids'] for r in rows]==[[1,2,3],[1,2,3],[0],[1,2,3],[0],[1,2,3],[0]]
    for update in range(1,3):step(b,update)
    # Exact 600-slot physical/rule validation, including instruction switches.
    # Compare observable-summary resource actions with original integer state formula.
    physical=0
    if device=='cpu':
        for method in RULE_METHODS:
            env,obs,_,masks=make_env(make_config(9911123,'joint'),[9911123,9912123],[[0,0],[200,1],[400,2]])
            for slot in range(600):
                actions=simple_rule_actions(env,method,obs)
                gid=int(env.context()[0][0])
                use_urgent=method=='R_single' or (method=='R_instruction' and gid==1)
                score=(env.aoi*env.q).sum(-1) if use_urgent else torch.ones_like(env.beta)
                total=score.sum(-1,keepdim=True)
                share=torch.where(total>0,score/total.clamp_min(1),torch.full_like(score,1/3))
                torch.testing.assert_close(actions[0],2*share-1,atol=1e-12,rtol=0)
                obs,_,masks,_,_=checked_step(env,actions);physical+=env.count
    result=dict(state='PASS',device=device,phase_updates=rows,
        exact_resume_across_warmup_boundary=True,frozen_weights_and_optimizers=True,
        paired_initial_weights_and_external_traces=True,equal_draw_fraction=equal_count/4000,
        independently_checked_physical_slots=physical)
    write(HERE/'preflight'/f'validation_{device.replace(":","_")}.json',result)
    print(json.dumps(result))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',choices=['cpu','cuda:0'],required=True)
    validate(p.parse_args().device)
