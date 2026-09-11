"""Transfer, branch replay, PPO likelihood, snapshot and exact-resume checks."""
import subprocess
from helpers import *
from selector_train import ContinuationTrainer,capture_fork,restore_fork
from training_checkpoint import load_checkpoint
from evaluation import FIELDS

def equal(a,b):
    assert type(a) is type(b),(type(a),type(b))
    if torch.is_tensor(a):assert torch.equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:equal(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):equal(x,y)
    else:assert a==b,(a,b)

def branch_replays(work):
    checked=0
    for seed in SEEDS:
        cfg=config('selector',seed)
        t=ContinuationTrainer(cfg,'cpu');t.initialize_transfer({'preflight':'branch'})
        for scene in ['fixed_0','fixed_2','switch300_1_to_2']:
            schedule=read(PARENT/'manifest.json')['scenarios'][scene]
            for route in ['original','quality_equal']:
                env,obs,_,masks=make_env(cfg,EVAL_SEEDS,schedule)
                class ForcedSelector:
                    def act(self,*args,**kwargs):
                        choice=(env.context()[0]==2).long() if route=='quality_equal' else torch.zeros(env.count,dtype=torch.long)
                        return choice[:,None],None
                actors=t.actors+[ForcedSelector()]
                path=(PARENT/f'evaluation/seed_{seed}/joint_at_10000000/{scene}.npz' if route=='original' else
                    PARENT.parent/f'2026-09-10_post_long_diagnosis/targeted_followup/evaluation/seed_{seed}/quality_rule_resources_only/{scene}.npz')
                with np.load(path) as z:expected={k:z[k].copy() for k in ['trace','modes','resource_fractions','aoi_after']}
                for slot in range(600):
                    act=learned_actions(env,actors,obs,masks,'selector')
                    obs,_,masks,info,values=checked_step(env,act);values['instruction_id']=arr(info['gid'])
                    row=np.column_stack([values[f] for f in FIELDS])
                    np.testing.assert_array_equal(row,expected['trace'][slot])
                    np.testing.assert_array_equal(arr(info['mode']),expected['modes'][slot])
                    np.testing.assert_array_equal(arr(env.beta),expected['resource_fractions'][slot])
                    np.testing.assert_array_equal(arr(env.aoi),expected['aoi_after'][slot])
                checked+=len(EVAL_SEEDS)
        print('Replay PASS',seed,flush=True)
    return checked

def main():
    work=HERE/'preflight';work.mkdir(exist_ok=False)
    verify_reference()
    for f,h in read(HERE/'provenance.json')['parent_input_hashes'].items():assert digest(f)==h,f
    # Parent state, optimizer history, finite episode position and random states migrate together.
    for seed in SEEDS:
        a=ContinuationTrainer(config('joint_continue',seed),'cpu');a.initialize_transfer({'test':'transfer'})
        b=ContinuationTrainer(config('selector',seed),'cpu');b.initialize_transfer({'test':'transfer'})
        assert a.initial_actors==b.initial_actors and a.initial_critic==b.initial_critic
        assert a.env.step_index==b.env.step_index==400
        for name in ['q','aoi','tau','noise_us','noise_sat','noise_du','instructions']:
            equal(getattr(a.env,name),getattr(b.env,name))
        # Learning rates deliberately differ: compare inherited Adam moments/steps.
        for x,y in zip(a.actors,b.actors):equal(x.actor_optimizer.state_dict()['state'],y.actor_optimizer.state_dict()['state'])
        del a,b
    cfg=config('selector');t=ContinuationTrainer(cfg,'cpu');t.initialize_transfer({'test':'likelihood'});t.collect()
    b=t.buffer;active=torch.ones((t.batch,1))
    logp,_,dist=t.selector.evaluate_actions(b.obs[:-1,:,0].flatten(0,1),b.flat_rnn,b.selector_actions.flatten(0,1),b.masks[:-1].flatten(0,1),torch.ones((t.batch,2)),active)
    np.testing.assert_allclose(arr(logp),arr(b.selector_log_probs.flatten(0,1)),atol=1e-6,rtol=0)
    assert torch.isfinite(logp).all() and b.selector_actions.unique().tolist()==[0.,1.]
    assert all(not p.requires_grad for a in t.actors for p in a.actor.parameters())
    before=t.initial_selector;t.update();assert network_hash(t.selector.actor)!=before
    del t
    replay_episodes=branch_replays(work)
    jobs=[];ev=[]
    for arm in METHODS:
        cfg=config(arm);cfg['algo_args']['train']['num_env_steps']=12000;cfg['training_design']['evaluation_steps']=[4000,12000]
        path=work/f'configs/seed_85/{arm}.json';write(path,cfg)
        jobs.append(dict(method=arm,id=f'seed_85/{arm}',output=f'jobs/seed_85/{arm}',config=str(path.relative_to(work))))
        for step in [4000,12000]:ev.append(f'seed_85/{arm}_at_{step}')
    ev+=['rules/'+r for r in RULE_METHODS]
    manifest=dict(purpose='selector_transfer_smoke',seeds=[85],jobs=jobs,milestones=[4000,12000],steps_per_method=12000,total_training_steps=24000,batch=4000,
        evaluation_seeds=[20262471,20262472],scenarios={'fixed_1':[[0,1]],'fixed_2':[[0,2]],'switch300_2_to_1':[[0,2],[300,1]]},evaluation_items=ev,evaluation_episodes=len(ev)*6)
    write(work/'manifest.json',manifest);write(work/'rule_selection.json',read(HERE/'rule_selection.json'))
    for j in jobs:
        arm=j['method'];path=work/j['config'];folder=work/j['output'];resumed=work/'resumed'/arm
        def command(out,extra=()):
            return [sys.executable,'-u',str(HERE/'source/formal_train.py'),'--config',str(path),'--output',str(out),
                '--run-manifest',str(work/'manifest.json'),'--device','cpu','--checkpoint-every','1',*extra]
        with (work/f'{arm}.log').open('w') as log:
            assert subprocess.run(command(folder),stdout=log,stderr=subprocess.STDOUT).returncode==0,arm
            assert subprocess.run(command(resumed,('--stop-after-update','1')),stdout=log,stderr=subprocess.STDOUT).returncode==75,arm
            assert subprocess.run(command(resumed,('--resume',)),stdout=log,stderr=subprocess.STDOUT).returncode==0,arm
        whole,_=load_checkpoint(folder/'checkpoints');restored,_=load_checkpoint(resumed/'checkpoints');equal(whole,restored)
        print('Exact resume PASS',arm,flush=True)
    import evaluation,aggregate
    evaluation.HERE=work
    evaluation.config=lambda arm='joint_continue',seed=85:read(work/f'configs/seed_{seed}/{arm}.json')
    for item in ev:
        assert evaluation.evaluate_item(item,[])==0
        assert evaluation.evaluate_item(item,[])==0
    aggregate.main(root=work,check_inputs=False)
    for f,h in read(HERE/'provenance.json')['parent_input_hashes'].items():assert digest(f)==h,f
    write(HERE/'preflight_results.json',dict(state='PASS',matched_transferred_weights_and_optimizer_history_all_seeds=True,
        inherited_finite_episode_and_exogenous_sequences=True,exact_original_and_quality_equal_replay_episodes=replay_episodes,
        selector_likelihood_replayed=True,only_selector_actor_trained=True,exact_checkpoint_resume=METHODS,
        snapshot_evaluation_idempotent=True,full_evaluation_audit_episodes=manifest['evaluation_episodes'],old_inputs_unchanged=True))
    print('Preflight PASS',flush=True)

if __name__=='__main__':main()
