"""Validate the actual milestone export/evaluation pipeline and exact resume."""
import subprocess
from helpers import *
from training_checkpoint import load_checkpoint
from tensor_train import TensorTrainer
from freeze import jobs_for
from supervisor import ready
import evaluation,aggregate

def equal(a,b):
    assert type(a) is type(b)
    if torch.is_tensor(a):assert torch.equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:equal(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):equal(x,y)
    else:assert a==b

def main():
    verify_reference();work=HERE/'preflight';work.mkdir(exist_ok=False)
    jobs,eval_jobs=jobs_for([85],[4000,12000])
    m=dict(purpose='10m_pipeline_smoke_only',seeds=[85],jobs=jobs,evaluation_jobs=eval_jobs,evaluation_items=[j['id'] for j in eval_jobs],
        steps_per_method=12000,resource_warmup_steps=4000,total_training_steps=24000,batch=4000,milestones=[4000,12000],
        evaluation_seeds=[20262471,20262472],scenarios={'fixed_1':[[0,1]],'fixed_2':[[0,2]],'switch300_2_to_1':[[0,2],[300,1]]},evaluation_episodes=42)
    write(work/'manifest.json',m);write(work/'rule_selection.json',read(HERE/'rule_selection.json'))
    for name in ['tensor_env.py','tensor_train.py','training_checkpoint.py','episode_source.py','reference_manifest.json']:
        assert digest(HERE/'source'/name)==digest(PARENT/'source'/name),name
    for seed in SEEDS:
        a=TensorTrainer(config('joint',seed),'cpu');b=TensorTrainer(config('staged',seed),'cpu')
        assert a.initial_actors==b.initial_actors and a.initial_critic==b.initial_critic
        assert a.total_updates==b.total_updates==2500
        b.set_training_update(150);assert b.fixed_resources
        b.set_training_update(151);assert not b.fixed_resources
        for update in [1,150,151,2500]:
            b.actors[0].lr_decay(update,b.total_updates)
            np.testing.assert_allclose(b.actors[0].actor_optimizer.param_groups[0]['lr'],1e-4*(1-(update-1)/2500),rtol=1e-9)
        del a,b
    for job in jobs:
        arm=job['method'];cfg=config(arm);cfg['algo_args']['train']['num_env_steps']=12000
        cfg['training_design']['resource_warmup_steps']=4000 if arm=='staged' else 0
        cfg['training_design']['evaluation_steps']=[4000,12000]
        path=work/job['config'];write(path,cfg)
        def command(output,extra=()):
            return [sys.executable,'-u',str(HERE/'source/formal_train.py'),'--config',str(path),'--output',str(output),
                '--run-manifest',str(work/'manifest.json'),'--device','cpu','--checkpoint-every','1',*extra]
        folder=work/job['output'];resumed=work/'resumed'/arm
        with (work/f'{arm}.log').open('w') as log:
            assert subprocess.run(command(folder),stdout=log,stderr=subprocess.STDOUT).returncode==0
            assert subprocess.run(command(resumed,('--stop-after-update','1')),stdout=log,stderr=subprocess.STDOUT).returncode==75
            assert subprocess.run(command(resumed,('--resume',)),stdout=log,stderr=subprocess.STDOUT).returncode==0
        whole,_=load_checkpoint(folder/'checkpoints');restored,_=load_checkpoint(resumed/'checkpoints');equal(whole,restored)
        for step in [4000,12000]:
            a=folder/'milestones'/f'steps_{step}';b=resumed/'milestones'/f'steps_{step}'
            for name in ['actor_agent0.pt','actor_agent1.pt','actor_agent2.pt','actor_agent3.pt','critic_agent.pt','value_normalizer.pt']:
                equal(torch.load(a/name,map_location='cpu',weights_only=True),torch.load(b/name,map_location='cpu',weights_only=True))
    assert all(ready(j,work) for j in eval_jobs)
    assert not ready(dict(kind='evaluating',model_dir='missing'),work)
    evaluation.HERE=work
    for j in eval_jobs:
        assert evaluation.evaluate_item(j['id'],[])==0
        assert evaluation.evaluate_item(j['id'],[])==0
    aggregate.main(root=work,check_inputs=False)
    for f,h in read(HERE/'provenance.json')['parent_input_hashes'].items():assert digest(f)==h,f
    write(HERE/'preflight_results.json',dict(state='PASS',unchanged_environment_optimizer_and_resume_kernels=True,
        paired_initial_weights_all_three_seeds=True,actual_budget_updates=2500,warmup_last_update=150,joint_first_update=151,
        exact_resume_with_snapshots=['joint','staged'],snapshot_evaluator_idempotent=True,full_aggregate_audit=True,
        evaluation_episodes=42,smoke_training_steps=48000,old_artifacts_unchanged=True))
    print('Long-training preflight PASS',flush=True)

if __name__=='__main__':main()
