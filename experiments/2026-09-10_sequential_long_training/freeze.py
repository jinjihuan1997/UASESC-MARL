"""Freeze the requested 0.6M + 9.4M schedule and paired 10M joint controls."""
import ast
from helpers import *
from training_checkpoint import runtime_signature

def jobs_for(seeds,milestones):
    jobs=[]
    for core,(seed,arm) in enumerate((s,a) for s in seeds for a in METHODS):
        item=f'seed_{seed}/{arm}'
        jobs.append(dict(kind='training',id=item,seed=seed,method=arm,device='cpu',core=core,config=f'configs/{item}.json',output=f'jobs/{item}'))
    evaluation=[dict(kind='evaluating',id=f'rules/{rule}',output=f'evaluation/rules/{rule}') for rule in RULE_METHODS]
    for step in milestones:
        for seed in seeds:
            for arm in METHODS:
                item=f'seed_{seed}/{arm}_at_{step}'
                evaluation.append(dict(kind='evaluating',id=item,output=f'evaluation/{item}',model_dir=f'jobs/seed_{seed}/{arm}/milestones/steps_{step}'))
    return jobs,evaluation

def main():
    assert not (HERE/'manifest.json').exists()
    assert read(HERE/'preflight_results.json')['state']=='PASS'
    verify_reference()
    for file,h in read(HERE/'provenance.json')['parent_input_hashes'].items(): assert digest(file)==h,file
    for seed in SEEDS:
        cfgs=[config(a,seed) for a in METHODS]
        assert cfgs[0]['algo_args']==cfgs[1]['algo_args'] and cfgs[0]['env_args']==cfgs[1]['env_args']
        for arm,c in zip(METHODS,cfgs):
            old=read(PARENT/f'configs/seed_{seed}/{arm}.json')
            a=copy.deepcopy(c['algo_args']);a['train']['num_env_steps']=old['algo_args']['train']['num_env_steps'];assert a==old['algo_args']
            env=copy.deepcopy(c['env_args'])
            for key in ['semantic_profile_path','semantic_registry_path']:env[key]=old['env_args'][key]
            assert env==old['env_args']
            assert c['training_design']['resource_warmup_steps']==(600000 if arm=='staged' else 0)
            assert c['training_design']['evaluation_steps']==MILESTONES
    jobs,evaluation=jobs_for(SEEDS,MILESTONES);scenarios=read(PARENT/'manifest.json')['scenarios']
    prior_probe=read(PARENT/'resource_probe.json')
    # Reserve the six faster measured CPU cores for training; evaluation uses two others.
    estimate=max(prior_probe['layouts']['cpu_9']['per_job_seconds_per_update'][:6])
    files=list(HERE.glob('*.py'))+[HERE/f for f in ['PROTOCOL.md','provenance.json','preflight_results.json','rule_selection.json','run_training.sh']]
    for folder in ['source','configs']:files.extend(f for f in (HERE/folder).rglob('*') if f.is_file())
    for f in files:
        if f.suffix=='.py':ast.parse(f.read_text())
    m=dict(schema=1,purpose='sequential_long_10m_warmup600k_three_seed',created_utc=stamp(),methods=METHODS,seeds=SEEDS,
        jobs=jobs,evaluation_jobs=evaluation,evaluation_items=[j['id'] for j in evaluation],milestones=MILESTONES,
        steps_per_method=10000000,resource_warmup_steps=600000,total_training_steps=60000000,total_training_jobs=6,batch=4000,
        evaluation_seeds=EVAL_SEEDS,calibration_seeds=CAL_SEEDS,scenarios=scenarios,evaluation_episodes=len(evaluation)*len(scenarios)*len(EVAL_SEEDS),
        runtime=runtime_signature(),resources=dict(layout='6_cpu_training_plus_2_cpu_evaluation',cores=list(range(8)),training_cores=list(range(6)),
            seconds_per_update_estimate=estimate,estimate_source=str(PARENT/'resource_probe.json'),benchmark_training_seconds=estimate*2500),
        input_hashes={str(f.relative_to(HERE)):digest(f) for f in sorted(set(files))})
    write(HERE/'manifest.json',m);write(HERE/'status.json',dict(state='prepared',total_training_steps=60000000))
    print('Frozen: 6 models x 10M steps; staged 600k + 9400k;',m['evaluation_episodes'],'evaluation episodes',flush=True)

if __name__=='__main__':main()
