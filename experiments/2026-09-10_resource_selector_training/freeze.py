"""Freeze the approved two-arm, three-seed, 2M-step continuation experiment."""
import ast
from helpers import *
from training_checkpoint import runtime_signature

def main():
    assert not (HERE/'manifest.json').exists()
    assert read(HERE/'preflight_results.json')['state']=='PASS'
    probe=read(HERE/'resource_probe.json');assert probe['state']=='PASS'
    verify_reference()
    for f,h in read(HERE/'provenance.json')['parent_input_hashes'].items():assert digest(f)==h,f
    layout=probe['selected'];jobs=[]
    for core,(seed,arm) in enumerate((s,a) for s in SEEDS for a in METHODS):
        item=f'seed_{seed}/{arm}';device='cuda:0' if layout=='four_cpu_two_gpu' and seed==966 else 'cpu'
        jobs.append(dict(kind='training',id=item,seed=seed,method=arm,device=device,core=core,config=f'configs/{item}.json',output=f'jobs/{item}'))
    ev=[dict(kind='evaluating',id=f'rules/{r}',output=f'evaluation/rules/{r}') for r in RULE_METHODS]
    for step in MILESTONES:
        for seed in SEEDS:
            for arm in METHODS:
                item=f'seed_{seed}/{arm}_at_{step}'
                ev.append(dict(kind='evaluating',id=item,output=f'evaluation/{item}',model_dir=f'jobs/seed_{seed}/{arm}/milestones/steps_{step}'))
    for seed in SEEDS:
        a,b=[config(arm,seed) for arm in METHODS]
        assert a['env_args']==b['env_args'] and a['continuation']['parent_sha256']==b['continuation']['parent_sha256']
        assert a['algo_args']['algo']==b['algo_args']['algo']
        assert a['algo_args']['train']==b['algo_args']['train']
    files=list(HERE.glob('*.py'))+[HERE/n for n in ['PROTOCOL.md','run_training.sh','preflight_results.json','preflight_amendments.md','provenance.json','resource_probe.json','rule_selection.json']]
    for folder in ['source','configs','initial']:files.extend(f for f in (HERE/folder).rglob('*') if f.is_file())
    for f in files:
        if f.suffix=='.py':ast.parse(f.read_text())
    estimate=probe['layouts'][layout]['seconds_per_update']
    scenarios=read(PARENT/'manifest.json')['scenarios']
    m=dict(schema=1,purpose='resource_selector_vs_joint_continuation_2m_three_seed',created_utc=stamp(),methods=METHODS,seeds=SEEDS,
        jobs=jobs,evaluation_jobs=ev,evaluation_items=[j['id'] for j in ev],milestones=MILESTONES,
        steps_per_method=2000000,parent_steps_per_method=10000000,total_training_steps=12000000,total_training_jobs=6,batch=4000,
        evaluation_seeds=EVAL_SEEDS,evaluation_split='previously_used_development_seeds',scenarios=scenarios,
        evaluation_episodes=len(ev)*len(scenarios)*len(EVAL_SEEDS),runtime=runtime_signature(),
        resources=dict(layout=layout,cores=list(range(8)),training_cores=list(range(6)),seconds_per_update_estimate=estimate,
            estimated_training_seconds=500*estimate,estimate_excludes='Evaluation, I/O and thermal pauses'),
        input_hashes={str(f.relative_to(HERE)):digest(f) for f in sorted(set(files))})
    write(HERE/'manifest.json',m);write(HERE/'status.json',dict(state='prepared',total_training_steps=12000000))
    print('Frozen',layout,'6 jobs x 2M;',m['evaluation_episodes'],'evaluation episodes',flush=True)

if __name__=='__main__':main()
