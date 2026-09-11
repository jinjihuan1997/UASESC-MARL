"""Freeze all implementation, protocol and resource choices before formal learning."""
import ast
from helpers import *
from training_checkpoint import runtime_signature

def main():
    assert not (HERE/'manifest.json').exists()
    for name in ['validation.json','preflight_results.json','calibration_results.json','resource_probe.json']:
        assert read(HERE/name)['state']=='PASS',name
    for file,h in read(HERE/'provenance.json')['parent_input_hashes'].items(): assert digest(file)==h,file
    verify_reference();probe=read(HERE/'resource_probe.json');layout=probe['layouts'][probe['selected']];jobs=[]
    for core,(seed,arm) in enumerate((s,a) for s in SEEDS for a in METHODS):
        item=f'seed_{seed}/{arm}'
        jobs.append(dict(id=item,seed=seed,method=arm,device=layout['devices'][core],core=core,
            config=f'configs/{item}.json',output=f'jobs/{item}'))
    for seed in SEEDS:
        assert len({j['device'] for j in jobs if j['seed']==seed})==1
        cfgs=[config(a,seed) for a in METHODS];assert all(c['algo_args']==cfgs[0]['algo_args'] and c['env_args']==cfgs[0]['env_args'] for c in cfgs)
    scenarios=read(PARENT/'manifest.json')['scenarios']
    items=[j['id'] for j in jobs]+[f'rules/{name}' for name in RULE_METHODS]
    files=list(HERE.glob('*.py'))+[HERE/n for n in ['PROTOCOL.md','provenance.json','validation.json','calibration_results.json','rule_selection.json','preflight_results.json','resource_probe.json','run_training.sh']]
    for folder in ['source','configs']: files.extend(f for f in (HERE/folder).rglob('*') if f.is_file())
    for f in files:
        if f.suffix=='.py': ast.parse(f.read_text())
    m=dict(schema=1,purpose='sequential_mdp_v1_three_arm_three_seed_1m',created_utc=stamp(),
        methods=METHODS,seeds=SEEDS,jobs=jobs,rules=RULE_METHODS,steps_per_method=1000000,
        total_training_steps=9000000,total_training_jobs=9,batch=4000,runtime=runtime_signature(),
        evaluation_items=items,evaluation_seeds=EVAL_SEEDS,calibration_seeds=CAL_SEEDS,scenarios=scenarios,
        evaluation_episodes=len(items)*len(scenarios)*len(EVAL_SEEDS),
        resources=dict(layout=probe['selected'],cores=list(range(9)),seconds_per_update_estimate=layout['per_job_seconds_per_update'],
            benchmark_training_seconds=layout['estimated_training_seconds']),
        input_hashes={str(f.relative_to(HERE)):digest(f) for f in sorted(set(files))})
    write(HERE/'manifest.json',m);write(HERE/'status.json',dict(state='prepared',total_training_steps=9000000))
    print('Frozen 9 models / 9M training steps /',m['evaluation_episodes'],'evaluation episodes /',probe['selected'],flush=True)

if __name__=='__main__': main()
