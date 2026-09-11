"""Freeze six new training jobs and eighteen paired evaluation objects."""
import ast
from helpers import *
from training_checkpoint import runtime_signature


def main():
    assert not (HERE/'manifest.json').exists()
    for name in ['validation.json','calibration_results.json','preflight_results.json']:assert read(HERE/name)['state']=='PASS'
    provenance=read(HERE/'provenance.json')
    for f,h in provenance['parent_input_hashes'].items():assert digest(PARENT/f)==h,f
    verify_reference();pm=read(PARENT/'manifest.json')
    for seed in SEEDS:
        for method in METHODS:
            a=config('ref10',method==METHODS[1],seed);b=config('ref8',method==METHODS[1],seed)
            assert a['algo_args']==b['algo_args'] and a['main_args']==b['main_args']
            assert [k for k in a['env_args'] if a['env_args'][k]!=b['env_args'][k]]==['aoi_reward_ref']
            assert a['env_args']['aoi_reward_ref']==10 and b['env_args']['aoi_reward_ref']==8
    jobs=[]
    for core,(seed,method) in enumerate((s,m) for s in SEEDS for m in METHODS):
        i=f'ref8/seed_{seed}/{method}'
        jobs.append(dict(id=i,objective='ref8',seed=seed,method=method,device='cpu',core=core,
            config=f'configs/{i}.json',output=f'jobs/{i}'))
    items=[f'{o}/seed_{s}/{method}' for o in OBJECTIVES for s in SEEDS for method in METHODS]
    items += [f'{o}/rules/{method}' for o in OBJECTIVES for method in RULE_METHODS]
    files=list(HERE.glob('*.py'))+[HERE/n for n in ['PROTOCOL.md','provenance.json','validation.json','calibration_results.json','rule_selection.json','preflight_results.json','run_pilot.sh']]
    for folder in ['source','configs','calibration','jobs/ref10']:
        files.extend(p for p in (HERE/folder).rglob('*') if p.is_file())
    for p in files:
        if p.suffix=='.py':ast.parse(p.read_text())
    manifest=dict(schema=1,purpose='aoi_reference_10_to_8_three_seed_1m_control',created_utc=stamp(),
        objectives=OBJECTIVES,seeds=SEEDS,methods=METHODS,rules=RULE_METHODS,jobs=jobs,
        steps_per_method=1000000,total_training_steps=6000000,total_training_jobs=6,batch=4000,
        runtime=runtime_signature(),evaluation_items=items,evaluation_episodes=4680,
        calibration_seeds=CAL_SEEDS,evaluation_seeds=EVAL_SEEDS,scenarios=pm['scenarios'],
        resources=dict(layout='six_cpu_jobs_one_wave',cores=list(range(6)),seconds_per_update_estimate=pm['resources']['seconds_per_update_estimate']),
        input_hashes={str(p.relative_to(HERE)):digest(p) for p in sorted(set(files))})
    write(HERE/'manifest.json',manifest)
    write(HERE/'status.json',dict(state='prepared',total_training_jobs=6,total_training_steps=6000000))
    print('Frozen 6 new models / 6M steps / 4680 paired evaluation episodes',flush=True)


if __name__=='__main__':main()
