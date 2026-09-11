"""Freeze fully validated protocol, calibration, source and twelve jobs."""
import ast
from helpers import *
from training_checkpoint import runtime_signature
from protocol import SCENARIOS


def main():
    assert not (HERE/'manifest.json').exists()
    for name in ['validation.json','calibration_results.json','preflight_results.json']:
        assert read(HERE/name)['state']=='PASS'
    provenance=read(HERE/'provenance.json')
    for f,h in provenance['parent_input_hashes'].items():assert digest(PARENT/f)==h,f
    verify_reference()
    jobs=[]
    for objective in OBJECTIVES:
        for core,(seed,method) in enumerate((s,m) for s in SEEDS for m in METHODS):
            item=f'{objective}/seed_{seed}/{method}'
            jobs.append(dict(id=item,objective=objective,seed=seed,method=method,device='cpu',core=core,
                             config=f'configs/{item}.json',output=f'jobs/{item}'))
    for seed in SEEDS:
        for o in OBJECTIVES:
            a=config(o,False,seed);b=config(o,True,seed)
            assert a['algo_args']==b['algo_args']
            assert [k for k in a['env_args'] if a['env_args'][k]!=b['env_args'][k]]==['actor_observe_instruction']
        for hidden in (False,True):
            a=config(OBJECTIVES[0],hidden,seed);b=config(OBJECTIVES[1],hidden,seed)
            assert a['algo_args']==b['algo_args']
            assert [k for k in a['env_args'] if a['env_args'][k]!=b['env_args'][k]]==['aoi_tail_threshold']
    files=list(HERE.glob('*.py'))+[HERE/n for n in ['run_pilot.sh','PROTOCOL.md','provenance.json','validation.json','calibration_results.json','rule_selection.json','preflight_results.json']]
    for folder in [HERE/'source',HERE/'configs']:files.extend(p for p in folder.rglob('*') if p.is_file())
    for p in files:
        if p.suffix=='.py':ast.parse(p.read_text())
    pre=read(HERE/'preflight_results.json')
    m=dict(schema=1,purpose='reward_threshold_three_seed_1m_controlled_pilot',created_utc=stamp(),
           objectives=OBJECTIVES,seeds=SEEDS,methods=METHODS,rules=RULE_METHODS,jobs=jobs,
           steps_per_method=1000000,total_training_steps=12000000,total_training_jobs=12,
           batch=4000,checkpoint_every_updates=25,runtime=runtime_signature(),
           calibration_seeds=CAL_SEEDS,evaluation_seeds=EVAL_SEEDS,scenarios=SCENARIOS,evaluation_episodes=4680,
           resources=dict(layout='cpu6_two_waves',parallel_jobs=6,cores=list(range(6)),evaluation_device='cpu',
                          seconds_per_update_estimate=pre['slowest_seconds_per_update'],
                          estimated_training_seconds=pre['estimated_training_seconds'],
                          prior_cpu_gpu_benchmark_sha256=digest(PARENT/'benchmark_results.json')),
           input_hashes={str(p.relative_to(HERE)):digest(p) for p in sorted(set(files))})
    write(HERE/'manifest.json',m);write(HERE/'status.json',dict(state='prepared',total_training_jobs=12,total_training_steps=12000000,updated_utc=stamp()))
    print('Frozen 12 jobs; 12M training steps; 4680 final evaluation episodes.',flush=True)


if __name__=='__main__':main()
