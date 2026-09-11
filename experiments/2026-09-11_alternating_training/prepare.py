"""Prepare fixed seeds/configs and seal source after preflight, before training."""
import argparse
from helpers import *

def make_config(seed,arm):
    cfg=read(HERE.parent/'2026-09-10_sequential_long_training/configs/seed_85/joint.json')
    cfg['main_args']['exp_name']=arm
    cfg['algo_args']['seed']['seed']=seed
    cfg['algo_args']['train'].update(num_env_steps=1000000,use_linear_lr_decay=False,model_dir=None)
    cfg['algo_args']['model'].update(lr=1e-4,critic_lr=4e-4)
    cfg['env_args'].update(semantic_registry_path=str(HERE/'source/reference/inputs/mode_registry.json'),
                           semantic_profile_path=str(HERE/'source/reference/inputs/profile.npz'))
    phases=[dict(name=name,end_step=end,resource_source=resource,trainable_actor_ids=ids)
        for name,end,resource,ids in [
            ('varied_budget_mode_warmup',400000,'warmup_mixture',[1,2,3]),
            ('sut_only_400k_to_600k',600000,'actor',[0]),
            ('uav_only_600k_to_700k',700000,'actor',[1,2,3]),
            ('sut_only_700k_to_800k',800000,'actor',[0]),
            ('uav_only_800k_to_900k',900000,'actor',[1,2,3]),
            ('sut_only_900k_to_1000k',1000000,'actor',[0])]]
    cfg['training_design']=dict(arm=arm,phases=phases if arm=='alternating' else [],
        termination='finite_600_slot_task',observation_age_ref=8.0,
        evaluation_steps=[200000,400000,600000,800000,1000000],
        version='alternating_budget_coverage_v1')
    return cfg

def configs():
    assert not (HERE/'manifest.json').exists(),'Manifest already sealed'
    seeds=read(HERE/'seed_selection.json')
    for seed in seeds['training']:
        for arm in METHODS:write(HERE/'configs'/f'seed_{seed}'/f'{arm}.json',make_config(seed,arm))
    for arm in METHODS:
        cfg=make_config(seeds['preflight_seed'],arm)
        write(HERE/'preflight/configs'/f'{arm}.json',cfg)
    write(HERE/'preflight/manifest.json',dict(purpose='preflight_only_not_formal_training'))
    print('Configurations prepared; no training started')

def seal(layout):
    assert not (HERE/'manifest.json').exists(),'Do not overwrite a sealed run'
    pre=read(HERE/'preflight/validation.json')
    assert pre['state']=='PASS'
    for name,h in pre['validated_source_hashes'].items():assert digest(HERE/name)==h,name
    seeds=read(HERE/'seed_selection.json')
    scenarios=read(HERE.parent/'2026-09-11_selected_seed_comparison/comparison.json')['scenarios']
    jobs=[]
    for i,seed in enumerate(seeds['training']):
        for a,arm in enumerate(METHODS):
            core=2*i+a
            device='cuda:0' if layout=='cpu4_gpu2' and i==2 else 'cpu'
            jobs.append(dict(kind='training',id=f'seed_{seed}/{arm}',seed=seed,method=arm,
                device=device,core=core,config=f'configs/seed_{seed}/{arm}.json',output=f'jobs/seed_{seed}/{arm}'))
    evals=[]
    for j in jobs:
        for step in [200000,400000,600000,800000,1000000]:
            item=f"seed_{j['seed']}/{j['method']}_at_{step}"
            evals.append(dict(kind='evaluating',id=item,output='evaluation/'+item,
                model_dir=j['output']+f'/milestones/steps_{step}'))
    for method in RULE_METHODS:
        item='rules/'+method
        evals.append(dict(kind='evaluating',id=item,output='evaluation/'+item))
    manifest=dict(schema=1,purpose='paired_three_seed_joint_vs_alternating_budget_coverage',
        created_utc=stamp(),seeds=seeds['training'],methods=METHODS,jobs=jobs,evaluation_jobs=evals,
        steps_per_method=1000000,total_training_steps=6000000,batch=4000,
        milestones=[200000,400000,600000,800000,1000000],evaluation_seeds=seeds['validation'],
        reserved_final_test_seeds=seeds['reserved_final_test'],scenarios=scenarios,
        evaluation_split='new_validation_only_final_test_not_run',
        evaluation_episodes=len(evals)*len(scenarios)*len(seeds['validation']),
        selection_rule='Primary comparison is all three seeds at fixed 1000000 steps; earlier checkpoints are diagnostics. No seed replacement or test-based checkpoint selection.',
        warmup_evaluation='Alternating 200k and 400k use equal resources to evaluate mode learning; SUT is untrained. These are labelled diagnostics, not full-controller comparisons.',
        simple_baselines=RULE_METHODS,rule_calibration='Reuse original independent calibration mapping; no new fitting or candidate reward search',
        resources=dict(layout=layout,cores=list(range(8)),training_cores=list(range(6)),
            maximum_active_workers=8,maximum_training_workers=6,seconds_per_update_estimate=3.0,
            external_gpu_process_observed=1377302,benchmark='preflight/benchmark.json'),
        preflight_sha256=digest(HERE/'preflight/validation.json'))
    files=list((HERE/'source').rglob('*'))+list((HERE/'configs').rglob('*.json'))
    files += list(HERE.glob('*.py'))+list(HERE.glob('*.sh'))
    files += [HERE/'seed_selection.json',HERE/'rule_selection.json',HERE/'PROTOCOL.md',HERE/'preflight/validation.json',HERE/'preflight/benchmark.json']
    manifest['input_hashes']={str(p.relative_to(HERE)):digest(p) for p in sorted(set(files))
        if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'}
    write(HERE/'manifest.json',manifest)
    print(json.dumps(dict(state='SEALED',seeds=seeds['training'],jobs=len(jobs),layout=layout,
        manifest_sha256=digest(HERE/'manifest.json'))))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['configs','seal'])
    p.add_argument('--layout',choices=['cpu6','cpu4_gpu2'],default='cpu6');a=p.parse_args()
    configs() if a.command=='configs' else seal(a.layout)
