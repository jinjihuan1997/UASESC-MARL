"""Freeze six CC jobs paired to the current SC study; preserve every input."""
import json
from pathlib import Path
import shutil
from common import ROOT, configuration, stamp, verify_reference, write
from training_checkpoint import digest, runtime_signature

METHODS=['CC_IC_HAPPO','CC_IC_MAPPO']
CORES=[0,4,6]


def read(path): return json.loads(Path(path).read_text())


def prepare(output,seeds=None,steps=None,smoke=False):
    parent=read(ROOT/'parent_provenance.json')
    sc=read(Path(parent['parent_run'])/'manifest.json')
    if digest(Path(parent['parent_run'])/'manifest.json')!=parent['parent_manifest_sha256']:
        raise RuntimeError('Parent SC protocol changed')
    seeds=sc['seeds'] if seeds is None else seeds
    if seeds!=sc['seeds']: raise ValueError('CC must use the same three training seeds as SC')
    steps=(8000 if smoke else 10_000_000) if steps is None else steps
    if (smoke and not 8000<=steps<=80000) or (not smoke and steps!=sc['steps_per_method']):
        raise ValueError('Unapproved CC training budget')
    verify_reference()
    if not smoke:
        if not read(ROOT/'evidence/verification.json')['passed']:
            raise ValueError('CC integration validation required before formal preparation')
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    source=output/'source';source.mkdir()
    for file in list(ROOT.glob('*.py'))+list(ROOT.glob('*.sh')):
        shutil.copy2(file,source/file.name)
    for name in ('reference_manifest.json','parent_provenance.json','PROTOCOL.md'):
        shutil.copy2(ROOT/name,source/name)
    for name in ('reference','inputs','tests'):
        shutil.copytree(ROOT/name,source/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    if (ROOT/'evidence').exists(): shutil.copytree(ROOT/'evidence',source/'evidence')
    jobs=[]
    for seed in seeds:
        for method in METHODS:
            config=configuration(method,seed,steps)
            config['env_args'].update(semantic_registry_path=str(source/'inputs/mode_registry.json'),
                semantic_profile_path=str(source/'inputs/profile.npz'),cc_delivery_moments_path=str(source/'inputs/delivery_moments.npz'))
            key=f'seed_{seed}/{method}';path=output/'configs'/f'{key}.json';write(path,config)
            jobs.append(dict(id=key,method=method,seed=seed,device='cpu',config=str(path.relative_to(output)),output=f'jobs/{key}'))
    scenarios={k:sc['scenarios'][k] for k in ('fixed_0','multi_0_1_2')} if smoke else sc['scenarios']
    evaluation_seeds=[20260931] if smoke else sc['evaluation_seeds']
    input_hashes={str(p.relative_to(output)):digest(p) for base in (source,output/'configs') for p in sorted(base.rglob('*')) if p.is_file()}
    manifest=dict(schema=1,created_utc=stamp(),purpose='smoke_cc_not_paper_results' if smoke else 'instruction_priority_cc_three_seed_v1',
        runtime=runtime_signature(),seeds=seeds,seed_selection='Paired to frozen SC seeds; no reselection',steps_per_method=steps,
        total_training_jobs=6,total_training_steps=6*steps,logical_environments=10,rollout_length=400,batch=4000,
        checkpoint_every_updates=25,jobs=jobs,method_order=METHODS,rules=[],scenarios=scenarios,evaluation_seeds=evaluation_seeds,
        total_evaluation_jobs=6,total_evaluation_slots=6*len(scenarios)*len(evaluation_seeds)*600,
        resources=dict(cpu_cores=CORES,gpu_host_cores=[],gpu_device=None,cpu_parallel_jobs=len(CORES),gpu_parallel_jobs=0,
                       evaluation_cpu_cores=CORES,library_threads=1),
        thermal_policy=sc['thermal_policy'],reference_status=str(Path(parent['parent_run'])/'status.json'),
        resource_lock=str(ROOT/'.cc_resource.lock'),input_hashes=input_hashes,parent=parent,
        environment_protocol='CC average load, joint packet/quality moments, single attempt, no ARQ; cache released after attempt; AoI reset only on success',
        scientific_scope='Fixed SC average model vs measured/analytic CC average model; equal instruction/reward/budget protocol; not an end-to-end codec fairness claim',
        evaluation='Deterministic CPU policy, identical external SC trajectories and independent CC delivery stream; final-budget models')
    write(output/'manifest.json',manifest)
    write(output/'status.json',dict(state='prepared',updated_utc=stamp(),seeds=seeds,completed_training_steps=0,total_training_steps=6*steps))
    return manifest


def verify_run(output):
    output=Path(output);manifest=read(output/'manifest.json')
    changed=[k for k,v in manifest['input_hashes'].items() if not (output/k).is_file() or digest(output/k)!=v]
    if changed: raise RuntimeError(f'Frozen CC inputs changed: {changed[:10]}')
    if manifest['schema']!=1 or len(manifest['seeds'])!=3 or len(manifest['jobs'])!=6:
        raise ValueError('Expected two CC methods by three seeds')
    if runtime_signature()!=manifest['runtime']: raise ValueError('Software runtime changed')
    return manifest


def verify_model(folder,expected_steps):
    folder=Path(folder);status=read(folder/'status.json')
    if status['state']!='complete' or status['completed_steps']!=expected_steps: raise ValueError(f'Incomplete model: {folder}')
    for name,expected in status['checkpoint_hashes'].items():
        if digest(folder/name)!=expected: raise RuntimeError('CC model changed')
    return status
