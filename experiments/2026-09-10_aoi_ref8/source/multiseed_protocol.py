"""Frozen three-seed experiment preparation and integrity checks."""
import json
from pathlib import Path
import secrets
import shutil

from common import ROOT, configuration, stamp, verify_reference, write
from protocol import METHODS, RULES, SCENARIOS
from training_checkpoint import digest, runtime_signature


CPU_CORES = [17, 19, 11, 13, 9, 10]
GPU_CORES = [16, 18]
CPU_METHODS = ['IC_HAPPO', 'HAPPO_hidden_instruction', 'IC_MAPPO', 'MAPPO_hidden_instruction', 'HAPPO_no_task_aux_reward']
GPU_METHODS = ['HAPPO_fixed_mode_rule', 'HAPPO_equal_resources']
PROJECT = Path('/home/king/Downloads/Projects/2_th_paper_TMC')
REFERENCE_STATUS = PROJECT/'experiments/2026-09-08_single_seed_comparison/runs/seed1_clean_restart_20260908/status.json'


def read(path):
    return json.loads(Path(path).read_text())


def prepare(output, seeds=None, steps=None, smoke=False):
    output = Path(output).resolve()
    randomly_selected = seeds is None
    seeds = sorted(secrets.SystemRandom().sample(range(1, 1000), 3)) if seeds is None else list(seeds)
    if len(seeds) != 3 or len(set(seeds)) != 3 or any(not 0 <= s < 2**31 for s in seeds):
        raise ValueError('Exactly three distinct nonnegative seeds below 2**31 are required')
    external_seeds = [s+1000*i for s in seeds for i in range(10)]
    if len(set(external_seeds)) != 30:
        raise ValueError('Selected seeds cause overlap in the 10-environment seed streams')
    steps = (8000 if smoke else 10_000_000) if steps is None else steps
    if smoke and not 8000 <= steps <= 80000:
        raise ValueError('Smoke budget must be between 8000 and 80000 steps per job')
    if not smoke and steps != 10_000_000:
        raise ValueError('The formal protocol fixes 10,000,000 steps per method and seed')
    configuration(seed=seeds[0], steps=steps)
    verify_reference()
    evidence = {}
    for path in sorted((ROOT/'evidence').glob('*.json')):
        if not read(path).get('passed'):
            raise ValueError(f'Inherited validation did not pass: {path}')
        evidence[path.name] = digest(path)
    if len(evidence) != 7:
        raise ValueError('Missing inherited environment/model/resume validation records')
    output.mkdir(parents=True, exist_ok=False)
    source = output/'source'
    source.mkdir()
    for file in list(ROOT.glob('*.py'))+list(ROOT.glob('*.sh')):
        shutil.copy2(file, source/file.name)
    shutil.copy2(ROOT/'reference_manifest.json', source/'reference_manifest.json')
    for name in ('PROTOCOL.md', 'parent_provenance.json'):
        shutil.copy2(ROOT/name, source/name)
    for name in ('reference', 'tests', 'evidence'):
        shutil.copytree(ROOT/name, source/name, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    scenarios = {name: SCENARIOS[name] for name in ('fixed_0', 'multi_0_1_2')} if smoke else SCENARIOS
    evaluation_seeds = [20260931] if smoke else list(range(20261301, 20261321))
    jobs = []
    for seed in seeds:
        for method in GPU_METHODS+CPU_METHODS:
            device = 'cuda:0' if method in GPU_METHODS else 'cpu'
            job_id = f'seed_{seed}/{method}'
            config = configuration(method, seed, steps)
            config['algo_args']['device'].update(cuda=device != 'cpu', cuda_deterministic=device != 'cpu')
            config['env_args'].update(semantic_registry_path=str(source/'reference/inputs/mode_registry.json'),
                                     semantic_profile_path=str(source/'reference/inputs/profile.npz'))
            path = output/'configs'/f'{job_id}.json'
            write(path, config)
            jobs.append(dict(id=job_id, method=method, seed=seed, device=device,
                             config=str(path.relative_to(output)), output=f'jobs/{job_id}'))
    inputs = {str(p.relative_to(output)): digest(p) for parent in (source, output/'configs')
              for p in sorted(parent.rglob('*')) if p.is_file()}
    manifest = dict(schema=1, created_utc=stamp(), purpose='smoke_only_not_paper_results' if smoke else 'instruction_priority_formal_three_seed_sc',
        seeds=seeds, seed_selection='three seeds sampled once and frozen' if randomly_selected else 'provided explicitly',
        runtime=runtime_signature(),
        steps_per_method=steps, total_training_jobs=len(jobs), total_training_steps=len(jobs)*steps,
        logical_environments=10, rollout_length=400, batch=4000, checkpoint_every_updates=25,
        jobs=jobs, method_order=[m[0] for m in METHODS], rules=RULES,
        resources=dict(cpu_cores=CPU_CORES, gpu_host_cores=GPU_CORES, gpu_device='cuda:0',
                       cpu_parallel_jobs=6, gpu_parallel_jobs=2, evaluation_cpu_cores=[16, 17, 18, 19], library_threads=1),
        thermal_policy=dict(cpu_hot_c=85, cpu_hot_seconds=5, cpu_emergency_c=95,
            gpu_hot_c=83, cool_below_c=75, cool_seconds=60, minimum_pause_seconds=180),
        inherited_validation=evidence,
        scenarios=scenarios, evaluation_seeds=evaluation_seeds,
        total_evaluation_jobs=len(jobs)+len(RULES),
        total_evaluation_slots=(len(jobs)+len(RULES))*len(scenarios)*len(evaluation_seeds)*600,
        reference_status=str(REFERENCE_STATUS),
        resource_lock=str(ROOT/'.three_seed_resource.lock'), input_hashes=inputs,
        scientific_scope='instruction_priority_v1 with original actor feasibility masks; final average SC table; CC excluded; fresh initialization; final-budget checkpoints',
        evaluation='frozen CPU reference; rules once; mean and sample SD across three training seeds')
    write(output/'manifest.json', manifest)
    write(output/'status.json', dict(state='prepared', updated_utc=stamp(), seeds=seeds,
        total_training_jobs=len(jobs), completed_training_jobs=0, total_training_steps=len(jobs)*steps,
        completed_training_steps=0, output=str(output)))
    return manifest


def verify_run(output):
    output = Path(output).resolve()
    manifest = read(output/'manifest.json')
    changed = [name for name, sha in manifest['input_hashes'].items()
               if not (output/name).is_file() or digest(output/name) != sha]
    if changed:
        raise RuntimeError(f'Frozen inputs changed: {changed[:10]}')
    if manifest['schema'] != 1 or len(manifest['seeds']) != 3 or len(manifest['jobs']) != 21:
        raise ValueError('Not a complete three-seed, seven-method experiment')
    if manifest['runtime'] != runtime_signature():
        raise ValueError('Software runtime changed from the prepared experiment')
    return manifest


def verify_model(folder, expected_steps):
    folder = Path(folder)
    status = read(folder/'status.json')
    if status['state'] != 'complete' or status['completed_steps'] != expected_steps:
        raise ValueError(f'Incomplete model: {folder}')
    for name, sha in status['checkpoint_hashes'].items():
        if digest(folder/name) != sha:
            raise RuntimeError(f'Model file changed: {folder/name}')
    return status
