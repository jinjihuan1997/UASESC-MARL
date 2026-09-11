"""Read-only verification of migration evidence and checkpoint/source hashes."""
from pathlib import Path
import hashlib
import json
import torch
from common import ROOT, verify_reference


def read(path): return json.loads(Path(path).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_trial(path):
    status=read(path/'status.json');manifest=read(path/'manifest.json')
    assert status['state']=='complete'
    for name,expected in manifest['sources'].items(): assert sha(path/'source'/name)==expected
    for name,expected in manifest['reference']['input_hashes'].items(): assert sha(path/'source'/name)==expected
    for name,expected in status['checkpoint_hashes'].items():
        checkpoint=path/name;assert sha(checkpoint)==expected
        values=torch.load(checkpoint,map_location='cpu',weights_only=True)
        assert all(torch.isfinite(v).all() for v in values.values())
    assert all(a!=b for a,b in zip(manifest['initial_actor_hashes'],status['final_actor_hashes']))
    assert manifest['initial_critic_hash']!=status['final_critic_hash']
    config=read(path/'config.json')
    assert config['algo_args']['train']['model_dir'] is None
    assert config['algo_args']['train']['n_rollout_threads']==10
    assert config['algo_args']['train']['episode_length']==400
    assert status['completed_steps']==config['algo_args']['train']['num_env_steps']
    return status,manifest


def main():
    reference=verify_reference();results=ROOT/'results'
    for name in ('cpu_equivalence.json','gpu_equivalence.json','update_equivalence_cpu.json','update_equivalence_cuda_0.json'):
        assert read(results/name)['passed'],name
    gpu_equivalence=read(results/'gpu_equivalence.json')
    suite=read(results/'gpu_suite/status.json');assert suite['state']=='complete' and len(suite['methods'])==7
    initial_root=ROOT.parent/'2026-09-08_single_seed_comparison/runs/restart_benchmark_jobs4/run/jobs'
    for method in suite['methods']:
        status,manifest=verify_trial(results/'gpu_suite'/method)
        initial=read(initial_root/method/'status.json')
        assert manifest['initial_actor_hashes']==initial['initial_actor_hashes']
        assert manifest['initial_critic_hash']==initial['initial_critic_hash']
    gpu,_=verify_trial(results/'train_gpu_IC_HAPPO')
    cpu,_=verify_trial(results/'train_tensor_cpu_IC_HAPPO')
    old=read(results/'reference_hybrid/jobs/IC_HAPPO/status.json')
    old_manifest=read(results/'reference_hybrid/manifest.json')
    for name,expected in old_manifest['input_hashes'].items(): assert sha(results/'reference_hybrid'/name)==expected
    for name,expected in old['checkpoint_hashes'].items(): assert sha(name)==expected
    new_config=read(results/'train_gpu_IC_HAPPO/config.json')
    old_config=read(results/'reference_hybrid/configs/IC_HAPPO.json')
    for cfg in (new_config,old_config):
        for k in ('semantic_registry_path','semantic_profile_path'): cfg['env_args'].pop(k)
        cfg['algo_args']['logger'].pop('log_dir');cfg['algo_args']['train'].pop('hybrid_inference')
    assert new_config==old_config
    mixed=read(results/'mixed_cpu_gpu_80k/status.json');assert mixed['state']=='complete'
    for method in mixed['results']:
        status,manifest=verify_trial(results/'mixed_cpu_gpu_80k'/method)
        assert len((results/'mixed_cpu_gpu_80k'/method/'training_metrics.jsonl').read_text().splitlines())==20
        cfg=read(results/'mixed_cpu_gpu_80k'/method/'config.json')
        assert cfg['algo_args']['device']['cuda']==(status['device']!='cpu')
    summary=dict(passed=True,reference_files_verified=len(reference['input_hashes']),
        gpu_transition_steps=sum(r['steps_per_environment']*r['environments'] for r in gpu_equivalence['results']),
        gpu_discrete_checks=sum(r['discrete_checks'] for r in gpu_equivalence['results']),
        gpu_environment_max_absolute_float_error=max(r['max_absolute_float_error'] for r in gpu_equivalence['results']),
        cuda_update_parameter_max_absolute_error=max(r['max_parameter_absolute_difference'] for r in read(results/'update_equivalence_cuda_0.json')['results']),
        gpu_methods_verified=7,gpu_steps_per_method=16000,
        same_single_core_benchmark=dict(cpu_core=18,threads_per_library=1,
            reference_hybrid_steps_per_second=old['performance']['steady_steps_per_second'],
            tensor_gpu_steps_per_second=gpu['steady_steps_per_second'],
            tensor_cpu_steps_per_second=cpu['steady_steps_per_second'],
            tensor_gpu_speedup_vs_reference=gpu['steady_steps_per_second']/old['performance']['steady_steps_per_second']),
        mixed_probe=dict(steps_per_method=mixed['steps_per_method'],wall_seconds=mixed['wall_seconds'],
            temperature_peak_c=mixed['temperature_peak_c'],assignments=mixed['assignments'],
            per_method_steady_steps_per_second={k:v['steady_steps_per_second'] for k,v in mixed['results'].items()}),
        limitations=['Early engineering tests, not full 10M-step learning evidence',
            'GPU environment uses CPU setup and exogenous RNG preparation once per episode',
            'CPU and CUDA stochastic policy draws differ; full trained weights are not assumed identical',
            'Single-core single-method speedup is not a complete seven-method scheduling speedup',
            'Formal reference training remained active during sequential benchmark measurements'])
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__': main()
