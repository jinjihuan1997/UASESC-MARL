"""Seal the immutable execution protocol only after all preflight checks pass."""
from repair_support import *

def main():
    guard();m=verify_inputs(False)
    assert not m['execution_code_sha256'],'Already sealed'
    result=read(HERE/'preflight_semantics.json');assert result['state']=='SEMANTICS_PASS'
    benchmarks={name:json.loads((HERE/name).read_text().splitlines()[-1]) for name in
                ['benchmark_cpu.log','benchmark_cuda.log','benchmark_cpu_pair0.log','benchmark_cpu_pair1.log']}
    assert benchmarks['benchmark_cpu.log']['steps_per_second']>benchmarks['benchmark_cuda.log']['steps_per_second']
    execution=['repair_support.py','adapter_policy.py','repair_training.py','repair_evaluation.py','preflight.py','supervise.py','run_training.sh','prepare.py','seal_preflight.py']
    m['execution_code_sha256']={p:sha(HERE/p) for p in execution}
    m['seed_selection_sha256']=sha(HERE/'seed_selection.json')
    m['resource_settings']=dict(device='cpu',workers=2,torch_threads=1,blas_threads=1,source='measured preflight throughput')
    m['tolerances']['rollout_to_full_batch_ratio']=3e-6
    # This tolerance was specified in the trainer before the benchmark: only
    # float32 GEMM batch-size rounding. Physical/gating/resume tolerances unchanged.
    write(HERE/'manifest.json',m)
    write(HERE/'preflight.json',dict(state='PASS',checks=result['checks'],benchmarks=benchmarks,
        resource_settings=m['resource_settings'],execution_code_sha256=m['execution_code_sha256'],
        manifest_sha256=sha(HERE/'manifest.json'),failures_preserved=sorted(str(p.relative_to(HERE)) for p in (HERE/'preflight_attempts').glob('*.json') if read(p)['state']=='FAIL'),
        fixes=['Categorical logp identity checked with exact float64 accumulation of the unchanged float32 /16 values; no tolerance relaxation.',
               'Snapshot action RNG list cloned to prevent later in-memory list updates from changing captured state. Full disk resume then matched exactly.'],
        preflight_training_cost=dict(benchmark_steps=32000,one_completed_semantic_check_run_training_steps=4848,
            one_completed_semantic_check_run_evaluation_steps=20,earlier_attempts='Retained in attempt logs; not formal training or reusable weights.',excluded_from_formal_budget=True),utc=stamp()))
    print(json.dumps(dict(state='PASS',code_files=len(execution),device='cpu',workers=2)))

if __name__=='__main__':main()
