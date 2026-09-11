"""Six-job timing smoke, exact checkpoint resume and real evaluator smoke."""
import subprocess
import time
from helpers import *
from three_seed_train import gpu_temperature
from run_suite import temperature
import evaluation
from aggregate import audit_trace


def command(cfg,folder,manifest,*extra):
    return [sys.executable,'-u',str(HERE/'source/formal_train.py'),'--config',str(cfg),'--output',str(folder),
            '--run-manifest',str(manifest),'--device','cpu','--checkpoint-every','2',*extra]


def main():
    work=HERE/'preflight';work.mkdir(exist_ok=False)
    manifest=work/'manifest.json';write(manifest,dict(purpose='timing_and_correctness_only',steps_per_method=40000,
        scenarios={'fixed_1':[[0,1]]},evaluation_seeds=[20262151,20262152]))
    processes=[];logs=[]
    for core,(seed,method) in enumerate((s,m) for s in SEEDS for m in METHODS):
        c=config('candidate_tail4',method=='HAPPO_hidden_instruction',seed);c['algo_args']['train']['num_env_steps']=40000
        item=f'candidate_tail4/seed_{seed}/{method}';cfg=work/'configs'/f'{item}.json';write(cfg,c)
        log=(work/f'core_{core}.log').open('w');logs.append(log)
        processes.append((item,subprocess.Popen(['taskset','-c',str(core),*command(cfg,work/'jobs'/item,manifest)],stdout=log,stderr=subprocess.STDOUT)))
    peak=0.
    try:
        while any(p.poll() is None for _,p in processes):
            cpu,gpu=temperature(),gpu_temperature()
            if cpu is None or cpu>=85 or gpu>=83:raise RuntimeError('Preflight temperature limit or sensor failure')
            peak=max(peak,cpu)
            for item,p in processes:
                if p.poll() is not None and p.returncode:raise RuntimeError(f'Preflight failed: {item}')
            time.sleep(.5)
    finally:
        for _,p in processes:
            if p.poll() is None:p.terminate()
        for _,p in processes:p.wait(timeout=60)
        for log in logs:log.close()
    timings={}
    for item,p in processes:
        assert p.returncode==0
        rows=[json.loads(r) for r in (work/'jobs'/item/'timing.jsonl').read_text().splitlines()][1:]
        timings[item]=float(np.median([r['total_seconds'] for r in rows]))
    # Original versus candidate starts from identical actor/critic weights.
    from tensor_train import TensorTrainer
    a=TensorTrainer(config('original_tail10'),'cpu');b=TensorTrainer(config('candidate_tail4'),'cpu')
    assert a.initial_actors==b.initial_actors and a.initial_critic==b.initial_critic
    assert torch.equal(a.buffer.obs,b.buffer.obs) and torch.equal(a.env.noise_us,b.env.noise_us)
    del a,b
    # Explicit uninterrupted/resume comparison. All arrays and RNG restored.
    cfg=work/'resume_config.json';c=config('candidate_tail4');c['algo_args']['train']['num_env_steps']=16000;write(cfg,c)
    with (work/'resume_test.log').open('w') as log:
        subprocess.run(command(cfg,work/'whole',manifest),stdout=log,stderr=subprocess.STDOUT,check=True)
        partial=subprocess.run(command(cfg,work/'resumed',manifest,'--stop-after-update','2'),stdout=log,stderr=subprocess.STDOUT)
        assert partial.returncode==75
        subprocess.run(command(cfg,work/'resumed',manifest,'--resume'),stdout=log,stderr=subprocess.STDOUT,check=True)
    for f in (work/'whole').glob('*.pt'):
        x=torch.load(f,map_location='cpu',weights_only=True);y=torch.load(work/'resumed'/f.name,map_location='cpu',weights_only=True)
        assert x.keys()==y.keys() and all(torch.equal(x[k],y[k]) for k in x)
    write(work/'rule_selection.json',read(HERE/'rule_selection.json'))
    evaluation.HERE=work
    for item in ['candidate_tail4/seed_85/IC_HAPPO','candidate_tail4/rules/R_instruction','candidate_tail4/rules/R_myopic']:
        assert evaluation.evaluate_item(item,[])==0
        assert evaluation.evaluate_item(item,[])==0
        with np.load(work/'evaluation'/item/'fixed_1.npz') as z:
            audit_trace(z['trace'],z['aoi_after'],config('candidate_tail4'))
    write(HERE/'preflight_results.json',dict(state='PASS',six_jobs_completed=True,steps_per_smoke=40000,
        median_update_seconds=timings,slowest_seconds_per_update=max(timings.values()),
        estimated_training_seconds=500*max(timings.values()),cpu_peak_c=peak,
        source_only_initialization_paired=True,exact_checkpoint_resume=True,evaluator_episodes=6,
        evaluator_resume_idempotent=True,both_scoring_standards_reaggregated=True))
    print('Preflight PASS',flush=True)


if __name__=='__main__':main()
