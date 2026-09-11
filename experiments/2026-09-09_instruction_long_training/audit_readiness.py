"""Read-only verification of completed smoke runs and a prepared formal queue."""
from pathlib import Path
import argparse
import copy
import datetime
import json
import statistics
from common import ROOT, write, stamp, verify_reference
from multiseed_protocol import read, verify_run, verify_model
from training_checkpoint import digest


def audit(formal):
    verify_reference()
    m=verify_run(formal)
    if read(formal/'status.json')['state'] != 'prepared':
        raise ValueError('Readiness audit expects an unstarted formal queue')
    smoke=ROOT/'runs/smoke_40k_v2';sm=verify_run(smoke)
    if read(smoke/'status.json')['state']!='complete':
        raise RuntimeError('Final allocation smoke has not completed')
    verification=read(smoke/'report/verification.json')
    assert verification['training_jobs']==21 and verification['evaluation_items']==25
    assert verification['all_executed_constraints_passed']
    # All worker, supervisor, protocol and shell sources exercised in the final
    # smoke must match the formal snapshot. This audit script is not a worker.
    matches={}
    for p in (smoke/'source').iterdir():
        if p.suffix in ['.py','.sh'] or p.name=='PROTOCOL.md':
            assert digest(p)==digest(formal/'source'/p.name),p.name
            matches[p.name]=digest(p)
    secs={}
    for job in sm['jobs']:
        s=verify_model(smoke/job['output'],sm['steps_per_method'])
        rows=[json.loads(line) for line in (smoke/job['output']/'timing.jsonl').read_text().splitlines()]
        assert [r['update'] for r in rows]==list(range(1,sm['steps_per_method']//sm['batch']+1))
        secs.setdefault(job['method'],[]).append(s['compute_seconds']/s['completed_steps'])
    for seed in m['seeds']:
        for full,hidden in [('IC_HAPPO','HAPPO_hidden_instruction'),('IC_MAPPO','MAPPO_hidden_instruction')]:
            a=read(formal/f'configs/seed_{seed}/{full}.json')
            b=read(formal/f'configs/seed_{seed}/{hidden}.json')
            a=copy.deepcopy(a);a['main_args']['exp_name']=b['main_args']['exp_name']
            a['env_args']['actor_observe_instruction']=False
            assert a==b
            a=read(smoke/f'jobs/seed_{seed}/{full}/manifest.json')
            b=read(smoke/f'jobs/seed_{seed}/{hidden}/manifest.json')
            assert a['initial_actor_hashes']==b['initial_actor_hashes']
            assert a['initial_critic_hash']==b['initial_critic_hash']
    for rule in m['rules']:
        c=read(smoke/f'evaluation/rules/{rule}/config.json')
        assert not c.get('fixed_mode_rule',False) and not c.get('fixed_resources',False)
    first=ROOT/'runs/smoke_40k'
    pause_record=read(first/'pause_validation.json')
    assert pause_record['passed']
    resumed=[]
    for p in (first/'jobs').glob('*/*/resume_events.jsonl'):
        if p.read_text().strip():resumed.append(str(p.parent.relative_to(first)))
    expected={f"jobs/{r['job']}" for r in pause_record['checks'] if r['state']=='paused'}
    assert set(resumed)==expected
    assert len(pause_record['checks'])==6
    assert read(smoke/'duplicate_start_validation.json')['passed']
    pools={'cpu':[0.]*len(m['resources']['cpu_cores']), 'cuda:0':[0.]*len(m['resources']['gpu_host_cores'])}
    for job in m['jobs']:
        times=pools[job['device']];index=min(range(len(times)),key=times.__getitem__)
        times[index]+=m['steps_per_method']*statistics.mean(secs[job['method']])
    estimated_train=max(max(values) for values in pools.values())
    rows=[json.loads(line) for line in (smoke/'resources.jsonl').read_text().splitlines()]
    eval_times=[datetime.datetime.fromisoformat(r['utc']).timestamp() for r in rows if r['phase']=='evaluating']
    finish=datetime.datetime.fromisoformat(verification['utc']).timestamp()
    estimated_eval=(finish-min(eval_times))*m['total_evaluation_slots']/verification['evaluation_slots']
    result=dict(passed=True,utc=stamp(),formal_run=str(formal),formal_state='prepared_not_started',
        formal_manifest_sha256=digest(formal/'manifest.json'),model_inputs_unchanged=True,
        smoke_source_matches=matches,completed_smoke_training_jobs=21,completed_smoke_evaluation_items=25,
        actual_cpu_gpu_pause_resume_workers=resumed,duplicate_start_protected=True,
        checkpointed_workers=len(pause_record['checks']),
        completed_workers_skipped_on_resume=sum(r['state']=='complete' for r in pause_record['checks']),
        paired_methods_configs_and_initializations_checked=True,rules_use_complete_system=True,
        seconds_per_step_by_method={k:statistics.mean(v) for k,v in secs.items()},
        timing_estimate=dict(training_hours=estimated_train/3600,evaluation_hours=estimated_eval/3600,
            total_hours=(estimated_train+estimated_eval)/3600,
            caveat='Short-run throughput projection with fixed device assignment and greedy queue order, not a completion guarantee; excludes future thermal pauses and unrelated machine load.'),
        observed_temperatures=dict(cpu_max_c=max(r['cpu_max_c'] for r in rows),gpu_max_c=max(r['gpu_max_c'] for r in rows)))
    write(ROOT/'readiness.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['smoke_source_matches','seconds_per_step_by_method']},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--formal',type=Path,required=True)
    audit(p.parse_args().formal.resolve())
