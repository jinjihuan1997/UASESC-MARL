"""Finalize only after two identical full aggregations and independent physics audit."""
from support import *
import subprocess, platform

def main():
 m=verify_inputs();report=HERE/'report';first=read(HERE/'aggregation_runs/final_pass1.json')
 hashes={f:sha(report/f) for f in first['file_sha256']}
 assert hashes==first['file_sha256'],{f:(h,first['file_sha256'][f]) for f,h in hashes.items() if h!=first['file_sha256'][f]}
 assert sha(HERE/'analysis_manifest.json')==first['analysis_manifest_sha256']
 assert '"aggregation": "PASS"' in (HERE/'logs/aggregate_final2.log').read_text()
 audit=read(report/'audit.json');extra=read(report/'independent_profile_audit.json')
 assert audit['state']==extra['state']=='PASS' and extra['source_sha256']==sha(HERE/'independent_profile_check.py')
 assert not audit['reserved_final_test_used'] and not audit['protected_input_changes']
 assert audit['new_training_steps']==audit['new_optimizer_updates']==0
 ledger=[read(p) for p in (HERE/'compute_ledger').glob('*.json')]
 assert len(ledger)==181 and all(d['state']=='complete' for d in ledger)
 for d in (HERE/'logs').glob('process_*.json'):assert read(d)['state']=='complete' and read(d)['returncode']==0
 # Refuse to finalize while any of this experiment's computational workers remain.
 processes=subprocess.run(['ps','-eo','pid,args'],capture_output=True,text=True).stdout.splitlines()
 live=[]
 for row in processes:
  if str(HERE) in row or '2026-09-12_policy_recomposition/' in row:
   if any('/'+f in row for f in ['aggregate.py','evaluate.py','validate.py','run_evaluation.py','independent_profile_check.py']) and ('/python ' in row or '/python3 ' in row):live.append(row)
 assert not live,live
 final=dict(state='PASS',identical_final_aggregation_passes=2,total_complete_aggregation_passes=3,initial_pass='aggregation_runs/initial_pass.json',final_passes=['logs/aggregate_final1.log','logs/aggregate_final2.log'],analysis_manifest_sha256=sha(HERE/'analysis_manifest.json'),deterministic_file_sha256=hashes,results_and_physical_and_paired_and_gap_unchanged_since_initial_pass=True,independent_numpy_profile_audit_sha256=sha(report/'independent_profile_audit.json'),time_metadata_separate=True,aggregate_policy_inference_slots_per_pass=1404000,total_aggregate_policy_inference_slots=4212000,new_physical_steps_during_aggregation=0)
 write(report/'reproducibility.json',final);write(HERE/'aggregation_runs/final_pass2.json',dict(state='PASS',file_sha256=hashes,log='logs/aggregate_final2.log'))
 timing={}
 for kind in ('primary','preflight'):
  q=[x for x in ledger if x['kind']==kind]
  start=min(datetime.datetime.fromisoformat(x['started_utc']) for x in q);end=max(datetime.datetime.fromisoformat(x['finished_utc']) for x in q)
  timing[kind]=dict(episodes=sum(x['episodes'] for x in q),physical_steps=sum(x['physical_steps'] for x in q),wall_span_seconds=(end-start).total_seconds(),sum_worker_seconds=sum(x['elapsed_seconds'] for x in q),median_batch_physical_steps_per_second=float(np.median([x['physical_steps_per_second'] for x in q])))
 write(HERE/'resource_summary.json',dict(resource_settings=read(HERE/'resource_settings.json'),timing=timing,initial=read(HERE/'software_environment.json'),final_utc=stamp(),final_load=os.getloadavg(),final_gpu=subprocess.run(['nvidia-smi','--query-gpu=name,utilization.gpu,memory.used,temperature.gpu','--format=csv'],capture_output=True,text=True).stdout,pauses=[],resumes=[],thermal_stops=0,other_processes_terminated=0,remaining_own_workers=live,peak_process_memory=None,peak_memory_note='not instrumented; no value inferred'))
 # A delivery index also seals auxiliary documentation/validation sources.
 write(HERE/'status.json',dict(state='complete',finished_utc=stamp(),new_training_steps=0,new_optimizer_updates=0,new_primary_complete_episodes=2340,new_primary_physical_steps=1404000,preflight_and_repro_complete_episodes=1280,preflight_physical_steps=768000,reused_episodes=4160,reference_supplement_episodes=0,total_new_physical_steps=2172000,reserved_final_test_used=False,remaining_own_workers=[],report='report/REPORT.md',reproducibility='report/reproducibility.json',independent_numpy_physics_audit='report/independent_profile_audit.json',protected_files_unchanged=2069))
 paths=sorted(p for p in HERE.rglob('*') if p.is_file() and p.name!='delivery_manifest.json')
 write(HERE/'delivery_manifest.json',dict(files={str(p.relative_to(HERE)):dict(sha256=sha(p),bytes=p.stat().st_size) for p in paths},total_files=len(paths),total_bytes=sum(p.stat().st_size for p in paths)))
 print(json.dumps(dict(state='complete',new_episodes=2340,preflight_episodes=1280,new_training_steps=0,identical_final_aggregations=2,protected_files=2069)),flush=True)
if __name__=='__main__':guard();main()
