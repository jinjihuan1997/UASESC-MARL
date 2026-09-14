"""Repeat aggregation, compare deterministic products, recheck frozen inputs, stop."""
from light_support import *
import subprocess

PRODUCTS=['results.json','paired_differences.json','physical_statistics.json','independent_audit.json','episode_scores.npz','complexity.json','REPORT.md','DETAILS.md']

def digest():return {p:sha(HERE/'report'/p) for p in PRODUCTS}
def command(script,logname):
 cmd=[sys.executable,str(HERE/script)];start=stamp();t=time.perf_counter()
 with (HERE/logname).open('a') as f:rc=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT).returncode
 d=dict(command=cmd,start_utc=start,elapsed_seconds=time.perf_counter()-t,returncode=rc,log=logname)
 append(HERE/'finalization_execution.jsonl',d)
 if rc:raise RuntimeError(f'{script} returned {rc}, see {logname}')

def main():
 m=verify();assert read(HERE/'report/complexity.json')['state']=='MEASURED';assert read(HERE/'preflight.json')['state']=='PASS'
 sealed=['aggregate.py','write_report.py','finalize.py'];seal=dict(source_sha256={p:sha(HERE/p) for p in sealed},manifest_sha256=sha(HERE/'manifest.json'))
 if (HERE/'analysis_execution_seal.json').exists():assert read(HERE/'analysis_execution_seal.json')==seal
 else:write(HERE/'analysis_execution_seal.json',seal)
 # One successful complete numeric pass has already executed in this run.
 # On a new invocation lacking those outputs, perform its first pass explicitly.
 if not all((HERE/'report'/p).exists() for p in PRODUCTS[:6]):command('aggregate.py','aggregate_first_complete.log')
 command('write_report.py','report_first.log');first=digest();write(HERE/'aggregation_first_digest.json',first)
 write(HERE/'status.json',dict(state='independent_reaggregation_running',new_training_steps=0,new_optimizer_updates=0))
 command('aggregate.py','aggregate_second_complete.log');command('write_report.py','report_second.log');second=digest()
 assert first==second,{p:(first[p],second[p]) for p in first if first[p]!=second[p]}
 audit=read(HERE/'report/independent_audit.json');assert audit['state']=='PASS' and audit['audited_complete_trace_slots']==2964000
 verify();before=read(HERE/'git_status.json');after={}
 for directory in before:
  after[directory]={}
  for key,args in [('head',['rev-parse','HEAD']),('status',['status','--porcelain=v1'])]:
   r=subprocess.run(['git','-C',directory,*args],capture_output=True,text=True,env={**os.environ,'GIT_OPTIONAL_LOCKS':'0'});after[directory][key]=dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)
 write(HERE/'git_status_end.json',after);assert before==after,'Existing Git state changed; inspect recorded before/end files'
 now={p:sha(ROOT/p) for p in m['protected_sha256']};assert now==m['protected_sha256'];write(HERE/'protected_hashes_end.json',now)
 assert not FORBIDDEN
 costs=audit['costs'];physical=sum(d['physical_steps'] for d in costs.values());assert physical==946802
 assert costs['latency']['sample_decisions']==131670 and costs['reference_preflight_replay']['replayed_sample_decisions']==312000
 reproducibility=dict(state='PASS',two_complete_numeric_aggregations=True,all_products_byte_identical=True,first_sha256=first,second_sha256=second,analysis_execution_seal_sha256=sha(HERE/'analysis_execution_seal.json'),execution_metadata_separate='finalization_execution.jsonl',old_inputs_unchanged=True,git_state_unchanged=True,independent_reconstructed_trace_slots_each_pass=2964000,independent_reconstructed_trace_slots_two_passes=5928000,new_policy_requested_decisions_replayed_each_pass=624000,new_policy_requested_decisions_replayed_two_passes=1248000,additional_physical_environment_steps_from_aggregation=0)
 write(HERE/'report/reproducibility.json',reproducibility)
 audit.update(state='COMPLETE_AUDITED',git_state_unchanged=True,protected_files_sha256_before_equals_after=True,protected_files=len(now),new_training_steps=0,new_optimizer_updates=0,new_fitting_updates=0,reserved_final_test_used=False,total_new_physical_steps=physical,
  final_numerical_hashes=second,preflight_failure_count=1,formal_evaluation_failure_count=0,latency_failure_count=0,aggregation_syntax_error_attempts=1,report_render_error_attempts=1,
  incomplete_items=[],strong_baselines_missing=[],new_baselines=4,reference_methods=9,unique_formal_and_supplement_episodes=1560,
  checked_call_semantics='Runtime gradient/optimizer prohibition, frozen policy state hashes, local input AST checks, source seals, every-slot original checker and independent NumPy reconstruction',
  no_new_foreign_students=True,no_new_test_set=True,limitations=['Development validation only; conditional on fixed parents','Empirical CPU timing, no hard real-time guarantee','Fixed average profile predictions, no decoded-video claim','No communication compression implemented','Historical model/predictor/profile costs retained'])
 write(HERE/'report/audit.json',audit)
 write(HERE/'status.json',dict(state='complete',all_requested_methods_evaluated=True,primary_complete_episodes=1040,primary_physical_steps=624000,supplement_complete_episodes=520,supplement_physical_steps=312000,reused_complete_episodes=3380,reused_new_physical_steps=0,preflight_complete_episodes=16,preflight_physical_steps=10802,total_new_physical_steps=physical,new_training_steps=0,new_optimizer_updates=0,new_fitting_updates=0,reserved_final_test_used=False,independent_audit='PASS',reproducibility='PASS',protected_input_hashes='UNCHANGED',git_state='UNCHANGED',report='report/REPORT.md',report_sha256=sha(HERE/'report/REPORT.md'),missing=[],stopped_at_requested_scope=True))
 print('COMPLETE: fixed evaluation, timing, independent audit, duplicate aggregation, no training.',flush=True)

if __name__=='__main__':guard();main()
