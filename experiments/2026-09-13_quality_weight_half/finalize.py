from weight_support import *
PRODUCTS=['results.json','paired_differences.json','gap_decomposition.json','physical_statistics.json','independent_audit.json','episode_scores.npz','REPORT.md','DETAILS.md']
def main():
 m=verify();assert read(HERE/'status.json')['state'] in ['evaluation_complete_pending_analysis','analysis_running','complete']
 seal=dict(sources={p:sha(HERE/p) for p in ['aggregate.py','write_report.py','finalize.py']},manifest_sha256=sha(HERE/'manifest.json'))
 if (HERE/'analysis_seal.json').exists():assert read(HERE/'analysis_seal.json')==seal
 else:write(HERE/'analysis_seal.json',seal)
 write(HERE/'status.json',dict(state='analysis_running',new_training_steps=0,new_optimizer_updates=0));digests=[]
 for repeat in [1,2]:
  for script in ['aggregate.py','write_report.py']:
   cmd=[sys.executable,str(HERE/script)];start=stamp();t=time.perf_counter()
   with (HERE/f'{script}_{repeat}.log').open('a') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
   write(HERE/f'execution/{script}_{repeat}.json',dict(command=cmd,start=start,seconds=time.perf_counter()-t,returncode=r.returncode));assert r.returncode==0,script
  digests.append({p:sha(HERE/'report'/p) for p in PRODUCTS})
 assert digests[0]==digests[1],{p:(digests[0][p],digests[1][p]) for p in PRODUCTS if digests[0][p]!=digests[1][p]}
 verify();before=read(HERE/'git_status_start.json');after={}
 for directory in before:
  after[directory]={}
  for key,args in [('head',['rev-parse','HEAD']),('status',['status','--porcelain=v1'])]:
   r=subprocess.run(['git','-C',directory,*args],capture_output=True,text=True,env={**os.environ,'GIT_OPTIONAL_LOCKS':'0'});after[directory][key]=dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)
 write(HERE/'git_status_end.json',after);assert before==after
 end={p:sha(ROOT/p) for p in m['protected_sha256']};assert end==m['protected_sha256'];write(HERE/'protected_hashes_end.json',end)
 audit=read(HERE/'report/independent_audit.json');costs=audit['costs'];assert costs['preflight']['physical_steps']==28800 and costs['preflight']['failed_records']==1
 total=sum(d['physical_steps'] for d in costs.values());assert total==3616800
 reproducibility=dict(state='PASS',two_complete_aggregations=True,all_numeric_and_report_files_identical=True,first_sha256=digests[0],second_sha256=digests[1],independently_reconstructed_slots_each_pass=3588000,local_greedy_decisions_replayed_each_pass=624000,aggregation_physical_advances=0,metadata_separate=True)
 write(HERE/'report/reproducibility.json',reproducibility)
 audit.update(state='COMPLETE_AUDITED',protected_hashes_unchanged=True,git_unchanged=True,total_new_physical_steps=total,renormalized=False,quality_multiplier=.5,actors_and_predictors_frozen=True,new_training_steps=0,new_optimizer_updates=0,new_fitting_updates=0,numeric_reproducibility='PASS',remaining=[],limitations=['RL frozen at old objective, no new learning','Fitted SUT greedy predictor remains calibrated on old objective','Development validation only, fixed models','Fixed average profile predictions'])
 write(HERE/'report/audit.json',audit);assert not FORBIDDEN
 write(HERE/'status.json',dict(state='complete',formal_episodes=5980,formal_physical_steps=3588000,preflight_physical_episodes=48,preflight_physical_steps=28800,total_new_physical_steps=total,new_training_steps=0,new_optimizer_updates=0,new_fitting_updates=0,reserved_final_test_used=False,protected_hashes='UNCHANGED',git_state='UNCHANGED',independent_audit='PASS',reproducibility='PASS',report='report/REPORT.md',report_sha256=sha(HERE/'report/REPORT.md'),remaining=[],stopped=True));print('COMPLETE',flush=True)
if __name__=='__main__':guard();main()
