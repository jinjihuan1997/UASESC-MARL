"""Two full deterministic aggregations, protected-input recheck and stop status."""
from study import *
import subprocess

def reference_index():
 m=manifest();refs={};old=read(RECOMPOSE/'reference_index.json')
 for key,value in old.items():
  if not any(x in key for x in ['greedy_modes_3','greedy_modes_16','R_instruction','R_equal_instruction']):continue
  p=ROOT/value['path'];assert sha(p)==value['trace_sha256'];meta=value['metadata']
  with np.load(p,allow_pickle=False) as z:assert z['seeds'].tolist()==m['validation']
  refs[key]=dict(path=value['path'],trace_sha256=sha(p),environment_split='original20_development_validation_NOT_gate_dev',used_for_gate_score=False,metadata_sha256=sha(p.with_suffix('.json')))
 for parent in read(RECOMPOSE/'manifest.json')['parents']:
  for method in ['C1','C3']:
   for scene in m['scenarios']:
    p=RECOMPOSE/f'evaluation/seed_{parent}/{method}/{scene}.npz';meta=read(p.with_suffix('.json'));assert sha(p)==meta['trace_sha256']
    with np.load(p,allow_pickle=False) as z:assert z['seeds'].tolist()==m['validation']
    refs[f'{parent}/{method}/{scene}']=dict(path=str(p.relative_to(ROOT)),trace_sha256=sha(p),environment_split='original20_development_validation_NOT_gate_dev',used_for_gate_score=False,metadata_sha256=sha(p.with_suffix('.json')))
 write(HERE/'reference_index.json',dict(state='HASH_VERIFIED_REFERENCE_ONLY',entries=refs,reused_for_new_gate_score_episodes=0,why_not_paired='Existing C1/C3 use the three old parent lineages and original20 development seeds. New students/gate_dev are not those lineages or environments. No cross-split subtraction is performed.',prior_independent_physical_audit_sha256=sha(RECOMPOSE/'report/independent_profile_audit.json')))

def finalize():
 m=verify();g=read(HERE/'gate_report.json');assert g['state']=='STOP_AFTER_A';guard()
 # Syntax and actual stop guards without creating actors or running gradients.
 for p in HERE.glob('*.py'):compile(p.read_text(),str(p),'exec')
 from warmup_critic import warmup
 from train_joint import train
 checks={}
 for name,fn in [('critic_warmup',lambda:warmup(m['students'][0])),('joint_HAPPO',lambda:train(m['students'][0],'B0','cpu'))]:
  try:fn()
  except AssertionError:checks[name]='BLOCKED_BY_FAILED_A_GATE'
  else:raise AssertionError(f'{name} bypassed gate')
 assert not (HERE/'warmup').exists() and not (HERE/'B').exists()
 write(HERE/'conditional_stop_check.json',dict(state='PASS',checks=checks,gradient_steps=0,environment_steps=0))
 reference_index();files=['results.json','physical_statistics.json','diagnostics_by_scene.json','gap_decomposition.json','post_quality_comparison.json','audit.json','REPORT.md'];passes=[]
 for n in [1,2]:
  with (HERE/f'logs/final_aggregate_{n}.log').open('a') as log:subprocess.run([sys.executable,str(HERE/'aggregate.py')],check=True,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
  passes.append({f:sha(HERE/'report'/f) for f in files})
 assert passes[0]==passes[1]
 before=read(HERE/'git_status.json');after={}
 for p,data in before.items():
  after[p]={}
  for name,args in [('head',['rev-parse','HEAD']),('status',['status','--porcelain=v1'])]:
   r=subprocess.run(['git','-C',p,*args],capture_output=True,text=True,env={**os.environ,'GIT_OPTIONAL_LOCKS':'0'});after[p][name]=dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)
   assert r.returncode==data[name]['returncode'] and r.stdout==data[name]['stdout'],f'Git state changed: {p}/{name}'
 write(HERE/'git_status_end.json',after);verify()
 write(HERE/'report/reproducibility.json',dict(state='PASS',deterministic_aggregation_passes=2,outputs_equal=True,passes=passes,aggregate_source_sha256=sha(HERE/'aggregate.py'),protected_files_unchanged=len(m['protected_sha256']),git_HEAD_and_status_unchanged=True,conditional_stop_guards_sha256=sha(HERE/'conditional_stop_check.json'),metadata_timestamps_excluded_from_numeric_outputs=True,source_sha256={p.name:sha(p) for p in sorted(HERE.glob('*.py'))},reserved_final_test_used=False))
 r=read(HERE/'report/results.json');write(HERE/'status.json',dict(state='COMPLETED_STOPPED_AFTER_A',conditional_requested_run_completed=True,stage_A='complete',gate='0/3',stage_B='NOT_RUN_STAGE_A_GATE_FAILED',critic_warmup='NOT_RUN_STAGE_A_GATE_FAILED',RL_environment_steps=0,formal_collection_steps=504000,formal_supervised_optimizer_updates=76320,gate_evaluation_episodes=1690,gate_evaluation_steps=1014000,total_physical_steps=r['costs']['total_physical_steps'],protected_files_unchanged=True,reserved_final_test_used=False,report_sha256=sha(HERE/'report/REPORT.md'),reproducibility_sha256=sha(HERE/'report/reproducibility.json')))
 print('COMPLETE: Stage A executed; 0/3 gate; Stage B not run; two aggregations exact.')

if __name__=='__main__':finalize()
