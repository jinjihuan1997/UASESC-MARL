from train_support import *
SHORT=HERE.parent/'2026-09-14_quality_half_retraining'
def main():
 guard();assert not (HERE/'manifest.json').exists();old=read(SHORT/'manifest.json');assert read(SHORT/'status.json')['state']=='complete' and read(SHORT/'report/reproducibility.json')['state']=='PASS'
 for p,h in old['protected_sha256'].items():assert sha(ROOT/p)==h
 for p,h in read(SHORT/'report/reproducibility.json')['deterministic_output_sha256'].items():assert sha(SHORT/'report'/p)==h
 parent={}
 for seed in old['seeds']:
  folder=SHORT/f'jobs/seed_{seed}';assert read(folder/'status.json')['state']=='complete'
  state,entry=checkpoint.load_checkpoint(folder/'checkpoints');assert state['update']==250 and state['device']=='cpu' and state['total_updates']==250
  c=read(SHORT/f'configs/seed_{seed}.json');c['algo_args']['train']['num_env_steps']=10000000;c['main_args']['exp_name']='joint_quality_half_long_continuation';c['training_design']['evaluation_steps']=list(range(1000000,10000001,1000000));c['audit_directory']=str(HERE/f'jobs/seed_{seed}/trajectory_audits')
  write(HERE/f'configs/seed_{seed}.json',c);parent[str(seed)]=dict(path=str((folder/'checkpoints'/entry['file']).relative_to(ROOT)),sha256=entry['sha256'],identity=state['identity'],update=250,steps=1000000)
 paths={ROOT/p for p in old['protected_sha256']};paths.update(p for p in SHORT.rglob('*') if p.is_file());m={k:copy.deepcopy(v) for k,v in old.items() if k not in ['protected_sha256','config_sha256']}
 m.update(purpose='quality_half_joint_happo_extend_1m_to_10m',created_utc=stamp(),protocol_sha256=sha(HERE/'PROTOCOL.md'),from_scratch=False,old_models_not_loaded=False,continuation_source=str(SHORT.relative_to(ROOT)),parent_checkpoints=parent,start_steps_per_seed=1000000,steps_per_seed=10000000,additional_steps_per_seed=9000000,total_training_steps=27000000,cumulative_training_steps=30000000,updates=2500,start_update=250,milestones=list(range(1000000,10000001,1000000)),evaluation_milestones=[2000000,4000000,6000000,8000000,10000000],bootstrap_seed=202609141300,config_sha256={str(p.relative_to(HERE)):sha(p) for p in sorted((HERE/'configs').glob('*.json'))},protected_sha256={str(p.relative_to(ROOT)):sha(p) for p in sorted(paths)})
 write(HERE/'manifest.json',m);write(HERE/'status.json',dict(state='prepared',additional_training_steps=27000000,cumulative_target_steps=30000000));before=read(SHORT/'git_status_end.json');current={}
 for directory in before:
  current[directory]={}
  for key,args in [('head',['rev-parse','HEAD']),('status',['status','--porcelain=v1'])]:
   r=subprocess.run(['git','-C',directory,*args],capture_output=True,text=True,env={**os.environ,'GIT_OPTIONAL_LOCKS':'0'});current[directory][key]=dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)
 write(HERE/'git_status_start.json',current);print('prepared',len(paths),'protected files',flush=True)
if __name__=='__main__':main()
