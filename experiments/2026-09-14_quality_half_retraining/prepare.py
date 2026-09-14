from train_support import *
def main():
 assert not (HERE/'manifest.json').exists();p=ws.verify();assert read(PREVIOUS/'status.json')['state']=='complete';seeds=p['parents'];derived={str(s):[s+1000*i for i in range(10)] for s in seeds};flat=sum(derived.values(),[]);assert len(set(flat))==30 and not set(flat)&set(p['validation']+p['reserved'])
 for seed in seeds:
  c=base.cfg(seed);c['quality_multiplier']=.5;c['effective_reward_weights_by_instruction']=p['effective_weights'];c['audit_directory']=str(HERE/f'jobs/seed_{seed}/trajectory_audits');c['main_args']['exp_name']='joint_quality_half_from_scratch'
  write(HERE/f'configs/seed_{seed}.json',c)
 paths={ROOT/k for k in p['protected_sha256']};paths.update(q for q in PREVIOUS.rglob('*') if q.is_file())
 m=dict(purpose='fresh_three_seed_joint_happo_quality_half',created_utc=stamp(),seeds=seeds,validation=p['validation'],reserved=p['reserved'],scenarios=p['scenarios'],effective_weights=p['effective_weights'],original_weights=p['original_weights'],training_environment_seeds=derived,quality_multiplier=.5,renormalized=False,steps_per_seed=1000000,total_training_steps=3000000,batch=4000,updates=250,milestones=[200000,400000,600000,800000,1000000],evaluation_episodes=1500,evaluation_physical_steps=900000,bootstrap_seed=202609141225,bootstrap_replicates=4000,equivalence_groups=p['equivalence_groups'],protocol_sha256=sha(HERE/'PROTOCOL.md'),config_sha256={str(q.relative_to(HERE)):sha(q) for q in sorted((HERE/'configs').glob('*.json'))},protected_sha256={str(q.relative_to(ROOT)):sha(q) for q in sorted(paths)},parent_results=str(PREVIOUS.relative_to(ROOT)),from_scratch=True,old_models_not_loaded=True)
 write(HERE/'manifest.json',m)
 before=read(PREVIOUS/'git_status_end.json');current={}
 for directory in before:
  current[directory]={}
  for key,args in [('head',['rev-parse','HEAD']),('status',['status','--porcelain=v1'])]:
   r=subprocess.run(['git','-C',directory,*args],capture_output=True,text=True,env={**os.environ,'GIT_OPTIONAL_LOCKS':'0'});current[directory][key]=dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)
 write(HERE/'git_status_start.json',current);write(HERE/'software_environment.json',dict(executable=sys.executable,python=sys.version,torch=torch.__version__,numpy=np.__version__,cpu_count=os.cpu_count(),affinity=sorted(os.sched_getaffinity(0)),load=os.getloadavg(),gpu=subprocess.run(['nvidia-smi','--query-gpu=name,utilization.gpu,memory.used,temperature.gpu','--format=csv'],capture_output=True,text=True).stdout))
 write(HERE/'status.json',dict(state='prepared',target_training_steps=3000000,from_scratch=True));print('PREPARED',len(paths),'protected inputs')
if __name__=='__main__':guard();main()
