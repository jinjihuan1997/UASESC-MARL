from weight_support import *
def main():
 assert not (HERE/'manifest.json').exists();lm=base.verify();assert read(LIGHT/'status.json')['state']=='complete';oldw=np.asarray(base.cfg()['env_args']['reward_weights_by_instruction']);new=oldw.copy();new[:,0]*=.5
 static=lm['methods']+lm['old_baselines'];learned=lm['learned_methods'];greedy=['G_equal_local16','G_urgency_local16','greedy_modes_16','greedy_modes_3'];adapt=[g+'_Qhalf_local' for g in greedy]
 paths={ROOT/p for p in lm['protected_sha256']};paths.update(p for p in LIGHT.rglob('*') if p.is_file())
 instances=[dict(method=s,parent=None) for s in static]+[dict(method=s,parent=p) for s in learned for p in lm['parents']]+[dict(method=s,parent=None) for s in adapt]
 assert len(instances)==23
 for p in lm['parents']:write(HERE/f'configs/seed_{p}.json',dict(base_configuration=base.cfg(p),quality_multiplier=.5,effective_reward_weights_by_instruction=new.tolist(),renormalized=False))
 m=dict(schema=1,created_utc=stamp(),parents=lm['parents'],validation=lm['validation'],reserved=lm['reserved'],scenarios=lm['scenarios'],original_weights=oldw.tolist(),effective_weights=new.tolist(),quality_multiplier=.5,renormalized=False,static_methods=static,learned_methods=learned,greedy_methods=greedy,adaptive_methods=adapt,instances=instances,equivalence_groups=lm['equivalence_groups'],models=lm['models'],parent_experiment=str(LIGHT.relative_to(ROOT)),protocol_sha256=sha(HERE/'PROTOCOL.md'),protected_sha256={str(p.relative_to(ROOT)):sha(p) for p in sorted(paths)},bootstrap_seed=202609132105,bootstrap_replicates=4000,planned=dict(fixed_controller_episodes=4940,adaptive_local_episodes=1040,formal_episodes=5980,formal_physical_steps=3588000,preflight_episodes=28,preflight_physical_steps=16800),resources=dict(device='cpu',max_workers=3,torch_threads=1,blas_threads=1),new_training_steps=0,new_fitting_updates=0)
 write(HERE/'manifest.json',m)
 before=read(LIGHT/'git_status_end.json');current={}
 for directory in before:
  current[directory]={}
  for key,args in [('head',['rev-parse','HEAD']),('status',['status','--porcelain=v1'])]:
   r=subprocess.run(['git','-C',directory,*args],capture_output=True,text=True,env={**os.environ,'GIT_OPTIONAL_LOCKS':'0'});current[directory][key]=dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)
 write(HERE/'git_status_start.json',current)
 write(HERE/'software_environment.json',dict(executable=sys.executable,python=sys.version,torch=torch.__version__,numpy=np.__version__,load=os.getloadavg(),cpu_count=os.cpu_count(),gpu=subprocess.run(['nvidia-smi','--query-gpu=name,utilization.gpu,memory.used,temperature.gpu','--format=csv'],capture_output=True,text=True).stdout,cpu=subprocess.run(['lscpu'],capture_output=True,text=True).stdout))
 write(HERE/'status.json',dict(state='prepared',new_training_steps=0,new_optimizer_updates=0));print('Prepared',len(paths),'protected files; formal 5980 episodes')
if __name__=='__main__':guard();main()
