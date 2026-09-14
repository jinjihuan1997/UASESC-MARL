from study import *
import subprocess,platform

def main():
 assert not (HERE/'manifest.json').exists()
 rm=read(RECOMPOSE/'manifest.json');assert read(RECOMPOSE/'status.json')['state']=='complete'
 for p,h in rm['protected_input_sha256'].items():assert sha(ROOT/p)==h
 teacher=read(GREEDY/'protocol.json')
 for p,h in teacher['source_hashes'].items():assert sha(ROOT/'experiments'/p.split('/experiments/',1)[1])==h
 predictor=GREEDY/'modes_16/predictor.npz';assert sha(predictor)==read(GREEDY/'modes_16/fit.json')['predictor_sha256']
 known=set(rm['validation']+teacher['validation_seeds']+teacher['evaluation_seeds']);reserved=rm['reserved_final_test']
 excluded=set(known)|set(reserved)|set(teacher['fit_seeds'])|set(rm['parents'])
 rng=np.random.default_rng(202609131350)
 def draw(n=1):
  out=[]
  while len(out)<n:
   x=int(rng.integers(400000000,2000000000))
   if x not in excluded:excluded.add(x);out.append(x)
  return out if n!=1 else out[0]
 students=draw(3);gate=draw(10);a0=draw(120);pf=draw(10)
 seeds={}
 for s in students:
  seeds[str(s)]=dict(dagger={f'A{r}':draw(80) for r in range(1,4)},warmup=draw(60),B=draw(10),fit=draw(),critic_init=draw(),critic_fit=draw(),B_permutations=draw(),actions={**{f'A{r}':draw(4) for r in range(1,4)},'warmup':draw(4),'B':draw(4),'gate':[draw(4) for _ in range(3)],'final_B':[draw(4) for _ in range(3)]})
 config=frozen_support.cfg_for(rm['parents'][0]);config['algo_args']['train']['model_dir']=None
 config['training_design']=dict(arm='joint',termination='finite_600_slot_task',study='teacher_init_instruction_kl')
 assert config['env_args']['instruction_mode_strategy']=='random_switch_once'
 assert (config['env_args']['instruction_switch_min_step'],config['env_args']['instruction_switch_max_step'])==(100,500)
 assert config['algo_args']['model']['hidden_sizes']==[256,256]
 write(HERE/'config.json',config)
 paths={ROOT/p for p in rm['protected_input_sha256']};paths.update(p for p in RECOMPOSE.rglob('*') if p.is_file())
 # Read all relevant human evidence and retain digest/character inventory.
 evidence=[]
 for root in (FROZEN,GREEDY,REPAIR,RECOMPOSE):
  for p in sorted(root.glob('*.md'))+sorted((root/'report').glob('*.md')):
   body=p.read_text();evidence.append(dict(path=str(p.relative_to(ROOT)),sha256=sha(p),characters=len(body)))
 allowed=dict(A0=a0,dagger=sum([sum(d['dagger'].values(),[]) for d in seeds.values()],[]),warmup=sum([d['warmup'] for d in seeds.values()],[]),B=sum([d['B'] for d in seeds.values()],[]),gate=gate,preflight=pf,validation=rm['validation'])
 m=dict(schema=1,created_utc=stamp(),students=students,student_seed_generation=202609131350,gate_dev=gate,validation=rm['validation'],known_development_seeds=sorted(known),reserved=reserved,A0=a0,preflight=pf,seeds=seeds,bootstrap_seed=draw(),preflight_model_seed=draw(),preflight_action_seeds=draw(4),preflight_stat_seed=draw(),scenarios=rm['scenarios'],equivalence_groups=rm['equivalence_groups'],allowed_environment_seeds=allowed,protocol_sha256=sha(HERE/'PROTOCOL.md'),config_sha256=sha(HERE/'config.json'),teacher=dict(method='greedy_modes_16',predictor=str(predictor.relative_to(ROOT)),predictor_sha256=sha(predictor),source_sha256=sha(GREEDY/'online_policy.py'),teacher_protocol_sha256=sha(GREEDY/'protocol.json')),protected_sha256={str(p.relative_to(ROOT)):sha(p) for p in sorted(paths)},evidence=evidence,reference_commit=rm['reference_commit'],recomposition_path=str(RECOMPOSE.relative_to(ROOT)),budget=dict(A0_episodes=120,dagger_episodes=720,unique_collection_steps=504000,gate_episodes=1690,gate_steps=1014000,conditional_warmup_episodes=180,conditional_warmup_steps=108000,conditional_RL_steps=6000000,conditional_B_evaluation_episodes=8460,conditional_B_evaluation_steps=5076000),imitation=dict(epochs=[20,10,10,10],batch_size=1024,lr=.001,adam_eps=1e-8,max_grad_norm=1.,target_concentration=1000.,smoothing=.0001),gate_thresholds=dict(D_overall_vs_teacher=-.10,D_fixed_vs_teacher=-.20,S_overall_vs_D=-.50,S_fixed_vs_D=-1.),resources=dict(collection_device='cpu',evaluation_device='cpu',fit_device='cuda:0',maximum_cpu_workers=3,maximum_gpu_workers=1,torch_threads=1,blas_threads=1))
 write(HERE/'manifest.json',m);write(HERE/'seed_selection.json',dict(students=students,A0=a0,gate_dev=gate,preflight=pf,seeds=seeds,bootstrap_seed=m['bootstrap_seed']))
 gitref=ROOT.parent/'.github-sync/UASESC-MARL_20260911/checkout'
 def git(p,*args):
  r=subprocess.run(['git','-C',str(p),*args],capture_output=True,text=True,env={**os.environ,'GIT_OPTIONAL_LOCKS':'0'});return dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)
 write(HERE/'git_status.json',{str(p):dict(head=git(p,'rev-parse','HEAD'),status=git(p,'status','--porcelain=v1')) for p in [ROOT,ROOT/'HARL/HARL',ROOT/'CRL-SemCom-VidCI',ROOT/'UASESE-MARL',gitref]})
 assert git(gitref,'rev-parse','HEAD')['stdout'].strip()==m['reference_commit']
 write(HERE/'software_environment.json',dict(utc=stamp(),python=sys.version,executable=sys.executable,torch=torch.__version__,numpy=np.__version__,cuda=torch.version.cuda,cpu_count=os.cpu_count(),load=os.getloadavg(),gpu=subprocess.run(['nvidia-smi','--query-gpu=name,utilization.gpu,memory.used,temperature.gpu','--format=csv'],capture_output=True,text=True).stdout))
 write(HERE/'status.json',dict(state='prepared',stage='preflight',B_authorized_by_gate=False))
 print(dict(students=students,protected_files=len(paths),gate_dev=gate))
if __name__=='__main__':main()
