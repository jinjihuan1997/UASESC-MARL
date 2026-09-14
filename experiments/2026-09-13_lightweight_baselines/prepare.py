from light_support import *
from lightweight_policies import METHODS
import subprocess,platform

def main():
 assert not (HERE/'manifest.json').exists();prior=read(RECOMPOSE/'manifest.json');fm=read(FROZEN/'manifest.json');split=read(FROZEN/'seed_selection.json');rm=read(REPAIR/'manifest.json')
 assert prior['validation']==fm['evaluation_seeds']==split['validation']==rm['validation'] and prior['scenarios']==fm['scenarios']==rm['scenarios']
 assert prior['parents']==split['training']==[104948945,111868397,160441552] and prior['reserved_final_test']==split['reserved_final_test']
 for p,h in prior['protected_input_sha256'].items():assert sha(ROOT/p)==h
 paths={ROOT/p for p in prior['protected_input_sha256']};paths.update(p for p in RECOMPOSE.rglob('*') if p.is_file())
 previous=HERE.parent/'2026-09-13_teacher_init_instruction_kl'
 if (previous/'manifest.json').exists():
  pm=read(previous/'manifest.json')
  for p,h in pm['protected_sha256'].items():assert sha(ROOT/p)==h
  paths.update(ROOT/p for p in pm['protected_sha256']);paths.update(p for p in previous.rglob('*') if p.is_file())
 for parent in prior['parents']:write(HERE/f'configs/seed_{parent}.json',old.cfg_for(parent))
 assert all(cfg0['env_args']==old.cfg_for(prior['parents'][0])['env_args'] for cfg0 in [old.cfg_for(s) for s in prior['parents']])
 refs={};old_index=read(RECOMPOSE/'reference_index.json')
 for key,value in old_index.items():
  if not any('/'+name+'/' in '/'+key for name in ['greedy_modes_16','greedy_modes_3','R_instruction','R_equal_instruction','original_rl']):continue
  p=ROOT/value['path'];assert sha(p)==value['trace_sha256'];refs[key]=dict(path=value['path'],sha256=sha(p),metadata_sha256=sha(p.with_suffix('.json')),source='repair',original_independent_audit_sha256=sha(RECOMPOSE/'report/independent_profile_audit.json'))
 for seed in prior['parents']:
  for method in ['C1','C3']:
   for scene in prior['scenarios']:
    p=RECOMPOSE/f'evaluation/seed_{seed}/{method}/{scene}.npz';meta=read(p.with_suffix('.json'));assert sha(p)==meta['trace_sha256'];refs[f'{seed}/{method}/{scene}']=dict(path=str(p.relative_to(ROOT)),sha256=sha(p),metadata_sha256=sha(p.with_suffix('.json')),source='recomposition',original_independent_audit_sha256=sha(RECOMPOSE/'report/independent_profile_audit.json'))
 assert len(refs)==169
 legacy={}
 for method in ['R_equal_single','R_single']:
  for scene in prior['scenarios']:
   p=FROZEN/f'evaluation/rules/{method}/{scene}.npz';meta=read(p.with_suffix('.json'));assert sha(p)==meta['trace_sha256'];legacy[f'{method}/{scene}']=dict(path=str(p.relative_to(ROOT)),sha256=sha(p),metadata_sha256=sha(p.with_suffix('.json')));paths.update([p,p.with_suffix('.json')])
 for row in refs.values():paths.update([ROOT/row['path'],(ROOT/row['path']).with_suffix('.json')])
 evidence={}
 for directory in [FROZEN,GREEDY,REPAIR,RECOMPOSE]:
  for p in sorted(directory.glob('*.md'))+sorted((directory/'report').glob('*.md')):
   content=p.read_text();evidence[str(p.relative_to(ROOT))]=dict(sha256=sha(p),characters=len(content));paths.add(p)
 for p in [FROZEN/'helpers.py',FROZEN/'manifest.json',FROZEN/'seed_selection.json',FROZEN/'source/tensor_env.py',GREEDY/'online_policy.py',GREEDY/'protocol.json']:
  content=p.read_text();paths.add(p);evidence[str(p.relative_to(ROOT))]=dict(sha256=sha(p),characters=len(content))
 baselines=['greedy_modes_16','greedy_modes_3','R_instruction','R_equal_instruction','R_equal_single','R_single']
 instances=[dict(method=s,parent=None) for s in METHODS+baselines]+[dict(method=s,parent=p) for s in ['original_rl','C1','C3'] for p in prior['parents']]
 bench=dict(device='cpu',torch_threads=1,blas_threads=1,batch_sizes=[1,20],warmup=30,repeats=300,fixtures=dict(source='R_instruction',scenes=list(prior['scenarios']),slots=[0,300,599],single_row='fixture_index mod20'),order_seed=202609131830,instances=instances,timing_scope='SUT input handling and resource decision + original allocate_resources budget and post-observation/mask generation + 3 local UAV mode decisions; no physical transition/log/audit/disk/model loading',planned_decision_batches=12540,planned_sample_decisions=131670)
 write(HERE/'reference_index.json',dict(reused=refs,legacy_to_supplement=legacy,missing=[],planned_supplements=dict(methods=['R_equal_single','R_single'],complete_episodes=520,physical_steps=312000,reason='Old numeric traces lack cache/timestamps/served and requested-action records needed for full independent audit and descriptive statistics; preserve exact old definitions and verify all old fields.')))
 m=dict(schema=1,created_utc=stamp(),parents=prior['parents'],validation=prior['validation'],reserved=prior['reserved_final_test'],scenarios=prior['scenarios'],methods=METHODS,old_baselines=baselines,learned_methods=['original_rl','C1','C3'],equivalence_groups=prior['equivalence_groups'],models=prior['models'],reference_commit=prior['reference_commit'],recomposition=str(RECOMPOSE.relative_to(ROOT)),protocol_sha256=sha(HERE/'PROTOCOL.md'),config_sha256={str(p.relative_to(HERE)):sha(p) for p in sorted((HERE/'configs').glob('*.json'))},reference_index_sha256=sha(HERE/'reference_index.json'),protected_sha256={str(p.relative_to(ROOT)):sha(p) for p in sorted(paths)},evidence=evidence,benchmark=bench,bootstrap_seed=202609131831,bootstrap_replicates=4000,planned=dict(primary_episodes=1040,primary_steps=624000,supplement_episodes=520,supplement_steps=312000,reused_episodes=3380,reused_steps=2028000,preflight_complete_episodes=16,preflight_prefix_steps=1200,preflight_physical_steps=10800,reference_preflight_replay_samples=156000),resources=dict(device='cpu',workers=3,torch_threads=1,blas_threads=1))
 write(HERE/'manifest.json',m)
 def git(p,args):
  r=subprocess.run(['git','-C',str(p),*args],capture_output=True,text=True,env={**os.environ,'GIT_OPTIONAL_LOCKS':'0'});return dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)
 gitref=ROOT.parent/'.github-sync/UASESC-MARL_20260911/checkout'
 write(HERE/'git_status.json',{str(p):dict(head=git(p,['rev-parse','HEAD']),status=git(p,['status','--porcelain=v1'])) for p in [ROOT,ROOT/'HARL/HARL',ROOT/'CRL-SemCom-VidCI',ROOT/'UASESE-MARL',gitref]})
 write(HERE/'software_environment.json',dict(python=sys.version,executable=sys.executable,torch=torch.__version__,numpy=np.__version__,platform=platform.platform(),cpu_model=subprocess.run(['lscpu'],capture_output=True,text=True).stdout,cpu_count=os.cpu_count(),load=os.getloadavg(),gpu=subprocess.run(['nvidia-smi','--query-gpu=name,utilization.gpu,memory.used,temperature.gpu','--format=csv'],capture_output=True,text=True).stdout))
 write(HERE/'status.json',dict(state='prepared',new_training_steps=0,new_optimizer_updates=0));print('Prepared',len(paths),'protected inputs; 169 reusable traces and 26 prescribed supplements')
if __name__=='__main__':guard();main()
