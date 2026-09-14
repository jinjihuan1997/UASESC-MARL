from support import *
import subprocess

def main():
 assert not (HERE/'manifest.json').exists()
 pm=read(OLD/'manifest.json');old.verify_inputs();fm=frozen.verify()
 assert pm['parents']==[104948945,111868397,160441552] and pm['scenarios']==fm['scenarios']
 select=read(FROZEN/'seed_selection.json');assert pm['validation']==select['validation'] and pm['reserved_final_test']==select['reserved_final_test']
 audit=read(OLD/'report/audit.json');assert audit['state']=='PASS' and not audit['final_test_used']
 assert audit['results_sha256']==sha(OLD/'report/results.json') and audit['physical_statistics_sha256']==sha(OLD/'report/physical_statistics.json')
 assert read(OLD/'report/reproducibility.json')['state']=='PASS'
 paths={WORKSPACE/p for p in pm['protected_input_sha256']}
 paths.update(p for p in OLD.rglob('*') if p.is_file())
 configsha={};models={}
 for seed in pm['parents']:
  cfg=old.cfg_for(seed)
  for arm in pm['arms']:
   oc=read(OLD/f'configs/seed_{seed}/{arm}.json');assert oc['env_args']==cfg['env_args']
  cp=HERE/f'configs/seed_{seed}.json';write(cp,cfg);configsha[str(cp.relative_to(HERE))]=sha(cp)
  models[str(seed)]={}
  for label,folder in [('original',old.model_dir(seed))]+[(arm,OLD/f'jobs/seed_{seed}/{arm}/milestones/steps_1000000') for arm in pm['arms']]:
   status=read(folder/'status.json');assert status['state']=='complete' and status['completed_steps']==1000000
   for f,h in status['checkpoint_hashes'].items():assert sha(folder/f)==h
   models[str(seed)][label]=dict(path=str(folder.relative_to(WORKSPACE)),status=status,files={p.name:sha(p) for p in folder.iterdir() if p.is_file()})
 refs={};methodmap={'C0':'residual_quality_at_1000000','original_rl':'original_rl','residual_all':'residual_all_at_1000000','quality_rule_hybrid':'quality_rule_hybrid'}
 for seed in pm['parents']+[None]:
  for method,label in (methodmap.items() if seed else [(x,x) for x in pm['rules']]):
   item=f'seed_{seed}/{label}' if seed else f'rules/{label}';folder=OLD/'evaluation'/item
   status=read(folder/'status.json');assert status['state']=='complete' and sha(folder/'summary.json')==status['summary_sha256']
   for scene in pm['scenarios']:
    key=f'{item}/{scene}';file=folder/f'{scene}.npz';meta=read(folder/f'{scene}.json')
    assert sha(file)==meta['trace_sha256']==audit['trace_sha256'][key]
    assert meta['identity']['manifest_sha256']==sha(OLD/'manifest.json')
    assert key in audit['independent_offline_profile_executor_reward_audit']
    with np.load(file,allow_pickle=False) as z:assert z['seeds'].tolist()==pm['validation'] and z['trace'].shape==(600,20,len(FIELDS))
    refs[f'{seed or "rules"}/{method}/{scene}']=dict(path=str(file.relative_to(WORKSPACE)),trace_sha256=meta['trace_sha256'],metadata=meta,prior_audit_key=key)
 assert len(refs)==208
 write(HERE/'reference_index.json',refs);write(HERE/'seed_selection.json',dict(parents=pm['parents'],validation=pm['validation'],reserved_final_test=pm['reserved_final_test'],bootstrap_seed=pm['bootstrap_seed']))
 m=dict(schema=1,created_utc=stamp(),reference_commit=pm['reference_commit'],parents=pm['parents'],validation=pm['validation'],reserved_final_test=pm['reserved_final_test'],scenarios=pm['scenarios'],rules=pm['rules'],equivalence_groups=pm['equivalence_groups'],controllers=['C0','C1','C2','C3'],new_controllers=['C1','C2','C3'],protocol_sha256=sha(HERE/'PROTOCOL.md'),parent_manifest_sha256=sha(FROZEN/'manifest.json'),repair_manifest_sha256=sha(OLD/'manifest.json'),reference_index_sha256=sha(HERE/'reference_index.json'),seed_selection_sha256=sha(HERE/'seed_selection.json'),config_sha256=configsha,models=models,protected_input_sha256={str(p.relative_to(WORKSPACE)):sha(p) for p in sorted(paths)},bootstrap_seed=pm['bootstrap_seed'],bootstrap_replicates=4000,new_training_steps=0,new_optimizer_updates=0,new_primary_episodes=2340,new_primary_physical_steps=1404000,reused_episodes=4160,planned_preflight_episodes=1280,reference_supplements=[],device='cpu',torch_threads=1,blas_threads=1,maximum_workers=3,model_source_ids={'0':'P_original','1':'P_all','2':'P_quality'},resource_source_ids={'0':'S_original','1':'original_equal_action'})
 write(HERE/'manifest.json',m)
 gitref=WORKSPACE.parent/'.github-sync/UASESC-MARL_20260911/checkout'
 def git(root,*args):
  r=subprocess.run(['git','-C',str(root),*args],capture_output=True,text=True,env={**os.environ,'GIT_OPTIONAL_LOCKS':'0'});return dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)
 roots=[WORKSPACE,gitref,WORKSPACE/'HARL/HARL',WORKSPACE/'CRL-SemCom-VidCI',WORKSPACE/'UASESE-MARL']
 write(HERE/'git_inspection.json',{str(r):dict(head=git(r,'rev-parse','HEAD'),status=git(r,'status','--porcelain=v1')) for r in roots})
 assert git(gitref,'rev-parse','HEAD')['stdout'].strip()==pm['reference_commit']
 write(HERE/'software_environment.json',dict(python=sys.version,executable=sys.executable,torch=torch.__version__,numpy=np.__version__,cuda=torch.version.cuda,cpu_count=os.cpu_count(),load=os.getloadavg(),gpu=subprocess.run(['nvidia-smi','--query-gpu=name,utilization.gpu,memory.used,temperature.gpu','--format=csv'],capture_output=True,text=True).stdout))
 write(HERE/'status.json',dict(state='prepared',new_training_steps=0,new_optimizer_updates=0,reserved_final_test_used=False))
 print(dict(protected_files=len(paths),verified_reference_traces=len(refs),missing_references=[]))
if __name__=='__main__':main()
