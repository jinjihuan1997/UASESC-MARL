"""Explicit budget-extension migration; science/optimizer/RNG tensors untouched."""
from train_support import *
SHORT=HERE.parent/'2026-09-14_quality_half_retraining'

def same(a,b,path=''):
 assert type(a)==type(b),(path,type(a),type(b))
 if isinstance(a,torch.Tensor):assert torch.equal(a,b),path
 elif isinstance(a,(dict,list,tuple)):
  assert len(a)==len(b),path
  for k in (a if isinstance(a,dict) else range(len(a))):same(a[k],b[k],path+'/'+str(k))
 else:assert a==b,(path,a,b)

def identity(seed):return dict(config_sha256=sha(HERE/f'configs/seed_{seed}.json'),run_manifest_sha256=sha(HERE/'manifest.json'),device='cpu')

def migrate(seed,trainer):
 meta=manifest()['parent_checkpoints'][str(seed)];path=ROOT/meta['path'];assert sha(path)==meta['sha256'];state=torch.load(path,map_location='cpu',weights_only=True)
 assert state['identity']==meta['identity'] and state['update']==250 and state['total_updates']==250 and state['runtime']==checkpoint.runtime_signature()
 expected=copy.deepcopy(state['config']);expected['algo_args']['train']['num_env_steps']=trainer.config['algo_args']['train']['num_env_steps'];expected['training_design']['evaluation_steps']=trainer.config['training_design']['evaluation_steps'];expected['main_args']['exp_name']=trainer.config['main_args']['exp_name'];expected['audit_directory']=trainer.config['audit_directory'];same(expected,trainer.config)
 assert trainer.config['algo_args']['train']['use_linear_lr_decay'] is False
 converted=checkpoint.cpu_tree(state);converted.update(config=copy.deepcopy(trainer.config),total_updates=trainer.total_updates,identity=identity(seed))
 for k in state:
  if k not in ['config','total_updates','identity']:same(state[k],converted[k],k)
 return converted,state

def stage(seed):
 out=HERE/f'jobs/seed_{seed}';assert not out.exists();trainer=Trainer(config(seed),'cpu');new,old=migrate(seed,trainer);assert checkpoint.restore(trainer,new,identity(seed))==250
 restored=checkpoint.capture(trainer,250,identity(seed));same(new,restored)
 entry=checkpoint.save_checkpoint(out/'checkpoints',restored)
 write(out/'config.json',trainer.config);write(out/'manifest.json',dict(seed=seed,method='joint_quality_half_long_continuation',identity=identity(seed),source_snapshot=str(FROZEN/'source'),wrapper=str(HERE/'train_support.py'),from_scratch=False,full_state_continuation=True,initial_actor_hashes=trainer.initial_actors,initial_critic_hash=trainer.initial_critic,continuation_actor_hashes=[kernel.network_hash(a.actor) for a in trainer.actors],prior_training_steps=1000000,new_training_budget=9000000,cumulative_target=10000000,parent_checkpoint=manifest()['parent_checkpoints'][str(seed)],migration_changed_fields=['identity','config','total_updates']))
 model_snapshot.export_snapshot(trainer,out,250,identity(seed));write(out/'status.json',dict(state='ready_to_continue',completed_steps=1000000,target_steps=10000000,recoverable_update=250,checkpoint_sha256=entry['sha256'],device='cpu'))
 return dict(seed=seed,state='PASS',old_sha256=manifest()['parent_checkpoints'][str(seed)]['sha256'],new_sha256=entry['sha256'],all_nonmetadata_state_exact=True)
