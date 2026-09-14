"""Conditional joint HAPPO entry. Fixed rollout-old instruction KL rollback."""
from study import *
from student_policy import Student
from tensor_train import TensorRollout,TensorValueNorm
from harl.algorithms.actors.happo import HAPPO
from harl.algorithms.critics.v_critic import VCritic
from harl.utils.ratio_tools import aggregate_action_ratio
from happo_kernel import strict_actor_step
from kl_control import distribution_snapshot,analytic_kl,summarize_kl,Candidate
from checkpointing import env_state,restore_env,cpu_tree
from warmup_critic import strict_critic_step
import argparse

class Joint:
 def __init__(self,seed,arm,device):
  require_A_gate();self.m=verify();assert seed in self.m['students'] and arm in ['B0','B1']
  self.seed=seed;self.arm=arm;self.ss=self.m['seeds'][str(seed)];self.device=torch.device(device);self.folder=HERE/f'B/{seed}/{arm}'
  self.env=make_env(self.ss['B'],'B',device=device);self.student=Student(seed,self.env,device)
  init=HERE/f'warmup/{seed}/initialization.pt';d=torch.load(init,map_location=device,weights_only=False)
  assert d['manifest_sha256']==sha(HERE/'manifest.json') and d['epoch']==20
  for net,weights in zip(self.student.actors,d['actors']):net.load_state_dict(weights)
  assert self.student.hashes()==d['actor_hashes']
  self.args={**cfg()['algo_args']['model'],**cfg()['algo_args']['algo']};self.args.update(lr=1e-4,critic_lr=4e-4)
  self.actors=[HAPPO(self.args,o,a,self.device) for o,a in zip(self.env.observation_space,self.env.action_space)]
  for wrapper,net in zip(self.actors,self.student.actors):
   wrapper.actor=net;wrapper.actor_optimizer=torch.optim.Adam(net.parameters(),lr=1e-4,eps=self.args['opti_eps'],weight_decay=self.args['weight_decay'])
  self.critic=VCritic(self.args,self.env.share_observation_space[0],self.device);self.critic.critic.load_state_dict(d['critic'])
  self.normal=TensorValueNorm(1,device=self.device);self.normal.load_state_dict(d['normalizer'])
  self.buffer=TensorRollout(self.env,400,cfg()['algo_args']['model']);self.buffer.store_observation(0,*self.env.observe())
  self.streams=ActionStreams(self.ss['actions']['B'],device);self.completed=0;self.counts=dict(actor_accepted=[0]*4,actor_rejected=[0]*4,actor_skipped=[0]*4,critic=0)
  self.init_sha=sha(init)
  self.initialization=dict(manifest_sha256=sha(HERE/'manifest.json'),warmup_sha256=self.init_sha,actors=self.student.hashes(),critic=network_hash(self.critic.critic),normalizer_hash=hashlib.sha256(b''.join(arr(v).tobytes() for v in self.normal.state_dict().values())).hexdigest(),fresh_optimizers=True,actor_optimizer_states=[len(a.actor_optimizer.state) for a in self.actors],critic_optimizer_states=len(self.critic.critic_optimizer.state))
  write(self.folder/'initialization.json',self.initialization)

 @torch.no_grad()
 def collect(self):
  self.student.set_mode(False);self.critic.prep_rollout();b=self.buffer;e=self.env;records=[];external=[]
  external.append(dict(offset=0,episode=e.episode_indices,slot=e.step_index,hashes=frozen.external_hashes(e)))
  for t in range(400):
   obs,state,mask=e.observe();b.store_observation(t,obs,state,mask);b.values[t].copy_(self.critic.get_values(b.state[t],b.rnn,b.masks[t])[0])
   before={k:arr(getattr(e,k)).copy() for k in ['q','tau','aoi']};slot=e.step_index;episode=e.episode_indices.copy()
   with self.streams.use(0):resource,lp=self.student.forward(0,obs[:,0],mask[:,0],False)
   b.actions[0][t].copy_(resource);b.log_probs[0][t].copy_(lp)
   post,_,am=e.allocate_resources(resource);b.obs[t,:,1:].copy_(post[:,1:]);b.available[t,:,1:].copy_(am[:,1:]);actions=[resource]
   for i in range(1,4):
    with self.streams.use(i):a,lp=self.student.forward(i,post[:,i],am[:,i],False)
    b.actions[i][t].copy_(a);b.log_probs[i][t].copy_(lp);actions.append(a)
   alpha=self.student.distribution(0,obs[:,0],mask[:,0]).concentration
   probabilities=torch.stack([self.student.distribution(i,post[:,i],am[:,i]).probs.squeeze(1) for i in range(1,4)],1)
   nxt,shared,avail,info,values=frozen_support.checked_step(e,actions)
   row=dict(physical_slot=np.full(e.count,slot),episode=np.array(episode),instruction_id=arr(info['gid']),trace=np.column_stack([values[f] for f in FIELDS]),modes=arr(info['mode']),requested_modes=np.column_stack([arr(a.argmax(-1)) for a in actions[1:]]),raw_resource_action=arr(resource),resource_fractions=arr(e.beta),budgets=arr(info['budget']),usage=arr(info['usage']),unused_budgets=arr(info['budget']-info['usage']),served=arr(info['served']),predicted_quality=arr(info['quality']),quality_requirement=arr(info['req']),uav_masks=arr(am[:,1:]),aoi_before=before['aoi'],cache_before=before['q'],tau_before=before['tau'],aoi_after=arr(e.aoi),cache_after=arr(e.q),tau_after=arr(e.tau),sut_concentration=arr(alpha),uav_probabilities=arr(probabilities))
   records.append({k:v.copy() for k,v in row.items()});done=e.step_index==600
   if done:
    nxt,shared,avail=e.reset();external.append(dict(offset=t+1,episode=e.episode_indices,slot=0,hashes=frozen.external_hashes(e)))
   b.store_observation(t+1,nxt,shared,avail);b.rewards[t].copy_(torch.from_numpy(values['training_reward']).to(self.device)[:,None]);b.masks[t+1].fill_(0 if done else 1)
  b.values[-1].copy_(self.critic.get_values(b.state[-1],b.rnn,b.masks[-1])[0])
  assert all(torch.isfinite(x).all() for x in [b.values,b.rewards,b.obs,b.state])
  old=[distribution_snapshot(self.student,i,b.actor_inputs(i)[0],b.actor_inputs(i)[4]) for i in range(4)]
  path=self.folder/f'rollouts/update_{self.completed+1:03}.npz'
  npz(path,**{k:np.stack([r[k] for r in records]) for k in records[0]},seeds=np.array(self.ss['B']),fields=np.array(FIELDS),obs=arr(b.obs),state=arr(b.state),available=arr(b.available),masks=arr(b.masks),values=arr(b.values),**{f'actions_{i}':arr(b.actions[i]) for i in range(4)},**{f'logp_{i}':arr(b.log_probs[i]) for i in range(4)})
  write(path.with_suffix('.json'),dict(state='complete',manifest_sha256=sha(HERE/'manifest.json'),trace_sha256=sha(path),external=external,physical_steps=4000,original_checker='PASS_EVERY_SLOT',actors_before_update=self.student.hashes()))
  record_cost('RL_collection',f'{self.seed}/{self.arm}/{self.completed+1}',physical_steps=4000,RL_environment_steps=4000,seeds=self.ss['B'],episodes_completed=sum(int(r['physical_slot'][0])==599 for r in records)*10,teacher_queries=[0]*4)
  return old

 def update(self,old):
  b=self.buffer;a=self.args;N=4000;self.student.set_mode(True);returns=b.returns(self.normal,.99,.95);adv=returns-self.normal.denormalize(b.values[:-1]);adv=((adv-adv.mean())/(adv.std(unbiased=False)+1e-5)).flatten(0,1);factor=torch.ones(N,1,device=self.device)
  # A deterministic update-index stream precomputes ALL permutations before any rejection.
  rng=torch.Generator().manual_seed(int(np.random.SeedSequence([self.ss['B_permutations'],self.completed]).generate_state(1)[0]))
  order=torch.randperm(4,generator=rng).tolist();actor_perms=[[torch.randperm(N,generator=rng).to(self.device) for _ in range(5)] for _ in range(4)];critic_perms=[torch.randperm(N,generator=rng).to(self.device) for _ in range(5)]
  permutation_hash=hashlib.sha256(b''.join(arr(x).tobytes() for row in actor_perms for x in row)+b''.join(arr(x).tobytes() for x in critic_perms)).hexdigest()
  metrics=[];accepted_this=[0]*4;rejected_this=[0]*4;skipped_this=[0]*4
  for i in order:
   actor=self.actors[i];actor.prep_training();obs,rnn,actions,mask,available,active=b.actor_inputs(i);oldlog=b.log_probs[i].flatten(0,1);gids=obs[:,23:26].argmax(-1) if i==0 else obs[:,67:70].argmax(-1)
   with torch.no_grad():
    current=actor.evaluate_actions(obs,rnn,actions,mask,available,active)[0];assert torch.equal(current,oldlog)
    initialkl=analytic_kl(old[i],self.student,i,obs,available);assert initialkl.abs().max()<1e-9
   stopped=False;records=[]
   for epoch in range(5):
    for indices in actor_perms[i][epoch].chunk(2):
     if stopped:skipped_this[i]+=1;continue
     candidate=Candidate(actor.actor,actor.actor_optimizer)
     sample=(obs[indices],rnn[indices],actions[indices],mask[indices],active[indices],oldlog[indices],adv[indices],available[indices],factor[indices])
     item=strict_actor_step(actor,sample)
     with torch.no_grad():kl=summarize_kl(analytic_kl(old[i],self.student,i,obs,available),gids)
     reject=self.arm=='B1' and kl['exceeds'];item.update(epoch=epoch+1,KL=kl,rejected=reject)
     if reject:candidate.rollback();stopped=True;rejected_this[i]+=1
     else:accepted_this[i]+=1
     records.append(item)
   with torch.no_grad():
    newlog=actor.evaluate_actions(obs,rnn,actions,mask,available,active)[0];ratio=aggregate_action_ratio(newlog-oldlog,a['action_aggregation'],clip=20.).clamp(0,1e3)
    assert torch.isfinite(ratio).all()
    if accepted_this[i]==0:assert torch.equal(ratio,torch.ones_like(ratio))
    factor=(factor*ratio).clamp(0,1e3);assert torch.isfinite(factor).all()
    finalkl=summarize_kl(analytic_kl(old[i],self.student,i,obs,available),gids)
    if self.arm=='B1':assert not finalkl['exceeds']
   metrics.append(dict(actor=i,candidates=records,accepted=accepted_this[i],rejected=rejected_this[i],skipped=skipped_this[i],final_KL=finalkl,final_factor_min=float(factor.min()),final_factor_max=float(factor.max())))
  self.critic.prep_training();cm=[]
  for perm in critic_perms:
   for ix in perm.chunk(2):cm.append(strict_critic_step(self.critic,self.normal,(b.state[:-1].flatten(0,1)[ix],b.flat_rnn[ix],b.values[:-1].flatten(0,1)[ix],returns.flatten(0,1)[ix],b.masks[:-1].flatten(0,1)[ix])))
  for key,values in [('actor_accepted',accepted_this),('actor_rejected',rejected_this),('actor_skipped',skipped_this)]:self.counts[key]=[x+y for x,y in zip(self.counts[key],values)]
  self.counts['critic']+=len(cm)
  with torch.no_grad():
   states=b.state[:-1].flatten(0,1);value=self.normal.denormalize(self.critic.get_values(states,b.flat_rnn,b.masks[:-1].flatten(0,1))[0]);target=returns.flatten(0,1);mse=float((value-target).square().mean());variance=float(target.var(unbiased=False));ev=1-float((value-target).var(unbiased=False))/variance if variance else None
  row=dict(update=self.completed+1,environment_steps=(self.completed+1)*4000,actor_order=order,permutation_sha256=permutation_hash,actors=metrics,critic=cm,value_MSE=mse,value_explained_variance=ev,cumulative_counts=self.counts.copy());append(self.folder/'training_metrics.jsonl',row)
  record_cost('RL_optimization',f'{self.seed}/{self.arm}/{self.completed+1}',physical_steps=0,actor_steps_accepted=accepted_this,actor_steps_rejected=rejected_this,actor_steps_skipped=skipped_this,critic_updates=len(cm),gradient_steps=sum(accepted_this)+sum(rejected_this)+len(cm))
  self.completed+=1;b.after_update();return row

 def snapshot(self,path):
  b=self.buffer;d=dict(student_seed=self.seed,manifest_sha256=sha(HERE/'manifest.json'),warmup_sha256=self.init_sha,arm=self.arm,actors=self.student.state(),actor_hashes=self.student.hashes(),critic=self.critic.critic.state_dict(),normalizer=self.normal.state_dict(),actor_optimizers=[a.actor_optimizer.state_dict() for a in self.actors],critic_optimizer=self.critic.critic_optimizer.state_dict(),completed=self.completed,environment_steps=self.completed*4000,counts=self.counts,environment=env_state(self.env),action_rng=self.streams.states,buffer={k:getattr(b,k) for k in ['obs','state','available','actions','log_probs','values','rewards','masks']},torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [])
  save(path,cpu_tree(d))

 def restore(self,path):
  d=torch.load(path,map_location=self.device,weights_only=False);assert d['manifest_sha256']==sha(HERE/'manifest.json') and d['warmup_sha256']==self.init_sha and d['arm']==self.arm and d['student_seed']==self.seed
  for a,w,o in zip(self.actors,d['actors'],d['actor_optimizers']):a.actor.load_state_dict(w);a.actor_optimizer.load_state_dict(o)
  self.critic.critic.load_state_dict(d['critic']);self.critic.critic_optimizer.load_state_dict(d['critic_optimizer']);self.normal.load_state_dict(d['normalizer']);restore_env(self.env,d['environment']);self.streams.states=[x.cpu() for x in d['action_rng']]
  for k,v in d['buffer'].items():
   dest=getattr(self.buffer,k)
   if isinstance(dest,list):
    for a,b in zip(dest,v):a.copy_(b)
   else:dest.copy_(v)
  self.completed=d['completed'];self.counts=d['counts'];assert self.completed<=250 and self.student.hashes()==d['actor_hashes']

def train(seed,arm,device):
 require_A_gate();assert read(HERE/'B_preflight.json')['state']=='PASS', 'Conditional B integration preflight is required before formal RL.'
 t=Joint(seed,arm,device);resume=t.folder/'resume.pt'
 if resume.exists():t.restore(resume)
 while t.completed<250:
  t.update(t.collect());t.snapshot(resume)
  if t.completed%50==0:
   p=t.folder/f'milestones/steps_{t.completed*4000}/student.pt';assert not p.exists();t.snapshot(p);write(p.parent/'status.json',dict(state='complete',environment_steps=t.completed*4000,checkpoint_sha256=sha(p)))
  write(t.folder/'status.json',dict(state='complete' if t.completed==250 else 'running',environment_steps=t.completed*4000,counts=t.counts));print(seed,arm,t.completed,flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--student',type=int,required=True);p.add_argument('--arm',choices=['B0','B1'],required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args();guard();train(a.student,a.arm,a.device)
