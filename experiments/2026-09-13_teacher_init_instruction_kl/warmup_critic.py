"""Conditional original-structure critic warmup; cannot run before 3/3 gate."""
from study import *
from student_policy import Student
from trajectory import run_episode_batch
from audit_trajectories import load_arrays
from tensor_train import TensorValueNorm
from harl.algorithms.critics.v_critic import VCritic
import argparse

def strict_critic_step(critic,normalizer,sample):
 states,rnn,old,returns,masks=sample
 assert all(torch.isfinite(x).all() for x in sample)
 values,_=critic.get_values(states,rnn,masks);assert torch.isfinite(values).all()
 loss=critic.cal_value_loss(values,old,returns,normalizer);assert torch.isfinite(loss)
 critic.critic_optimizer.zero_grad();(loss*critic.value_loss_coef).backward()
 grad=torch.nn.utils.clip_grad_norm_(critic.critic.parameters(),critic.max_grad_norm,error_if_nonfinite=True)
 assert all(p.grad is None or torch.isfinite(p.grad).all() for p in critic.critic.parameters())
 critic.critic_optimizer.step();assert all(torch.isfinite(p).all() for p in critic.critic.parameters())
 return dict(loss=float(loss.detach()),gradient_norm=float(grad))

@torch.no_grad()
def reconstruct_states(z,identity):
 env=make_env(identity['seeds'],identity['purpose'],identity['schedule']);states=[];switched=torch.zeros(env.count,dtype=torch.bool)
 for t in range(600):
  for k in ['q','tau','aoi']:
   src=z['initial_'+k] if t==0 else z[{'q':'cache_after','tau':'tau_after','aoi':'aoi_after'}[k]][t-1]
   setattr(env,k,env.tensor(src,getattr(env,k).dtype))
  env.step_index=t;env.allocation_pending=False
  if t:switched|=env.instructions[t]!=env.instructions[t-1]
  env.switched=switched.clone();env.previous_instruction=env.instructions[t].clone()
  env.beta=env.tensor(np.full((env.count,3),1/3) if t==0 else z['resource_fractions'][t-1]);env.update_channels(t);env.update_tables()
  obs,state,_=env.observe();np.testing.assert_array_equal(arr(obs[:,0]),z['sut_obs'][t]);states.append(arr(state[:,0]).copy())
 return np.stack(states)

def warmup(seed,device='cuda:0'):
 require_A_gate();m=verify();assert seed in m['students'];ss=m['seeds'][str(seed)];folder=HERE/f'warmup/{seed}';actor_path=HERE/f'students/{seed}/A3/checkpoint.pt'
 if (folder/'status.json').exists():
  st=read(folder/'status.json');assert st['state']=='complete' and st['checkpoint_sha256']==sha(folder/'initialization.pt');return
 paths=[]
 for b in range(3):
  path=folder/f'data/batch_{b:02}.npz';run_episode_batch(path,ss['warmup'][b*20:(b+1)*20],'warmup',student_seed=seed,checkpoint=actor_path,deterministic=False,action_seeds=ss['actions']['warmup'],kind='critic_warmup_collection');paths.append(path)
 xs=[];ys=[]
 for path in paths:
  z=load_arrays(path);meta=read(path.with_suffix('.json'));xs.append(reconstruct_states(z,meta['identity']).reshape(-1,202));rewards=z['trace'][:,:,FIELDS.index('training_reward')].astype(np.float32)
  ret=np.zeros_like(rewards);tail=np.zeros(rewards.shape[1],dtype=np.float32)
  for t in range(599,-1,-1):tail=rewards[t]+np.float32(.99)*tail;ret[t]=tail
  ys.append(ret.reshape(-1,1))
 env=make_env(m['preflight'][:1],'preflight');student=Student(seed,env,device);student.load(actor_path);student.set_mode(False);actor_hashes=student.hashes()
 params={**cfg()['algo_args']['model'],**cfg()['algo_args']['algo']};params['critic_lr']=4e-4
 with torch.random.fork_rng(devices=[]):
  torch.manual_seed(ss['critic_init']);critic=VCritic(params,env.share_observation_space[0],torch.device('cpu'))
 critic.critic.to(device);critic.device=torch.device(device);critic.tpdv['device']=torch.device(device);critic.critic.tpdv['device']=torch.device(device)
 normal=TensorValueNorm(1,device=device);data=torch.as_tensor(np.concatenate(xs),device=device);returns=torch.as_tensor(np.concatenate(ys),device=device);assert len(data)==36000
 rng=torch.Generator().manual_seed(ss['critic_fit']);completed=0;updates=0;resume=folder/'resume.pt'
 if resume.exists():
  d=torch.load(resume,map_location=device,weights_only=False);assert d['actor_hashes']==actor_hashes and d['manifest_sha256']==sha(HERE/'manifest.json');critic.critic.load_state_dict(d['critic']);critic.critic_optimizer.load_state_dict(d['critic_optimizer']);normal.load_state_dict(d['normalizer']);rng.set_state(d['shuffle_rng'].cpu());completed=d['epoch'];updates=d['critic_steps']
 for epoch in range(completed,20):
  metrics=[]
  for idx in torch.randperm(36000,generator=rng).split(1024):
   idx=idx.to(device);n=len(idx);rnn=torch.zeros(n,1,256,device=device);mask=torch.ones(n,1,device=device)
   with torch.no_grad():old=critic.get_values(data[idx],rnn,mask)[0]
   metrics.append(strict_critic_step(critic,normal,(data[idx],rnn,old,returns[idx],mask)));updates+=1
  assert student.hashes()==actor_hashes
  state=dict(student_seed=seed,manifest_sha256=sha(HERE/'manifest.json'),actor_hashes=actor_hashes,actors=student.state(),critic=copy.deepcopy(critic.critic.state_dict()),normalizer=copy.deepcopy(normal.state_dict()),critic_optimizer=critic.critic_optimizer.state_dict(),shuffle_rng=rng.get_state(),epoch=epoch+1,critic_steps=updates,data_sha256={str(p.relative_to(HERE)):sha(p) for p in paths})
  save(resume,state);append(folder/'fit.jsonl',dict(epoch=epoch+1,critic_steps=len(metrics),loss=float(np.mean([x['loss'] for x in metrics])),gradient_norm=float(np.mean([x['gradient_norm'] for x in metrics]))));record_cost('critic_warmup_fit',f'{seed}/epoch_{epoch+1}',physical_steps=0,critic_updates=len(metrics))
 save(folder/'initialization.pt',state);write(folder/'status.json',dict(state='complete',epochs=20,critic_updates=updates,actor_hashes_unchanged=True,checkpoint_sha256=sha(folder/'initialization.pt')))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--student',type=int,required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args();guard();warmup(a.student,a.device)
