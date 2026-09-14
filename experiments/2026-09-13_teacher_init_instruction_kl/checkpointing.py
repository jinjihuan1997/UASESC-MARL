"""Exact per-episode frozen environment/source reconstruction and RNG state."""
from study import *
from training_checkpoint import cpu_tree,device_tree

def env_state(env):
 assert not env.allocation_pending
 return dict(tensors={k:cpu_tree(v) for k,v in vars(env).items() if isinstance(v,torch.Tensor) or k=='last'},step_index=env.step_index,episode_indices=env.episode_indices,source=[dict(seed=e._base_seed,episode=e._episode_index) for e in env.source.envs])
def restore_env(env,d):
 for e,s in zip(env.source.envs,d['source']):e.seed(s['seed']);e._episode_index=s['episode']-1
 env.reset()
 for k,v in d['tensors'].items():setattr(env,k,device_tree(v,env.device))
 env.step_index=d['step_index'];env.episode_indices=d['episode_indices'];env.allocation_pending=False

def preflight():
 from student_policy import Student
 from tensor_train import TensorRollout,TensorValueNorm
 m=manifest();seeds=m['preflight'][:2];env=make_env(seeds,'preflight');student=Student(m['preflight_model_seed'],env);student.load(HERE/'preflight/prototype.pt');streams=ActionStreams(m['preflight_action_seeds'],'cpu')
 count=0
 @torch.no_grad()
 def step(e,stream):
  nonlocal count
  obs,_,mask=e.observe();actions=[]
  with stream.use(0):a,_=student.forward(0,obs[:,0],mask[:,0],False)
  post,_,am=e.allocate_resources(a);actions=[a]
  for i in range(1,4):
   with stream.use(i):x,_=student.forward(i,post[:,i],am[:,i],False)
   actions.append(x)
  obs,_,mask,info,values=frozen_support.checked_step(e,actions);count+=len(seeds)
  ret=dict(actions=[x.clone() for x in actions],aoi=e.aoi.clone(),q=e.q.clone(),tau=e.tau.clone(),reward=values['common_reward'].copy(),gid=info['gid'].clone(),slot=e.step_index)
  if e.step_index==600:e.reset()
  return ret
 for _ in range(590):step(env,streams)
 saved=dict(environment=env_state(env),action_rng=copy.deepcopy(streams.states));save(HERE/'preflight/cross_termination.pt',saved)
 a=[step(env,streams) for _ in range(25)]
 env2=make_env(seeds,'preflight');d=torch.load(HERE/'preflight/cross_termination.pt',weights_only=False);restore_env(env2,d['environment']);stream2=ActionStreams(m['preflight_action_seeds'],'cpu');stream2.states=d['action_rng']
 b=[step(env2,stream2) for _ in range(25)]
 assert state_equal(a,b) and env.episode_indices==env2.episode_indices==[1,1]
 torch.testing.assert_close(env.noise_us,env2.noise_us,atol=0,rtol=0)
 buffer=TensorRollout(env,3,cfg()['algo_args']['model']);normal=TensorValueNorm(1,device='cpu')
 buffer.values.zero_();buffer.rewards[:, :,0]=torch.tensor([1.,2.,3.])[:,None];buffer.masks.fill_(1);buffer.masks[2]=0
 returns=buffer.returns(normal,.99,.95);np.testing.assert_allclose(arr(returns[:,0,0]),[1+.99*.95*2,2,3],atol=1e-6,rtol=0)
 record_cost('preflight_resume','across_600_slot_terminal',physical_steps=count,complete_episodes=2,replayed_partial_steps=50,new_gradient_steps=0,seeds=seeds)
 write(HERE/'preflight/resume.json',dict(state='PASS',physical_steps=count,resume_exact=True,episode_indices=[1,1],GAE_reset_mask='PASS',same_source_next_episode=True))
if __name__=='__main__':guard();preflight()
