"""Fixed milestone evaluations under the same half-quality objective."""
from train_support import *
from harl.models.policy_models.stochastic_policy import StochasticPolicy
import argparse

class Policy:
 def __init__(self,seed,steps,env):
  cfg=config(seed);args={**cfg['algo_args']['model'],**cfg['algo_args']['algo']};self.env=env
  folder=HERE/f'jobs/seed_{seed}/milestones/steps_{steps}';status=read(folder/'status.json')
  assert status['state']=='complete' and status['completed_steps']==steps
  self.networks=[]
  for i in range(4):
   path=folder/f'actor_agent{i}.pt';assert sha(path)==status['checkpoint_hashes'][path.name]
   net=StochasticPolicy(args,env.observation_space[i],env.action_space[i],env.device)
   net.load_state_dict(torch.load(path,map_location=env.device,weights_only=True),strict=True);net.eval().requires_grad_(False)
   assert kernel.network_hash(net)==status['actor_hashes'][i];self.networks.append(net)
  self.hashes=status['actor_hashes'];self.assert_frozen()
 def call(self,i,obs,mask):return self.networks[i](obs,torch.zeros((len(obs),1,256),device=self.env.device),torch.ones((len(obs),1),device=self.env.device),mask,deterministic=True)[0]
 def assert_frozen(self):
  assert [kernel.network_hash(n) for n in self.networks]==self.hashes
  assert not any(p.requires_grad for n in self.networks for p in n.parameters());assert not any(q.training for n in self.networks for q in n.modules())

@torch.inference_mode()
def rollout(seed,steps,scene):
 m=manifest();seeds=m['validation'];env=ws.make_env(scene,seeds);obs,_,mask=env.observe();ctrl=Policy(seed,steps,env)
 folder=HERE/f'evaluation/seed_{seed}/steps_{steps}';out=folder/f'{scene}.npz';meta=out.with_suffix('.json')
 identity=dict(manifest_sha256=sha(HERE/'manifest.json'),source_sha256=sha(Path(__file__)),seed=seed,steps=steps,scene=scene,seeds=seeds,actor_hashes=ctrl.hashes,deterministic_both_sides=True)
 if meta.exists():
  old=read(meta);assert old['state']=='complete' and old['identity']==identity and old['trace_sha256']==sha(out);return old
 initial={k:arr(getattr(env,k)).copy() for k in ['q','tau','aoi']};external=base.frozen.external_hashes(env);rows=[];count=0;start=time.perf_counter();error=None
 try:
  for t in range(600):
   pre=obs[:,0].clone();gid=pre[:,23:26].argmax(-1);resource=ctrl.call(0,pre,mask[:,0]);before={k:getattr(env,k).clone() for k in ['q','tau','aoi']}
   post,_,am=env.allocate_resources(resource);assert env.step_index==t
   for k,v in before.items():assert torch.equal(v,getattr(env,k))
   acts=[resource]+[ctrl.call(u+1,post[:,u+1],am[:,u+1]) for u in range(3)];requested=torch.stack([a.argmax(-1) for a in acts[1:]],-1)
   obs,_,mask,info,values=base.old.checked_step(env,acts);count+=20;assert torch.equal(gid,info['gid']);values['instruction_id']=arr(info['gid'])
   rows.append(dict(trace=np.column_stack([values[f] for f in FIELDS]),modes=arr(info['mode']).astype(np.int8),requested_modes=arr(requested).astype(np.int8),resource_fractions=arr(env.beta).copy(),budgets=arr(info['budget']).copy(),usage=arr(info['usage']).copy(),unused_budgets=arr(info['budget']-info['usage']),predicted_quality=arr(info['quality']).copy(),quality_requirement=arr(info['req']).copy(),raw_resource_action=arr(resource).copy(),uav_masks=arr(am[:,1:]).astype(bool),aoi_after=arr(env.aoi).astype(np.uint16),cache_after=arr(env.q).astype(bool),tau_after=arr(env.tau).astype(np.int16),served=arr(info['served']).astype(bool),sut_obs=arr(pre).copy(),post_uav_obs=arr(post[:,1:]).copy()))
  ctrl.assert_frozen();z=base.Arrays({k:np.stack([r[k] for r in rows]) for k in rows[0]});z.update(fields=np.array(FIELDS),seeds=np.array(seeds),initial_q=initial['q'],initial_aoi=initial['aoi'],initial_tau=initial['tau'])
  audited=ws.independent_check(z,ws.PHYS.tables(env));npz(out,**z)
  result=dict(state='complete',identity=identity,trace_sha256=sha(out),external_hashes=external,original_checker='PASS_every_slot',independent=audited,policy_unchanged=True,scores_by_environment=(z['trace'][:,:,0].mean(0)*100).tolist())
  write(meta,result);print(seed,steps,scene,'PASS',np.mean(result['scores_by_environment']),flush=True);return result
 except BaseException as e:
  error=repr(e);write(HERE/f'failures/evaluation_{time.time_ns()}.json',dict(identity=identity,error=error,traceback=traceback.format_exc(),physical_steps=count));raise
 finally:append(HERE/f'evaluation_costs/{seed}.jsonl',dict(seed=seed,steps=steps,scene=scene,physical_steps=count,complete_episodes=20 if count==12000 else 0,seconds=time.perf_counter()-start,error=error))

def main():
 a=argparse.ArgumentParser();a.add_argument('--seed',type=int,required=True);a.add_argument('--cpu',type=int);a.add_argument('--steps',type=int);x=a.parse_args()
 if x.cpu is not None:os.sched_setaffinity(0,{x.cpu})
 guard();m=verify();assert x.seed in m['seeds'];assert read(HERE/'preflight.json')['state']=='PASS'
 assert read(HERE/f'jobs/seed_{x.seed}/status.json')['state']=='complete'
 for step in ([x.steps] if x.steps else [200000,400000,600000,800000,1000000]):
  assert step in [200000,400000,600000,800000,1000000]
  for scene in (m['scenarios'] if step==1000000 else ['fixed_0','fixed_1','fixed_2']):rollout(x.seed,step,scene)
if __name__=='__main__':main()
