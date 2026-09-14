from diag_support import *
import argparse

def gae(reward,values):
 r=torch.as_tensor(reward,dtype=torch.float32);v=torch.as_tensor(values,dtype=torch.float32);out=torch.empty_like(r)
 for lo,hi in [(0,400),(400,600)]:
  acc=torch.zeros_like(r[0])
  for t in range(hi-1,lo-1,-1):
   live=float(t<599);delta=r[t]+.99*v[t+1]*live-v[t];acc=delta+.99*.95*live*acc;out[t]=acc
 return out.numpy()

@torch.inference_mode()
def rollout(seed,steps,scene,repeat):
 m=manifest();det=repeat is None;env=ws.make_env(scene);obs,state,mask=env.observe();policy=Policy(seed,steps,env);action_seed=None if det else m['action_seeds'][repeat]
 if action_seed is not None:torch.manual_seed(action_seed)
 out=HERE/f'evaluation/seed_{seed}/steps_{steps}/{scene}'/('det.npz' if det else f's{repeat:02d}.npz')
 identity=dict(manifest_sha256=sha(HERE/'manifest.json'),seed=seed,steps=steps,scene=scene,repeat=repeat,action_seed=action_seed,source_sha256=sha(Path(__file__)),model_hashes=policy.initial_hashes)
 if out.with_suffix('.json').exists():
  old=read(out.with_suffix('.json'));assert old['identity']==identity and old['trace_sha256']==sha(out);return
 initial={k:arr(getattr(env,k)).copy() for k in ['q','tau','aoi']};external=base.frozen.external_hashes(env);rows=[];value=[];start=time.perf_counter();count=0;error=None
 try:
  for t in range(600):
   pre=obs[:,0].clone();prestate=state[:,0].clone();gid=pre[:,23:26].argmax(-1);value.append(arr(policy.value(prestate)));snr=arr(env.snr_db).copy();resource,lp0=policy.call(0,pre,mask[:,0],det)
   before={k:getattr(env,k).clone() for k in ['q','tau','aoi']};post,_,am=env.allocate_resources(resource);assert env.step_index==t
   for k,v in before.items():assert torch.equal(v,getattr(env,k))
   acts=[resource];logp=[lp0.sum(-1)]
   for u in range(3):a,lp=policy.call(u+1,post[:,u+1],am[:,u+1],det);acts.append(a);logp.append(lp.sum(-1))
   requested=torch.stack([a.argmax(-1) for a in acts[1:]],-1)
   _,_,reward,done,info,_=env.commit_modes(acts[1:],auto_reset=False);values=base.frozen.check_tensor(env,info,t,{k:arr(v) for k,v in before.items()});values['instruction_id']=arr(info['gid']);np.testing.assert_allclose(arr(reward[:,0,0]),values['common_reward'],atol=1e-6,rtol=1e-6);obs,state,mask=env.observe();count+=20;assert torch.equal(gid,info['gid'])
   rows.append(dict(trace=np.column_stack([values[f] for f in FIELDS]),modes=arr(info['mode']).astype(np.int8),requested_modes=arr(requested).astype(np.int8),resource_fractions=arr(env.beta).copy(),budgets=arr(info['budget']).copy(),usage=arr(info['usage']).copy(),unused_budgets=arr(info['budget']-info['usage']),predicted_quality=arr(info['quality']).copy(),quality_requirement=arr(info['req']).copy(),raw_resource_action=arr(resource).copy(),uav_masks=arr(am[:,1:]).astype(bool),aoi_after=arr(env.aoi).astype(np.uint16),cache_after=arr(env.q).astype(bool),tau_after=arr(env.tau).astype(np.int16),served=arr(info['served']).astype(bool),sut_obs=arr(pre).copy(),post_uav_obs=arr(post[:,1:]).copy(),critic_state=arr(prestate).copy(),old_logp=arr(torch.stack(logp,-1)),snr_db=snr))
  value.append(arr(policy.value(state[:,0])));policy.assert_frozen();z=base.Arrays({k:np.stack([r[k] for r in rows]) for k in rows[0]});z.update(fields=np.array(FIELDS),seeds=np.array(m['validation']),initial_q=initial['q'],initial_aoi=initial['aoi'],initial_tau=initial['tau'],critic_values=np.stack(value));z['gae_raw']=gae(z['trace'][:,:,0],z['critic_values'])
  audited=ws.independent_check(z,ws.PHYS.tables(env));npz(out,**z);write(out.with_suffix('.json'),dict(state='complete',identity=identity,external_hashes=external,trace_sha256=sha(out),independent=audited,every_slot_original_checker=True,models_unchanged=True,new_optimizer_updates=0));print(seed,steps,scene,repeat,'PASS',flush=True)
 except BaseException as ex:
  error=repr(ex);write(HERE/f'failures/{time.time_ns()}.json',dict(error=error,identity=identity,traceback=traceback.format_exc(),physical_steps=count));raise
 finally:append(HERE/f'costs/{os.getpid()}.jsonl',dict(kind='evaluation',seed=seed,steps=steps,scene=scene,repeat=repeat,physical_steps=count,complete_episodes=20 if count==12000 else 0,seconds=time.perf_counter()-start,error=error))

def main():
 a=argparse.ArgumentParser();a.add_argument('--seed',type=int,required=True);a.add_argument('--cpu',type=int,default=0);x=a.parse_args();os.sched_setaffinity(0,{x.cpu});guard();m=verify();assert read(HERE/'preflight.json')['state']=='PASS'
 for steps in [1000000,6000000]:
  scenes=['fixed_2'] if steps==1000000 else m['sampling_scenes']
  for scene in scenes:
   if steps==6000000:rollout(x.seed,steps,scene,None)
   for repeat in range(16 if scene=='fixed_2' else 8):rollout(x.seed,steps,scene,repeat)
if __name__=='__main__':main()
