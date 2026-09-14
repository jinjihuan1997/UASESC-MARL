"""Actual native closed-loop evaluation under the one-factor reward change."""
from weight_support import *
import argparse

@torch.inference_mode()
def rollout(method,parent,scene,seeds=None,kind='formal',repeat=0):
 m=manifest();seeds=list(m['validation'] if seeds is None else seeds);indices=[m['validation'].index(s) for s in seeds];E=len(seeds)
 folder=HERE/('evaluation' if kind=='formal' else 'preflight_traces')/str(parent or 'rules')/method
 out=folder/(scene+('' if kind=='formal' else f'_{repeat}')+'.npz');meta=out.with_suffix('.json')
 identity=dict(manifest_sha256=sha(HERE/'manifest.json'),method=method,parent=parent,scene=scene,seeds=seeds,kind=kind,repeat=repeat,source_sha256={p:sha(HERE/p) for p in ['weight_support.py','evaluate.py']})
 if meta.exists():
  d=read(meta);assert d['state']=='complete' and d['identity']==identity and d['trace_sha256']==sha(out);return d
 env=make_env(scene,seeds);obs,_,mask=env.observe();ctrl=Controller(method,env,parent);external=base.frozen.external_hashes(env);initial={k:arr(getattr(env,k)).copy() for k in ['q','tau','aoi']};rows=[];steps=0;error=None;start=time.perf_counter()
 try:
  for t in range(600):
   pre=obs[:,0].clone();gid=pre[:,23:26].argmax(-1);resource=ctrl.resource(pre,mask[:,0]);before={k:getattr(env,k).clone() for k in ['q','tau','aoi']}
   post,_,am=env.allocate_resources(resource);assert env.step_index==t
   for k,v in before.items():assert torch.equal(v,getattr(env,k))
   acts=[resource]+[ctrl.mode(u,post[:,u+1],am[:,u+1]) for u in range(3)];requested=torch.stack([a.argmax(-1) for a in acts[1:]],-1)
   obs,_,mask,info,values=base.old.checked_step(env,acts);steps+=E;assert torch.equal(gid,info['gid']);values['instruction_id']=arr(info['gid'])
   rows.append(dict(trace=np.column_stack([values[f] for f in FIELDS]),modes=arr(info['mode']).astype(np.int8),requested_modes=arr(requested).astype(np.int8),resource_fractions=arr(env.beta).copy(),budgets=arr(info['budget']).copy(),usage=arr(info['usage']).copy(),unused_budgets=arr(info['budget']-info['usage']),predicted_quality=arr(info['quality']).copy(),quality_requirement=arr(info['req']).copy(),raw_resource_action=arr(resource).copy(),uav_masks=arr(am[:,1:]).astype(bool),aoi_after=arr(env.aoi).astype(np.uint16),cache_after=arr(env.q).astype(bool),tau_after=arr(env.tau).astype(np.int16),served=arr(info['served']).astype(bool),sut_obs=arr(pre).copy(),post_uav_obs=arr(post[:,1:]).copy()))
  ctrl.assert_frozen();z=Arrays({k:np.stack([r[k] for r in rows]) for k in rows[0]});z.update(fields=np.array(FIELDS),seeds=np.array(seeds),initial_q=initial['q'],initial_tau=initial['tau'],initial_aoi=initial['aoi'])
  tab=PHYS.tables(make_env(scene,seeds));independent=independent_check(z,tab)
  legacy=old_physical_equal(z,ctrl.origin,parent,scene,indices) if not ctrl.updated else None
  npz(out,**z);d=dict(state='complete',identity=identity,trace_sha256=sha(out),policy_hash=ctrl.initial_hash,policy_unchanged=True,external_hashes=external,original_checker='PASS_every_slot',independent=independent,old_physical_comparison=legacy,scores_by_environment=(z['trace'][:,:,0].mean(0)*100).tolist(),old_objective_on_this_trajectory=((z['trace'][:,:,0]+z['trace'][:,:,FIELDS.index('quality_credit')]).mean(0)*100).tolist(),environment_creations=CREATIONS[-2:],new_training_steps=0,new_optimizer_updates=0)
  write(meta,d);print(method,parent,scene,'PASS',np.mean(d['scores_by_environment']),flush=True);return d
 except BaseException as ex:
  error=repr(ex);write(HERE/f'failures/{time.time_ns()}.json',dict(identity=identity,error=error,traceback=traceback.format_exc(),steps=steps));raise
 finally:cost(kind,method=method,parent=parent,scene=scene,seeds=seeds,physical_steps=steps,complete_episodes=E if steps==600*E else 0,seconds=time.perf_counter()-start,error=error,output=str(out.relative_to(HERE)))

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--method',required=True);a.add_argument('--parent',type=int);args=a.parse_args();guard();m=verify();assert read(HERE/'preflight.json')['state']=='PASS';assert dict(method=args.method,parent=args.parent) in m['instances']
 for scene in m['scenarios']:rollout(args.method,args.parent,scene)
