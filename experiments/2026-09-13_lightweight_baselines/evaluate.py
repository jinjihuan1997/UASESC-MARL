"""Native closed-loop evaluation, complete numeric traces and exact old checks."""
from light_support import *
from reference_policies import policy,policy_hash
from independent_physics import tables,check
import argparse

def old_compare(z,path,length=600):
 previous=arrays(path);fields=previous['fields'].tolist()
 for k in previous:
  if k in ['fields','seeds']:continue
  if k=='trace':actual=np.stack([z['trace'][:,:,FIELDS.index(f)] for f in fields],-1)
  else:actual=z[k]
  np.testing.assert_array_equal(actual[:length],previous[k][:length],err_msg=f'{path}:{k}')
 assert z['seeds'].tolist()==previous['seeds'].tolist()

@torch.inference_mode()
def rollout(method,scene,kind='primary',seeds=None,repeat=None,length=600):
 m=manifest();seeds=list(m['validation'] if seeds is None else seeds);suffix=f'_{repeat}' if repeat is not None else '';folder=HERE/('evaluation' if kind=='primary' else 'supplement' if kind=='supplement' else 'preflight_traces')/method
 path=folder/f'{scene}{suffix}.npz';meta=path.with_suffix('.json');identity=dict(manifest_sha256=sha(HERE/'manifest.json'),method=method,scenario=scene,seeds=seeds,length=length,repeat=repeat,kind=kind,source_sha256={p:sha(HERE/p) for p in ['light_support.py','lightweight_policies.py','reference_policies.py','evaluate.py','independent_physics.py']})
 if meta.exists():
  d=read(meta);assert d['state']=='complete' and d['identity']==identity and d['trace_sha256']==sha(path);return d
 env=make_env(m['scenarios'][scene],seeds);E=len(seeds);obs,_,mask=env.observe();ctrl=policy(method,env);h=policy_hash(ctrl);external=frozen.external_hashes(env);initial={k:arr(getattr(env,k)).copy() for k in ['q','tau','aoi']}
 record=[];steps=0;start=time.perf_counter();error=None
 try:
  for t in range(length):
   pre=obs[:,0].clone();gid=pre[:,23:26].argmax(-1);resource=ctrl.resource(pre,mask[:,0]);before={k:getattr(env,k).clone() for k in ['q','tau','aoi']}
   post,_,am=env.allocate_resources(resource);assert env.step_index==t
   for k,v in before.items():assert torch.equal(v,getattr(env,k))
   acts=[resource]
   for u in range(3):
    assert torch.equal(post[:,u+1,67:70].argmax(-1),gid);acts.append(ctrl.mode(u,post[:,u+1],am[:,u+1]))
   requested=torch.stack([a.argmax(-1) for a in acts[1:]],-1);valid=(env.quality>=env.context()[2][...,None]-1e-9)&(env.load<=env.budget[...,None]+1e-9)&env.q.any(-1)[...,None]
   rf=valid.gather(-1,requested[...,None]).squeeze(-1)
   obs,_,mask,info,values=old.checked_step(env,acts);steps+=E;assert torch.equal(gid,info['gid']);values['instruction_id']=arr(info['gid'])
   row=dict(trace=np.column_stack([values[f] for f in FIELDS]),modes=arr(info['mode']).astype(np.int8),requested_modes=arr(requested).astype(np.int8),resource_fractions=arr(env.beta),budgets=arr(info['budget']),usage=arr(info['usage']),unused_budgets=arr(info['budget']-info['usage']),predicted_quality=arr(info['quality']),quality_requirement=arr(info['req']),raw_resource_action=arr(resource),uav_masks=arr(am[:,1:]).astype(bool),aoi_after=arr(env.aoi).astype(np.uint16),cache_after=arr(env.q).astype(bool),tau_after=arr(env.tau).astype(np.int16),served=arr(info['served']).astype(bool),sut_obs=arr(pre),post_uav_obs=arr(post[:,1:]),adapter_enabled=np.zeros((E,3),bool),truly_feasible=arr(valid),requested_feasible=arr(rf),fallback=arr((info['mode']>=0)&(info['mode']!=requested)))
   record.append({k:v.copy() for k,v in row.items()})
  ctrl.assert_frozen();assert policy_hash(ctrl)==h
  z=Arrays({k:np.stack([row[k] for row in record]) for k in record[0]});z.update(fields=np.array(FIELDS),seeds=np.array(seeds),initial_q=initial['q'],initial_tau=initial['tau'],initial_aoi=initial['aoi'])
  independent=check(z,tables(make_env(m['scenarios'][scene],seeds))) if length==600 else dict(state='prefix_each_slot_original_audit',physical_slots=steps)
  legacy=None
  if method in ['R_equal_single','R_single']:
   legacy=FROZEN/f'evaluation/rules/{method}/{scene}.npz';old_compare(z,legacy,length)
  npz(path,**z);d=dict(state='complete',identity=identity,trace_sha256=sha(path),external_hashes=external,original_checker='PASS_every_slot',independent_audit=independent,score_x100=float(z['trace'][:,:,0].mean()*100),scores_by_environment=(z['trace'][:,:,0].mean(0)*100).tolist(),policy_hash_before=h,policy_hash_after=policy_hash(ctrl),old_trace_match='ALL_OLD_FIELDS_EXACT' if legacy else None,old_trace_sha256=sha(legacy) if legacy else None,environment_creations=ENV_CREATIONS[-2:],new_training_steps=0,new_optimizer_updates=0)
  write(meta,d);print(method,scene,'complete',d['score_x100'],flush=True);return d
 except BaseException as ex:
  error=repr(ex);write(HERE/f'failures/{time.time_ns()}.json',dict(identity=identity,error=error,traceback=traceback.format_exc(),physical_steps=steps))
  if record:npz(HERE/f'failures/partial_{time.time_ns()}.npz',**{k:np.stack([row[k] for row in record]) for k in record[0]})
  raise
 finally:cost(kind,method=method,scene=scene,physical_steps=steps,complete_episodes=E if steps==600*E else 0,partial_physical_steps=0 if steps==600*E else steps,seeds=seeds,seconds=time.perf_counter()-start,error=error,output=str(path.relative_to(HERE)))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--method',required=True);p.add_argument('--supplement',action='store_true');a=p.parse_args();guard();m=verify();assert read(HERE/'preflight.json')['state']=='PASS'
 assert a.method in (['R_equal_single','R_single'] if a.supplement else m['methods'])
 for scene in m['scenarios']:rollout(a.method,scene,'supplement' if a.supplement else 'primary')
