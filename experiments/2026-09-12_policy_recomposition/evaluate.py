"""Closed-loop deterministic evaluation and per-slot native/independent checks."""
from support import *
from composition_policy import Composition
import argparse, traceback

def code_identity():return {p:sha(HERE/p) for p in ('support.py','composition_policy.py','evaluate.py')}

@torch.inference_mode()
def rollout(seed,controller,scenario,folder,policy=None):
 m=manifest();folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
 file=folder/f'{scenario}.npz';meta=folder/f'{scenario}.json'
 identity=dict(manifest_sha256=sha(HERE/'manifest.json'),code_sha256=code_identity(),parent=seed,controller=controller,scenario=scenario,device='cpu')
 if meta.exists():
  oldmeta=read(meta);assert oldmeta['identity']==identity and sha(file)==oldmeta['trace_sha256'];return oldmeta
 attempt=f'{time.time_ns()}_{os.getpid()}'
 ledger=HERE/'compute_ledger'/f'{attempt}.json';done_slots=0;started=time.perf_counter()
 entry=dict(attempt=attempt,output=str(file.relative_to(HERE)),kind='primary' if folder.is_relative_to(HERE/'evaluation') else 'preflight',parent=seed,controller=controller,scenario=scenario,episodes=0,physical_steps=0,batch_size=20,started_utc=stamp(),pid=os.getpid(),command=sys.argv,state='running')
 write(ledger,entry)
 try:
  env=make_env(seed,m['scenarios'][scenario]);obs,_,masks=env.observe();external=frozen.external_hashes(env)
  if policy is None:policy=Composition(seed,env,controller)
  else:assert policy.controller==controller and policy.seed==seed
  initial={k:arr(getattr(env,k)).copy() for k in ('aoi','q','tau')}
  keys=['trace','modes','requested_modes','resource_fractions','budgets','usage','unused_budgets','aoi_after','cache_after','tau_after','served','predicted_quality','quality_requirement','adapter_enabled','uav_masks','raw_resource_action','sut_obs','post_uav_obs','resource_source','policy_source','budget_feasible','truly_feasible','requested_feasible','fallback']
  records={k:[] for k in keys};tabs={k:[] for k in ('quality','load','req')}
  for slot in range(600):
   sut_obs=obs[:,0].clone();gid=policy.gid(sut_obs,23)
   equal=frozen.simple_rule_actions(env,'R_equal_instruction',obs)[0]
   resource,rs=policy.resource(sut_obs,masks[:,0],equal)
   before={k:getattr(env,k).clone() for k in ('q','aoi','tau')};assert env.step_index==slot and not env.allocation_pending
   post,_,postmask=env.allocate_resources(resource)
   assert env.step_index==slot and env.allocation_pending
   for k,v in before.items():assert torch.equal(v,getattr(env,k)),k
   modes=[];sources=[]
   for i in range(3):
    assert torch.equal(policy.gid(post[:,i+1],67),gid)
    action,source=policy.mode(i,post[:,i+1],postmask[:,i+1]);modes.append(action);sources.append(source)
   acts=[resource]+modes
   bf=env.load<=env.budget[...,None]+1e-9
   valid=bf&(env.quality>=env.context()[2][...,None]-1e-9)&env.q.any(-1)[...,None]
   requested=torch.stack([a.argmax(-1) for a in modes],-1)
   rf=valid.gather(-1,requested[...,None]).squeeze(-1)
   for k,v in [('quality',env.quality),('load',env.load),('req',env.context()[2])]:tabs[k].append(arr(v).copy())
   obs,_,masks,info,values=old.checked_step(env,acts);done_slots+=1
   assert torch.equal(info['gid'],gid)
   values['instruction_id']=arr(info['gid'])
   records['trace'].append(np.column_stack([values[f] for f in FIELDS]))
   additions=dict(modes=arr(info['mode']).astype(np.int8),requested_modes=arr(requested).astype(np.int8),resource_fractions=arr(env.beta),budgets=arr(info['budget']),usage=arr(info['usage']),unused_budgets=arr(info['budget']-info['usage']),predicted_quality=arr(info['quality']),quality_requirement=arr(info['req']),raw_resource_action=arr(resource),adapter_enabled=np.column_stack([arr(s!=0) for s in sources]),uav_masks=arr(postmask[:,1:]).astype(bool),aoi_after=arr(env.aoi).astype(np.uint16),cache_after=arr(env.q).astype(bool),tau_after=arr(env.tau).astype(np.int16),served=arr(info['served']).astype(bool),sut_obs=arr(sut_obs),post_uav_obs=arr(post[:,1:]),resource_source=arr(rs),policy_source=np.column_stack([arr(s) for s in sources]),budget_feasible=arr(bf),truly_feasible=arr(valid),requested_feasible=arr(rf),fallback=arr((info['mode']>=0)&(info['mode']!=requested)))
   for k,v in additions.items():records[k].append(v.copy())
   if done_slots%50==0:write(ledger,{**entry,'physical_steps':done_slots*20})
  policy.assert_frozen()
  records={k:np.stack(v) for k,v in records.items()};records.update(fields=np.asarray(FIELDS),seeds=np.asarray(m['validation']),initial_aoi=initial['aoi'],initial_q=initial['q'],initial_tau=initial['tau'])
  save_npz(file,records)
  with np.load(file,allow_pickle=False) as z:check=prior_audit_module().physical_audit(z,{k:np.stack(v) for k,v in tabs.items()},cfg_for(seed))
  scores=records['trace'][:,:,0].mean(0)*100
  result=dict(identity=identity,external_hashes=external,trace_sha256=sha(file),scores_by_environment=scores.tolist(),score_x100=float(scores.mean()),episodes=20,slots_each=600,original_checker='PASS_every_slot',independent_audit=check,network_hashes=policy.hashes(),model_index=manifest()['models'][str(seed)],source_ids=dict(resource=m['resource_source_ids'],uav=m['model_source_ids']),forbidden_training_calls=FORBIDDEN_CALLS,environment_creations=ENV_CREATIONS[-1:])
  write(meta,result)
  elapsed=time.perf_counter()-started
  write(ledger,{**entry,'state':'complete','episodes':20,'physical_steps':12000,'elapsed_seconds':elapsed,'physical_steps_per_second':12000/elapsed,'finished_utc':stamp()})
  print(json.dumps(dict(controller=controller,parent=seed,scenario=scenario,score_x100=float(scores.mean()),seconds=elapsed)),flush=True)
  return result
 except BaseException as error:
  write(ledger,{**entry,'state':'failed','physical_steps':done_slots*20,'episodes':20 if done_slots==600 else 0,'error':repr(error),'traceback':traceback.format_exc()});raise

def main():
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--controller',required=True);p.add_argument('--scene');p.add_argument('--preflight',action='store_true');args=p.parse_args()
 guard();m=verify_inputs()
 if not args.preflight:assert read(HERE/'preflight.json')['state']=='PASS' and (HERE/'execution_seal.json').exists()
 assert args.seed in m['parents'] and args.controller in m['controllers']
 scenes=[args.scene] if args.scene else list(m['scenarios'])
 folder=HERE/('preflight_traces' if args.preflight else 'evaluation')/f'seed_{args.seed}'/args.controller
 for scene in scenes:assert scene in m['scenarios'];rollout(args.seed,args.controller,scene,folder)
 write(folder/'status.json',dict(state='complete',manifest_sha256=sha(HERE/'manifest.json'),completed_scenarios=scenes,new_training_steps=0,new_optimizer_updates=0))
if __name__=='__main__':main()
