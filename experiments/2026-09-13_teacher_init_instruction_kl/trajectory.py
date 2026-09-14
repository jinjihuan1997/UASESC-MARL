"""Real one-slot closed loop; teacher labels only at the executed resource budget."""
from study import *
from student_policy import Student
from teacher_interface import Teacher

@torch.no_grad()
def run_episode_batch(path,seeds,purpose,schedule=None,student_seed=None,checkpoint=None,deterministic=True,action_seeds=None,labels=False,kind='collection'):
 path=Path(path);meta=path.with_suffix('.json');m=manifest();started=time.perf_counter()
 identity=dict(manifest_sha256=sha(HERE/'manifest.json'),seeds=list(seeds),purpose=purpose,schedule=schedule,student_seed=student_seed,checkpoint_sha256=sha(checkpoint) if checkpoint else None,deterministic=deterministic,action_seeds=action_seeds,labels=labels,code_sha256={p:sha(HERE/p) for p in ['study.py','student_policy.py','teacher_interface.py','trajectory.py']})
 if meta.exists():
  d=read(meta);assert d['identity']==identity and d['state']=='complete' and sha(path)==d['trace_sha256'];return d
 steps=0;error=None;teacher=None;records={};env=None
 try:
  env=make_env(seeds,purpose,schedule);E=len(seeds);obs,_,mask=env.observe();external=frozen.external_hashes(env)
  student=Student(student_seed,env) if student_seed is not None else None
  if student:
   assert checkpoint is not None;student.load(checkpoint);student.set_mode(False);initial_hash=student.hashes()
  teacher=Teacher() if labels or student is None else None
  streams=ActionStreams(action_seeds,'cpu') if not deterministic else None
  initial={k:arr(getattr(env,k)).copy() for k in ('aoi','q','tau')}
  keys=['trace','modes','requested_modes','resource_fractions','budgets','usage','unused_budgets','aoi_after','cache_after','tau_after','served','predicted_quality','quality_requirement','uav_masks','raw_resource_action','sut_obs','post_uav_obs','mode_actions','sut_logp','uav_logp','sut_concentration','uav_probabilities','adapter_enabled']
  if labels:keys+=['teacher_resource','teacher_requested','teacher_effective','teacher_active','teacher_skip_reason']
  records={k:[] for k in keys};table={k:[] for k in ('quality','load','req')}
  for slot in range(600):
   pre=obs.clone();gid=pre[:,0,23:26].argmax(-1)
   ta=teacher.resource(pre[:,0]) if teacher else None
   if student:
    with streams.use(0) if streams else contextlib.nullcontext():resource,slp=student.forward(0,pre[:,0],mask[:,0],deterministic)
    concentration=student.distribution(0,pre[:,0],mask[:,0]).concentration
   else:resource=ta;slp=torch.zeros_like(resource);concentration=torch.zeros_like(resource)
   before={k:getattr(env,k).clone() for k in ('q','aoi','tau')}
   post,_,am=env.allocate_resources(resource)
   assert env.step_index==slot
   for k,v in before.items():assert torch.equal(v,getattr(env,k)),k
   modeacts=[];mlp=[];probabilities=[];tl=[]
   native_valid=(env.quality>=env.context()[2][...,None]-1e-9)&(env.load<=env.budget[...,None]+1e-9)&env.q.any(-1)[...,None]
   for u in range(3):
    assert torch.equal(post[:,u+1,67:70].argmax(-1),gid)
    label=teacher.mode_label(u,post[:,u+1],am[:,u+1]) if teacher else None
    if label:
     requested,effective,active,reason=label
     # Audit-only real quantities never enter the teacher or label decision.
     assert native_valid[:,u].gather(-1,effective[:,None]).squeeze(-1)[active].all()
     executed_raw=torch.where(native_valid[:,u].gather(-1,requested[:,None]).squeeze(-1),requested,native_valid[:,u].long().argmax(-1))
     assert torch.equal(executed_raw[active],effective[active])
     tl.append(label)
    if student:
     with streams.use(u+1) if streams else contextlib.nullcontext():a,lp=student.forward(u+1,post[:,u+1],am[:,u+1],deterministic)
     probs=student.distribution(u+1,post[:,u+1],am[:,u+1]).probs.squeeze(1)
    else:a=torch.nn.functional.one_hot(label[0],16).float();lp=torch.zeros_like(a);probs=a
    modeacts.append(a);mlp.append(lp);probabilities.append(probs)
   for k,v in [('quality',env.quality),('load',env.load),('req',env.context()[2])]:table[k].append(arr(v).copy())
   obs,_,mask,info,values=frozen_support.checked_step(env,[resource]+modeacts);steps+=E
   assert torch.equal(gid,info['gid']);values['instruction_id']=arr(gid)
   add=dict(trace=np.column_stack([values[f] for f in FIELDS]),modes=arr(info['mode']).astype(np.int8),requested_modes=np.column_stack([arr(x.argmax(-1)) for x in modeacts]).astype(np.int8),resource_fractions=arr(env.beta),budgets=arr(info['budget']),usage=arr(info['usage']),unused_budgets=arr(info['budget']-info['usage']),aoi_after=arr(env.aoi).astype(np.uint16),cache_after=arr(env.q).astype(bool),tau_after=arr(env.tau).astype(np.int16),served=arr(info['served']).astype(bool),predicted_quality=arr(info['quality']),quality_requirement=arr(info['req']),uav_masks=arr(am[:,1:]).astype(bool),raw_resource_action=arr(resource),sut_obs=arr(pre[:,0]),post_uav_obs=arr(post[:,1:]),mode_actions=np.stack([arr(a) for a in modeacts],1),sut_logp=arr(slp),uav_logp=np.stack([arr(a) for a in mlp],1),sut_concentration=arr(concentration),uav_probabilities=np.stack([arr(x) for x in probabilities],1),adapter_enabled=np.zeros((E,3),dtype=bool))
   if labels:
    add.update(teacher_resource=arr(ta),teacher_requested=np.column_stack([arr(x[0]) for x in tl]).astype(np.int8),teacher_effective=np.column_stack([arr(x[1]) for x in tl]).astype(np.int8),teacher_active=np.column_stack([arr(x[2]) for x in tl]),teacher_skip_reason=np.column_stack([arr(x[3]) for x in tl]))
   for k,v in add.items():records[k].append(v.copy())
  if student:assert student.hashes()==initial_hash
  if teacher:teacher.assert_frozen()
  data={k:np.stack(v) for k,v in records.items()};data.update(seeds=np.asarray(seeds),fields=np.asarray(FIELDS),initial_aoi=initial['aoi'],initial_q=initial['q'],initial_tau=initial['tau'])
  npz(path,**data)
  with np.load(path,allow_pickle=False) as z:audit=prior_auditor().physical_audit(z,{k:np.stack(v) for k,v in table.items()},cfg())
  result=dict(state='complete',identity=identity,trace_sha256=sha(path),episodes=E,physical_steps=steps,score_x100_by_environment=(data['trace'][:,:,0].mean(0)*100).tolist(),external_hashes=external,initial_actor_hashes=initial_hash if student else None,final_actor_hashes=student.hashes() if student else None,teacher_queries=teacher.queries if teacher else [0]*4,original_checker='PASS_every_slot',independent_executor_reward_audit=audit,student_inference_teacher_queries=0 if student and not labels else None,elapsed_seconds=time.perf_counter()-started)
  write(meta,result);print(json.dumps(dict(path=str(path.relative_to(HERE)),episodes=E,score_x100=float(np.mean(result['score_x100_by_environment'])))),flush=True);return result
 except BaseException as exc:
  error=repr(exc);write(HERE/f'failures/{time.time_ns()}.json',dict(identity=identity,error=error,physical_steps=steps,traceback=traceback.format_exc()))
  if records and len(next(iter(records.values())))>0:
   npz(HERE/f'failures/partial_{time.time_ns()}.npz',**{k:np.stack(v) for k,v in records.items() if v})
  raise
 finally:
  record_cost(kind,str(path.relative_to(HERE)),physical_steps=steps,complete_episodes=len(seeds) if steps==600*len(seeds) else 0,teacher_queries=teacher.queries if teacher else [0]*4,error=error,seconds=time.perf_counter()-started,seeds=list(seeds),purpose=purpose)
