"""Common-state, complete decision-path timing; no physical transition or fit."""
from light_support import *
from reference_policies import policy,policy_hash,tensors
import subprocess

def fixtures(batch):
 m=manifest();out=[]
 for scene in m['scenarios']:
  p=REPAIR/f'evaluation/rules/R_instruction/{scene}.npz';z=arrays(p);meta=read(p.with_suffix('.json'));assert sha(p)==meta['trace_sha256']
  for slot in m['benchmark']['fixtures']['slots']:
   row=len(out)%20;indices=np.arange(20) if batch==20 else np.array([row]);seeds=[m['validation'][int(i)] for i in indices];env=make_env(m['scenarios'][scene],seeds)
   obs,_,mask=state_before(env,z,slot,indices)
   out.append(dict(env=env,sut_obs=obs[:,0].clone(),sut_mask=mask[:,0].clone(),beta=env.beta.clone(),scene=scene,slot=slot,seeds=seeds,source_path=str(p.relative_to(ROOT)),source_sha256=sha(p)))
 return out

def storage(ctrl,method,parent):
 unique={};parameters=0
 for t in tensors(ctrl):
  if isinstance(t,torch.Tensor):
   s=t.untyped_storage();unique[('torch',s.data_ptr())]=s.nbytes()
   if isinstance(t,torch.nn.Parameter):parameters+=t.numel()
  else:unique[('numpy',t.__array_interface__['data'][0])]=t.nbytes
 files=[];needs_fit=False;offline='No offline fitting';resource_candidates=0;uav_utility=0;attribute_candidates=0
 if method.startswith('greedy_'):
  files=[GREEDY/method.removeprefix('greedy_')/'predictor.npz'];needs_fit=True;offline='Existing fitted SUT HGB predictor; no refitting this run';resource_candidates=3;uav_utility=16 if method.endswith('16') else 3
 elif method.startswith('G_'):uav_utility=16
 elif method in ['R_equal_minload','R_equal_maxquality']:attribute_candidates=16
 elif method in ['original_rl','C1','C3']:
  parentdir=old.model_dir(parent);files=[parentdir/f'actor_agent{i}.pt' for i in range(4)];needs_fit=True;offline='Existing parent joint 1000000 physical training steps'
  if method in ['C1','C3']:
   files += [REPAIR/f'jobs/seed_{parent}/{arm}/milestones/steps_1000000/adapter_agent{i}.pt' for arm in ['residual_all','residual_quality'] for i in range(1,4)];offline+=' plus two pre-existing 1000000-step adapter experiments (3000000 cumulative steps per parent)'
 else:offline='No network fitting; original fixed rule/mode mapping, including its historical calibration, is preserved'
 if method.startswith('greedy_'):sutfields=list(range(27));uavfields=list(range(63))+[67,68,69,74]
 elif method.startswith('G_'):sutfields=[13,14,15] if method=='G_urgency_local16' else [];uavfields=list(range(63))+[67,68,69,74]
 elif method in ['R_equal_minload','R_equal_maxquality']:sutfields=[];uavfields=[0]+list(range(1,11))+list(range(31,63))+[74]
 elif method in ['original_rl','C1','C3']:sutfields=list(range(76));uavfields=list(range(76))
 else:sutfields=list(range(13,16))+list(range(23,26));uavfields=[]
 return dict(offline_training_or_fit_dependency=needs_fit,offline_description=offline,new_training_or_fitting=0,pretrained_predictor=method.startswith('greedy_'),dependency_files={str(p.relative_to(ROOT)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in files},dependency_file_bytes=sum(p.stat().st_size for p in files),unique_persistent_tensor_array_bytes=sum(unique.values()),neural_parameter_elements=parameters,resource_candidate_score_evaluations_per_environment_slot=resource_candidates,UAV_utility_candidate_evaluations_each=uav_utility,UAV_attribute_candidates_each=attribute_candidates,RL_output_categories_each=16 if method in ['original_rl','C1','C3'] else 0,simple_rule_requested_modes_each=1 if method in frozen.RULE_METHODS else 0,policy_used_SUT_fields=sutfields,policy_used_UAV_fields=uavfields,common_interface=dict(float32_observations=4*76,float32_masks=4*16,uncompressed_bytes=4*(4*76+4*16)),signal_compression_implemented=False,storage_scope='Unique persistent owned tensor/array storage and dependency files. Excludes shared Python/Torch runtime, common profile/environment/input fixture storage, temporary workspaces and Python object overhead; not a process RSS claim.')

def stat(ns,batch=1):
 ns=np.asarray(ns,dtype=np.float64);return dict(P50_ms=float(np.quantile(ns,.5)/1e6),P95_ms=float(np.quantile(ns,.95)/1e6),mean_ms=float(ns.mean()/1e6),sample_decisions_per_second=float(batch*1e9/ns.mean()),measured_batches=len(ns))

@torch.inference_mode()
def main():
 m=verify();assert read(HERE/'preflight.json')['state']=='PASS';b=m['benchmark'];source=sha(HERE/'benchmark_latency.py');seal=dict(manifest_sha256=sha(HERE/'manifest.json'),source_sha256=source,protocol=b)
 if (HERE/'benchmark_execution_seal.json').exists():assert read(HERE/'benchmark_execution_seal.json')==seal
 else:write(HERE/'benchmark_execution_seal.json',seal)
 order=list(range(len(b['instances'])));np.random.default_rng(b['order_seed']).shuffle(order);write(HERE/'benchmark_resource_start.json',dict(utc=stamp(),load=os.getloadavg(),affinity=sorted(os.sched_getaffinity(0)),torch_threads=torch.get_num_threads(),order=order,cpu=subprocess.run(['lscpu'],capture_output=True,text=True).stdout))
 results={};storages={};input_index={}
 for batch in b['batch_sizes']:
  pool=fixtures(batch);input_index[str(batch)]=[{k:v for k,v in f.items() if k not in ['env','sut_obs','sut_mask','beta']}|dict(sut_obs_sha256=hashlib.sha256(arr(f['sut_obs']).tobytes()).hexdigest()) for f in pool]
  for j in order:
   item=b['instances'][j];method=item['method'];parent=item['parent'];key=f'{parent or "rules"}/{method}';out=HERE/f'latency/batch_{batch}/{parent or "rules"}_{method}.npz';meta=out.with_suffix('.json');identity=dict(**seal,batch_size=batch,method=method,parent=parent)
   if meta.exists():
    d=read(meta);assert d['identity']==identity and d['trace_sha256']==sha(out);results.setdefault(key,{})[str(batch)]=d['timing'];storages[key]=d['storage'];continue
   start=time.perf_counter();ctrl=policy(method,pool[0]['env'],parent);load_seconds=time.perf_counter()-start;h=policy_hash(ctrl);footprint=storage(ctrl,method,parent);times=[];warm_done=measured=0;error=None
   try:
    for iteration in range(b['warmup']+b['repeats']):
     f=pool[iteration%len(pool)];env=f['env'];env.allocation_pending=False;env.beta=f['beta'].clone();env.update_budget()
     begin=time.perf_counter_ns();resource=ctrl.resource(f['sut_obs'],f['sut_mask']);rtime=time.perf_counter_ns();post,_,mask=env.allocate_resources(resource);atime=time.perf_counter_ns();modes=[ctrl.mode(u,post[:,u+1],mask[:,u+1]) for u in range(3)];end=time.perf_counter_ns()
     assert env.step_index==f['slot'] and all(a.shape==(batch,16) for a in modes)
     if iteration<b['warmup']:warm_done+=1
     else:times.append([end-begin,rtime-begin,atime-rtime,end-atime]);measured+=1
    ctrl.assert_frozen();assert policy_hash(ctrl)==h;timing=np.asarray(times,dtype=np.int64);npz(out,nanoseconds=timing,columns=np.array(['total','resource_decision','budget_post_observation_interface','UAV_mode_decision']))
    summary={name:stat(timing[:,col],batch) for col,name in enumerate(['total','resource_decision','budget_post_observation_interface','UAV_mode_decision'])}
    d=dict(state='complete',identity=identity,trace_sha256=sha(out),timing=summary,storage=footprint,model_load_seconds_excluded_and_reported=load_seconds,policy_state_hash=h,policy_hash_after=policy_hash(ctrl),warmup_batches=warm_done,measured_batches=measured,physical_commit_steps=0)
    write(meta,d);results.setdefault(key,{})[str(batch)]=summary;storages[key]=footprint;print(key,'batch',batch,summary['total'],flush=True)
   except BaseException as ex:
    error=repr(ex);write(HERE/f'latency_failures/{time.time_ns()}.json',dict(identity=identity,error=error,traceback=traceback.format_exc(),completed_warmup=warm_done,completed_measured=measured));raise
   finally:cost('latency',method=method,parent=parent,batch_size=batch,warmup_batches=warm_done,measured_batches=measured,sample_decisions=(warm_done+measured)*batch,physical_steps=0,error=error)
 write(HERE/'latency/input_index.json',input_index);assert len(results)==19 and all(set(v)=={'1','20'} for v in results.values()) and not FORBIDDEN
 write(HERE/'report/complexity.json',dict(state='MEASURED',protocol=b,timings=results,storage=storages,common_profile_bytes=(FROZEN/'source/reference/inputs/profile.npz').stat().st_size,timing_raw_files_sha256={str(p.relative_to(HERE)):sha(p) for p in sorted((HERE/'latency').rglob('*.npz'))},input_index_sha256=sha(HERE/'latency/input_index.json'),new_training_steps=0,new_optimizer_updates=0,communication_reduction_claimed=False,scope='CPU single-thread empirical latency at shared saved inputs; not a hard real-time deadline guarantee.'))
if __name__=='__main__':guard();main()
