"""Real short training, full-state replay, reward isolation and device timing."""
from train_support import *
import gc

def same(a,b,path=''):
 assert type(a)==type(b),(path,type(a),type(b))
 if isinstance(a,torch.Tensor):assert torch.equal(a,b),path
 elif isinstance(a,(dict,list,tuple)):
  assert len(a)==len(b),path
  for k in (a if isinstance(a,dict) else range(len(a))):same(a[k],b[k],path+'/'+str(k))
 else:assert a==b,(path,a,b)

def cfg(seed,name):
 c=config(seed);c['audit_directory']=str(HERE/'preflight_traces'/name);return c

def iteration(t,i,check_logp=False):
 t.set_training_update(i);t.synchronize();start=time.perf_counter();t.collect();t.synchronize();mid=time.perf_counter()
 if check_logp:
  with torch.no_grad():
   for n,a in enumerate(t.actors):
    obs,rnn,action,mask,available,active=t.buffer.actor_inputs(n)
    # Match the actual execution batch size; no GEMM shape change in this test.
    lp=torch.cat([a.evaluate_actions(obs[j:j+10],rnn[j:j+10],action[j:j+10],mask[j:j+10],available[j:j+10],active[j:j+10])[0] for j in range(0,4000,10)])
    torch.testing.assert_close(lp,t.buffer.log_probs[n].flatten(0,1),rtol=0,atol=1e-6)
    assert float((torch.exp((lp-t.buffer.log_probs[n].flatten(0,1)).sum(-1))-1).abs().max())<=1e-6
 metrics=t.update();t.synchronize();end=time.perf_counter();assert all(torch.isfinite(v).all() for v in metrics.values())
 t.buffer.after_update();append(HERE/'preflight_costs.jsonl',dict(device=str(t.device),update=i,physical_steps=t.batch,collect_seconds=mid-start,update_seconds=end-mid,includes_logp_check=check_logp))
 return dict(collect_seconds=mid-start,update_seconds=end-mid,total_seconds=end-start,steps_per_second=t.batch/(end-start))

def main():
 guard();m=verify();result=dict(state='RUNNING',checks={},started=stamp());write(HERE/'preflight.json',result)
 seeds=m['seeds'];first=seeds[0]
 for seed in seeds:
  t=Trainer(cfg(seed,f'init_{seed}'),'cpu');result['checks'][f'paired_fresh_initialization_{seed}']=dict(actors=t.initial_actors,critic=t.initial_critic)
  np.testing.assert_array_equal(arr(t.env.reward_weights),m['effective_weights']);del t;gc.collect()
 t=Trainer(cfg(first,'cpu'),'cpu');identity=dict(test='same_boundary_restore',config_sha256=sha(HERE/f'configs/seed_{first}.json'))
 initial=t.initial_actors.copy();critic=t.initial_critic;ext=base.frozen.external_hashes(t.env)
 cpu1=iteration(t,1,True);first_actions=[v.clone() for v in t.buffer.actions];first_rewards=t.buffer.rewards.clone()
 assert all(kernel.network_hash(a.actor)!=h for a,h in zip(t.actors,initial));assert kernel.network_hash(t.critic.critic)!=critic
 cp=checkpoint.save_checkpoint(HERE/'preflight_checkpoint',checkpoint.capture(t,1,identity));state,meta=checkpoint.load_checkpoint(HERE/'preflight_checkpoint')
 cpu2=iteration(t,2);assert (t.buffer.masks==0).sum()==10;assert t.env.step_index==200
 target=checkpoint.capture(t,2,identity);assert int(t.env.audited_physical_steps)==8000
 del t;gc.collect();t=Trainer(cfg(first,'cpu'),'cpu');assert checkpoint.restore(t,state,identity)==1
 iteration(t,2);same(target,checkpoint.capture(t,2,identity));result['checks']['exact_resume_across_600_slot_terminal']='PASS'
 result['checks']['all_actors_and_critic_updated']='PASS';result['checks']['sampling_logp_and_initial_ratio']='PASS';del t;gc.collect()
 # Before any update, reward-only intervention cannot alter sampled actions.
 c=cfg(first,'original_reward_control');c['quality_multiplier']=1.;t=Trainer(c,'cpu');assert base.frozen.external_hashes(t.env)==ext
 t.set_training_update(1);t.collect()
 for a,b in zip(first_actions,t.buffer.actions):assert torch.equal(a,b)
 # Quality reward credit is accumulated as a per-vector-step mean in the original trainer.
 old=t.buffer.rewards;diff=old-first_rewards;assert float(diff.min())>=-1e-7 and float(diff.max())>0
 append(HERE/'preflight_costs.jsonl',dict(device='cpu',physical_steps=4000,kind='unchanged_actions_old_reward_control',optimizer_updates=0))
 result['checks']['reward_only_action_and_exogenous_parity']='PASS';del t;gc.collect()
 cpu=(cpu2['total_seconds']);gpu=None
 if torch.cuda.is_available():
  t=Trainer(cfg(first,'cuda'),'cuda:0');iteration(t,1);gpu_result=iteration(t,2);gpu=gpu_result['total_seconds'];result['checks']['cuda_training_and_physical_audit']='PASS';del t;torch.cuda.empty_cache();gc.collect()
 # Retain original CPU execution to pair fresh initialization exactly; GPU timing is disclosed.
 cores=sorted(os.sched_getaffinity(0));plan=dict(device='cpu',concurrency=3,torch_threads=1,blas_threads=1,cpu_ids=cores[:3],cpu_seconds_per_4000=cpu,gpu_seconds_per_4000=gpu,reason='Same CPU arithmetic and fresh initialization as original joint runs; measured throughput recorded',nominal_serial_seconds_per_seed=250*cpu)
 if gpu is not None and gpu<cpu:plan['note']='GPU is faster in isolated timing; CPU retained for original exact initialization/numerical pairing, three independent jobs run concurrently.'
 write(HERE/'resource_plan.json',plan)
 costs=[json.loads(x) for x in (HERE/'preflight_costs.jsonl').read_text().splitlines()];result.update(state='PASS',finished=stamp(),cpu_timing=cpu2,gpu_seconds_per_4000=gpu,physical_steps=sum(x['physical_steps'] for x in costs),reserved_final_test_used=False,resource_plan=plan)
 verify();write(HERE/'preflight.json',result);print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':
 try:main()
 except BaseException as e:
  d=dict(state='FAIL',error=repr(e),traceback=traceback.format_exc(),utc=stamp());write(HERE/f'failures/preflight_{time.time_ns()}.json',d);write(HERE/'preflight.json',d);raise
