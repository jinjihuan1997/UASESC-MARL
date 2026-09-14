from continuation import *
import gc
def iterate(t,i):
 start=time.perf_counter();t.set_training_update(i);t.collect();middle=time.perf_counter();metrics=t.update();assert all(torch.isfinite(x).all() for x in metrics.values());t.buffer.after_update();end=time.perf_counter()
 row=dict(update=i,physical_steps=4000,actor_optimizer_steps=40,critic_optimizer_steps=10,collect_seconds=middle-start,update_seconds=end-middle,total_seconds=end-start);append(HERE/'preflight_costs.jsonl',row);return row

def main():
 guard();m=verify();os.sched_setaffinity(0,{14});seed=m['seeds'][0];c=config(seed);c['audit_directory']=str(HERE/'preflight_traces');t=Trainer(c,'cpu');new,old=migrate(seed,t);checkpoint.restore(t,new,identity(seed));first=iterate(t,251)
 assert t.env.step_index==200 and (t.buffer.masks==0).sum()==10;after1=checkpoint.capture(t,251,identity(seed));checkpoint.save_checkpoint(HERE/'preflight_checkpoint',after1)
 second=iterate(t,252);target=checkpoint.capture(t,252,identity(seed));del t;gc.collect()
 t=Trainer(c,'cpu');saved,entry=checkpoint.load_checkpoint(HERE/'preflight_checkpoint');checkpoint.restore(t,saved,identity(seed));iterate(t,252);same(target,checkpoint.capture(t,252,identity(seed)));del t;gc.collect()
 # Compare first continuation update to the same payload under the old budget.
 oldconfig=copy.deepcopy(c);oldconfig['algo_args']['train']['num_env_steps']=1000000;oldconfig['training_design']['evaluation_steps']=old['config']['training_design']['evaluation_steps'];oldconfig['main_args']['exp_name']=old['config']['main_args']['exp_name'];t=Trainer(oldconfig,'cpu');oldcopy=checkpoint.cpu_tree(old);oldcopy.update(config=copy.deepcopy(t.config),identity=identity(seed));checkpoint.restore(t,oldcopy,identity(seed));iterate(t,251);reference=checkpoint.capture(t,251,identity(seed))
 for k in after1:
  if k not in ['config','total_updates']:same(after1[k],reference[k],k)
 del t;gc.collect();write(HERE/'preflight.json',dict(state='PASS',full_state_migration_and_next_update_exact=True,checkpoint_replay_across_real_termination=True,physical_steps=16000,actor_optimizer_steps=160,critic_optimizer_steps=40,timing=second,source_semantics='same next update as old completed state; only budget and output metadata differ',reserved_final_test_used=False))
 staged=[stage(seed) for seed in m['seeds']];write(HERE/'migration_results.json',dict(state='PASS',jobs=staged));plan=dict(device='cpu',cpu_ids=[14,15,16],concurrency=3,torch_threads=1,blas_threads=1,measured_seconds_per_4000=second['total_seconds'],estimated_new_training_seconds=2250*second['total_seconds'],other_processes_untouched=True,device_reason='Retain CPU full optimizer/RNG/execution state; prior same-day CPU/GPU measurements also favored CPU')
 write(HERE/'resource_plan.json',plan);verify();print(json.dumps(plan),flush=True)
if __name__=='__main__':
 try:main()
 except BaseException as e:
  d=dict(state='FAIL',error=repr(e),traceback=traceback.format_exc(),utc=stamp());write(HERE/f'failures/{time.time_ns()}.json',d);write(HERE/'preflight.json',d);raise
