"""Offline threshold sensitivity on saved student states; never a controller."""
from study import *
from student_policy import physical_shares
from teacher_interface import Teacher
from audit_trajectories import load_arrays,independent_tables

@torch.inference_mode()
def main():
 m=verify();out={};teacher=Teacher();examples=[]
 for student in m['students']:
  for scene in m['scenarios']:
   path=HERE/f'gate/{student}/D/{scene}.npz';z=load_arrays(path);meta=read(path.with_suffix('.json'));env=make_env(m['gate_dev'],'gate',m['scenarios'][scene]);tab=independent_tables(env)
   obs=z['sut_obs'].reshape(-1,76);ta=torch.cat([teacher.resource(torch.from_numpy(obs[i:i+2048])) for i in range(0,len(obs),2048)]).reshape(600,10,3)
   tb=arr(physical_shares(ta))*60000;sb=z['budgets'];beforeq=np.concatenate([z['initial_q'][None],z['cache_after'][:-1]])
   qvalid=tab['quality']>=tab['req'][...,None]-1e-9;sv=qvalid&(tab['load']<=sb[...,None]+1e-9)&beforeq.any(-1)[...,None];tv=qvalid&(tab['load']<=tb[...,None]+1e-9)&beforeq.any(-1)[...,None]
   modes=z['modes'].astype(int);has=modes>=0;loads=np.take_along_axis(tab['load'],np.maximum(modes,0)[...,None],-1).squeeze(-1)
   # Hold the actually executed mode fixed. This is NOT a rerun or the full teacher reward.
   caps=[]
   for budget in [sb,tb]:
    left=budget.copy();cap=np.zeros_like(modes)
    for j in range(10):
     take=has&(beforeq.sum(-1)>j)&(loads<=left+1e-9);left-=np.where(take,loads,0);cap+=take
    caps.append(cap)
   actual=z['served'].sum(-1);np.testing.assert_array_equal(caps[0][has],actual[has]);gid=z['trace'][:,:,FIELDS.index('instruction_id')].astype(int);parts={}
   for g in range(3):
    take=gid==g;n=int(take.sum())
    if not n:continue
    parts[str(g)]=dict(slots=n,by_uav=[dict(no_delivery_slots=int((~has[:,:,u]&take).sum()),feasible_mode_set_changed_slots=int(((sv[:,:,u]!=tv[:,:,u]).any(-1)&take).sum()),teacher_budget_supports_more_deliveries_same_executed_mode=int(((caps[1][:,:,u]>caps[0][:,:,u])&take&has[:,:,u]).sum()),teacher_budget_supports_fewer_deliveries_same_executed_mode=int(((caps[1][:,:,u]<caps[0][:,:,u])&take&has[:,:,u]).sum()),same_mode_capacity_difference_sum=int((caps[1][:,:,u]-caps[0][:,:,u])[take&has[:,:,u]].sum())) for u in range(3)])
   if scene.startswith('fixed_'):
    candidates=np.argwhere((caps[1]>caps[0])&has)
    if len(candidates):
     t,e,u=map(int,candidates[0]);examples.append(dict(student=student,scene=scene,environment_seed=int(z['seeds'][e]),slot=t,uav=u+1,instruction_id=int(gid[t,e]),executed_mode=int(modes[t,e,u]),predicted_load=float(loads[t,e,u]),student_budget=float(sb[t,e,u]),teacher_budget_at_same_student_SUT_observation=float(tb[t,e,u]),cached_blocks=int(beforeq[t,e,u].sum()),student_deliveries=int(caps[0][t,e,u]),same_mode_capacity_under_teacher_budget=int(caps[1][t,e,u]),causal_closed_loop_effect='NOT_IDENTIFIED'))
   out[f'{student}/{scene}']=parts
 teacher.assert_frozen();record_cost('offline_budget_diagnostic','all_A3_D_gate_saved_states',physical_steps=0,teacher_queries=teacher.queries,forward_example_evaluations=0)
 write(HERE/'report/budget_threshold_diagnostics.json',dict(state='PASS',source_sha256=sha(HERE/'budget_diagnostics.py'),results=out,examples=examples,physical_steps=0,interpretation='Posthoc capacity sensitivity holding the executed mode and cache fixed; not actual teacher closed-loop deliveries, not a training label and not a new controller. No cause-specific reward decomposition is identified.'))

if __name__=='__main__':guard();main()
