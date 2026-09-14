"""Static capacity on archived quality-task states; no resource replacement rollout."""
from analysis_support import *

def run():
 guard();env=ws.make_env('fixed_2');tab=ws.PHYS.tables(env);result={}
 for seed in SEEDS:
  result[str(seed)]={}
  for step in STEPS:
   z=arrays(trace(seed,step));old=np.concatenate([z['initial_q'][None],z['cache_after'][:-1]],0);q=old.sum(-1);load=tab['load'][...,5];qualityok=tab['quality'][...,5]>=tab['req']-1e-9
   uniform=env.p.delta_T*env.p.backhaul_availability*min(env.p.B_uav_sut,env.p.B_sut_sat/3)
   def capacity(b):
    left=np.broadcast_to(b,q.shape).copy();count=np.zeros_like(q)
    for k in range(10):
     take=(q>k)&qualityok&(load<=left+1e-9);left-=np.where(take,load,0);count+=take
    return count
   actual=capacity(z['budgets']);equal=capacity(uniform)
   result[str(seed)][str(step)]=[dict(uav=u+1,cache_blocks_mean=float(q[...,u].mean()),mid_deliverable_blocks_under_actual_budget=float(actual[...,u].mean()),mid_deliverable_blocks_under_static_equal_budget=float(equal[...,u].mean()),static_equal_capacity_difference=float((equal[...,u]-actual[...,u]).mean()),actual_budget_below_four_mid_blocks_fraction=float((z['budgets'][...,u]<4*load[...,u]-1e-9).mean()),same_input_equal_affords_more_mid_blocks_fraction=float((equal[...,u]>actual[...,u]).mean()),same_input_equal_affords_fewer_mid_blocks_fraction=float((equal[...,u]<actual[...,u]).mean())) for u in range(3)]
 assert env.step_index==0
 dump(HERE/'report/budget_capacity.json',result)
if __name__=='__main__':run()
