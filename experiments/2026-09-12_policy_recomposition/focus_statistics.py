"""All-13-scenario quality-slot focus; do not confuse this with fixed_2."""
def quality_focus(physical,parents,scenarios):
 output={'scope':'all slots whose actual current instruction is quality, across all 13 complete scenarios','denominator':'quality UAV decision slots, includes no-delivery slots','by_parent':{}}
 for seed in parents:
  methods={}
  for method in ('C0','C1','C2','C3'):
   rows=[physical[f'{seed}/{method}/{s}']['2'] for s in scenarios if '2' in physical[f'{seed}/{method}/{s}']]
   uavs=[]
   for u in range(3):
    d=[x['by_uav'][u] for x in rows];n=sum(x['decision_slots'] for x in d);total=sum(x['deliveries_total'] for x in d)
    combined=dict(uav=u+1,quality_decision_slots=n,deliveries_total=total,undelivered_slot_fraction=sum(x['undelivered_uav_slots'] for x in d)/n,predicted_psnr_per_delivery=sum(x['predicted_psnr_per_delivery']*x['deliveries_total'] for x in d if x['deliveries_total'])/total,maximum_aoi=max(x['aoi_max_over_run'] for x in d))
    for k in ['resource_share_mean','budget_mean','usage_mean','unused_budget_mean','aoi_mean','mean_slot_max_aoi','fraction_aoi_above4','fraction_aoi_above6','mean_excess_aoi_above4','quality_credit_mean','additive_age_mean_cost','additive_age_tail_cost','nonadditive_uav_max_aoi_cost','additive_resource_cost']:
     combined[k]=sum(x[k]*x['decision_slots'] for x in d)/n
    groups=[]
    for i,template in enumerate(d[0]['exact_equivalence_groups']):
     items=[x['exact_equivalence_groups'][i] for x in d]
     g={'modes':template['modes']}
     for key in ['requested_count','executed_count','budget_infeasible_count','budget_feasible_count','fully_feasible_count','fully_feasible_but_requested_other_count']:g[key]=sum(x[key] for x in items)
     g['executed_fraction_of_all_quality_uav_slots']=g['executed_count']/n
     g['requested_given_fully_feasible_fraction']=1-g['fully_feasible_but_requested_other_count']/g['fully_feasible_count'] if g['fully_feasible_count'] else None
     groups.append(g)
    combined['exact_equivalence_groups']=groups;uavs.append(combined)
   methods[method]=uavs
  output['by_parent'][str(seed)]=methods
 return output
