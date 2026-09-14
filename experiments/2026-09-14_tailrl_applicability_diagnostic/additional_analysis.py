"""Safety failure attribution, paired action patterns, and auditable weight accounting."""
from diag_support import *
from controlled_stats import get_metrics, successes, ci, tail_matrix, normalized, weight_summary

def chosen_diff(a,order):
 n=max(1,math.ceil(len(order)*.1));e=np.arange(order.shape[1])[None,:]
 return a[order[-n:],e].mean(0)-a[order[len(order)//2-1:len(order)//2+1],e].mean(0)

def main():
 guard();m=manifest();boot=np.random.default_rng(m['bootstrap_seed']).integers(0,20,(4000,20));results={}
 for seed in m['parents']:
  for steps in [1000000,6000000]:
   for scene in (['fixed_2'] if steps==1000000 else m['sampling_scenes']):
    zs,x,d,ref,w,dw=get_metrics(seed,steps,scene);N=len(zs);s=successes(x,d,.005);good=s['feasible'];failure={};order=np.argsort(x['quality'],axis=0,kind='stable');p={}
    for k in ['mean_aoi','max_aoi','age_tail_cost','resource']:
     failure[k]=dict(failing_feasible_episodes=int((good&(x[k]>d[k]*1.05+1e-12)).sum()),all_feasible_episodes=int(good.sum()),mean_delta_on_feasible=float((x[k]-d[k])[good].mean()) if good.any() else None,mean_relative_change_on_feasible=float((x[k]/d[k]-1)[good].mean()) if good.any() else None)
    for k in ['quality','psnr','mean_aoi','max_aoi','resource','deliveries']:
     diff=chosen_diff(x[k],order);p[k]=ci(diff,boot)
    groups=m['equivalence_groups'];counts=x['requested_mode_counts'];ec=np.stack([counts[:,:,:,group].sum(-1) for group in groups],-1);diff=chosen_diff(ec,order);ratios=x['resource_fractions'];shares=chosen_diff(ratios,order)
    equiv=[dict(group=g,by_uav=[ci(diff[:,u,j],boot) for u in range(3)]) for j,g in enumerate(groups)]
    split=[]
    for h in range(2):
     sl=slice(h*N//2,(h+1)*N//2);o=np.argsort(x['quality'][sl],axis=0,kind='stable');split.append(dict(feasible_probability=ci(good[sl],boot),safe_probability=ci(s['safe'][sl],boot),equivalent_mode_top_minus_median=chosen_diff(ec[sl],o).mean(0).tolist(),resource_top_minus_median=chosen_diff(x['resource'][sl],o).mean().item()))
    # Compare actual time-level positive weights, as well as trajectory-mean proxies.
    gae=np.stack([z['gae_raw'] for z in zs]);time_mask=np.stack([z['trace'][:,:,FIELDS.index('instruction_id')]==2 for z in zs]);gn=normalized(gae);tail=normalized(tail_matrix(x['quality']));ta=np.broadcast_to(tail[:,None,:],gae.shape);success_time=np.broadcast_to(good[:,None,:],gae.shape)[time_mask]
    per_slot={k:weight_summary(a[time_mask],success_time) for k,a in [('frozen_GAE',gn),('tail_quality',ta)]}
    for v in per_slot.values():v.pop('positive_effective_sample_size');v['count_unit']='correlated decision slots, not independent statistical samples'
    gae_average=(gn*time_mask).sum(1)/time_mask.sum(1);conditional_GAE=gae_average-gae_average.mean(0)
    success_info=dict(feasible_count=int(good.sum()),safe_count=int(s['safe'].sum()),environment_success_counts=good.sum(0).tolist(),quality_delta_mean=float((x['quality']-d['quality'])[good].mean()) if good.any() else None,quality_stage_reward_difference=float((x['reward']-d['reward'])[good].mean()) if good.any() else None,full_reward_difference=float((w['reward']-dw['reward'])[good].mean()) if good.any() else None,psnr_delta=float((x['psnr']-d['psnr'])[good].mean()) if good.any() else None)
    # The same quality prefix must not be counted twice as independent action evidence.
    prefix_hash=hashlib.sha256(b''.join(np.ascontiguousarray(z['requested_modes'][:300]).tobytes()+np.ascontiguousarray(z['raw_resource_action'][:300]).tobytes() for z in zs[:8])).hexdigest()
    results[f'{seed}/{steps}/{scene}']=dict(safety_failure=failure,success_detail=success_info,top10_minus_median=p,equivalent_mode_changes=equiv,resource_share_changes=[ci(shares[:,u],boot) for u in range(3)],split_repetition=split,per_slot_positive_weight=per_slot,condition_centered_GAE_trajectory_weight=weight_summary(normalized(conditional_GAE),good),first300_prefix_hash=prefix_hash,distribution_variance_note='Ranking within environment; raw across-environment Q quantiles additionally reflect channel differences')
    print('extra',seed,steps,scene,flush=True)
 for seed in m['parents']:
  prefixes=[results[f'{seed}/6000000/{s}']['first300_prefix_hash'] for s in m['sampling_scenes']];assert len(set(prefixes))==1
 write(HERE/'report/additional_results.json',results)
if __name__=='__main__':main()
