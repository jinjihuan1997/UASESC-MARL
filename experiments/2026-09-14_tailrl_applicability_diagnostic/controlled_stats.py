from diag_support import *

def trace_path(seed,steps,scene,rep=None):
 if rep is None and steps==1000000:return SHORT/f'evaluation/seed_{seed}/steps_1000000/{scene}.npz'
 return HERE/f'evaluation/seed_{seed}/steps_{steps}/{scene}'/('det.npz' if rep is None else f's{rep:02d}.npz')
def get_metrics(seed,steps,scene):
 N=16 if scene=='fixed_2' else 8;zs=[arrays(trace_path(seed,steps,scene,r)) for r in range(N)];d=arrays(trace_path(seed,steps,scene));ref=arrays(trace_path(seed,1000000,scene));qs=[episode_metrics(z,2) for z in zs];whole=[episode_metrics(z,None) for z in zs]
 return zs,{k:np.stack([q[k] for q in qs]) for k in qs[0]},episode_metrics(d,2),episode_metrics(ref,2),{k:np.stack([q[k] for q in whole]) for k in whole[0]},episode_metrics(d,None)
def successes(x,d,eps):
 improve=x['quality']-d['quality']>eps;feasible=improve&(x['mean_aoi']<=6)&(x['violation']==0);safe=feasible.copy();zero=feasible.copy()
 for k in ['mean_aoi','max_aoi','age_tail_cost','resource']:
  safe&=x[k]<=d[k]*1.05+1e-12;zero&=x[k]<=d[k]+1e-12
 return dict(improvement=improve,feasible=feasible,safe=safe,zero_regression=zero,strict_service=feasible&(x['max_aoi_ever']<=6))
def ci(x,boot):
 x=np.asarray(x,dtype=float)
 if x.ndim>1:x=x.mean(tuple(range(x.ndim-1)))
 return dict(mean=float(x.mean()),ci95=np.quantile(x[boot].mean(-1),[.025,.975]).tolist())
def tail_matrix(x):return np.stack([tail_advantage(x[:,e]) for e in range(x.shape[1])],1)
def weight_summary(a,success):
 good=a[success];bad=a[~success];pg=np.maximum(good,0);pb=np.maximum(bad,0);total=np.maximum(a,0).sum();den=pb.mean() if len(pb) else 0
 return dict(success_count=len(good),failure_count=len(bad),mean_success=float(good.mean()) if len(good) else None,mean_failure=float(bad.mean()) if len(bad) else None,mean_positive_success=float(pg.mean()) if len(pg) else None,mean_positive_failure=float(pb.mean()) if len(pb) else None,positive_weight_ratio=float(pg.mean()/den) if len(pg) and den>0 else None,ratio_undefined_reason=None if len(pg) and den>0 else 'no success or zero positive failure weight',success_fraction_of_positive_weight=float(pg.sum()/total) if total>0 else None,positive_effective_sample_size=float(total**2/(np.maximum(a,0)**2).sum()) if total>0 else None)
def normalized(a):return (a-a.mean())/(a.std()+1e-5)
def patterns(x,delta):
 N,E=delta.shape;order=np.argsort(delta,axis=0,kind='stable');out={};features=['resource_fractions','budgets','usage','unused_budgets','requested_mode_counts','deliveries','mean_aoi','max_aoi','resource','quality','psnr']
 for label,idx in [('top1',order[-max(1,math.ceil(N*.01)):]),('top5',order[-max(1,math.ceil(N*.05)):]),('top10',order[-max(1,math.ceil(N*.1)):]),('median',order[N//2-1:N//2+1]),('bottom50',order[:N//2])]:
  out[label]=dict(samples_per_environment=idx.shape[0],**{k:np.mean(x[k][idx,np.arange(E)[None,:]],axis=(0,1)).tolist() for k in features})
 return out

def analyze():
 m=manifest();boot=np.random.default_rng(m['bootstrap_seed']).integers(0,20,(4000,20));results={};quantities={};audit={}
 for seed in m['parents']:
  for steps in [1000000,6000000]:
   for scene in (['fixed_2'] if steps==1000000 else m['sampling_scenes']):
    zs,x,d,ref,w,dw=get_metrics(seed,steps,scene);N=len(zs);key=f'{seed}/{steps}/{scene}';delta=x['quality']-d['quality'];dqref=x['quality']-ref['quality'];checks=[]
    tab=ws.PHYS.tables(ws.make_env(scene));expected_external=tab['external']
    for rep,z in enumerate(zs):
     p=trace_path(seed,steps,scene,rep);meta=read(p.with_suffix('.json'));assert meta['trace_sha256']==sha(p) and meta['external_hashes']==expected_external;assert z['seeds'].tolist()==m['validation'];checks.append(ws.independent_check(z,tab))
     np.testing.assert_array_equal(z['initial_q'],zs[0]['initial_q']);np.testing.assert_array_equal(z['initial_tau'],zs[0]['initial_tau']);np.testing.assert_array_equal(z['initial_aoi'],zs[0]['initial_aoi']);np.testing.assert_array_equal(z['snr_db'],zs[0]['snr_db'])
    if steps==6000000:
     zd=arrays(trace_path(seed,steps,scene));md=read(trace_path(seed,steps,scene).with_suffix('.json'));assert md['trace_sha256']==sha(trace_path(seed,steps,scene));checks.append(ws.independent_check(zd,tab))
    tailQ=tail_matrix(x['quality']);tailR=tail_matrix(x['reward']);baseR=x['reward']-x['reward'].mean(0);baseQ=x['quality']-x['quality'].mean(0)
    g=[]
    for z in zs:
     mask=z['trace'][:,:,FIELDS.index('instruction_id')]==2;a=normalized(z['gae_raw']);g.append((a*mask).sum(0)/mask.sum(0))
    baseGAE=np.stack(g);weak=successes(x,d,.001);main=successes(x,d,.005);strong=successes(x,d,.01)
    out=dict(quality=distribution(x['quality'].ravel()),delta_quality=distribution(delta.ravel()),delta_psnr=distribution((x['psnr']-d['psnr']).ravel()),delta_quality_vs_fixed_1m=distribution(dqref.ravel()),deterministic_quality=distribution(d['quality']),full_reward_difference=ci(w['reward']-dw['reward'],boot),by_environment={},success={},success_vs_fixed_1m={},weight_analysis={},action_patterns=patterns(x,delta))
    for eps in m['quality_thresholds']:
     out['success'][str(eps)]={k:ci(v,boot) for k,v in successes(x,d,eps).items()};out['success_vs_fixed_1m'][str(eps)]={k:ci(v,boot) for k,v in successes(x,ref,eps).items()}
    for eps in m['psnr_thresholds']:out.setdefault('psnr_success',{})[str(eps)]=ci(x['psnr']-d['psnr']>eps,boot)
    for e,envseed in enumerate(m['validation']):out['by_environment'][str(envseed)]=dict(delta_quality=distribution(delta[:,e]),quality=distribution(x['quality'][:,e]),success_probability={k:float(v[:,e].mean()) for k,v in main.items()},best_delta_quality=float(delta[:,e].max()))
    for label,a in [('expected_total',baseR),('expected_quality',baseQ),('frozen_GAE_trajectory_average',baseGAE),('tail_total',tailR),('tail_quality',tailQ)]:
     out['weight_analysis'][label]=dict(raw={k:weight_summary(a,s) for k,s in main.items() if k in ['feasible','safe']},common_scale={k:weight_summary(normalized(a),s) for k,s in main.items() if k in ['feasible','safe']})
    # Environment effects held fixed within each column, action randomness changes rows.
    within=x['quality']-x['quality'].mean(0);withinR=x['reward']-x['reward'].mean(0);noisevar=float(x['quality'].var(0).mean());between=float(x['quality'].mean(0).var());snr=x['snr_db'][0].mean(-1)
    out['environment_vs_policy']=dict(exogenous_all_repeats_exact=True,within_environment_Q_variance=noisevar,between_environment_mean_Q_variance=between,between_minus_sampling_variance=max(between-noisevar/N,0),Q_envmean_vs_SNR_correlation=corr(x['quality'].mean(0),snr),within_environment_Q_vs_total_correlation=corr(within.ravel(),withinR.ravel()),environment_fraction_with_any_positive_delta=float((delta.max(0)>0).mean()),environment_fraction_with_any_main_safe_success=float(main['safe'].any(0).mean()),first_half_safe_probability=ci(main['safe'][:N//2],boot),second_half_safe_probability=ci(main['safe'][N//2:],boot))
    for first in [0,1]:
     sl=slice(first*N//2,(first+1)*N//2);out.setdefault('split_action_patterns',{})[str(first)]=patterns({k:v[sl] for k,v in x.items()},delta[sl])
    out['reward_correlations']={k:corr(x['reward'].ravel(),x[k].ravel()) for k in ['quality','mean_aoi','age_mean_cost','age_max_cost','age_tail_cost','resource']};out['reward_correlations']['delta_quality']=corr(x['reward'].ravel(),delta.ravel())
    n=max(1,math.ceil(N*.1));qt=np.argsort(x['quality'],axis=0)[-n:];rt=np.argsort(x['reward'],axis=0)[-n:];out['top10_Q_given_top10_R_within_environment']=float(np.mean([len(set(qt[:,e])&set(rt[:,e]))/n for e in range(20)]))
    if scene!='fixed_2':
     gid=0 if scene.endswith('_0') else 1;post=[episode_metrics(z,gid) for z in zs];dp=episode_metrics(arrays(trace_path(seed,steps,scene)),gid);out['post_quality_difference']={k:ci(np.stack([a[k] for a in post])-dp[k],boot) for k in ['quality','reward','mean_aoi','max_aoi','resource']};out['post_quality_caution']='Includes both inherited state changes and current non-quality stochastic actions; not isolated carryover causality'
    results[key]=out;quantities[key]=dict(Q=x['quality'],deltaQ=delta,deltaQ_vs_fixed_1m=dqref,total=x['reward'],full_total=w['reward'],meanAoI=x['mean_aoi'],maxAoI=x['max_aoi'],resource=x['resource'],success_feasible=main['feasible'],success_safe=main['safe'],A_base=baseGAE,A_tailQ=tailQ,A_tailR=tailR);npz(HERE/f'analysis_arrays/{seed}_{steps}_{scene}.npz',**quantities[key]);audit[key]=checks;print('analyzed',key,flush=True)
 write(HERE/'report/controlled_results.json',results);write(HERE/'report/controlled_audit.json',dict(state='PASS',groups=audit,original_checker_all_slots=True,actual_seeds=m['validation'],reserved_final_test_used=False))
 # Same environment/action repetitions and fixed 1M baseline across checkpoints.
 trend={}
 for seed in m['parents']:
  a=results[f'{seed}/1000000/fixed_2'];b=results[f'{seed}/6000000/fixed_2'];trend[str(seed)]=dict(early=a['success_vs_fixed_1m'],late=b['success_vs_fixed_1m'],sampled_reward_change=ci(quantities[f'{seed}/6000000/fixed_2']['total']-quantities[f'{seed}/1000000/fixed_2']['total'],boot),sampled_quality_change=ci(quantities[f'{seed}/6000000/fixed_2']['Q']-quantities[f'{seed}/1000000/fixed_2']['Q'],boot))
 write(HERE/'report/checkpoint_tail_trend.json',trend)
if __name__=='__main__':guard();analyze()
