"""Deterministic complete-episode aggregation with independent physical replay."""
from light_support import *
from independent_physics import tables,check
from lightweight_policies import Lightweight
from benchmark_latency import stat

GROUPS={'overall_13':'13场景综合','balance_fixed':'固定均衡','aoi_fixed':'固定AoI','quality_fixed':'固定质量','switch_10':'10个切换场景'}
REWARD_PARTS=['quality_credit','age_mean_cost','age_max_cost','age_tail_cost','resource_cost','service_violation_cost','recv_aoi_bonus']

def ci(x,boot):
 x=np.asarray(x,dtype=np.float64)
 if x.ndim>1:x=x.mean(tuple(range(x.ndim-1)))
 return dict(mean=float(x.mean()),ci95=np.quantile(x[boot].mean(-1),[.025,.975]).tolist(),environment_clusters=20)

def summary(x,m,boot,indices,learned):
 out=dict(conditional_mean={g:ci(x[:,ix].mean(1),boot) for g,ix in indices.items()},by_scenario={s:ci(x[:,i],boot) for i,s in enumerate(m['scenarios'])})
 if learned:out['by_parent']={str(p):dict(groups={g:ci(x[k,ix].mean(0),boot) for g,ix in indices.items()},by_scenario={s:ci(x[k,i],boot) for i,s in enumerate(m['scenarios'])}) for k,p in enumerate(m['parents'])}
 else:out['baseline_repetitions']=1;out['not_independent_parent_replicates']=True
 return out

def physics_stats(z,m,take=None):
 if take is None:take=np.ones(z['trace'].shape[:2],bool)
 n=int(take.sum());v=z['trace'][take];a=z['aoi_after'][take].astype(float);served=z['served'][take];count=served.sum(-1);d=int(count.sum());modes=z['modes'][take]
 mapping=np.full(17,-1,dtype=int)
 for i,g in enumerate(m['equivalence_groups']):
  for mode in g:mapping[mode+1]=i
 return dict(slots=n,field_sums=dict(zip(FIELDS,v.sum(0).tolist())),score_x100=float(v[:,0].mean()*100),
   predicted_psnr_per_delivery=float(v[:,FIELDS.index('predicted_quality_sum')].sum()/d) if d else None,
   deliveries=d,deliveries_by_uav=count.sum(0).tolist(),
   resource_share_sum=z['resource_fractions'][take].sum(0).tolist(),budget_sum=z['budgets'][take].sum(0).tolist(),usage_sum=z['usage'][take].sum(0).tolist(),unused_budget_sum=z['unused_budgets'][take].sum(0).tolist(),
   aoi_by_uav_sum=a.mean(-1).sum(0).tolist(),max_aoi_by_uav_sum=a.max(-1).sum(0).tolist(),p95_aoi_by_uav_sum=np.quantile(a,.95,axis=-1).sum(0).tolist(),tail_excess4_by_uav_sum=np.maximum(a-4,0).mean(-1).sum(0).tolist(),
   no_delivery_uav_slots=(count==0).sum(0).tolist(),fallback_slots=((modes>=0)&(modes!=z['requested_modes'][take])).sum(0).tolist(),
   requested_mode_counts=[np.bincount(z['requested_modes'][take][:,u],minlength=16).tolist() for u in range(3)],
   executed_mode_counts_minus1_then_0_to_15=[np.bincount(modes[:,u]+1,minlength=17).tolist() for u in range(3)],
   equivalence_counts_no_delivery_then_groups=[np.bincount(mapping[modes[:,u]+1]+1,minlength=len(m['equivalence_groups'])+1).tolist() for u in range(3)])

def merge_physics(items):
 n=sum(x['slots'] for x in items);sums={f:sum(x['field_sums'][f] for x in items) for f in FIELDS};d=sum(x['deliveries'] for x in items)
 out=dict(slots=n,field_means={f:s/n for f,s in sums.items()},score_x100=sums['common_reward']/n*100,deliveries=d,deliveries_per_slot=d/n,
  predicted_psnr_per_delivery=sums['predicted_quality_sum']/d if d else None,reward_parts_x100={f:sums[f]/n*100 for f in REWARD_PARTS})
 for key in ['resource_share','budget','usage','unused_budget','aoi_by_uav','max_aoi_by_uav','p95_aoi_by_uav','tail_excess4_by_uav']:
  out[key+'_mean']=(np.array([x[key+'_sum'] for x in items]).sum(0)/n).tolist()
 for key in ['no_delivery_uav_slots','fallback_slots','deliveries_by_uav','requested_mode_counts','executed_mode_counts_minus1_then_0_to_15','equivalence_counts_no_delivery_then_groups']:
  out[key]=np.array([x[key] for x in items]).sum(0).tolist()
 out['no_delivery_fraction_per_uav']=(np.asarray(out['no_delivery_uav_slots'])/n).tolist()
 out['no_delivery_denominator_each_uav']=n
 return out

def replay_new_actions(z,method):
 """Recompute every new requested action solely from its saved allowed inputs."""
 ctrl=Lightweight(method);E=len(z['seeds']);N=600*E
 pre=z['sut_obs'].reshape(N,76);post=z['post_uav_obs'].reshape(N,3,76);masks=z['uav_masks'].reshape(N,3,16)
 for start in range(0,N,1024):
  end=min(start+1024,N);resource=arr(ctrl.resource(torch.from_numpy(pre[start:end])))
  np.testing.assert_array_equal(resource,z['raw_resource_action'].reshape(N,3)[start:end])
  for u in range(3):
   action=ctrl.mode(u,torch.from_numpy(post[start:end,u]),torch.from_numpy(masks[start:end,u])).argmax(-1)
   np.testing.assert_array_equal(arr(action),z['requested_modes'].reshape(N,3)[start:end,u])
 return dict(requested_action_replay_sample_decisions=N,inputs='saved SUT obs and each UAV own post-allocation obs/mask only')

def cost_summary():
 rows=[]
 for p in sorted((HERE/'costs').glob('*.jsonl')):rows.extend(json.loads(s) for s in p.read_text().splitlines())
 numeric={}
 for r in rows:
  a=numeric.setdefault(r['kind'],dict(records=0,physical_steps=0,complete_episodes=0,partial_physical_steps=0,sample_decisions=0,replayed_sample_decisions=0,warmup_batches=0,measured_batches=0,failures=0))
  a['records']+=1;a['failures']+=int(r.get('error') is not None)
  for k in list(a):
   if k not in ['records','failures']:a[k]+=int(r.get(k,0))
 return numeric,rows

def timing_summary(m):
 oldtiming=read(HERE/'report/complexity.json');assert oldtiming['state']=='MEASURED'
 for p,h in oldtiming['timing_raw_files_sha256'].items():assert sha(HERE/p)==h
 out={}
 for method in m['methods']+m['old_baselines']+m['learned_methods']:
  parents=m['parents'] if method in m['learned_methods'] else ['rules'];out[method]={}
  for batch in [1,20]:
   ns=np.concatenate([arrays(HERE/f'latency/batch_{batch}/{p}_{method}.npz')['nanoseconds'] for p in parents])
   out[method][str(batch)]=stat(ns[:,0],batch)
  key=f'{parents[0]}/{method}';out[method]['storage']=oldtiming['storage'][key]
  out[method]['latency_parent_pooling']='Equal 300 measurements per fixed parent, pooled before quantile; not new training repeats' if method in m['learned_methods'] else 'Single deterministic baseline'
 return out

@torch.inference_mode()
def main():
 m=verify();assert read(HERE/'preflight.json')['state']=='PASS';assert sha(HERE/'reference_index.json')==m['reference_index_sha256']
 scenarios=list(m['scenarios']);parents=m['parents'];methods=m['methods']+m['old_baselines']+m['learned_methods'];refs=read(HERE/'reference_index.json');boot=np.random.default_rng(m['bootstrap_seed']).integers(0,20,(4000,20))
 indices=dict(overall_13=list(range(13)),balance_fixed=[scenarios.index('fixed_0')],aoi_fixed=[scenarios.index('fixed_1')],quality_fixed=[scenarios.index('fixed_2')],switch_10=[i for i,s in enumerate(scenarios) if not s.startswith('fixed_')])
 score={k:np.zeros((3,13,20)) for k in methods};tabcache={};audits={};physical={};episode={};trace_hash={};seen=set();initial={};external={};items=[]
 for method in m['methods']+m['old_baselines']:
  for s in scenarios:
   key=f'rules/{method}/{s}'
   p=HERE/f'evaluation/{method}/{s}.npz' if method in m['methods'] else HERE/f'supplement/{method}/{s}.npz' if method in ['R_equal_single','R_single'] else ROOT/refs['reused'][key]['path']
   items.append((key,method,None,s,p))
 for parent in parents:
  for method in m['learned_methods']:
   for s in scenarios:
    key=f'{parent}/{method}/{s}';items.append((key,method,parent,s,ROOT/refs['reused'][key]['path']))
 assert len(items)==247
 for key,method,parent,scene,p in items:
  meta=read(p.with_suffix('.json'));z=arrays(p);h=sha(p);assert meta['trace_sha256']==h
  if key in refs['reused']:
   r=refs['reused'][key];assert h==r['sha256'] and sha(p.with_suffix('.json'))==r['metadata_sha256']
  else:
   assert meta['state']=='complete' and meta['identity']['manifest_sha256']==sha(HERE/'manifest.json');assert meta['original_checker']=='PASS_every_slot'
   assert meta['policy_hash_before']==meta['policy_hash_after']
  assert z['seeds'].tolist()==m['validation'];seen.update(z['seeds'].tolist());assert not seen&set(m['reserved'])
  if scene not in tabcache:
   env=make_env(m['scenarios'][scene]);tabcache[scene]=tables(env);initial[scene]={k:arr(getattr(env,k)).copy() for k in ['q','tau','aoi']};external[scene]=tabcache[scene]['external']
  assert meta['external_hashes']==external[scene]
  for k,v in initial[scene].items():np.testing.assert_array_equal(v,z['initial_'+k])
  gids=np.zeros(600,dtype=int)
  for t,g in m['scenarios'][scene]:gids[t:]=g
  np.testing.assert_array_equal(z['trace'][:,:,FIELDS.index('instruction_id')],np.repeat(gids[:,None],20,axis=1))
  a=check(z,tabcache[scene])
  if method in m['methods']:a.update(replay_new_actions(z,method))
  if method in ['R_equal_single','R_single']:
   assert meta['old_trace_match']=='ALL_OLD_FIELDS_EXACT';assert meta['old_trace_sha256']==refs['legacy_to_supplement'][f'{method}/{scene}']['sha256']
  ep=z['trace'][:,:,0].mean(0)*100
  np.testing.assert_allclose(ep,meta['scores_by_environment'],rtol=0,atol=0)
  pi=parents.index(parent) if parent else slice(None);si=scenarios.index(scene);score[method][pi,si]=ep
  physical[key]=dict(overall=physics_stats(z,m),by_instruction={str(g):physics_stats(z,m,np.repeat((gids==g)[:,None],20,axis=1)) for g in sorted(set(gids))})
  episode[key]=dict(score_x100=float(ep.mean()),episode_scores_x100=ep.tolist(),episode_mean_raw_reward=(ep/100).tolist(),episode_total_reward=(ep*6).tolist(),trace_path=str(p.relative_to(ROOT)))
  audits[key]=a;trace_hash[key]=h
  if len(audits)%13==0:print('AGGREGATED',key,len(audits),flush=True)
 timings=timing_summary(m)
 scores={method:summary(score[method],m,boot,indices,method in m['learned_methods']) for method in methods}
 comparisons=[(a,b) for a in m['methods'] for b in ['greedy_modes_16','R_instruction']]+[(a,b) for a in m['learned_methods'] for b in m['methods']+m['old_baselines']]
 paired={a+'_minus_'+b:summary(score[a]-score[b],m,boot,indices,a in m['learned_methods']) for a,b in comparisons}
 physical_summary={}
 for method in methods:
  keys=[k for k in physical if k.split('/')[1]==method];ps={}
  for group,ix in indices.items():ps[group]=merge_physics([physical[k]['overall'] for k in keys if k.split('/')[-1] in [scenarios[i] for i in ix]])
  physical_summary[method]=dict(groups=ps,by_instruction={str(g):merge_physics([physical[k]['by_instruction'][str(g)] for k in keys if str(g) in physical[k]['by_instruction']]) for g in range(3)})
  if method in m['learned_methods']:
   physical_summary[method]['by_parent']={str(p):{group:merge_physics([physical[k]['overall'] for k in keys if k.startswith(str(p)+'/') and k.split('/')[-1] in [scenarios[i] for i in ix]]) for group,ix in indices.items()} for p in parents}
  np.testing.assert_allclose(ps['overall_13']['score_x100'],scores[method]['conditional_mean']['overall_13']['mean'],atol=1e-12,rtol=0)
 duplicate={}
 for s in scenarios:
  a=arrays(HERE/f'evaluation/R_equal_minload/{s}.npz');b=arrays(HERE/f'supplement/R_equal_single/{s}.npz')
  fields=['trace','modes','resource_fractions','budgets','usage','unused_budgets','aoi_after','cache_after','tau_after','served']
  duplicate[s]=dict(all_compared_physical_arrays_exact=all(np.array_equal(a[k],b[k]) for k in fields),compared_fields=fields,requested_mode_mismatches=int(np.sum(a['requested_modes']!=b['requested_modes'])),physical_field_max_abs_difference={k:float(np.max(np.abs(a[k].astype(float)-b[k].astype(float)))) for k in fields})
 costs,ledger=cost_summary();assert costs['primary']['physical_steps']==624000 and costs['primary']['complete_episodes']==1040
 assert costs['supplement']['physical_steps']==312000 and costs['supplement']['complete_episodes']==520
 for r in ledger:assert set(r.get('seeds',[]))<=set(m['validation']) and not set(r.get('seeds',[]))&set(m['reserved'])
 # All actual trajectory seed arrays, including repeated prechecks, are checked.
 new_seed_files={}
 for folder in ['evaluation','supplement','preflight_traces','failures']:
  for p in sorted((HERE/folder).glob('**/*.npz')):
   with np.load(p,allow_pickle=False) as z:
    if 'seeds' in z.files:
     seeds=z['seeds'].tolist();assert set(seeds)<=set(m['validation']) and not set(seeds)&set(m['reserved']);new_seed_files[str(p.relative_to(HERE))]=seeds
 for instance in read(HERE/'latency/input_index.json').values():
  for fixture in instance:assert set(fixture['seeds'])<=set(m['validation']) and not set(fixture['seeds'])&set(m['reserved'])
 dominators={a:[b for b in methods if b!=a and scores[b]['conditional_mean']['overall_13']['mean']>=scores[a]['conditional_mean']['overall_13']['mean'] and timings[b]['1']['P50_ms']<=timings[a]['1']['P50_ms'] and (scores[b]['conditional_mean']['overall_13']['mean']>scores[a]['conditional_mean']['overall_13']['mean'] or timings[b]['1']['P50_ms']<timings[a]['1']['P50_ms'])] for a in methods}
 numerical=dict(score_units='Mean native common reward per physical slot times 100; larger is better',scenarios=m['scenarios'],seeds=m['validation'],parents=parents,methods=methods,scores=scores,episode_results=episode,latency_summary=timings,point_estimate_score_latency_dominators=dominators,duplicate_minload_vs_original_equal_single=duplicate,statistical_scope=dict(bootstrap_replicates=4000,bootstrap_seed=m['bootstrap_seed'],cluster='environment seed; all 13 scenarios, methods and 3 fixed parents share resampling',interval='percentile 95%, conditional on fixed parents and reused development set; no population-level training or equivalence claim',no_multiple_comparison_adjustment=True))
 write(HERE/'report/results.json',numerical);write(HERE/'report/paired_differences.json',paired)
 write(HERE/'report/physical_statistics.json',dict(quality_label='Fixed average profile predicted PSNR, weighted by actual delivery count, not decoded-video measurement',equivalence_groups=m['equivalence_groups'],summary=physical_summary,by_scenario_parent_instruction=physical,tail_definitions=dict(global_p95='per-slot 95th percentile of 30 source AoIs, then averaged',uav_p95='per-slot 95th percentile of each UAV ten source AoIs, then averaged',tail_cost='positive excess of source AoI above 4, averaged per native reward'),no_delivery_denominator='UAV physical slots; separate from the distribution conditional on delivery'))
 write(HERE/'report/independent_audit.json',dict(state='PASS',trace_audits=audits,trace_sha256=trace_hash,external_hashes_by_scenario=external,protected_inputs_checked=len(m['protected_sha256']),input_hashes_unchanged=True,actual_new_seed_files=new_seed_files,actual_environment_seeds=sorted(seen),reserved_final_test_intersection=[],reserved_final_test_used=False,costs=costs,reused=dict(traces=169,complete_episodes=3380,prior_physical_steps=2028000,new_physical_steps=0),audited_complete_trace_slots=sum(a['physical_slots'] for a in audits.values()),new_training_steps=0,new_optimizer_updates=0,new_fitting_updates=0,forbidden_calls=FORBIDDEN))
 npz(HERE/'report/episode_scores.npz',methods=np.array(methods),parents=np.array(parents),scenarios=np.array(scenarios),seeds=np.array(m['validation']),score_x100=np.stack([score[k] for k in methods]),bootstrap_environment_indices=boot)
 verify();assert not FORBIDDEN
 print('COMPLETE NUMERIC AGGREGATION',len(audits),flush=True)

if __name__=='__main__':guard();main()
