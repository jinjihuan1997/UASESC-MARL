"""Independent physical reconstruction and paired weight-sensitivity statistics."""
from weight_support import *
STATS=base.load_module('weight_prior_statistics',LIGHT/'aggregate.py')
GROUPS=STATS.GROUPS

@torch.inference_mode()
def main():
 m=verify();assert read(HERE/'preflight.json')['state']=='PASS';methods=m['static_methods']+m['learned_methods']+m['adaptive_methods'];scenes=list(m['scenarios']);boot=np.random.default_rng(m['bootstrap_seed']).integers(0,20,(4000,20));parents=m['parents']
 ix=dict(overall_13=list(range(13)),balance_fixed=[scenes.index('fixed_0')],aoi_fixed=[scenes.index('fixed_1')],quality_fixed=[scenes.index('fixed_2')],switch_10=[i for i,s in enumerate(scenes) if not s.startswith('fixed_')])
 score={method:np.zeros((3,13,20)) for method in methods};original_objective={method:np.zeros((3,13,20)) for method in methods};audits={};phys={};hashes={};episodes={};tablecache={};seen=set();external={};changed={}
 for item in m['instances']:
  method=item['method'];parent=item['parent'];pi=parents.index(parent) if parent else slice(None)
  for si,scene in enumerate(scenes):
   key=f'{parent or "rules"}/{method}/{scene}';p=HERE/'evaluation'/f'{key}.npz';meta=read(p.with_suffix('.json'));z=arrays(p)
   assert meta['state']=='complete' and meta['trace_sha256']==sha(p) and meta['identity']['manifest_sha256']==sha(HERE/'manifest.json');assert meta['original_checker']=='PASS_every_slot' and meta['policy_unchanged']
   assert z['seeds'].tolist()==m['validation'];seen.update(z['seeds'].tolist());assert not seen&set(m['reserved'])
   if scene not in tablecache:tablecache[scene]=PHYS.tables(make_env(scene));external[scene]=tablecache[scene]['external']
   assert external[scene]==meta['external_hashes'];a=independent_check(z,tablecache[scene])
   gids=np.zeros(600,dtype=int)
   for t,g in m['scenarios'][scene]:gids[t:]=g
   np.testing.assert_array_equal(z['trace'][:,:,FIELDS.index('instruction_id')],np.repeat(gids[:,None],20,axis=1))
   if not method.endswith('_Qhalf_local'):a['old_comparison']=old_physical_equal(z,method,parent,scene)
   else:
    origin=method.removesuffix('_Qhalf_local');oldz=arrays(legacy_path(origin,None,scene));changed[key]=dict(requested_uav_slots_changed=int(np.sum(z['requested_modes']!=oldz['requested_modes'])),executed_uav_slots_changed=int(np.sum(z['modes']!=oldz['modes'])),denominator=36000)
    # Pure local score replay from stored post-allocation input, no environment.
    spec=replace(LIGHTPOL.ONLINE.PublicSpec(),weights=tuple(tuple(row) for row in m['effective_weights']));g=LIGHTPOL.ONLINE.UAVGreedy([0,5,10] if origin=='greedy_modes_3' else list(range(16)),spec=spec)
    post=z['post_uav_obs'].reshape(12000,3,76);mask=z['uav_masks'].reshape(12000,3,16);req=z['requested_modes'].reshape(12000,3)
    for start in range(0,12000,1000):
     for u in range(3):np.testing.assert_array_equal(arr(g.act(torch.from_numpy(post[start:start+1000,u]),torch.from_numpy(mask[start:start+1000,u])).argmax(-1)),req[start:start+1000,u])
    a['matched_local_score_action_replay']='PASS_all_slots'
   ep=z['trace'][:,:,0].mean(0)*100;q=z['trace'][:,:,FIELDS.index('quality_credit')];original=(z['trace'][:,:,0]+q).mean(0)*100
   np.testing.assert_allclose(ep,meta['scores_by_environment'],atol=0,rtol=0);np.testing.assert_allclose(original,meta['old_objective_on_this_trajectory'],atol=0,rtol=0)
   score[method][pi,si]=ep;original_objective[method][pi,si]=original
   episodes[key]=dict(new_score_x100=ep.tolist(),new_mean_raw_reward=(ep/100).tolist(),original_objective_x100_on_same_trajectory=original.tolist())
   phys[key]=dict(overall=STATS.physics_stats(z,m),by_instruction={str(g):STATS.physics_stats(z,m,np.repeat((gids==g)[:,None],20,axis=1)) for g in sorted(set(gids))})
   audits[key]=a;hashes[key]=sha(p)
  print('Aggregated',parent,method,len(audits),flush=True)
 assert len(audits)==299
 summaries={method:STATS.summary(score[method],m,boot,ix,method in m['learned_methods']) for method in methods}
 old_summaries={method:STATS.summary(original_objective[method],m,boot,ix,method in m['learned_methods']) for method in methods}
 pairs=[(a,b) for a in m['learned_methods'] for b in m['static_methods']+m['adaptive_methods']]+[(a,a.removesuffix('_Qhalf_local')) for a in m['adaptive_methods']]
 paired={a+'_minus_'+b:STATS.summary(score[a]-score[b],m,boot,ix,a in m['learned_methods']) for a,b in pairs}
 ps={}
 for method in methods:
  keys=[k for k in phys if k.split('/')[1]==method]
  ps[method]=dict(groups={group:STATS.merge_physics([phys[k]['overall'] for k in keys if k.split('/')[-1] in [scenes[i] for i in indices]]) for group,indices in ix.items()},by_instruction={str(g):STATS.merge_physics([phys[k]['by_instruction'][str(g)] for k in keys if str(g) in phys[k]['by_instruction']]) for g in range(3)})
  if method in m['learned_methods']:ps[method]['by_parent']={str(parent):{group:STATS.merge_physics([phys[k]['overall'] for k in keys if k.startswith(str(parent)+'/') and k.split('/')[-1] in [scenes[i] for i in indices]]) for group,indices in ix.items()} for parent in parents}
  np.testing.assert_allclose(ps[method]['groups']['overall_13']['score_x100'],summaries[method]['conditional_mean']['overall_13']['mean'],atol=1e-12,rtol=0)
 decomposition={}
 for a,b in pairs:
  output={}
  for group in ix:
   x=ps[a]['groups'][group]['reward_parts_x100'];y=ps[b]['groups'][group]['reward_parts_x100'];parts={k:x[k]-y[k] for k in x};combined=parts['quality_credit']-sum(parts[k] for k in ['age_mean_cost','age_max_cost','age_tail_cost','resource_cost','service_violation_cost'])+parts['recv_aoi_bonus']
   delta=paired[a+'_minus_'+b]['conditional_mean'][group]['mean'];np.testing.assert_allclose(combined,delta,atol=1e-10,rtol=0);output[group]=dict(score_difference=delta,reward_component_differences=parts,reconstruction_error=float(abs(combined-delta)))
  decomposition[a+'_minus_'+b]=output
 ledger=[]
 for p in sorted((HERE/'costs').glob('*.jsonl')):ledger.extend(json.loads(row) for row in p.read_text().splitlines())
 costs={}
 for row in ledger:
  d=costs.setdefault(row['kind'],dict(records=0,physical_steps=0,physically_complete_episodes=0,failed_records=0,failed_physical_steps=0));d['records']+=1;d['physical_steps']+=row['physical_steps'];d['physically_complete_episodes']+=row['complete_episodes'];d['failed_records']+=int(row['error'] is not None);d['failed_physical_steps']+=row['physical_steps'] if row['error'] is not None else 0
  assert set(row['seeds'])<=set(m['validation']) and not set(row['seeds'])&set(m['reserved'])
 assert costs['formal']['physical_steps']==3588000 and costs['formal']['physically_complete_episodes']==5980 and costs['formal']['failed_records']==0
 seed_arrays={}
 for folder in ['evaluation','preflight_traces']:
  for p in sorted((HERE/folder).rglob('*.npz')):
   with np.load(p,allow_pickle=False) as z:
    seeds=z['seeds'].tolist();assert set(seeds)<=set(m['validation']) and not set(seeds)&set(m['reserved']);seed_arrays[str(p.relative_to(HERE))]=seeds
 result=dict(methods=methods,original_weights=m['original_weights'],effective_weights=m['effective_weights'],renormalized=False,scenarios=m['scenarios'],parents=parents,validation=m['validation'],new_objective_scores=summaries,original_objective_on_new_trajectories=old_summaries,episode_results=episodes,changed_modes=changed,statistical_scope=dict(bootstrap_replicates=4000,bootstrap_seed=m['bootstrap_seed'],unit='Environment seed; retain all scenarios, methods and fixed parents',interval='Percentile 95%, conditional fixed parents and development environments; no multiple comparison adjustment'),interpretation='Frozen-policy rescoring and newly weight-matched local greedy decisions; no relearning of RL or fitted SUT predictors')
 write(HERE/'report/results.json',result);write(HERE/'report/paired_differences.json',paired);write(HERE/'report/gap_decomposition.json',decomposition)
 write(HERE/'report/physical_statistics.json',dict(quality_label='Fixed average profile predicted PSNR, delivery weighted',summary=ps,by_scenario_parent_instruction=phys,equivalence_groups=m['equivalence_groups']))
 write(HERE/'report/independent_audit.json',dict(state='PASS',traces=audits,trace_sha256=hashes,external_hashes=external,actual_seed_arrays=seed_arrays,actual_environment_seeds=sorted(seen),reserved_final_test_intersection=[],reserved_final_test_used=False,costs=costs,audited_trace_slots=3588000,protected_files=len(m['protected_sha256']),new_training_steps=0,new_optimizer_updates=0,new_fitting_updates=0,matched_local_candidate_decisions_replayed=624000,forbidden_calls=FORBIDDEN))
 npz(HERE/'report/episode_scores.npz',methods=np.array(methods),parents=np.array(parents),scenarios=np.array(scenes),seeds=np.array(m['validation']),new_score_x100=np.stack([score[k] for k in methods]),old_score_on_same_trajectory_x100=np.stack([original_objective[k] for k in methods]),bootstrap_indices=boot)
 verify();assert not FORBIDDEN;print('FULL AGGREGATION COMPLETE',flush=True)
if __name__=='__main__':guard();main()
