"""Deterministic complete-episode aggregation, paired cluster CIs, physical audit."""
from support import *
from composition_policy import Composition
import argparse, subprocess
GROUPS={'overall_13':list(range(13)),'balance_fixed':[0],'aoi_fixed':[1],'quality_fixed':[2],'switch_10':list(range(3,13))}
TERMS=['quality_credit','age_mean_cost','age_max_cost','age_tail_cost','resource_cost','service_violation_cost','recv_aoi_bonus']
SIGN=np.asarray([1,-1,-1,-1,-1,-1,1])
CONTRASTS={'C1_minus_C0':{'C1':1,'C0':-1},'C2_minus_C0':{'C2':1,'C0':-1},'C3_minus_C0':{'C3':1,'C0':-1},'C3_minus_C1':{'C3':1,'C1':-1},'C3_minus_C2':{'C3':1,'C2':-1},'interaction_I':{'C3':1,'C1':-1,'C2':-1,'C0':1}}

def ci(x,boot):
 x=np.asarray(x,dtype=np.float64)
 if x.ndim>1:x=x.mean(tuple(range(x.ndim-1)))
 draws=x[boot].mean(-1)
 return dict(mean=float(x.mean()),ci95=np.quantile(draws,[.025,.975]).tolist(),paired_environment_clusters=20)

def summarize(x,m,boot):
 d=dict(conditional_mean={},by_parent={},by_scenario={})
 for group,idx in GROUPS.items():
  v=x[:,idx].mean(1);d['conditional_mean'][group]=ci(v,boot)
  for pi,seed in enumerate(m['parents']):d['by_parent'].setdefault(str(seed),{})[group]=ci(v[pi],boot)
 for si,scene in enumerate(m['scenarios']):d['by_scenario'][scene]=ci(x[:,si],boot)
 return d

def gids_for(schedule):
 gid=np.zeros(600,dtype=int)
 for t,g in schedule:gid[t:]=g
 return gid

def physical_statistics(z,tab,m,cfg):
 prior=prior_audit_module().compact_stats(z,m);trace=z['trace'];gids=trace[:,:,FIELDS.index('instruction_id')].astype(int)
 before_q=np.concatenate([z['initial_q'][None],z['cache_after'][:-1]])
 valid_budget=tab['load']<=z['budgets'][...,None]+1e-9
 valid=valid_budget&(tab['quality']>=tab['req'][...,None]-1e-9)&before_q.any(-1)[...,None]
 deliveries=z['served'].sum(-1);q=z['predicted_quality'];aoi=z['aoi_after'].astype(float)
 weights=np.asarray(cfg['env_args']['reward_weights_by_instruction'])[gids]
 for g,oldstats in prior.items():
  take=gids==int(g);n=int(take.sum());by_u=[]
  for u in range(3):
   count=deliveries[take,u];quality=q[take,u];a=aoi[take,u];delivered=count>0
   request=z['requested_modes'][take,u];mode=z['modes'][take,u]
   d=dict(uav=u+1,decision_slots=n,delivered_uav_slots=int(delivered.sum()),undelivered_uav_slots=int((~delivered).sum()),undelivered_slot_fraction=float((~delivered).mean()),deliveries_total=int(count.sum()),predicted_psnr_per_delivery=float((quality*count).sum()/count.sum()) if count.sum() else None,resource_share_mean=float(z['resource_fractions'][take,u].mean()),budget_mean=float(z['budgets'][take,u].mean()),usage_mean=float(z['usage'][take,u].mean()),unused_budget_mean=float(z['unused_budgets'][take,u].mean()),aoi_mean=float(a.mean()),aoi_max_over_run=float(a.max()),mean_slot_max_aoi=float(a.max(-1).mean()),aoi_p95=float(np.quantile(a,.95)),aoi_p99=float(np.quantile(a,.99)),fraction_aoi_above4=float((a>4).mean()),fraction_aoi_above6=float((a>6).mean()),mean_excess_aoi_above4=float(np.maximum(a-4,0).mean()),quality_credit_mean=float((weights[take,0]*np.maximum((quality-21)/12,0)*count/30).mean()),additive_age_mean_cost=float((weights[take,1]*.4*a.sum(-1)/(8*30)).mean()),additive_age_tail_cost=float((weights[take,1]*.3*np.maximum(a-4,0).sum(-1)/(8*30)).mean()),nonadditive_uav_max_aoi_cost=float((weights[take,1]*.3*a.max(-1)/8).mean()),additive_resource_cost=float((weights[take,2]*.2*z['usage'][take,u]/60000).mean()),budget_feasible_mode_fraction=valid_budget[take,u].mean(0).tolist(),fully_feasible_mode_fraction=valid[take,u].mean(0).tolist(),requested_mode_counts=np.bincount(request,minlength=16).tolist(),executed_mode_counts_no_delivery_then_0_to15=np.bincount(mode+1,minlength=17).tolist(),exact_equivalence_groups=[])
   for group in m['equivalence_groups']:
    chosen=np.isin(mode,group);req=np.isin(request,group);budget=valid_budget[take,u][:,group].any(-1);feasible=valid[take,u][:,group].any(-1)
    d['exact_equivalence_groups'].append(dict(modes=group,requested_count=int(req.sum()),executed_count=int(chosen.sum()),executed_fraction_of_all_uav_slots=float(chosen.mean()),executed_fraction_of_delivered_uav_slots=float(chosen.sum()/delivered.sum()) if delivered.sum() else None,budget_infeasible_count=int((~budget).sum()),budget_feasible_count=int(budget.sum()),fully_feasible_count=int(feasible.sum()),fully_feasible_but_requested_other_count=int((feasible&~req).sum()),requested_given_fully_feasible_fraction=float((feasible&req).sum()/feasible.sum()) if feasible.sum() else None))
   by_u.append(d)
  oldstats['by_uav']=by_u
 return prior

@torch.inference_mode()
def route_audit(z,controller,policy):
 gid=z['trace'][:,:,FIELDS.index('instruction_id')].astype(int)
 expected_res=((gid==2)&(controller in ('C2','C3'))).astype(np.int8)
 expected_pol=np.where(gid==2,2,np.where((gid==0)&(controller in ('C1','C3')),1,0)).astype(np.int8)
 np.testing.assert_array_equal(z['resource_source'],expected_res)
 np.testing.assert_array_equal(z['policy_source'],np.repeat(expected_pol[...,None],3,-1))
 np.testing.assert_array_equal(z['adapter_enabled'],z['policy_source']!=0)
 np.testing.assert_array_equal(z['sut_obs'][...,23:26].argmax(-1),gid)
 np.testing.assert_array_equal(z['post_uav_obs'][...,67:70].argmax(-1),np.repeat(gid[...,None],3,-1))
 # Independently execute saved local inputs. No critic/global state/reward supplied.
 for slot in range(600):
  sut=torch.from_numpy(z['sut_obs'][slot]);mask=torch.zeros(20,16);mask[:,:3]=1
  eq=torch.full((20,3),1/3,dtype=torch.float64)*2-1
  action,_=policy.resource(sut,mask,eq);np.testing.assert_array_equal(arr(action),z['raw_resource_action'][slot])
  for u in range(3):
   action,_=policy.mode(u,torch.from_numpy(z['post_uav_obs'][slot,:,u]),torch.from_numpy(z['uav_masks'][slot,:,u]))
   np.testing.assert_array_equal(arr(action.argmax(-1)),z['requested_modes'][slot,:,u])
 policy.assert_frozen()
 return dict(state='PASS',saved_local_input_policy_inference_slots=12000,physical_steps=0,exact_source_and_action_match=True)

def aggregate():
 m=verify_inputs();assert read(HERE/'preflight.json')['state']=='PASS'
 for name,h in read(HERE/'analysis_manifest.json')['code_sha256'].items():assert sha(HERE/name)==h,name
 assert sha(HERE/'reference_index.json')==m['reference_index_sha256'] and sha(HERE/'seed_selection.json')==m['seed_selection_sha256']
 refs=read(HERE/'reference_index.json');prioraudit=read(OLD/'report/audit.json')
 for p,h in read(old.GREEDY/'protocol.json')['source_hashes'].items():assert sha(HERE.parent/p.split('/experiments/',1)[1])==h
 report=HERE/'report';report.mkdir(exist_ok=True)
 methods=['C0','C1','C2','C3','original_rl','residual_all','quality_rule_hybrid']+m['rules']
 N=13;E=20;P=3
 scores={k:np.zeros((P,N,E)) for k in methods}
 # Contributions average over every slot, so instruction contributions add to total.
 components={k:np.zeros((P,N,E,3,len(TERMS))) for k in methods}
 carry={k:{} for k in methods};physical={};tracehash={};audits={};routechecks={};episode={};seen=set();external_hashes={};tabs={}
 items=[]
 for key,r in refs.items():
  seed,method,scene=key.split('/');items.append((None if seed=='rules' else int(seed),method,scene,WORKSPACE/r['path'],r))
 for seed in m['parents']:
  for c in m['new_controllers']:
   for scene in m['scenarios']:
    path=HERE/f'evaluation/seed_{seed}/{c}/{scene}.npz';items.append((seed,c,scene,path,None))
 # Scene-major saves memory; each table is generated from frozen independent streams.
 items.sort(key=lambda x:(list(m['scenarios']).index(x[2]),str(x[0]),x[1]))
 policy_cache={}
 for seed,method,scene,path,ref in items:
  si=list(m['scenarios']).index(scene);pi=m['parents'].index(seed) if seed else slice(None);key=f'{seed or "rules"}/{method}/{scene}'
  if scene not in tabs:
   tabs={scene:tables(scene)}
  tab=tabs[scene]
  if ref:
   meta=ref['metadata'];assert sha(path)==ref['trace_sha256']==prioraudit['trace_sha256'][ref['prior_audit_key']]
  else:
   meta=read(path.with_suffix('.json'));assert meta['identity']['manifest_sha256']==sha(HERE/'manifest.json')
   assert meta['identity']['code_sha256']=={p:sha(HERE/p) for p in meta['identity']['code_sha256']}
   assert sha(path)==meta['trace_sha256'] and not meta['forbidden_training_calls']
  assert meta['external_hashes']==tab['external'];external_hashes[key]=meta['external_hashes'];tracehash[key]=sha(path)
  with np.load(path,allow_pickle=False) as loaded:
   # Materialize once: compressed NPZ repeated indexing otherwise decompresses per slot.
   data={k:loaded[k] for k in loaded.files}
  class Arrays(dict):
   @property
   def files(self):return list(self)
  z=Arrays(data)
  assert z['seeds'].tolist()==m['validation'];seen.update(z['seeds'].tolist());assert not seen&set(m['reserved_final_test'])
  gids=z['trace'][:,:,FIELDS.index('instruction_id')].astype(int)
  want=np.repeat(gids_for(m['scenarios'][scene])[:,None],20,1);np.testing.assert_array_equal(gids,want)
  audits[key]=prior_audit_module().physical_audit(z,tab,cfg_for(seed or m['parents'][0]))
  if not ref:
   pk=(seed,method)
   if pk not in policy_cache:
    policy_cache[pk]=Composition(seed,make_env(seed,[[0,0]]),method)
   routechecks[key]=route_audit(z,method,policy_cache[pk])
   valid=(tab['load']<=z['budgets'][...,None]+1e-9)&(tab['quality']>=tab['req'][...,None]-1e-9)&np.concatenate([z['initial_q'][None],z['cache_after'][:-1]]).any(-1)[...,None]
   np.testing.assert_array_equal(z['truly_feasible'],valid)
   np.testing.assert_array_equal(z['budget_feasible'],tab['load']<=z['budgets'][...,None]+1e-9)
   np.testing.assert_array_equal(z['requested_feasible'],np.take_along_axis(valid,z['requested_modes'][...,None],-1).squeeze(-1))
   np.testing.assert_array_equal(z['fallback'],(z['modes']>=0)&(z['modes']!=z['requested_modes']))
  reward=z['trace'][:,:,0];score=reward.mean(0)*100;scores[method][pi,si]=score
  np.testing.assert_allclose(score,meta['scores_by_environment'],atol=0,rtol=0)
  episode[key]=dict(score_x100_by_environment=score.tolist(),mean_reward_raw_by_environment=(score/100).tolist())
  vals=z['trace'][:,:,[FIELDS.index(t) for t in TERMS]]*SIGN
  for g in range(3):components[method][pi,si,:,g]=(vals*(gids==g)[...,None]).mean(0)*100
  mask=np.maximum.accumulate(want[:,0]==2)&(want[:,0]!=2)
  if mask.any():
   carry[method].setdefault(scene,np.zeros((3,20)))[pi]=reward[mask].mean(0)*100
  physical[key]=physical_statistics(z,tab,m,cfg_for(seed or m['parents'][0]))
  print(json.dumps(dict(aggregated=key,traces=len(audits))),flush=True)
 boot=np.random.default_rng(m['bootstrap_seed']).integers(0,E,(4000,E))
 summary={k:summarize(v,m,boot) for k,v in scores.items()}
 contrasts=copy.deepcopy(CONTRASTS)
 for a in m['new_controllers']:
  for b in ['original_rl','residual_all','quality_rule_hybrid']+m['rules']:contrasts[f'{a}_minus_{b}']={a:1,b:-1}
 paired={k:summarize(sum(scores[a]*w for a,w in v.items()),m,boot) for k,v in contrasts.items()}
 gaps={}
 for a in m['new_controllers']:
  for b in ['C0','original_rl','residual_all','quality_rule_hybrid']+m['rules']:
   diff=components[a]-components[b];overall=diff.sum((-1,-2));np.testing.assert_allclose(overall,scores[a]-scores[b],atol=1e-10,rtol=0)
   d=dict(overall_score_difference=summarize(scores[a]-scores[b],m,boot),instruction_contributions={},scenario_contributions={},by_parent={})
   for g in range(3):
    v=diff[:,:,:,g].sum(-1).mean(1);frac=sum((gids_for(s)==g).sum() for s in m['scenarios'].values())/(13*600)
    d['instruction_contributions'][str(g)]=dict(actual_slot_fraction=frac,weighted_score_contribution=ci(v,boot),conditional_instruction_score_difference=ci(v/frac,boot),signed_reward_component_contributions={t:ci(diff[:,:,:,g,i].mean(1),boot) for i,t in enumerate(TERMS)})
   for si,scene in enumerate(m['scenarios']):d['scenario_contributions'][scene]=dict(score_difference=ci((scores[a]-scores[b])[:,si],boot),signed_reward_components={t:ci(diff[:,si,:,: ,i].sum(-1),boot) for i,t in enumerate(TERMS)})
   for pi,s in enumerate(m['parents']):d['by_parent'][str(s)]={str(g):dict(weighted_contribution=ci(diff[pi,:,:,g].sum(-1).mean(0),boot),signed_reward_components={t:ci(diff[pi,:,:,g,i].mean(0),boot) for i,t in enumerate(TERMS)}) for g in range(3)}
   d['decomposition_error']=float(abs(sum(v['weighted_score_contribution']['mean'] for v in d['instruction_contributions'].values())-d['overall_score_difference']['conditional_mean']['overall_13']['mean']))
   assert d['decomposition_error']<1e-10;gaps[f'{a}_minus_{b}']=d
 carry_stats={}
 for key,coefs in CONTRASTS.items():
  byscene={};weighteds=[];values=[]
  for scene in carry['C0']:
   v=sum(carry[a][scene]*w for a,w in coefs.items());g=gids_for(m['scenarios'][scene]);slots=int((np.maximum.accumulate(g==2)&(g!=2)).sum())
   byscene[scene]=dict(nonquality_slots_after_quality=slots,difference=ci(v,boot),by_parent={str(s):ci(v[i],boot) for i,s in enumerate(m['parents'])})
   weighteds.append(v*slots/(13*600));values.append((v,slots))
  carry_stats[key]=dict(by_scenario=byscene,slot_weighted_after_quality_mean=ci(sum(v*n for v,n in values)/sum(n for _,n in values),boot),contribution_already_in_overall_13=ci(sum(weighteds),boot),interpretation='pure resource-history contrast under identical current nonquality routing' if key in ('C2_minus_C0','C3_minus_C1') else 'may include different current routing; not purely quality-history effect')
 results=dict(schema=1,unit='mean original common reward per slot x100',raw_unit='original common reward',split='development_validation',parent_seeds=m['parents'],validation_seeds=m['validation'],scenarios=m['scenarios'],score_summary=summary,episode_results=episode,post_quality_carryover=carry_stats,statistical_unit='20 paired environment seeds; all 13 scenes, 3 fixed parents, all methods kept together',bootstrap_seed=m['bootstrap_seed'],bootstrap_replicates=4000,confidence_interval_scope='conditional on these three fixed parent models and reused development environments, not random training population',new_training_steps=0,new_optimizer_updates=0,historical_unique_training_steps=9000000,lineage_steps_per_parent={'C0':2000000,'C1':3000000,'C2':2000000,'C3':3000000},reserved_final_test_used=False,predicted_psnr_notice='Average-profile predicted PSNR, delivery-weighted, not measured decoded video quality')
 from focus_statistics import quality_focus
 focus=quality_focus(physical,m['parents'],m['scenarios']);write(report/'focus_case.json',focus)
 write(report/'results.json',results);write(report/'physical_statistics.json',physical);write(report/'paired_differences.json',paired);write(report/'gap_decomposition.json',gaps)
 ledger=[read(p) for p in sorted((HERE/'compute_ledger').glob('*.json'))];counts={}
 for kind in ('primary','preflight'):
  q=[x for x in ledger if x['kind']==kind];counts[kind]=dict(attempts=len(q),complete_episodes=sum(x['episodes'] for x in q),physical_steps=sum(x['physical_steps'] for x in q),failed_attempts=sum(x['state']=='failed' for x in q))
 assert counts['primary']['complete_episodes']==2340 and counts['primary']['physical_steps']==1404000
 assert len(audits)==325 and len(routechecks)==117
 unchanged=[]
 for p,h in m['protected_input_sha256'].items():
  if sha(WORKSPACE/p)!=h:unchanged.append(p)
 assert not unchanged
 gitbefore=read(HERE/'git_inspection.json');gitdiff={}
 for root,oldgit in gitbefore.items():
  run=subprocess.run(['git','-C',root,'status','--porcelain=v1'],capture_output=True,text=True,env={**os.environ,'GIT_OPTIONAL_LOCKS':'0'})
  if run.stdout!=oldgit['status']['stdout']:gitdiff[root]=dict(before=oldgit['status']['stdout'],after=run.stdout)
 assert not gitdiff
 audit=dict(state='PASS',protected_files=len(m['protected_input_sha256']),protected_input_changes=unchanged,git_status_changes=gitdiff,protocol_sha256=sha(HERE/'PROTOCOL.md'),manifest_sha256=sha(HERE/'manifest.json'),execution_seal_sha256=sha(HERE/'execution_seal.json'),new_training_steps=0,new_optimizer_updates=0,forbidden_training_calls=FORBIDDEN_CALLS,reserved_final_test_used=False,actual_trace_seeds=sorted(seen),reserved_intersection=sorted(seen&set(m['reserved_final_test'])),environment_creation_whitelist_asserted=True,new_primary=counts['primary'],preflight_and_reproducibility=counts['preflight'],reused=dict(complete_episodes=4160,physical_steps_previously_computed=2496000,traces=208,new_physical_steps=0),reference_supplements=dict(episodes=0,physical_steps=0,missing=[]),preflight_reference_inference_only_slots=read(HERE/'preflight.json')['reference_inference_only_slots'],aggregation_policy_inference_only_slots=1404000,profile_table_reconstruction_physical_steps=0,every_new_slot_original_checker=True,independent_offline_audits=audits,new_controller_saved_input_route_audits=routechecks,trace_sha256=tracehash,external_hashes=external_hashes,failures=[read(p) for p in sorted((HERE/'preflight_failures').glob('*.json'))],results_sha256=sha(report/'results.json'),physical_statistics_sha256=sha(report/'physical_statistics.json'),paired_sha256=sha(report/'paired_differences.json'),gap_sha256=sha(report/'gap_decomposition.json'),focus_sha256=sha(report/'focus_case.json'),analysis_manifest_sha256=sha(HERE/'analysis_manifest.json'))
 write(report/'audit.json',audit)
 from report_writer import render
 text=render(results,physical,paired,gaps,audit,focus);tmp=report/'REPORT.md.tmp';tmp.write_text(text);tmp.replace(report/'REPORT.md')
 print(json.dumps(dict(aggregation='PASS',new_episodes=2340,all_traces=325)),flush=True)

if __name__=='__main__':guard();aggregate()
