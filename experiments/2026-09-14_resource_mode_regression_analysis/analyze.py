"""Read saved trajectories and run frozen networks; never advance a physical slot."""
from analysis_support import *
from dataclasses import replace

def distribution(policy,i,obs,mask,actions):
 probs=[]; acts=[]; ent=[]; max_logp_error=0.
 for start in range(0,len(obs),512):
  o=torch.as_tensor(obs[start:start+512],dtype=torch.float32)
  m=torch.as_tensor(mask[start:start+512]);a=torch.as_tensor(actions[start:start+512],dtype=torch.float32)
  n=len(o);r=torch.zeros(n,1,256);ones=torch.ones(n,1)
  action,lp,_=policy.networks[i](o,r,ones,m,deterministic=True)
  elp,_,d=policy.networks[i].evaluate_actions(o,r,action,ones,m,ones)
  max_logp_error=max(max_logp_error,float((lp-elp).abs().max()))
  q=d.components[0]['dist']
  probs.append((q.concentration if i==0 else q.probs.squeeze(1)).cpu().numpy())
  ent.append(q.entropy().reshape(n,-1).sum(-1).cpu().numpy())
  acts.append(action.cpu().numpy())
 assert max_logp_error<=1e-6,max_logp_error
 return np.concatenate(probs),np.concatenate(acts),np.concatenate(ent),max_logp_error

def protected_paths():
 paths={ROOT/p for p in MAN['protected_sha256']}
 for root in [SHORT,LONG]:
  paths.update(p for p in root.glob('*.py'))
  paths.update(p for p in root.glob('*.json'))
  paths.update(p for p in root.glob('report/*') if p.is_file())
  paths.update(root.glob('configs/*.json'))
 for seed in SEEDS:
  for root in [SHORT,LONG]:paths.add(root/f'jobs/seed_{seed}/training_metrics.jsonl')
  for step in STEPS:
   model=LONG/f'jobs/seed_{seed}/milestones/steps_{step}'
   paths.add(model/'status.json');paths.update(model.glob('actor_agent*.pt'))
   for scene in ['fixed_0','fixed_1','fixed_2']:
    p=trace(seed,step,scene);paths.add(p);paths.add(p.with_suffix('.json'))
 return sorted(paths)

def profile_summary(tab):
 return {str(g):dict(ids=g,load=stats(tab['load'][...,g[0]].ravel()),predicted_psnr=stats(tab['quality'][...,g[0]].ravel())) for g in GROUPS}

def quality_row(z,tab):
 before=np.concatenate([z['initial_q'][None],z['cache_after'][:-1]],axis=0)
 valid=(tab['quality']>=tab['req'][...,None]-1e-9)&(tab['load']<=z['budgets'][...,None]+1e-9)&before.any(-1)[...,None]
 np.testing.assert_array_equal(valid|~valid.any(-1,keepdims=True),z['uav_masks'])
 # Original masks mark all actions in a no-service state. Keep true validity separate.
 qreq=z['post_uav_obs'][...,74]*33
 np.testing.assert_allclose(qreq,tab['req'],atol=2e-6,rtol=0)
 count=z['served'].sum(-1);requested=z['requested_modes'];executed=z['modes']
 live=before.any(-1);midvalid=valid[...,MID].any(-1);low=np.isin(requested,LOW)
 greedy=ws.LIGHTPOL.ONLINE.UAVGreedy(list(range(16)),replace(ws.LIGHTPOL.ONLINE.PublicSpec(),weights=tuple(map(tuple,MAN['effective_weights']))))
 gt=[]
 o=z['post_uav_obs'].reshape(-1,76);mask=z['uav_masks'].reshape(-1,16)
 for k in range(0,len(o),512):gt.append(greedy.act(torch.tensor(o[k:k+512]),torch.tensor(mask[k:k+512])).argmax(-1).numpy())
 g=np.concatenate(gt).reshape(600,20,3)
 uniformbudget=ENV.p.delta_T*ENV.p.backhaul_availability*min(ENV.p.B_uav_sut,ENV.p.B_sut_sat/3)
 uniformvalid=(tab['quality'][...,5]>=tab['req']-1e-9)&(tab['load'][...,5]<=uniformbudget+1e-9)&live
 result=[]
 for u in range(3):
  ix=(slice(None),slice(None),u);c=count[ix];sel=executed[ix];psnr=z['predicted_quality'][ix]
  assert int(c.sum())>0
  modes={str(group):float(np.isin(sel,group).mean()) for group in GROUPS}
  low_mid=low[ix]&midvalid[ix]
  # Pending cache together with enough remaining budget for the executed mode should never occur.
  selected_load=np.take_along_axis(tab['load'][ix],np.maximum(sel,0)[...,None],-1)[...,0]
  # cache_after includes newly admitted source blocks; it is not necessarily unserved pre-cache.
  unserved_pre=(before[ix]&~z['served'][ix]).any(-1)
  unexplained=unserved_pre&(z['unused_budgets'][ix]>=selected_load+1e-9)&(sel>=0)
  assert not unexplained.any()
  result.append(dict(uav=u+1,budget=stats(z['budgets'][ix].ravel()),share=stats(z['resource_fractions'][ix].ravel()),unused_budget=stats(z['unused_budgets'][ix].ravel()),usage_mean=float(z['usage'][ix].mean()),deliveries_per_slot=float(c.mean()),predicted_psnr_delivery_weighted=float((psnr*c).sum()/c.sum()),mean_aoi=float(z['aoi_after'][ix].mean()),mean_local_max_aoi=float(z['aoi_after'][ix].max(-1).mean()),p95_aoi=float(np.quantile(z['aoi_after'][ix],.95)),no_cache_fraction=float((~live[ix]).mean()),no_valid_mode_fraction=float((~valid[ix].any(-1)).mean()),mid_feasible_fraction=float(midvalid[ix].mean()),mid_infeasible_while_cache_fraction=float((live[ix]&~midvalid[ix]).mean()),low_selected_fraction=float(low[ix].mean()),mid_selected_fraction=float(np.isin(requested[ix],MID).mean()),low_when_mid_feasible_fraction=float(low_mid.mean()),low_conditional_on_mid_feasible=float(low_mid.sum()/max(1,midvalid[ix].sum())),local_greedy_mid_on_same_inputs_fraction=float(np.isin(g[ix],MID).mean()),local_greedy_mid_given_rl_low_and_mid_feasible=float((np.isin(g[ix],MID)&low_mid).sum()/max(1,low_mid.sum())),same_input_greedy_exact_agreement=float((g[ix]==requested[ix]).mean()),same_input_greedy_group_agreement=float(np.array([np.isin(g[ix],grp)&np.isin(requested[ix],grp) for grp in GROUPS]).any(0).mean()),uniform_static_mid_feasible_fraction=float(uniformvalid[ix].mean()),uniform_static_newly_mid_feasible_fraction=float((uniformvalid[ix]&~midvalid[ix]).mean()),mode_group_fractions=modes,unserved_pre_cache_with_budget_for_executed_mode=int(unexplained.sum())))
 return result

def policy_probe(seed,step,z,common=None):
 p=EVAL.Policy(seed,step,ENV);output={};summ={};max_action_error=0
 for i in range(4):
  if i==0:
   o=z['sut_obs'].reshape(-1,76);mask=np.zeros((len(o),16),np.float32);mask[:,:3]=1;a=z['raw_resource_action'].reshape(-1,3)
  else:
   o=z['post_uav_obs'][:,:,i-1].reshape(-1,76);mask=z['uav_masks'][:,:,i-1].reshape(-1,16);a=np.eye(16,dtype=np.float32)[z['requested_modes'][:,:,i-1].ravel()]
  par,act,entropy,le=distribution(p,i,o,mask,a)
  if common is None:
   error=float(abs(act-a).max());max_action_error=max(error,max_action_error)
   if i:np.testing.assert_array_equal(act.argmax(-1),a.argmax(-1))
   else:np.testing.assert_allclose(act,a,atol=1e-6,rtol=0)
  output[str(i)]=par;output[f'action_{i}']=act
  if i==0:summ[str(i)]=dict(total_concentration=stats(par.sum(-1)),concentration_components_mean=par.mean(0).tolist(),entropy=stats(entropy))
  else:
   gp=np.column_stack([par[:,g].sum(-1) for g in GROUPS]);chosen=act.argmax(-1);cg=np.array([next(k for k,g in enumerate(GROUPS) if a in g) for a in chosen]);winner=gp.argmax(-1)
   lowidx=GROUPS.index(LOW);mididx=GROUPS.index(MID)
   summ[str(i)]=dict(entropy=stats(entropy),group_probability_mean={str(g):float(gp[:,k].mean()) for k,g in enumerate(GROUPS)},mode_probability_mean=par.mean(0).tolist(),top_two_probability_margin=stats(np.sort(par,axis=-1)[:,-1]-np.sort(par,axis=-1)[:,-2]),raw_argmax_group_differs_from_highest_group_mass_fraction=float((cg!=winner).mean()),raw_argmax_low_while_mid_group_mass_largest_fraction=float(((cg==lowidx)&(winner==mididx)).mean()),argmax_low_fraction=float((cg==lowidx).mean()),argmax_mid_fraction=float((cg==mididx).mean()))
 p.assert_frozen()
 save(HERE/f'numerics/seed_{seed}/steps_{step}_{"common10M" if common else "own"}.npz',**output)
 return output,summ,max_action_error

def common_comparison(outs):
 result={}
 for left,right in [(1000000,4000000),(4000000,10000000),(1000000,10000000)]:
  d={}
  for i in range(4):
   p=outs[left][str(i)].astype(np.float64);q=outs[right][str(i)].astype(np.float64)
   if i==0:
    k=torch.distributions.kl_divergence(torch.distributions.Dirichlet(torch.tensor(p)),torch.distributions.Dirichlet(torch.tensor(q))).numpy()
    a=outs[left]['action_0'].astype(np.float64);b=outs[right]['action_0'].astype(np.float64)
    def shares(x):
     x=((x+1)/2).clip(0,1);x=x/x.sum(-1,keepdims=True);return ENV.p.beta_sat_lower_bound+(1-3*ENV.p.beta_sat_lower_bound)*x
    extra=dict(abs_resource_share_error_mean_by_uav=abs(shares(a)-shares(b)).mean(0).tolist())
   else:
    terms=np.zeros_like(p);nz=p>0;assert np.all(q[nz]>0);terms[nz]=p[nz]*(np.log(p[nz])-np.log(q[nz]));k=terms.sum(-1)
    a=outs[left][f'action_{i}'].argmax(-1);b=outs[right][f'action_{i}'].argmax(-1)
    extra=dict(exact_action_disagreement=float((a!=b).mean()),physical_group_disagreement=float((~np.stack([np.isin(a,g)&np.isin(b,g) for g in GROUPS]).any(0)).mean()),low_fraction_before=float(np.isin(a,LOW).mean()),low_fraction_after=float(np.isin(b,LOW).mean()),mid_fraction_before=float(np.isin(a,MID).mean()),mid_fraction_after=float(np.isin(b,MID).mean()))
   assert np.isfinite(k).all() and k.min()>-1e-8
   d[str(i)]=dict(kl_old_to_new=stats(k),**extra)
  result[f'{left}_to_{right}']=d
 return result

def logs(seed):
 rows=[]
 for root in [SHORT,LONG]:rows.extend(json.loads(s) for s in (root/f'jobs/seed_{seed}/training_metrics.jsonl').read_text().splitlines())
 assert [r['steps'] for r in rows]==list(range(4000,10000001,4000))
 out=[]
 for million in range(1,11):
  batch=rows[(million-1)*250:million*250];n=np.sum([r['instruction_counts'] for r in batch],0);r=np.sum([r['instruction_reward_sums'] for r in batch],0);m=np.sum([r['mode_budget_counts'] for r in batch],0)
  np.testing.assert_array_equal(m.sum((-1,-2)),np.repeat(n[:,None],3,axis=1))
  ae=np.asarray([r['actor_loss_entropy_grad_ratio'] for r in batch]);cv=np.asarray([r['critic_loss_grad'] for r in batch])
  assert np.isfinite(ae).all() and np.isfinite(cv).all()
  assert all(r['actor_learning_rates']==[.0001]*4 and r['critic_learning_rate']==.0004 and r['trained_actor_ids']==list(range(4)) for r in batch)
  modes=m.sum(-2)
  out.append(dict(end_steps=million*1000000,samples=n.tolist(),score_by_instruction=(r/n*100).tolist(),joint_mean_score=float(sum(x['mean_training_reward'] for x in batch)/250*100),actor_loss=stats(ae[:,0]),actor_entropy_mixed_summary=stats(ae[:,1]),actor_preclip_gradient_norm_mixed_summary=stats(ae[:,2]),ratio_mixed_summary=stats(ae[:,3]),critic_loss=stats(cv[:,0]),critic_gradient_norm=stats(cv[:,1]),quality_low_execution_fraction_by_uav=(modes[2][:,np.array(LOW)+1].sum(-1)/n[2]).tolist(),quality_mid_execution_fraction_by_uav=(modes[2][:,np.array(MID)+1].sum(-1)/n[2]).tolist(),all_instruction_resource_share_mean=np.mean([r['resource_share_mean'] for r in batch],0).tolist()))
 return out

def run():
 global ENV
 guard();start=time.time();print('hash inputs',flush=True)
 ph={str(p.relative_to(ROOT)):sha(p) for p in protected_paths()}
 for p,h in MAN['protected_sha256'].items():assert ph[p]==h,p
 manifest=dict(protocol_sha256=sha(HERE/'PROTOCOL.md'),protected_input_sha256=ph,seeds=SEEDS,steps=STEPS,validation=MAN['validation'],reserved=MAN['reserved'],existing_costs_only=True,new_training_steps=0,new_physical_steps=0,new_optimizer_updates=0,network_threads=1,model_probe_batch=512,software=dict(python=sys.version,torch=torch.__version__,numpy=np.__version__),old_analysis_source=sha(LONG/'report/results.json'),analysis_sources={p.name:sha(p) for p in HERE.glob('*.py')})
 dump(HERE/'manifest.json',manifest);dump(HERE/'status.json',dict(state='running'))
 ENV=ws.make_env('fixed_2');tab=ws.PHYS.tables(ENV);result=dict(profile=profile_summary(tab),models={},training_logs={});seen=set();audits={};inference=0
 for seed in SEEDS:
  rr={}
  for step in STEPS:
   fixed={}
   for scene in ['fixed_0','fixed_1','fixed_2']:
    path=trace(seed,step,scene);z=arrays(path);meta=read(path.with_suffix('.json'))
    assert meta['trace_sha256']==sha(path) and meta['original_checker']=='PASS_every_slot'
    assert list(z['seeds'])==MAN['validation'];seen.update(map(int,z['seeds']))
    v=z['trace'];f={k:float(v[:,:,FIELDS.index(k)].mean()) for k in FIELDS}
    np.testing.assert_allclose(v[:,:,0],v[:,:,15]-v[:,:,16:21].sum(-1)+v[:,:,21],atol=1e-9,rtol=0)
    fixed[scene]=dict(score_x100=f['common_reward']*100,score_by_environment=(v[:,:,0].mean(0)*100).tolist(),metrics=f)
   assert meta['external_hashes']==tab['external']
   pr=quality_row(z,tab);_,model,er=policy_probe(seed,step,z);inference+=48000
   rr[str(step)]=dict(fixed_tasks=fixed,quality_physical=pr,quality_policy=model,replayed_action_max_error=er)
   audits[f'{seed}/{step}']=dict(trace_sha256=meta['trace_sha256'],source_independent_check=meta['independent'],action_replay_max_error=er,mask_reconstruction='PASS')
   dump(HERE/f'report/seed_{seed}_partial.json',rr)
   print('quality analysed',seed,step,flush=True)
  z=arrays(trace(seed,10000000));outs={};summary={}
  for step in [1000000,4000000,10000000]:
   out,su,_=policy_probe(seed,step,z,common=True);outs[step]=out;summary[str(step)]=su;inference+=48000
  result['models'][str(seed)]=dict(checkpoints=rr,common_10M_input_policy_summaries=summary,common_input_drift=common_comparison(outs))
  result['training_logs'][str(seed)]=logs(seed)
  dump(HERE/'report/results.json',result)
  print('seed done',seed,flush=True)
 assert not seen&set(MAN['reserved']);assert ENV.step_index==0
 changed=[p for p,h in ph.items() if sha(ROOT/p)!=h];assert not changed,changed
 dump(HERE/'report/audit.json',dict(state='PASS',protected_input_count=len(ph),changed=changed,new_physical_steps=0,new_training_steps=0,new_optimizer_updates=0,profile_only_environment_initializations=1,frozen_actor_input_records=inference,frozen_actor_observation_forward_evaluations=2*inference,local_greedy_observation_queries=18*36000,reserved_final_test_used=False,actual_saved_seeds=sorted(seen),reserved_intersection=[],saved_trace_validation=audits,elapsed_seconds=time.time()-start,exceptions_log='failures.jsonl'))
 dump(HERE/'status.json',dict(state='numeric_analysis_complete',new_training_steps=0,new_physical_steps=0))

if __name__=='__main__':
 try:run()
 except BaseException as e:
  import traceback
  with (HERE/'failures.jsonl').open('a') as f:f.write(json.dumps(dict(error=repr(e),traceback=traceback.format_exc()))+'\n')
  raise
