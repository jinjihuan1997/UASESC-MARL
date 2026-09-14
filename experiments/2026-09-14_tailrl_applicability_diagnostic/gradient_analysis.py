"""Counterfactual score-function derivatives, no backward or optimizer step."""
from diag_support import *
from controlled_stats import trace_path, tail_matrix, normalized, successes

LABELS=['baseline_GAE','tail_quality','tail_total','expected_quality',
        'negative_AoI_cost','negative_resource_cost','feasible_success','safe_success',
        'negative_service_exceedance','tail_quality_feasible_part','baseline_GAE_feasible_part']

def cosine(a,b):
 den=float(np.linalg.norm(a)*np.linalg.norm(b))
 return float(np.dot(a,b)/den) if den>1e-18 else None

def inputs(z,i):
 obs=z['sut_obs'][:,:8] if i==0 else z['post_uav_obs'][:,:8,i-1]
 if i==0:
  act=z['raw_resource_action'][:,:8];mask=np.zeros((600,8,16),np.float32);mask[:,:,:3]=1
 else:
  act=np.eye(16,dtype=np.float32)[z['requested_modes'][:,:8,i-1]];mask=z['uav_masks'][:,:8,i-1]
 return [a.reshape(-1,a.shape[-1]) for a in [obs,act,mask]]

def all_weights(zs):
 xlist=[episode_metrics(z,2) for z in zs];x={k:np.stack([a[k][:8] for a in xlist]) for k in xlist[0]}
 seed=int(zs[0]['parent_seed']);d={k:v[:8] for k,v in episode_metrics(arrays(trace_path(seed,6000000,'fixed_2')),2).items()}
 s=successes(x,d,.005);weights={};raw={}
 raw['tail_quality']=tail_matrix(x['quality']);raw['tail_total']=tail_matrix(x['reward'])
 raw['expected_quality']=x['quality']-x['quality'].mean(0)
 age=-(x['age_mean_cost']+x['age_max_cost']+x['age_tail_cost']);res=-x['resource_cost']
 raw.update(negative_AoI_cost=age-age.mean(0),negative_resource_cost=res-res.mean(0))
 for label,key in [('feasible_success','feasible'),('safe_success','safe')]:
  val=s[key].astype(float);raw[label]=val-val.mean(0)
 val=-x['fraction_above6'];raw['negative_service_exceedance']=val-val.mean(0)
 for k,a in raw.items():weights[k]=np.broadcast_to(normalized(a)[:,None,:],(16,600,8)).copy()
 raw_gae=np.stack([z['gae_raw'][:,:8] for z in zs]);weights['baseline_GAE']=normalized(raw_gae)
 sm=s['feasible'][:,None,:]
 weights['tail_quality_feasible_part']=weights['tail_quality']*sm
 weights['baseline_GAE_feasible_part']=weights['baseline_GAE']*sm
 info=dict(episodes=128,physical_slots=76800,feasible_success_count=int(s['feasible'].sum()),safe_success_count=int(s['safe'].sum()),normalization='Each main estimator uses all 16x600x8 samples, common denominator; not historical 4000-sample batch replay',raw_estimator_std={k:float(v.std()) for k,v in raw.items()},gae_std=float(raw_gae.std()))
 return {k:weights[k].reshape(-1).astype(np.float32) for k in LABELS},info

def run():
 guard();m=verify();assert read(HERE/'evaluation_complete.json')['state']=='complete'
 plan=dict(checkpoint=6000000,scene='fixed_2',parents=m['parents'],environments=m['validation'][:8],action_repetitions=16,head_names=LABELS,batch_size=1536,logp_absolute_tolerance=1e-5,ratio_absolute_tolerance=1e-5,autograd_method='torch.autograd.grad; ascent gradient mean(logpi * detached weight)',optimizer_instances=0,optimizer_steps=0)
 if (HERE/'gradient_plan.json').exists():assert read(HERE/'gradient_plan.json')==plan
 else:write(HERE/'gradient_plan.json',plan)
 results={};cost={};started=time.perf_counter()
 for seed in m['parents']:
  zs=[]
  for r in range(16):
   z=arrays(trace_path(seed,6000000,'fixed_2',r));z['parent_seed']=seed;zs.append(z)
  weights,wi=all_weights(zs);env=ws.make_env('fixed_2');pol=Policy(seed,6000000,env);actors={};by_actor=[];forward=0;vjp=0
  for i,net in enumerate(pol.nets):
   net.requires_grad_(True);params=list(net.parameters());names=[n for n,p in net.named_parameters()];sizes=[p.numel() for p in params]
   data=[inputs(z,i) for z in zs];obs,act,mask=[torch.as_tensor(np.concatenate([a[j] for a in data])) for j in range(3)];old=np.concatenate([z['old_logp'][:,:8,i].reshape(-1) for z in zs]);total=len(obs)
   grads={k:np.zeros(sum(sizes),np.float64) for k in LABELS};le=0.;re=0.
   for start in range(0,total,plan['batch_size']):
    sl=slice(start,min(start+plan['batch_size'],total));n=len(obs[sl]);rnn=torch.zeros((n,1,256));ones=torch.ones((n,1))
    lp,entropy,_=net.evaluate_actions(obs[sl],rnn,act[sl],ones,mask[sl],ones);lp=lp.sum(-1);forward+=n
    now=lp.detach().numpy();assert np.isfinite(now).all();err=float(np.max(np.abs(now-old[sl])));rat=float(np.max(np.abs(np.exp(now-old[sl])-1)));le=max(le,err);re=max(re,rat)
    assert err<=plan['logp_absolute_tolerance'] and rat<=plan['ratio_absolute_tolerance'],(seed,i,start,err,rat)
    for h,k in enumerate(LABELS):
     w=torch.as_tensor(weights[k][sl]);objective=(lp*w).sum()/total
     gs=torch.autograd.grad(objective,params,retain_graph=h<len(LABELS)-1,allow_unused=True);vjp+=1
     flat=np.concatenate([(np.zeros(p.numel(),np.float32) if g is None else g.detach().numpy().reshape(-1)) for p,g in zip(params,gs)])
     assert np.isfinite(flat).all();grads[k]+=flat
   net.requires_grad_(False);pol.assert_frozen();npz(HERE/f'gradients/seed_{seed}_actor_{i}.npz',**grads)
   actors[str(i)]=dict(parameter_names=names,parameter_sizes=sizes,gradient_norm={k:float(np.linalg.norm(g)) for k,g in grads.items()},cosine={a:{b:cosine(grads[a],grads[b]) for b in LABELS} for a in LABELS},max_logp_error=le,max_ratio_error=re)
   by_actor.append(grads);print('GRADIENT PASS',seed,i,'logp',le,flush=True)
  joint={k:np.concatenate([a[k] for a in by_actor]) for k in LABELS}
  results[str(seed)]=dict(weights=wi,actors=actors,joint=dict(gradient_norm={k:float(np.linalg.norm(g)) for k,g in joint.items()},cosine={a:{b:cosine(joint[a],joint[b]) for b in LABELS} for a in LABELS}),models_unchanged=pol.hashes()==pol.initial_hashes)
  cost[str(seed)]=dict(actor_forward_samples=forward,vector_jacobian_products=vjp,physical_steps=0,optimizer_steps=0);write(HERE/'report/gradient_results.partial.json',results)
 write(HERE/'report/gradient_results.json',results);write(HERE/'report/gradient_audit.json',dict(state='PASS',plan=plan,cost=cost,seconds=time.perf_counter()-started,new_physical_steps=0,new_training_steps=0,new_optimizer_updates=0,model_and_normalization_hashes_unchanged=True,autograd_grad_only=True,forbidden_calls=FORBIDDEN))
if __name__=='__main__':
 try:run()
 except BaseException as e:write(HERE/f'failures/gradient_{time.time_ns()}.json',dict(error=repr(e),traceback=traceback.format_exc()));raise
