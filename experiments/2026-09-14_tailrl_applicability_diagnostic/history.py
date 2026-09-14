from diag_support import *
import re

def normalized(path):
 z=arrays(path)
 if 'env_seeds' in z:
  z.update(seeds=z['env_seeds'],requested_modes=z['sampled_modes'],modes=z['executed_modes'],predicted_quality=z['quality_predicted_uav'],served=z['served_ds'],raw_resource_action=z['resource_raw'])
 return z

def detail(x,reference=None):
 out={k:distribution(v) for k,v in x.items() if np.asarray(v).ndim==1 and k not in ['valid','no_delivery']};out['correlations']={k:corr(x['reward'],x[k]) for k in ['quality','mean_aoi','age_mean_cost','age_max_cost','age_tail_cost','resource']}
 n=len(x['quality']);take=max(1,math.ceil(n*.1));qtop=set(np.argsort(x['quality'],kind='stable')[-take:]);rtop=set(np.argsort(x['reward'],kind='stable')[-take:]);out['top10_Q_given_top10_R']=len(qtop&rtop)/take
 if reference is not None:
  delta=x['quality']-reference['quality'];out['delta_quality']=distribution(delta);out['delta_psnr']=distribution(x['psnr']-reference['psnr']);out['success']={}
  for eps in manifest()['quality_thresholds']:
   improve=delta>eps;feasible=improve&(x['mean_aoi']<=6)&(x['violation']==0);safe=feasible.copy()
   for k in ['mean_aoi','max_aoi','age_tail_cost','resource']:safe&=x[k]<=reference[k]*1.05+1e-12
   out['success'][str(eps)]=dict(improvement=float(improve.mean()),feasible=float(feasible.mean()),safe=float(safe.mean()),strict_service=float((feasible&(x['max_aoi_ever']<=6)).mean()))
 return out

def main():
 guard();m=verify();sources=[];summary={};raw={};inventory={};seen=set();recompose=base.RECOMPOSE;repair=base.REPAIR;credit=HERE.parent/'2026-09-12_credit_assignment_probe'
 for family,folder in [('repair',repair/'evaluation'),('recomposition',recompose/'evaluation'),('half_short',SHORT/'evaluation'),('half_objective',prior.PREVIOUS/'evaluation'),('credit',credit/'execution')]:
  for p in sorted(folder.rglob('*.npz')):
   parts=p.relative_to(folder).parts;match=re.search(r'(?:seed_)?(104948945|111868397|160441552)',str(p));parent=int(match.group(1)) if match else None;scene=p.stem if p.stem!='trajectory' else str(np.load(p,allow_pickle=False)['scenario'])
   if scene not in m['scenarios']:continue
   if family=='credit':method='/'.join(parts[1:-1])
   elif family=='half_short':method=parts[1]
   else:method=parts[-2]
   sources.append(dict(family=family,parent=parent,method=method,scene=scene,path=str(p.relative_to(ROOT))))
 for k,v in read(recompose/'reference_index.json').items():
  parent,method,scene=k.split('/')
  if method=='C0':sources.append(dict(family='recomposition',parent=int(parent),method='C0',scene=scene,path=v['path'],alias=True))
 for item in sources:
  p=ROOT/item['path'];z=normalized(p);seeds=z['seeds'].tolist();assert seeds==m['validation'];h=sha(p)
  if 'trajectory' in p.name and item['family']=='credit':
   if str(z['variant']) in ['B','C','D']:item['action_seed']=int(z['action_seed'])
  key=f"{item['family']}/{item['parent'] or 'rules'}/{item['method']}/{item['scene']}"
  if 'action_seed' in item:key+='/a'+str(item['action_seed'])
  if key in summary:key+='@'+h[:8]
  item.update(key=key,sha256=h,episodes=len(seeds),duplicate_payload_sha256=h in seen);seen.add(h);inventory[key]=item;summary[key]={};raw[key]={}
  for gid in [None,0,1,2]:
   x=episode_metrics(z,gid)
   if not x['valid'].any():continue
   g='overall' if gid is None else str(gid);ref=None
   if item['parent']:
    refpath=SHORT/f"evaluation/seed_{item['parent']}/steps_1000000/{item['scene']}.npz" if item['family']=='half_short' else repair/f"evaluation/seed_{item['parent']}/original_rl/{item['scene']}.npz"
    if refpath.exists():ref=episode_metrics(normalized(refpath),gid)
   summary[key][g]=detail(x,ref);raw[key][g]={k:v.tolist() for k,v in x.items() if k not in ['psnr','max_aoi_ever']};raw[key][g]['psnr']=[None if not np.isfinite(v) else float(v) for v in x['psnr']];raw[key][g]['max_aoi_ever']=x['max_aoi_ever'].tolist()
  if len(summary)%50==0:print('history',len(summary),flush=True)
 audit=read(recompose/'report/audit.json');assert audit['new_primary']['complete_episodes']==2340 and audit['preflight_and_reproducibility']['complete_episodes']==1280
 repetitions=[]
 for folder in ['preflight','preflight_traces','preflight_repeat']:
  for p in sorted((recompose/folder).rglob('*.npz')):
   with np.load(p,allow_pickle=False) as z:
    if 'trace' in z and 'seeds' in z:repetitions.append(dict(path=str(p.relative_to(ROOT)),sha256=sha(p),seeds=z['seeds'].tolist(),physical_slots=int(np.prod(z['trace'].shape[:2]))))
 logs={}
 for p in sorted((HERE/'log_snapshots').glob('*training_metrics.jsonl')):
  # This index describes only the current half-weight short/long prefixes.
  # Older nested-schema logs have their own complete parser in training_evidence.py.
  if not p.name.startswith(('short_','long_')):continue
  rows=[json.loads(x) for x in p.read_text().splitlines()];logs[p.name]=dict(updates=len(rows),last_steps=rows[-1]['steps'],first_steps=rows[0]['steps'],instruction_counts=np.array([r['instruction_counts'] for r in rows]).sum(0).tolist(),first20_mean_training_reward=float(np.mean([r['mean_training_reward'] for r in rows[:20]])),last20_mean_training_reward=float(np.mean([r['mean_training_reward'] for r in rows[-20:]])),actual_historical_per_sample_advantage_available=False)
 write(HERE/'history_index.json',inventory);write(HERE/'report/history_distributions.json',summary);write(HERE/'report/history_episode_metrics.json',raw);write(HERE/'report/history_inventory.json',dict(records=len(inventory),logical_episodes=sum(x['episodes'] for x in inventory.values()),unique_payloads=len(seen),recomposition_formal=audit['new_primary'],recomposition_preflight=audit['preflight_and_reproducibility'],preflight_trace_index=repetitions,preflight_not_counted_as_independent_evidence=True,training_logs=logs));print('HISTORY COMPLETE',len(inventory),flush=True)
if __name__=='__main__':main()
