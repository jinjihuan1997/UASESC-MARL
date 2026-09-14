from diag_support import *

def main():
 guard();m=manifest();logs={};traces={};inputs={}
 for folder in [FROZEN,base.REPAIR]:
  for p in sorted((folder/'jobs').rglob('training_metrics.jsonl')):
   if not any(f'seed_{s}/' in str(p) for s in m['parents']):continue
   text=p.read_text();rel=str(p.relative_to(ROOT));inputs[rel]=sha(p);dest=HERE/'log_snapshots'/rel.replace('/','__');textfile(dest,text)
 for p in sorted((HERE/'log_snapshots').glob('*training_metrics.jsonl')):
  rows=[json.loads(s) for s in p.read_text().splitlines()];r=rows[0];curve=[]
  for lo in range(0,len(rows),20):
   win=rows[lo:lo+20];means=[];counts=[];reward=[]
   for x in win:
    c=x.get('collection',x);v=c.get('mean_training_reward',c.get('mean_reward'));means.append(v)
    cnt=c.get('instruction_counts');sums=c.get('instruction_reward_sums')
    if cnt is not None:counts.append(cnt)
    if sums is not None:reward.append(sums)
   rec=dict(first_steps=win[0]['steps'],last_steps=win[-1]['steps'],updates=len(win),mean_training_reward=float(np.mean([x for x in means if x is not None])) if any(x is not None for x in means) else None)
   if counts:rec['instruction_counts']=np.array(counts).sum(0).tolist()
   if counts and reward:rec['instruction_reward_mean']=(np.array(reward).sum(0)/np.maximum(np.array(counts).sum(0),1)).tolist()
   curve.append(rec)
  logs[p.name]=dict(snapshot_sha256=sha(p),updates=len(rows),window20=curve,per_sample_historical_gradient_available=False)
 for folder in [SHORT,LONG]:
  for seed in m['parents']:
   for p in sorted((folder/f'jobs/seed_{seed}/trajectory_audits').glob('*.npz')):
    episode=int(p.stem.split('_')[-1]);steps_at_end=(episode+1)*6000
    if steps_at_end>6000000:continue
    meta=read(p.with_suffix('.json'));assert meta['state']=='PASS' and meta['trace_sha256']==sha(p)
    z=arrays(p);assert set(z['seeds'].tolist()).isdisjoint(m['reserved']);out={}
    for gid in [None,0,1,2]:
     x=episode_metrics(z,gid);valid=x['valid']
     if not valid.any():continue
     out['overall' if gid is None else str(gid)]={k:distribution(v[valid]) for k,v in x.items() if v.ndim==1 and k not in ['valid','no_delivery']}
    rel=str(p.relative_to(ROOT));inputs[rel]=sha(p);traces[rel]=dict(episode=episode,steps_at_end=steps_at_end,base_environment_seeds=z['seeds'].tolist(),physical_slots=int(np.prod(z['trace'].shape[:2])),recorded_independent_audit=meta['independent_audit'],metrics=out,interpretation='Previously collected nonstationary training: actors can change at 400-slot boundaries, not an N-rollout frozen-policy TailRL group')
 write(HERE/'training_evidence_index.json',inputs);write(HERE/'report/training_evidence.json',dict(logs=logs,archived_training_traces=traces,new_physical_steps=0,caution='All reward series retain their own training objective; historical original-weight and half-weight learning curves are not interchangeable'))
 print('training evidence',len(logs),'logs',len(traces),'traces',flush=True)
if __name__=='__main__':main()
