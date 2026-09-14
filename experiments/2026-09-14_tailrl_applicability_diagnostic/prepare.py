from diag_support import *
def main():
 guard();assert not (HERE/'manifest.json').exists();old=read(LONG/'manifest.json');models={};protected=dict(old['protected_sha256']);paths=set()
 for seed in old['seeds']:
  for steps in [1000000,6000000]:
   p=(SHORT if steps==1000000 else LONG)/f'jobs/seed_{seed}/milestones/steps_{steps}';s=read(p/'status.json');assert s['state']=='complete' and s['completed_steps']==steps
   models[f'{seed}/{steps}']=dict(path=str(p.relative_to(ROOT)),status=s)
   for f in p.iterdir():
    if f.is_file():paths.add(f)
  for family,folder in [('short',SHORT),('long',LONG)]:
   for name in ['training_metrics.jsonl','timing.jsonl']:
    p=folder/f'jobs/seed_{seed}/{name}';lines=[]
    for line in p.read_text().splitlines():
     try:r=json.loads(line)
     except ValueError:continue
     if r['steps']<=6000000:lines.append(line)
    textfile(HERE/f'log_snapshots/{family}_{seed}_{name}','\n'.join(lines)+'\n')
 for p in paths:protected[str(p.relative_to(ROOT))]=sha(p)
 rng=np.random.default_rng(202609141420);actions=rng.choice(np.arange(600000001,700000000,dtype=np.int64),16,replace=False).tolist()
 m=dict(created_utc=stamp(),parents=old['seeds'],validation=old['validation'],reserved=old['reserved'],scenarios=old['scenarios'],effective_weights=old['effective_weights'],equivalence_groups=old['equivalence_groups'],models=models,action_seeds=actions,bootstrap_seed=202609141421,bootstrap_repeats=4000,quality_thresholds=[.001,.005,.01],psnr_thresholds=[.1,.5,1.],sampling_checkpoints=[1000000,6000000],sampling_scenes=['fixed_2','switch300_2_to_0','switch300_2_to_1'],expected_new_episodes=3060,expected_new_physical_steps=1836000,protocol_sha256=sha(HERE/'PROTOCOL.md'),protected_sha256=protected,new_training_steps=0,new_optimizer_updates=0,active_long_training_status_at_start=read(LONG/'status.json'),log_snapshots={str(p.relative_to(HERE)):sha(p) for p in (HERE/'log_snapshots').glob('*')})
 write(HERE/'manifest.json',m);write(HERE/'status.json',dict(state='prepared',new_training_steps=0,new_optimizer_updates=0));write(HERE/'resource_start.json',dict(cpu_count=os.cpu_count(),load=os.getloadavg(),gpu=subprocess.run(['nvidia-smi','--query-gpu=name,utilization.gpu,memory.used,temperature.gpu','--format=csv'],capture_output=True,text=True).stdout));print('PREPARED',len(protected),flush=True)
if __name__=='__main__':main()
