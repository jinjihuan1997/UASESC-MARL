"""Repeat full aggregation from frozen raw inputs; never collects or trains."""
from diag_support import *
from concurrent.futures import ThreadPoolExecutor

OUTPUTS=['history_distributions.json','history_episode_metrics.json','history_inventory.json',
 'training_evidence.json','controlled_results.json','controlled_audit.json','checkpoint_tail_trend.json',
 'additional_results.json','results.json','REPORT.md']

def invoke(name,cpu):
 with (HERE/f'recompute_{name}.log').open('w') as out:
  subprocess.run(['taskset','-c',str(cpu),sys.executable,str(HERE/(name+'.py'))],stdout=out,stderr=subprocess.STDOUT,check=True,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})

def gradient_check():
 results=read(HERE/'report/gradient_results.json');worst=0
 for seed,v in results.items():
  for i,actor in v['actors'].items():
   z=arrays(HERE/f'gradients/seed_{seed}_actor_{i}.npz')
   for k,vec in z.items():
    np.testing.assert_allclose(np.linalg.norm(vec),actor['gradient_norm'][k],rtol=0,atol=0)
    for other,w in z.items():
     den=float(np.linalg.norm(vec)*np.linalg.norm(w));cos=float(np.dot(vec,w)/den) if den>1e-18 else None;assert cos==actor['cosine'][k][other]
 return 'PASS: saved per-actor gradient vectors reproduce every norm and cosine exactly'

def main():
 guard();verify();sources={p.name:sha(p) for p in sorted(HERE.glob('*.py'))};sources['PROTOCOL.md']=sha(HERE/'PROTOCOL.md')
 seal=dict(sources=sources,note='Implementation seal for repeat aggregation; scientific definitions and evaluation seeds were frozen in PROTOCOL and manifest before sampling')
 if (HERE/'execution_seal.json').exists():assert read(HERE/'execution_seal.json')==seal
 else:write(HERE/'execution_seal.json',seal)
 # Refresh the presentation-only report change before hashing numeric pass 1.
 invoke('aggregate',1);before={p:sha(HERE/'report'/p) for p in OUTPUTS};array_before={str(p.relative_to(HERE)):arrays(p) for p in sorted((HERE/'analysis_arrays').glob('*.npz'))}
 write(HERE/'reproducibility_pass1.json',dict(outputs=before,sources=sources))
 with ThreadPoolExecutor(max_workers=2) as pool:
  futures=[pool.submit(invoke,'history',0),pool.submit(invoke,'controlled_stats',1)]
  for f in futures:f.result()
 invoke('training_evidence',0);invoke('additional_analysis',1);invoke('aggregate',1)
 after={p:sha(HERE/'report'/p) for p in OUTPUTS};assert before==after,{p:(before[p],after[p]) for p in before if before[p]!=after[p]}
 for p,old in array_before.items():
  new=arrays(HERE/p);assert old.keys()==new.keys()
  for k in old:np.testing.assert_array_equal(old[k],new[k])
 gi=gradient_check();invoke('validate',1);audit=read(HERE/'report/audit.json');assert audit['state']=='PASS'
 write(HERE/'report/reproducibility.json',dict(state='PASS',full_aggregation_passes=2,all_output_sha256_equal=True,outputs=after,analysis_arrays_elementwise_equal=True,gradient_vector_reaggregation=gi,no_new_physical_steps=True,no_new_optimizer_steps=True,excluded_metadata=['gradient_audit elapsed seconds','resource timestamps','status timestamps']))
 write(HERE/'status.json',dict(state='complete',task='TailRL applicability diagnostic',scope_completed=True,new_complete_evaluation_episodes=3060,new_physical_steps=1836000,new_training_steps=0,new_optimizer_updates=0,protected_files_checked=audit['protected_files_checked'],reserved_final_test_used=False,report='report/REPORT.md',verdict='No evidence for likely non-regressing TailRL improvement; partial quality-only tail with AoI costs',finished_utc=stamp()))
 print('REPRODUCIBILITY PASS; DIAGNOSTIC COMPLETE',flush=True)
if __name__=='__main__':main()
