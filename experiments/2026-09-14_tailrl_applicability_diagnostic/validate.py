"""Final read-only input/provenance audit. No new rollouts."""
from diag_support import *

def main():
 guard();m=verify();hist=read(HERE/'history_index.json');extra=read(HERE/'training_evidence_index.json');hi=read(HERE/'report/history_inventory.json');old={k:v for k,v in m['protected_sha256'].items()}
 for v in hist.values():old[v['path']]=v['sha256']
 old.update(extra)
 for v in hi['preflight_trace_index']:old[v['path']]=v['sha256']
 for p,h in old.items():assert sha(ROOT/p)==h,p
 for p,h in m['log_snapshots'].items():assert sha(HERE/p)==h,p
 count=0;seeds=set();worst=0.;external={};identity={}
 for p in sorted((HERE/'evaluation').rglob('*.npz')):
  meta=read(p.with_suffix('.json'));assert meta['state']=='complete';assert sha(p)==meta['trace_sha256'];assert meta['identity']['manifest_sha256']==sha(HERE/'manifest.json');assert meta['identity']['source_sha256']==sha(HERE/'evaluate.py');assert meta['every_slot_original_checker'] and meta['models_unchanged']
  with np.load(p,allow_pickle=False) as z:
   assert z['trace'].shape[:2]==(600,20);assert z['seeds'].tolist()==m['validation'];seeds.update(map(int,z['seeds']))
   assert np.isfinite(z['old_logp']).all();assert z['trace'][:,:,FIELDS.index('instruction_id')].min()>=0
  external[str(p.relative_to(HERE))]=meta['external_hashes'];identity[str(p.relative_to(HERE))]=meta['trace_sha256'];count+=1
  worst=max(worst,meta['independent']['max_reward_reconstruction_error'])
 assert count==153;assert not(seeds&set(m['reserved']));assert count*20==m['expected_new_episodes'];assert count*12000==m['expected_new_physical_steps']
 rows=[json.loads(l) for p in (HERE/'costs').glob('*.jsonl') for l in p.read_text().splitlines()]
 assert sum(x['physical_steps'] for x in rows)==1836000;assert sum(x['complete_episodes'] for x in rows)==3060;assert all(x['error'] is None for x in rows)
 ga=read(HERE/'report/gradient_audit.json');assert ga['state']=='PASS' and ga['new_optimizer_updates']==0;co=read(HERE/'report/controlled_audit.json');assert co['state']=='PASS';assert read(HERE/'preflight.json')['state']=='PASS'
 c=read(HERE/'report/controlled_results.json');assert all(s['safe']['mean']==0 for v in c.values() for s in v['success'].values())
 source={str(p.relative_to(HERE)):sha(p) for p in sorted(HERE.glob('*.py'))};source['PROTOCOL.md']=sha(HERE/'PROTOCOL.md')
 failures=list((HERE/'failures').glob('*.json')) if (HERE/'failures').exists() else []
 report=dict(state='PASS',protected_files_checked=len(old),protected_file_changes=[],trace_count=count,new_complete_episodes=3060,new_physical_steps=1836000,new_training_steps=0,new_optimizer_updates=0,preflight_physical_steps=0,gradient_replay_new_physical_steps=0,existing_training_continued_independently=True,actual_new_environment_seeds=sorted(seeds),reserved_final_test_used=False,reserved_seed_intersection=[],original_checker_every_physical_slot=True,offline_independent_checker_all_trajectories=True,max_reward_reconstruction_error=worst,new_trace_sha256=identity,external_hashes=external,analysis_source_sha256=source,gradient_cost=ga['cost'],model_and_normalization_hashes_unchanged=True,historical_logical_trace_records=len(hist),historical_unique_payloads=hi['unique_payloads'],recomposition_formal_reuse=hi['recomposition_formal'],historical_preflight_not_independent=hi['recomposition_preflight'],execution_errors=[str(p.relative_to(HERE)) for p in failures],git_metadata_limitation='Workspace and empty HARL/.git are not an operative Git repository; file-hash provenance used',preflight_attempts_excluded_from_scientific_sample_size=True)
 write(HERE/'report/audit.json',report);print('AUDIT PASS',len(old),'protected files; 3060 new complete episodes; zero updates',flush=True)
if __name__=='__main__':main()
