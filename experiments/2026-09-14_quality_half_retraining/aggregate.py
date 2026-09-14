"""Independent full-trajectory checks, paired scores and honest cost accounting."""
from train_support import *
STATS=base.load_module('retraining_frozen_statistics',ws.LIGHT/'aggregate.py')

@torch.inference_mode()
def main():
 for filename,h in read(HERE/'analysis_seal.json')['sources'].items():assert sha(HERE/filename)==h
 m=verify();prior=ws.manifest();m.update(parents=m['seeds'],equivalence_groups=prior['equivalence_groups'])
 assert read(HERE/'preflight.json')['state']=='PASS';scenes=list(m['scenarios']);boot=np.random.default_rng(m['bootstrap_seed']).integers(0,20,(4000,20))
 ix=dict(overall_13=list(range(13)),balance_fixed=[scenes.index('fixed_0')],aoi_fixed=[scenes.index('fixed_1')],quality_fixed=[scenes.index('fixed_2')],switch_10=[i for i,s in enumerate(scenes) if not s.startswith('fixed_')])
 ref=arrays(PREVIOUS/'report/episode_scores.npz');assert ref['parents'].tolist()==m['seeds'] and ref['seeds'].tolist()==m['validation'] and ref['scenarios'].tolist()==scenes
 oldaudit=read(PREVIOUS/'report/independent_audit.json');assert oldaudit['state']=='PASS' and oldaudit['reserved_final_test_used'] is False
 for key,h in oldaudit['trace_sha256'].items():assert sha(PREVIOUS/'evaluation'/f'{key}.npz')==h
 scores={name:ref['new_score_x100'][i] for i,name in enumerate(ref['methods'].tolist())};scores['retrained_rl_Qhalf']=np.zeros((3,13,20))
 physical={};checks={};tables={};process={};episodes={};hashes={}
 for pi,seed in enumerate(m['seeds']):
  for step in [200000,400000,600000,800000,1000000]:
   for scene in (scenes if step==1000000 else ['fixed_0','fixed_1','fixed_2']):
    key=f'seed_{seed}/steps_{step}/{scene}';path=HERE/'evaluation'/f'{key}.npz';meta=read(path.with_suffix('.json'));z=arrays(path)
    assert meta['state']=='complete' and meta['trace_sha256']==sha(path) and meta['policy_unchanged'] and meta['original_checker']=='PASS_every_slot'
    assert meta['identity']['manifest_sha256']==sha(HERE/'manifest.json') and meta['identity']['source_sha256']==sha(HERE/'evaluate.py')
    assert z['seeds'].tolist()==m['validation'] and not set(z['seeds'])&set(m['reserved'])
    if scene not in tables:tables[scene]=ws.PHYS.tables(ws.make_env(scene))
    assert meta['external_hashes']==oldaudit['external_hashes'][scene]==tables[scene]['external']
    checks[key]=ws.independent_check(z,tables[scene]);hashes[key]=sha(path)
    gids=np.zeros(600,dtype=int)
    for t,g in m['scenarios'][scene]:gids[t:]=g
    np.testing.assert_array_equal(z['trace'][:,:,FIELDS.index('instruction_id')],np.repeat(gids[:,None],20,axis=1))
    value=z['trace'][:,:,0].mean(0)*100;np.testing.assert_array_equal(value,meta['scores_by_environment']);episodes[key]=value.tolist()
    if step==1000000:
     scores['retrained_rl_Qhalf'][pi,scenes.index(scene)]=value
     physical[key]=dict(overall=STATS.physics_stats(z,m),by_instruction={str(g):STATS.physics_stats(z,m,np.repeat((gids==g)[:,None],20,axis=1)) for g in sorted(set(gids))})
    else:process.setdefault(str(seed),{}).setdefault(str(step),{})[scene]=STATS.ci(value,boot)
   print('verified',seed,step,flush=True)
 summaries={name:STATS.summary(v,m,boot,ix,name in ['original_rl','C1','C3','retrained_rl_Qhalf']) for name,v in scores.items()}
 paired={f'retrained_rl_Qhalf_minus_{name}':STATS.summary(scores['retrained_rl_Qhalf']-v,m,boot,ix,True) for name,v in scores.items() if name!='retrained_rl_Qhalf'}
 phys=dict(groups={g:STATS.merge_physics([v['overall'] for k,v in physical.items() if k.split('/')[-1] in [scenes[i] for i in idx]]) for g,idx in ix.items()},by_parent={str(p):{g:STATS.merge_physics([v['overall'] for k,v in physical.items() if k.startswith(f'seed_{p}/') and k.split('/')[-1] in [scenes[i] for i in idx]]) for g,idx in ix.items()} for p in m['seeds']},by_instruction={str(g):STATS.merge_physics([v['by_instruction'][str(g)] for v in physical.values() if str(g) in v['by_instruction']]) for g in range(3)})
 np.testing.assert_allclose(phys['groups']['overall_13']['score_x100'],summaries['retrained_rl_Qhalf']['conditional_mean']['overall_13']['mean'],atol=1e-12,rtol=0)
 refphys=read(PREVIOUS/'report/physical_statistics.json')['summary'];gaps={}
 for name in scores:
  if name=='retrained_rl_Qhalf':continue
  gaps[name]={}
  for group in ix:
   a=phys['groups'][group]['reward_parts_x100'];b=refphys[name]['groups'][group]['reward_parts_x100'];parts={k:a[k]-b[k] for k in a};delta=parts['quality_credit']-sum(parts[k] for k in ['age_mean_cost','age_max_cost','age_tail_cost','resource_cost','service_violation_cost'])+parts['recv_aoi_bonus']
   expected=paired[f'retrained_rl_Qhalf_minus_{name}']['conditional_mean'][group]['mean'];np.testing.assert_allclose(delta,expected,atol=1e-10,rtol=0);gaps[name][group]=dict(score_difference=expected,reward_component_differences=parts)
 train={};actualseeds=set();training_audits={}
 for seed in m['seeds']:
  folder=HERE/f'jobs/seed_{seed}';status=read(folder/'status.json');assert status['state']=='complete' and status['completed_steps']==1000000
  rows=[json.loads(x) for x in (folder/'training_metrics.jsonl').read_text().splitlines()];assert len(rows)==250 and [r['steps'] for r in rows]==list(range(4000,1000001,4000))
  assert all(r['trained_actor_ids']==[0,1,2,3] and r['stage']=='joint' for r in rows)
  counts=np.array([r['instruction_counts'] for r in rows]).sum(0);assert counts.sum()==1000000
  state,entry=checkpoint.load_checkpoint(folder/'checkpoints');assert state['update']==250 and int(state['environment']['audited_physical_steps'])==1000000
  actual=[e['seed'] for e in state['source_episodes']];assert actual==m['training_environment_seeds'][str(seed)];actualseeds.update(actual)
  np.testing.assert_array_equal(arr(state['environment']['reward_weights']),m['effective_weights'])
  step_counts=[]
  for opt in state['actor_optimizers']+[state['critic_optimizer']]:
   steps=sorted(set(int(v['step']) for v in opt['state'].values()));assert steps==[2500];step_counts.append(steps[0])
  selected=sorted((folder/'trajectory_audits').glob('episode_*.npz'));assert len(selected)==7
  for path in selected:
   z=arrays(path);meta=read(path.with_suffix('.json'));assert meta['state']=='PASS' and meta['trace_sha256']==sha(path);assert z['seeds'].tolist()==actual
   # Reconstruct the corresponding reset from the declared training sequence.
   env=TrainingEnv(config(seed)['env_args'],10,seed,'cpu')
   # reset increments the episode index before drawing its independent streams.
   for p in env.source.envs:p._episode_index=int(meta['episode'])-1
   env.reset();training_audits[str(path.relative_to(HERE))]=ws.independent_check(z,ws.PHYS.tables(env))
  train[str(seed)]=dict(environment_steps=1000000,actor_optimizer_steps=sum(step_counts[:4]),critic_optimizer_steps=step_counts[4],instruction_counts=counts.tolist(),complete_episodes=1660,partial_episodes=10,partial_episode_steps_each=400,online_audited_physical_steps=1000000,offline_complete_trajectory_audited_steps=42000,initial_actor_hashes=state['initial_actor_hashes'],final_actor_hashes=status['final_actor_hashes'],checkpoint_sha256=entry['sha256'])
  del state
 costs=[]
 for p in sorted((HERE/'evaluation_costs').glob('*.jsonl')):costs.extend(json.loads(x) for x in p.read_text().splitlines())
 assert sum(x['physical_steps'] for x in costs)==900000 and sum(x['complete_episodes'] for x in costs)==1500 and all(x['error'] is None for x in costs)
 actualseeds.update(m['validation']);assert not actualseeds&set(m['reserved']);verify()
 result=dict(methods=list(scores),seeds=m['seeds'],validation=m['validation'],scenarios=m['scenarios'],reward_weights=m['effective_weights'],score_definition='mean original common reward under half-quality weights times 100',scores=summaries,learning_process_fixed_tasks_only=process,episode_scores=episodes,statistical_scope='4000 paired environment-cluster bootstrap; all 13 scenes and all fixed models retained, 95% percentile CI conditional on these models and development seeds; no training population claim',bootstrap_seed=m['bootstrap_seed'],reference_source=str(PREVIOUS),training=train)
 write(HERE/'report/results.json',result);write(HERE/'report/paired_differences.json',paired);write(HERE/'report/physical_statistics.json',dict(summary=phys,by_scenario_parent_instruction=physical,predicted_quality_label='fixed average profile predicted PSNR, delivery weighted'));write(HERE/'report/gap_decomposition.json',gaps)
 write(HERE/'report/audit.json',dict(state='PASS',new_rl_environment_steps=3000000,new_actor_optimizer_steps=30000,new_critic_optimizer_steps=7500,new_evaluation_steps=900000,new_evaluation_complete_episodes=1500,preflight=read(HERE/'preflight.json'),formal_training=train,evaluation_checks=checks,training_offline_checks=training_audits,trace_sha256=hashes,reference_trace_sha256=oldaudit['trace_sha256'],reference_reused_episodes=5980,reference_new_computation_steps=0,protected_files=len(m['protected_sha256']),protected_files_unchanged=True,actual_environment_seeds=sorted(actualseeds),reserved_final_test_used=False,reserved_final_test_intersection=[]))
 npz(HERE/'report/episode_scores.npz',methods=np.array(list(scores)),seeds=np.array(m['validation']),parents=np.array(m['seeds']),scenarios=np.array(scenes),scores_x100=np.stack(list(scores.values())),bootstrap_indices=boot)
 print('AGGREGATION PASS',flush=True)
if __name__=='__main__':guard();main()
