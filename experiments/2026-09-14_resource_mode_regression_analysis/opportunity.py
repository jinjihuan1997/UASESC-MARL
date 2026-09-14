from analysis_support import *

def calculate(z,tab):
 mode=z['modes'];request=z['requested_modes'];cnt=z['served'].sum(-1);gid=z['trace'][:,:,FIELDS.index('instruction_id')].astype(int)
 selected=np.maximum(mode,0);load=np.take_along_axis(tab['load'],selected[...,None],-1)[...,0];quality=np.take_along_axis(tab['quality'],selected[...,None],-1)[...,0]
 equal=(tab['load']==load[...,None]);best=np.where(equal,tab['quality'],-np.inf).argmax(-1)
 bestq=np.take_along_axis(tab['quality'],best[...,None],-1)[...,0];improve=(bestq>quality)&(cnt>0)
 assert not np.any(improve&(bestq<tab['req']))
 assert not np.any(improve&(load>z['budgets']+1e-9))
 np.testing.assert_allclose(quality[cnt>0],z['predicted_quality'][cnt>0],atol=1e-9,rtol=0)
 gain=np.where(improve,(np.maximum((bestq-21)/12,0)-np.maximum((quality-21)/12,0))*cnt,0)*np.asarray(MAN['effective_weights'])[gid,0,None]/30*100
 out=dict(score_opportunity_x100=float(gain.sum(-1).mean()),score_opportunity_per_environment=gain.sum(-1).mean(0).tolist(),fraction_all_uav_slots=float(improve.mean()),by_instruction={})
 for g in range(3):
  take=gid==g
  if not take.any():continue
  out['by_instruction'][str(g)]=dict(slots=int(take.sum()),fraction=float(improve[take].mean()),score_opportunity_x100=float(gain[take].sum(-1).mean()))
 out['uavs']=[]
 for u in range(3):
  ix=(slice(None),slice(None),u);take=improve[ix]
  replacements={}
  for a in range(16):
   for b in range(16):
    found=take&(mode[ix]==a)&(best[ix]==b)
    if found.any():replacements[f'{a}_to_{b}']=dict(slots=int(found.sum()),gain_x100=float(gain[ix][found].sum()/12000),mean_quality_difference=float((bestq[ix]-quality[ix])[found].mean()),exact_equal_load=True)
  out['uavs'].append(dict(uav=u+1,fraction=float(take.mean()),score_opportunity_x100=float(gain[ix].mean()),replacements=replacements))
 return out

def run():
 guard();paths=[]
 for seed in SEEDS:
  for scene in MAN['scenarios']:
   p=trace(seed,10000000,scene);paths.extend([p,p.with_suffix('.json')])
 ph={str(p.relative_to(ROOT)):sha(p) for p in paths}
 dump(HERE/'opportunity_manifest.json',dict(purpose='exploratory_same_load_algebra_only',protected_input_sha256=ph,source_sha256=sha(Path(__file__)),definition_sha256=sha(HERE/'EXPLORATORY_EXTENSION.md'),new_physical_steps=0))
 tabs={scene:ws.PHYS.tables(ws.make_env(scene)) for scene in MAN['scenarios']};r={};seen=set()
 for seed in SEEDS:
  out={}
  for scene in MAN['scenarios']:
   p=trace(seed,10000000,scene);z=arrays(p);meta=read(p.with_suffix('.json'))
   assert meta['trace_sha256']==sha(p) and meta['original_checker']=='PASS_every_slot' and meta['external_hashes']==tabs[scene]['external']
   assert list(z['seeds'])==MAN['validation'];seen.update(map(int,z['seeds']))
   out[scene]=calculate(z,tabs[scene])
  r[str(seed)]=dict(final_scenes=out,overall_opportunity_x100=float(np.mean([x['score_opportunity_x100'] for x in out.values()])),quality_timeline={str(step):calculate(arrays(trace(seed,step)),tabs['fixed_2']) for step in STEPS})
 assert not seen&set(MAN['reserved']);assert all(sha(ROOT/p)==h for p,h in ph.items())
 dump(HERE/'report/same_load_opportunity.json',r)
 dump(HERE/'report/opportunity_audit.json',dict(state='PASS',changed=[],protected_count=len(ph),profile_only_environment_initializations=len(tabs),new_physical_steps=0,new_training_steps=0,new_optimizer_updates=0,reserved_final_test_used=False,actual_saved_seeds=sorted(seen),source_metadata_checks='PASS',same_load_equality='exact equality, not tolerance'))
if __name__=='__main__':run()
