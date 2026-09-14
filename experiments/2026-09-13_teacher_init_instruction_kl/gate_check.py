"""Fixed all-three engineering gate; no adaptive threshold or seed selection."""
from study import *
GROUPS={'overall_13':list(range(13)),'balance_fixed':[0],'aoi_fixed':[1],'quality_fixed':[2],'switch_10':list(range(3,13))}
def ci(x,indices):
 x=np.asarray(x)
 if x.ndim>1:x=x.mean(tuple(range(x.ndim-1)))
 return dict(mean=float(x.mean()),ci95=np.quantile(x[indices].mean(-1),[.025,.975]).tolist())
def gate_scores():
 m=manifest();out={}
 for label,modes in [('teacher',['D'])]+[(str(s),['D','S0','S1','S2']) for s in m['students']]:
  out[label]={}
  for mode in modes:
   values=[]
   for scene in m['scenarios']:
    path=HERE/f'gate/{label}/{mode}/{scene}.npz';meta=read(path.with_suffix('.json'))
    assert meta['state']=='complete' and sha(path)==meta['trace_sha256'] and meta['identity']['manifest_sha256']==sha(HERE/'manifest.json')
    with np.load(path,allow_pickle=False) as z:
     assert z['seeds'].tolist()==m['gate_dev'];v=z['trace'][:,:,0].mean(0)*100
     np.testing.assert_allclose(v,meta['score_x100_by_environment'],atol=0,rtol=0)
    if label!='teacher':
     assert not sum(meta['teacher_queries'])
     tm=read(HERE/f'gate/teacher/D/{scene}.json');assert meta['external_hashes']==tm['external_hashes']
    values.append(v)
   out[label][mode]=np.stack(values)
 return out

def gate():
 m=verify();assert read(HERE/'preflight.json')['state']=='PASS';scores=gate_scores();N=10
 auditchecks={}
 for label,modes in [('teacher',['D'])]+[(str(s),['D','S0','S1','S2']) for s in m['students']]:
  for mode in modes:
   for scene in m['scenarios']:
    path=HERE/f'gate/{label}/{mode}/{scene}.npz';ap=HERE/f'audits/gate/{label}/{mode}/{scene}.json';a=read(ap)
    assert a['state']=='PASS' and a['identity']['trace_sha256']==sha(path) and a['identity']['source_sha256']==sha(HERE/'audit_trajectories.py')
    assert a['observation_reconstruction']=='ALL_SLOTS_EXACT' and a['model_hashes_unchanged']
    if label!='teacher':assert a['policy_action_logp_replay']=='ALL_SLOTS_EXACT'
    auditchecks[str(ap.relative_to(HERE))]=sha(ap)
 boot=np.random.default_rng(m['bootstrap_seed']).integers(0,N,(4000,N));items={}
 for seed in m['students']:
  d=scores[str(seed)]['D'];s=np.mean([scores[str(seed)][f'S{i}'] for i in range(3)],0);teacher=scores['teacher']['D']
  td={g:ci((d-teacher)[ids],boot) for g,ids in GROUPS.items()};sd={g:ci((s-d)[ids],boot) for g,ids in GROUPS.items()}
  conditions={'D_overall_vs_teacher':td['overall_13']['mean']>=-.10,'D_all_three_fixed_vs_teacher':all(td[g]['mean']>=-.20 for g in ['balance_fixed','aoi_fixed','quality_fixed']),'S_overall_vs_D':sd['overall_13']['mean']>=-.50,'S_all_three_fixed_vs_D':all(sd[g]['mean']>=-1. for g in ['balance_fixed','aoi_fixed','quality_fixed']),'all_implementation_checks':len(auditchecks)==169}
  items[str(seed)]=dict(passed=all(conditions.values()),conditions=conditions,D_minus_teacher=td,S_minus_D=sd,D_scores={g:ci(d[ids],boot) for g,ids in GROUPS.items()},S_scores={g:ci(s[ids],boot) for g,ids in GROUPS.items()},checkpoint_sha256=sha(HERE/f'students/{seed}/A3/checkpoint.pt'))
 out=dict(state='PASS_ALL_THREE' if all(v['passed'] for v in items.values()) else 'STOP_AFTER_A',manifest_sha256=sha(HERE/'manifest.json'),students=items,thresholds=m['gate_thresholds'],teacher_scores={g:ci(scores['teacher']['D'][ids],boot) for g,ids in GROUPS.items()},bootstrap_seed=m['bootstrap_seed'],bootstrap_replicates=4000,statistical_unit='environment seed preserving 13 scenes and 3 sampling repeats averaged within student',gate_not_final_test=True,B_permitted=all(v['passed'] for v in items.values()))
 out['implementation_audit_sha256']=auditchecks
 write(HERE/'gate_report.json',out);print(json.dumps(out,ensure_ascii=False),flush=True);return out
if __name__=='__main__':guard();gate()
