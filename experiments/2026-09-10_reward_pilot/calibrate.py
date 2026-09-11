"""Freeze the same nine-candidate rule family on new calibration seeds."""
from helpers import *


def main():
    assert not (HERE/'rule_selection.json').exists()
    values={o:{} for o in OBJECTIVES};pairing=None
    for name in RULES:
        env,_,_,_=make_env(config('original_tail10'),CAL_SEEDS)
        h=external_hashes(env)
        if pairing is None:pairing=h
        assert h==pairing
        rows=[]
        for slot in range(600):
            _,_,_,info,v=checked_step(env,rule_actions(env,name))
            age=arr(env.aoi).reshape(env.count,-1)
            rows.append(np.column_stack([arr(info['quality_term']),arr(info['load_term']),age.mean(-1),age.max(-1),
                np.maximum(age-10,0).mean(-1)/10,np.maximum(age-4,0).mean(-1)/10,
                v['deliveries'],v['predicted_quality_sum'],v['channel_uses'],(age>4).mean(-1),(age>6).mean(-1)]))
        path=HERE/'calibration'/f'{name}.npz';path.parent.mkdir(exist_ok=True)
        np.savez_compressed(path,trace=np.stack(rows),seeds=np.asarray(CAL_SEEDS))
        with np.load(path) as z:a=z['trace'].mean(0)
        w=np.asarray(env.p.reward_weights_by_instruction)
        for objective,tail_index in [('original_tail10',4),('candidate_tail4',5)]:
            age=.4*a[:,2]/10+.3*a[:,3]/10+.3*a[:,tail_index]
            reward=a[:,0,None]*w[:,0]-age[:,None]*w[:,1]-.2*a[:,1,None]*w[:,2]
            assert np.isfinite(reward).all() and np.all(a[:,6]>0)
            values[objective][name]=dict(rewards_by_seed=reward.tolist(),mean_rewards=reward.mean(0).tolist(),
                mean_aoi=float(a[:,2].mean()),psnr=float(a[:,7].sum()/a[:,6].sum()),
                deliveries_per_slot=float(a[:,6].mean()),fraction_above4=float(a[:,9].mean()),
                fraction_above6=float(a[:,10].mean()),trace_sha256=digest(path))
        print('Calibration:',name,flush=True)
    selection={}
    checks={}
    for objective,vs in values.items():
        scores=np.asarray([vs[n]['mean_rewards'] for n in RULES])
        choice=[RULES[i] for i in scores.argmax(0)]
        selection[objective]=dict(single=RULES[int(scores.mean(-1).argmax())],by_instruction=choice,
                                  selection='new_calibration_only',seeds=CAL_SEEDS)
        a,q=vs[choice[1]],vs[choice[2]]
        checks[objective]=dict(aoi_quality_rules_differ=choice[1]!=choice[2],
                              quality_minus_aoi_psnr=q['psnr']-a['psnr'],
                              quality_minus_aoi_age=q['mean_aoi']-a['mean_aoi'],
                              tail4_has_active_states=any(v['fraction_above4']>0 for v in vs.values()))
        assert checks[objective]['aoi_quality_rules_differ'] and checks[objective]['quality_minus_aoi_psnr']>0 and checks[objective]['quality_minus_aoi_age']>0
    write(HERE/'calibration_results.json',dict(state='PASS',rule_family=RULES,values=values,checks=checks,
                                             external_hashes=pairing,episodes=90,slots=54000))
    write(HERE/'rule_selection.json',selection)
    print('Calibration PASS',selection,flush=True)


if __name__=='__main__':main()
