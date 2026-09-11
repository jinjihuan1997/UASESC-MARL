"""Independent calibration; only the predeclared AoI reference differs."""
from helpers import *


def main():
    assert not (HERE/'rule_selection.json').exists()
    values={o:{} for o in OBJECTIVES};pairing=None;hashes={}
    for name in RULES:
        env,_,_,_=make_env(config('ref10'),CAL_SEEDS)
        h=external_hashes(env)
        if pairing is None:pairing=h
        assert h==pairing
        rows=[]
        for slot in range(600):
            _,_,_,info,v=checked_step(env,rule_actions(env,name))
            a=arr(env.aoi).reshape(len(CAL_SEEDS),-1)
            rows.append(np.column_stack([arr(info['quality_term']),a.mean(-1),a.max(-1),
                np.maximum(a-4,0).mean(-1),v['channel_uses'],v['deliveries'],v['predicted_quality_sum']]))
        path=HERE/'calibration'/f'{name}.npz';path.parent.mkdir(exist_ok=True)
        np.savez_compressed(path,trace=np.stack(rows),seeds=np.asarray(CAL_SEEDS))
        hashes[str(path.relative_to(HERE))]=digest(path)
        with np.load(path) as z:a=z['trace'].mean(0)
        w=np.asarray(env.p.reward_weights_by_instruction)
        for o,ref in [('ref10',10),('ref8',8)]:
            age=(.4*a[:,1]+.3*a[:,2]+.3*a[:,3])/ref
            reward=w[:,0]*a[:,0,None]-w[:,1]*age[:,None]-.02*a[:,4,None]/60000
            values[o][name]=dict(rewards_by_seed=reward.tolist(),mean_rewards=reward.mean(0).tolist(),
                mean_aoi=float(a[:,1].mean()),psnr=float(a[:,6].sum()/a[:,5].sum()),
                deliveries_per_slot=float(a[:,5].mean()))
        print('Calibrated',name,flush=True)
    selection={}
    for o in OBJECTIVES:
        scores=np.asarray([values[o][n]['mean_rewards'] for n in RULES])
        assert np.isfinite(scores).all()
        selection[o]=dict(single=RULES[int(scores.mean(-1).argmax())],
            by_instruction=[RULES[i] for i in scores.argmax(0)],seeds=CAL_SEEDS,selection='independent_calibration_only')
    write(HERE/'calibration_results.json',dict(state='PASS',values=values,input_hashes=hashes,
        external_hashes=pairing,episodes=90,slots=54000))
    write(HERE/'rule_selection.json',selection)
    print('Calibration PASS',selection,flush=True)


if __name__=='__main__':main()
