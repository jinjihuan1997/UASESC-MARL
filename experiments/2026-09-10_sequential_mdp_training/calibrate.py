"""Select equal-resource rules on the original calibration seeds only."""
from helpers import *

def main():
    results={}
    for rule in ['m0_equal','m5_equal','m10_equal']:
        scores=[]
        for gid in range(3):
            env,*_=make_env(config(),CAL_SEEDS,[[0,gid]]);sums=np.zeros(len(CAL_SEEDS))
            for slot in range(600):
                *_,values=checked_step(env,rule_actions(env,rule));sums+=values['common_reward']
            scores.append((sums/600).tolist())
        results[rule]=scores
    choices=list(results);mean=np.asarray([results[r] for r in choices]).mean(-1)
    selection=read(HERE/'rule_selection.json')
    selection['equal']=dict(single=choices[int(mean.mean(-1).argmax())],by_instruction=[choices[int(i)] for i in mean.argmax(0)],
        candidates=choices,calibration_seeds=CAL_SEEDS)
    write(HERE/'rule_selection.json',selection)
    write(HERE/'calibration_results.json',dict(state='PASS',seeds=CAL_SEEDS,results=results,selection=selection['equal'],full_resource_rules='unchanged parent selection'))
    print('Equal-resource calibration',selection['equal'],flush=True)

if __name__=='__main__': main()
