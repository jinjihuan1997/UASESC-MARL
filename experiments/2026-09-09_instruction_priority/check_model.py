"""Independent model checks and predeclared rule feasibility sweep."""
from pathlib import Path
import copy,json,hashlib
import numpy as np
import torch
from common import ROOT,configuration,write,verify_reference,stamp
from tensor_env import TensorSCEnv
from harl.envs.uav_escs.SC.uav_escs_env_sc import SCUAVEnv
from harl.envs.uav_escs.SC.rules import rule_actions
from evaluate import external_metrics,aggregate

def main():
    verify_reference();torch.set_num_threads(1)
    args=configuration()['env_args'];env=SCUAVEnv(args)
    for p in [np.array([1.,0.,0.]),np.array([0.,1.,0.]),np.array([.2,.3,.5])]:
        np.testing.assert_allclose(env._project_beta(2*p-1),.05+.85*p,atol=1e-12)
    a=copy.deepcopy(args);b=copy.deepcopy(args)
    a['backhaul_availability']=1.;b['backhaul_availability']=.6
    e1,e2=SCUAVEnv(a),SCUAVEnv(b)
    for e in [e1,e2]: e.seed(20260911);e.reset()
    np.testing.assert_array_equal(e1.gamma_bh,e2.gamma_bh)
    np.testing.assert_allclose(e2.Phi_bh,.6*e1.Phi_bh)
    assert e1.A_max==e2.A_max==600
    # Hold physical state and action fixed, alter only the service utility.
    base=copy.deepcopy(args);base['Q_req_by_instruction_snr_bucket']=[[21.]*4]*3
    rewards=[]
    for g in range(3):
        base['explicit_instruction_schedule']=[[0,g]]
        e=SCUAVEnv(base);e.seed(20260911);obs,_,available=e.reset()
        actions=[np.zeros(3)]+[np.eye(16)[0] for _ in range(3)]
        result=e.step(actions);info=result[4][0];row=external_metrics(e,info)
        chi=info['chi'];post=np.asarray(info['A_rcc']);quality=info['Q_hat_rec'];load=info['Lambda_sem']
        qt=(chi*np.maximum((quality-21)/12,0)).sum()/30
        ct=(chi*load).sum()/300000
        at=(.4*post.mean()+.3*post.max()+.3*np.maximum(post-10,0).mean())/10
        w=args['reward_weights_by_instruction'][g]
        np.testing.assert_allclose(row['base_reward'],w[0]*qt-w[1]*at-w[2]*ct,atol=1e-12)
        rewards.append(row['base_reward'])
    assert np.ptp(rewards)>.01
    checks=dict(passed=True,utc=stamp(),resource_vertices_reachable=True,
        resource_availability_preserves_snr=True,physical_aoi_cap_unchanged=True,
        independent_reward_formula=True,base_reward_by_instruction_same_action=rewards)
    write(ROOT/'model_checks.json',checks)
    rows=[]
    for availability in [.5,.6,.75,1.]:
        for g in range(3):
            for seed in [20260911,20260912,20260913]:
                candidate=copy.deepcopy(args);candidate.update(backhaul_availability=availability,
                    explicit_instruction_schedule=[[0,g]])
                e=SCUAVEnv(candidate);e.seed(seed);obs,_,available=e.reset();metrics=[]
                rng=np.random.default_rng(seed)
                for t in range(600):
                    actions=rule_actions(e,obs,available,'R_fixed',rng)
                    obs,_,_,done,info,available=e.step(actions)
                    metrics.append(external_metrics(e,info[0]))
                assert all(done)
                rows.append(dict(availability=availability,g=g,seed=seed,**aggregate(metrics)))
            subset=[r for r in rows if r['availability']==availability and r['g']==g]
            print(availability,g,'AoI',np.mean([r['mean_aoi'] for r in subset]),
                  'PSNR',np.mean([r['delivered_predicted_psnr'] for r in subset]),flush=True)
    candidate=[r for r in rows if r['availability']==.6]
    feasible=all(r['deliveries_per_slot']>0 and r['aoi_exceedance_fraction'] < (.01 if r['g']==1 else .05)
                 for r in candidate)
    write(ROOT/'parameter_sweep.json',dict(passed=feasible,rule='R_fixed',rows=rows,
        criterion='At eta=.6 each development seed: deliveries>0; AoI excess fraction<1% for aoi and<5% otherwise. Not a learning-performance gate.',
        candidate_availability=.6,utc=stamp()))
    if not feasible: raise RuntimeError('Predeclared feasibility gate failed; do not launch pilot')

if __name__=='__main__':main()
