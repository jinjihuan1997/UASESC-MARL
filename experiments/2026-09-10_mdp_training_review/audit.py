"""Read-only MDP/interface review with explicit constructed-state checks."""
import sys
from pathlib import Path
OUT=Path(__file__).resolve().parent
ROOT=OUT.parent.parent
REF=OUT.parent/'2026-09-10_aoi_ref8'
sys.path.insert(0,str(REF))
from helpers import *
from harl.models.base.distributions import FixedSimplex

def create():return make_env(config('ref8'),[20262501],[[0,2]])

def main():
    verify();env,obs,state,masks=create()
    paths=[ROOT/'Manuscript/main.tex',REF/'source/tensor_env.py',REF/'source/tensor_train.py',
           REF/'configs/ref8/seed_85/IC_HAPPO.json',
           REF/'source/reference/runtime/harl/models/base/distributions.py']
    hashes={str(p):digest(p) for p in paths}
    old_budget=arr(obs[:,1:,0]*env.p.Lambda_ref)
    share=torch.tensor([[.8,.1,.1]],dtype=env.dtype)
    mu=torch.zeros((1,env.M),dtype=env.dtype);mu[:,5]=1
    actions=[2*share-1]+[mu.clone() for _ in range(3)]
    _,_,_,info,_=checked_step(env,actions)
    budget=arr(info['budget'])
    np.testing.assert_allclose(budget,[[43800.,8100.,8100.]],atol=1e-8)
    assert not np.allclose(old_budget,budget)
    # Construct two states with equal visible physical variables/cache ages and
    # different clocks; these are interface counterexamples, not empirical returns.
    pair=[]
    for slot in [150,599]:
        e,*_=create();e.step_index=slot;e.aoi.fill_(3);e.q.fill_(True);e.tau.fill_(slot-1)
        o,s,av=e.observe();a=rule_actions(e,'m0_equal')
        result=e.step(a,auto_reset=False)
        pair.append(dict(obs=o,state=s,done=arr(result[3]),reward=arr(result[2])))
    assert torch.equal(pair[0]['obs'],pair[1]['obs'])
    assert torch.equal(pair[0]['state'],pair[1]['state'])
    assert not pair[0]['done'].any() and pair[1]['done'].all()
    np.testing.assert_allclose(pair[0]['reward'],pair[1]['reward'],atol=0,rtol=0)
    # Current SUT features can alias states with different cached urgency.
    e,*_=create();e.step_index=100;e.q[:,:,:5]=True;e.q[:,:,5:]=False;e.tau.fill_(99)
    e.aoi[0,0,:5]=9;e.aoi[0,0,5:]=2
    e.aoi[0,1,:5]=2;e.aoi[0,1,5:]=9
    e.aoi[0,2,:5]=5;e.aoi[0,2,5:]=9
    before=e.observe()[0][:,0].clone();urgency_before=(e.aoi*e.q).sum(-1).clone()
    e.aoi[0,0,:5]=2;e.aoi[0,0,5:]=9
    e.aoi[0,1,:5]=9;e.aoi[0,1,5:]=2
    after=e.observe()[0][:,0];urgency_after=(e.aoi*e.q).sum(-1)
    assert torch.equal(before,after) and not torch.equal(urgency_before,urgency_after)
    # Temperature is a retained constructor argument but does not affect the
    # current Dirichlet concentration, mean, log density or seeded samples.
    logits=torch.tensor([[.2,-.1,.4]])
    d1=FixedSimplex(logits,temperature=.1);d2=FixedSimplex(logits,temperature=2.)
    assert torch.equal(d1.concentration,d2.concentration)
    torch.manual_seed(1);a1=d1.sample()
    torch.manual_seed(1);a2=d2.sample()
    assert torch.equal(a1,a2)
    p=env.profiles
    groups=[]
    for i in range(env.M):
        for g in groups:
            if torch.equal(p[:,i],p[:,g[0]]):g.append(i);break
        else:groups.append([i])
    out=dict(state='PASS',type='interface_review_not_training_or_performance_trial',input_hashes=hashes,
        budget_timing=dict(uav_visible_budget=old_budget.tolist(),same_slot_executed_budget=budget.tolist()),
        terminal_alias=dict(slots=[150,599],actor_observations_identical=True,critic_inputs_identical=True,
            terminated=[False,True],scope='constructed equal visible physical states, not two recorded natural trajectories'),
        sut_information_alias=dict(observations_identical=True,cached_urgency_before=arr(urgency_before).tolist(),
            cached_urgency_after=arr(urgency_after).tolist(),scope='constructed interface counterexample'),
        dimensions=dict(sut=env.p.sut_obs_dim,uav=env.p.uav_obs_dim,padded=env.p.obs_dim_common,critic=env.p.share_obs_dim,
            repeated_quality_uav=160,nonrepeated_quality_uav=16,uav_after_exact_quality_dedup=75,critic_after_exact_quality_dedup=200),
        aoi_observation=dict(scale=600,examples={str(x):x/600 for x in [2,3,4]},candidate_scale=8),
        exact_profile_equivalence_groups=groups,remaining_classes=len(groups),
        simplex_temperature_changed_but_distribution_unchanged=True,
        gamma_099_relative_future_weights={str(k):.99**k for k in [100,300,599]},
        gamma_note='discount within a return; these are not the weighting of all late-episode actor minibatch samples',
        no_training=True,no_production_edits=True)
    for path,h in hashes.items():assert digest(path)==h
    verify();write(OUT/'audit.json',out)
    print(json.dumps(out,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
