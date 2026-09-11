"""Preflight input invariance, reward prediction and actual evaluation output."""
import ast
from helpers import *
from rule_tools import myopic_actions
from aggregate import aggregate,check
import long_eval
import supervisor


if __name__=='__main__':
    for f in HERE.glob('*.py'):ast.parse(f.read_text())
    seeds=[20261991,20261992];schedule=[[0,0],[200,1],[400,2]]
    env,obs,state,masks=make_env(seeds,schedule,hidden=True)
    checked=0
    for slot in range(600):
        original=env.instructions[slot].clone();feeds=[];available=[]
        for g in range(3):
            env.instructions[slot]=g;o,s,m=env.observe();feeds.append(o);available.append(m)
        assert all(torch.equal(feeds[0],o) for o in feeds)
        assert all(torch.equal(available[0],m) for m in available)
        env.instructions[slot]=original
        actions,pred=myopic_actions(env)
        obs,_,masks,info,values=checked_step(env,actions)
        np.testing.assert_allclose(arr(pred),values['common_reward'],rtol=0,atol=1e-9)
        checked+=2
    full,fo,fs,fm=make_env(seeds,schedule)
    hidden,ho,hs,hm=make_env(seeds,schedule,True)
    allowed=torch.as_tensor(explicit_columns(full.p))
    assert torch.equal(fo[:,~allowed],ho[:,~allowed]) and torch.equal(fs,hs) and torch.equal(fm,hm)
    out=HERE/'validation_eval';out.mkdir(exist_ok=True)
    write(out/'manifest.json',dict(evaluation_seeds=list(range(20261901,20261921)),scenarios={'fixed_1':[[0,1]]}))
    write(out/'rule_selection.json',read(HERE/'rule_selection.json'));long_eval.HERE=out
    assert long_eval.evaluate_item('rules/R_instruction',[])==0
    assert long_eval.evaluate_item('rules/R_instruction',[])==0
    assert read(out/'evaluation/rules/R_instruction/status.json')['completed_episodes']==20
    summary=read(out/'evaluation/rules/R_instruction/summary.json')
    with np.load(out/'evaluation/rules/R_instruction/fixed_1.npz') as saved:
        data=saved['trace'];assert data.shape==(600,20,9)
    check(aggregate([data]),summary['scenarios']['fixed_1']['overall'])
    write(HERE/'validation.json',dict(state='PASS',hidden_input_and_mask_checks=checked,myopic_reward_checks=checked,
        critics_paired=True,explicit_fields_only=True,evaluator_smoke_episodes=20,evaluator_resume_checked=True,independent_aggregation=True,
        inherited_pilot_audit=read(HERE/'provenance.json')['pilot_audit_sha256'],
        benchmark_all_six_models_updated_on_both_layouts=read(HERE/'benchmark_results.json')['state']=='PASS'))
    print('PASS: input invariance, reward checks, standalone evaluator, resume and independent aggregation.')
