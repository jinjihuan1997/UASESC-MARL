"""Produce action-controlled physical traces using one isolated version per process."""
from pathlib import Path
import sys,argparse,hashlib,json
p=argparse.ArgumentParser();p.add_argument('--version',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
sys.path.insert(0,str(a.version.resolve()))
from common import configuration,verify_reference
from harl.envs.uav_escs.SC.uav_escs_env_sc import SCUAVEnv
import numpy as np
verify_reference();results=[]
for case,overrides in [('default',{}),('partial',{'initial_cache_prob':.4}),('empty',{'initial_cache_prob':0.}),('quality_infeasible',{'Q_req_by_instruction_snr_bucket':[[100.]*4]*3})]:
    args=configuration()['env_args'];args.update(overrides,explicit_instruction_schedule=[[0,0],[200,1],[400,2]])
    env=SCUAVEnv(args);env.seed(20260911);env.reset();rng=np.random.default_rng(61231)
    for t in range(620):
        actions=[rng.uniform(-1,1,s.shape).astype(np.float32) for s in env.action_space]
        if t%7==0:actions=[np.zeros_like(x) for x in actions]
        _,_,reward,done,info,_=env.step(actions)
        h=hashlib.sha256()
        for array in [env.q_cache,env.tau_cache,env.A_rcc,env.beta_sut_sat,env.last_y,env.last_selected_modes,env.last_lambda_usage,env.gamma_bh,np.asarray(reward)]:
            h.update(np.ascontiguousarray(array).tobytes())
        results.append(dict(case=case,slot=t,physical_and_reward_sha256=h.hexdigest()))
        if all(done):env.reset()
a.output.write_text(json.dumps(dict(records=results),indent=2)+'\n')
