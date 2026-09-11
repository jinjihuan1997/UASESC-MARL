"""Local frozen runtime, paired evaluation, and explicitly simple rule controls."""
import copy
import hashlib
import json
import os
import sys
from pathlib import Path
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'): os.environ[key]='1'
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'source'))
from common import write,stamp,verify_reference,network_hash
from training_checkpoint import digest
from tensor_env import TensorSCEnv
from evaluate import load_actors
from metrics import arr,check_tensor
import numpy as np
import torch

METHODS=['joint','alternating']
RULE_METHODS=['R_equal_single','R_single','R_equal_instruction','R_instruction']

def read(path): return json.loads(Path(path).read_text())
def config(arm,seed): return read(HERE/'configs'/f'seed_{seed}'/f'{arm}.json')

def make_env(cfg,seeds,schedule=None,device='cpu'):
    args=copy.deepcopy(cfg['env_args'])
    args.update(instruction_mode_strategy='explicit_evaluation',explicit_instruction_schedule=schedule or [[0,0]])
    env=TensorSCEnv(args,count=len(seeds),seed=seeds[0],device=device)
    for source,seed in zip(env.source.envs,seeds): source.seed(seed)
    return (env,*env.reset())

def simple_rule_actions(env,method,obs):
    # Online resource rule uses SUT's existing summary; no candidate reward model.
    sut=obs[:,0]
    gid=sut[:,23:26].argmax(-1)
    urgency=(sut[:,13:16].to(env.dtype)*80).round().clamp_min(0)
    total=urgency.sum(-1,keepdim=True)
    urgent_share=torch.where(total>0,urgency/total.clamp_min(1),torch.full_like(urgency,1/3))
    equal=torch.full_like(urgency,1/3)
    instruction=method in ('R_equal_instruction','R_instruction')
    mode=torch.where(gid==1,0,5) if instruction else torch.zeros_like(gid)
    if method=='R_single':share=urgent_share
    elif method=='R_instruction':share=torch.where((gid==1)[:,None],urgent_share,equal)
    elif method in ('R_equal_single','R_equal_instruction'):share=equal
    else:raise ValueError(method)
    mu=torch.nn.functional.one_hot(mode,env.M).to(env.dtype)
    return [2*share-1]+[mu.clone() for _ in range(env.U)]

@torch.no_grad()
def learned_actions(env,actors,obs,available):
    rnn=torch.zeros((env.count,1,256),device=env.device)
    mask=torch.ones((env.count,1),device=env.device)
    resource=actors[0].act(obs[:,0],rnn,mask,available[:,0],deterministic=True)[0]
    post,_,masks=env.allocate_resources(resource)
    modes=[actors[i].act(post[:,i],rnn,mask,masks[:,i],deterministic=True)[0] for i in range(1,env.n_agents)]
    return [resource]+modes

def external_hashes(env):
    result=[]
    for i in range(env.count):
        h=hashlib.sha256()
        for name in ['pos_uav','pos_ds','q','tau','aoi']:h.update(arr(getattr(env,name))[i].tobytes())
        for name in ['noise_us','noise_sat','noise_du','potential_content','instructions']:h.update(arr(getattr(env,name))[:,i].tobytes())
        result.append(h.hexdigest())
    return result

def checked_step(env,actions):
    before={k:arr(getattr(env,k)).copy() for k in ('aoi','q','tau')}
    slot=env.step_index
    if env.allocation_pending:
        assert torch.equal(actions[0],env.allocated_action)
        obs,state,reward,done,info,masks=env.commit_modes(actions[1:],auto_reset=False)
    else:obs,state,reward,done,info,masks=env.step(actions,auto_reset=False)
    values=check_tensor(env,info,slot,before)
    np.testing.assert_allclose(arr(reward[:,0,0]),values['common_reward'],atol=1e-6,rtol=1e-6)
    return obs,state,masks,info,values

def verify():
    m=read(HERE/'manifest.json')
    for f,h in m['input_hashes'].items():assert digest(HERE/f)==h,f
    verify_reference()
    return m
