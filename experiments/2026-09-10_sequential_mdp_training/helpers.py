"""Versioned sequential-control experiment, fixed reward and exogenous pairing."""
import copy
import hashlib
import json
import os
import sys
from pathlib import Path
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'): os.environ[key]='1'
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
PARENT=HERE.parent/'2026-09-10_aoi_ref8'
sys.path.insert(0,str(HERE/'source'))
from common import write,stamp,verify_reference,network_hash
from training_checkpoint import digest
from tensor_env import TensorSCEnv
from harl.envs.uav_escs.SC.uav_escs_env_sc import SCUAVEnv
from harl.envs.uav_escs.reward_objective import reward_components
from evaluate import load_actors
from metrics import arr,check_tensor
import numpy as np
import torch

METHODS=['joint','mode_only','staged']
SEEDS=[85,218,966]
RULES=[f'm{m}_{a}' for m in (0,5,10) for a in ('equal','backlog','urgency')]
RULE_METHODS=['R_single','R_instruction','R_myopic','R_equal_single','R_equal_instruction','R_equal_myopic']
CAL_SEEDS=list(range(20262401,20262411))
EVAL_SEEDS=list(range(20262501,20262521))

def read(path): return json.loads(Path(path).read_text())
def config(arm='joint',seed=85): return read(HERE/'configs'/f'seed_{seed}'/f'{arm}.json')
def make_env(cfg,seeds,schedule=None,device='cpu'):
    args=copy.deepcopy(cfg['env_args'])
    args.update(instruction_mode_strategy='explicit_evaluation',explicit_instruction_schedule=schedule or [[0,0]])
    env=TensorSCEnv(args,count=len(seeds),seed=seeds[0],device=device)
    for source,seed in zip(env.source.envs,seeds): source.seed(seed)
    return (env,*env.reset())

def rule_actions(env,name):
    mode_text,allocation=name.split('_');mode=int(mode_text[1:])
    if allocation=='equal': score=torch.ones_like(env.beta)
    elif allocation=='backlog': score=env.q.sum(-1).to(env.dtype)
    elif allocation=='urgency': score=(env.aoi*env.q).sum(-1)
    else: raise ValueError(allocation)
    share=torch.where(score.sum(-1,keepdim=True)>0,score/score.sum(-1,keepdim=True).clamp_min(1e-12),torch.full_like(score,1/env.U))
    mu=torch.zeros((env.count,env.M),dtype=env.dtype,device=env.device);mu[:,mode]=1
    return [2*share-1]+[mu.clone() for _ in range(env.U)]

@torch.no_grad()
def learned_actions(env,actors,obs,available,arm):
    rnn=torch.zeros((env.count,1,256),device=env.device);mask=torch.ones((env.count,1),device=env.device)
    if arm=='mode_only': resource=torch.full((env.count,env.U),2/env.U-1,device=env.device)
    else: resource=actors[0].act(obs[:,0],rnn,mask,available[:,0],deterministic=True)[0]
    post,_,masks=env.allocate_resources(resource)
    modes=[actors[i].act(post[:,i],rnn,mask,masks[:,i],deterministic=True)[0] for i in range(1,env.n_agents)]
    return [resource]+modes

def external_hashes(env):
    result=[]
    for i in range(env.count):
        h=hashlib.sha256()
        for name in ['pos_uav','pos_ds','q','tau','aoi']: h.update(arr(getattr(env,name))[i].tobytes())
        for name in ['noise_us','noise_sat','noise_du','potential_content','instructions']: h.update(arr(getattr(env,name))[:,i].tobytes())
        result.append(h.hexdigest())
    return result

def snapshot(env): return {k:arr(getattr(env,k)).copy() for k in ('aoi','q','tau')}
def checked_step(env,actions):
    before=snapshot(env);slot=env.step_index
    if env.allocation_pending:
        assert torch.equal(actions[0],env.allocated_action)
        obs,state,reward,done,info,masks=env.commit_modes(actions[1:],auto_reset=False)
    else: obs,state,reward,done,info,masks=env.step(actions,auto_reset=False)
    values=check_tensor(env,info,slot,before)
    np.testing.assert_allclose(arr(reward[:,0,0]),values['common_reward'],atol=1e-6,rtol=1e-6)
    return obs,state,masks,info,values

def verify():
    m=read(HERE/'manifest.json')
    for f,h in m['input_hashes'].items(): assert digest(HERE/f)==h,f
    return m
