"""Independent transition comparison against the unchanged NumPy environment."""
from pathlib import Path
import argparse
import copy
import time
import numpy as np
import torch
from common import ROOT, configuration, configurations, verify_reference, write, stamp
from tensor_env import TensorSCEnv
from harl.envs.uav_escs.SC.uav_escs_env_sc import SCUAVEnv


def compare_case(name,args,device,steps,count=2):
    tensor=TensorSCEnv(args,count=count,seed=11,device=device)
    refs=[SCUAVEnv(copy.deepcopy(args)) for _ in range(count)]
    for i,e in enumerate(refs): e.seed(11+1000*i)
    actual=tensor.reset();expected=[e.reset() for e in refs]
    max_float_error=0.;discrete_checks=0
    def close(a,b,label,atol=2e-6,rtol=2e-6):
        nonlocal max_float_error
        x=a.detach().cpu().numpy() if isinstance(a,torch.Tensor) else np.asarray(a)
        y=np.asarray(b)
        np.testing.assert_allclose(x,y,atol=atol,rtol=rtol,err_msg=f'{name}: {label}')
        if x.dtype!=bool and x.size: max_float_error=max(max_float_error,float(np.max(np.abs(x-y))))
    def equal(a,b,label):
        nonlocal discrete_checks
        x=a.detach().cpu().numpy() if isinstance(a,torch.Tensor) else a
        np.testing.assert_array_equal(x,b,err_msg=f'{name}: {label}');discrete_checks+=1
    for i,e in enumerate(refs):
        for j in range(3): close(actual[j][i],expected[i][j],f'reset {j}')
    rng=np.random.default_rng(817)
    for step in range(steps):
        actions=[torch.as_tensor(rng.uniform(-1,1,(count,s.shape[0])).astype(np.float32),device=device) for s in tensor.action_space]
        # Exact action-priority ties exercise lower-index tie breaking.
        if step%7==0: actions=[torch.zeros_like(a) for a in actions]
        result=tensor.step(actions)
        for i,e in enumerate(refs):
            expected=e.step([a[i].cpu().numpy() for a in actions])
            info=result[4]
            equal(info['mode'][i],e.last_selected_modes,f'{step} modes')
            equal(info['served'][i].reshape(-1),e.last_y[e.owner_uav,np.arange(e.n_ds)],f'{step} served')
            equal(info['admission'][i].reshape(-1),e.last_admission[e.owner_uav,np.arange(e.n_ds)],f'{step} admission')
            close(info['usage'][i],e.last_lambda_usage,f'{step} load usage',atol=1e-7,rtol=1e-10)
            for key,reference in [('base_reward',e.last_base_reward),('common_reward',e.last_common_evaluation_reward),
                ('quality_term',e.last_quality_term),('load_term',e.last_load_term),('aoi_term',e.last_aoi_term)]:
                close(info[key][i],reference,f'{step} {key}',atol=1e-10,rtol=1e-10)
            close(result[2][i],expected[2],f'{step} reward')
            equal(result[3][i],expected[3],f'{step} done')
            terminal=bool(expected[3][0])
            if terminal:
                for key,j in [('terminal_observation',0),('terminal_state',1),('terminal_available',5)]:
                    close(info[key][i],expected[j],f'{step} {key}')
                next_obs,next_state,next_mask=e.reset()
            else: next_obs,next_state,next_mask=expected[0],expected[1],expected[5]
            close(result[0][i],next_obs,f'{step} obs')
            close(result[1][i],next_state,f'{step} shared obs')
            equal(result[5][i],next_mask,f'{step} masks')
            equal(tensor.q[i].reshape(-1),e.q_cache[e.owner_uav,np.arange(e.n_ds)],f'{step} cache')
            equal(tensor.tau[i].reshape(-1),e.tau_cache[e.owner_uav,np.arange(e.n_ds)],f'{step} timestamps')
            equal(tensor.aoi[i].reshape(-1),e.A_rcc,f'{step} AoI')
            close(tensor.psi[i],e.psi,f'{step} content',atol=0,rtol=0)
            for gpu,ref in [('gamma_us','gamma_uav_sut'),('gamma_sat','gamma_sut_sat'),('gamma_du','gamma_ds_uav'),('gamma_bh','gamma_bh')]:
                close(getattr(tensor,gpu)[i],getattr(e,ref),f'{step} {ref}',atol=1e-8,rtol=1e-10)
            close(tensor.quality[i],e.Q_hat_rec[:,0,:],f'{step} quality',atol=1e-8,rtol=1e-10)
            close(tensor.load[i],e.Lambda_sem[:,0,:],f'{step} load',atol=1e-7,rtol=1e-10)
    return dict(case=name,steps_per_environment=steps,environments=count,discrete_checks=discrete_checks,
                max_absolute_float_error=max_float_error,passed=True)


def run(device,steps):
    configs=configurations(steps=16000,device='cuda');cases=[]
    for method,config in configs.items():
        cases.append((method,config['env_args'],steps))
    original=configuration()['env_args']
    for name,overrides in [
        ('round_robin_partial_cache',dict(scheduler='round_robin',initial_cache_prob=.45,explicit_instruction_schedule=[[0,0],[1,2],[2,1],[20,0]])),
        ('no_feasible_quality',dict(Q_req_by_instruction_snr_bucket=[[100.]*4]*3)),
        ('insufficient_budget',dict(B_sut_sat=10.)),
        ('deterministic_channels',dict(sigma_uav_sut=0.,sigma_sut_sat=0.,sigma_ds_uav=0.)),
        ('no_explicit_context',dict(use_instruction_constraints=False)),
    ]:
        args=copy.deepcopy(original);args.update(overrides);cases.append((name,args,40))
    results=[]
    for name,args,n in cases:
        result=compare_case(name,args,device,n)
        results.append(result);print(result,flush=True)
    return results


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',default='cpu',choices=['cpu','cuda:0'])
    parser.add_argument('--steps',type=int,default=620)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();verify_reference();torch.set_num_threads(1)
    if args.output.exists(): raise FileExistsError(args.output)
    started=time.perf_counter()
    try:
        results=run(args.device,args.steps)
        write(args.output,dict(passed=True,device=args.device,utc=stamp(),wall_seconds=time.perf_counter()-started,results=results))
    except BaseException as exc:
        write(args.output,dict(passed=False,device=args.device,utc=stamp(),error=repr(exc)))
        raise
