"""Bounded CPU-only environment replay, independent of policy training and IPC."""
import argparse
import faulthandler
import hashlib
import json
import os
from pathlib import Path
import sys
import time

faulthandler.enable(all_threads=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--frozen-run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--steps', type=int, default=4000)
    p.add_argument('--ranks', type=int, nargs='+', default=[0,1,2,3,4,5,6,7,8,9])
    p.add_argument('--import-torch', action='store_true')
    p.add_argument('--cpu', type=int, default=18)
    args = p.parse_args()
    os.sched_setaffinity(0, {args.cpu})
    os.nice(10)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    frozen = args.frozen_run.resolve()/'frozen'
    sys.path.insert(0, str(frozen/'runtime'))
    if args.import_torch:
        import torch
        torch.set_num_threads(1)
    import numpy as np
    from harl.envs.uav_escs.SC.uav_escs_env_sc import SCUAVEnv
    config = json.loads((args.frozen_run/'configs/IC_HAPPO.json').read_text())
    start = time.monotonic()
    common = dict(python=sys.version, executable=sys.executable, numpy=np.__version__,
        torch_imported='torch' in sys.modules, cuda_initialized=(sys.modules['torch'].cuda.is_initialized() if 'torch' in sys.modules else False),
        cpu=args.cpu, pid=os.getpid(), steps_per_rank=args.steps, ranks=args.ranks,
        environment_file=str(Path(sys.modules[SCUAVEnv.__module__].__file__).resolve()))
    (out/'metadata.json').write_text(json.dumps(common,indent=2)+'\n')
    completed = []
    for rank in args.ranks:
        env = SCUAVEnv(config['env_args'])
        env.seed(1+rank*1000)
        obs,state,mask = env.reset()
        rng = np.random.default_rng(99000+rank)
        digest = hashlib.sha256()
        for step in range(args.steps):
            actions = [rng.dirichlet(np.ones(3)).astype(np.float32)]
            actions.extend(np.zeros(16,dtype=np.float32) for _ in range(1,env.n_agents))
            for agent in range(1,env.n_agents):
                choices = np.flatnonzero(mask[agent])
                actions[agent][int(rng.choice(choices)) if len(choices) else 0] = 1
            obs,state,rewards,done,info,mask = env.step(actions)
            for value in (obs,state,rewards,done,mask):
                array = np.asarray(value)
                if array.dtype.kind in 'fc' and not np.isfinite(array).all():
                    raise FloatingPointError(f'Non-finite output at rank {rank}, step {step}')
                digest.update(array.tobytes())
            if np.all(done):
                obs,state,mask = env.reset()
            if step%1000==999:
                print(json.dumps(dict(rank=rank,steps=step+1,seconds=time.monotonic()-start)),flush=True)
        env.close()
        completed.append(dict(rank=rank,steps=args.steps,sha256=digest.hexdigest()))
        (out/'progress.json').write_text(json.dumps(completed,indent=2)+'\n')
    (out/'result.json').write_text(json.dumps(dict(**common,completed=completed,
        elapsed_seconds=time.monotonic()-start,crash=False),indent=2)+'\n')
    print('CPU-only replay completed',flush=True)


if __name__=='__main__':
    main()
