"""Reproduce failure cleanup using the real frozen vector wrapper and fake envs."""
import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

BASE = Path(__file__).resolve().parent
FROZEN = BASE.parent/'2026-09-08_single_seed_comparison/runs/seed1_10m_hybrid_parallel/frozen'


class Env:
    def __init__(self, fail=False):
        from gymnasium.spaces import Box
        self.n_agents=1
        self.observation_space=self.share_observation_space=self.action_space=[Box(-1,1,(1,))]
        self.fail=fail

    def close(self):
        pass

    def step(self, action):
        if self.fail:
            os.kill(os.getpid(),signal.SIGKILL)  # Deliberate fault in this fake env only.
        return [[0.]],[[0.]],[[0.]],[False],[{}],[[1.]]


def child(mode, log):
    sys.path.insert(0,str(FROZEN/'runtime'))
    from harl.envs.env_wrappers import ShareSubprocVecEnv
    mp.set_start_method(mode,force=True)
    envs=ShareSubprocVecEnv([lambda:Env(False),lambda:Env(True)])
    def phase(name, **extra):
        log.write_text(json.dumps(dict(phase=name,time=time.time(),pid=os.getpid(),**extra)))
    envs.step_async([[[0.]],[[0.]]])
    phase('waiting_for_step')
    try:
        envs.step_wait()
    except BaseException as error:
        phase('closing_after_exception',error=repr(error),waiting_flag=envs.waiting)
        envs.close()
        phase('closed')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--child',choices=['fork','spawn'])
    p.add_argument('--log',type=Path)
    args=p.parse_args()
    if args.child:
        child(args.child,args.log)
        return
    sys.path.insert(0,str(BASE.parent/'2026-09-08_single_seed_comparison'))
    import queue_supervisor as q
    results=[]
    for mode in ['fork','spawn']:
        path=BASE/f'ipc_{mode}_phase.json'
        with (BASE/f'ipc_{mode}.log').open('x') as log:
            proc=subprocess.Popen([sys.executable,str(__file__),'--child',mode,'--log',str(path)],
                                  stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            identity=q.process_info(proc.pid)
            try:
                deadline=time.monotonic()+20
                while time.monotonic()<deadline and not path.exists() and proc.poll() is None:
                    time.sleep(.1)
                if not path.exists():
                    raise RuntimeError(f'Probe did not initialize: {mode}')
                time.sleep(2)
                state=json.loads(path.read_text())
                state.update(mode=mode,parent_alive=proc.poll() is None,
                    group=q.group_members(proc.pid),wchan=(Path('/proc')/str(proc.pid)/'wchan').read_text())
                results.append(state)
                assert state['parent_alive']
                assert state['phase']==('waiting_for_step' if mode=='fork' else 'closing_after_exception')
            finally:
                q.stop_group(identity,grace=.1)
                proc.wait(timeout=5)
    (BASE/'ipc_failure_results.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(results,indent=2))


if __name__=='__main__':
    main()
