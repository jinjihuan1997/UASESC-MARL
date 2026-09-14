from study import *
from trajectory import run_episode_batch
import argparse

def collect(student=None,stage='A0'):
 m=verify();assert read(HERE/'preflight.json')['state']=='PASS'
 if stage=='A0':seeds=m['A0'];purpose='A0';checkpoint=None;student=None
 else:
  assert stage in ('A1','A2','A3') and student in m['students'];seeds=m['seeds'][str(student)]['dagger'][stage];purpose='dagger';checkpoint=HERE/f'students/{student}/A{int(stage[1:])-1}/checkpoint.pt'
 folder=HERE/('data/A0' if stage=='A0' else f'data/{student}/{stage}')
 for offset in range(0,len(seeds),20):
  deterministic=stage=='A0' or offset<40
  actions=None if deterministic else m['seeds'][str(student)]['actions'][stage]
  run_episode_batch(folder/f'batch_{offset//20:02}.npz',seeds[offset:offset+20],purpose,student_seed=student,checkpoint=checkpoint,deterministic=deterministic,action_seeds=actions,labels=True,kind='A0_collection' if stage=='A0' else 'dagger_collection')
 write(folder/'status.json',dict(state='complete',stage=stage,student=student,episodes=len(seeds),steps=len(seeds)*600))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--student',type=int);p.add_argument('--stage',default='A0');a=p.parse_args();guard();collect(a.student,a.stage)
