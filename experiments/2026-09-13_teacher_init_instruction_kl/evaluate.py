"""Student-only deployment; teacher is a separately executed reference method."""
from study import *
from trajectory import run_episode_batch
import argparse

def gate_evaluate(student=None):
 m=verify();seeds=m['gate_dev'];schedule=m['scenarios']
 if student is None:
  for scene,sc in schedule.items():run_episode_batch(HERE/f'gate/teacher/D/{scene}.npz',seeds,'gate',sc,kind='gate_teacher')
 else:
  assert student in m['students'];checkpoint=HERE/f'students/{student}/A3/checkpoint.pt'
  assert read(checkpoint.parent/'status.json')['state']=='complete'
  for mode in ['D','S0','S1','S2']:
   for scene,sc in schedule.items():
    run_episode_batch(HERE/f'gate/{student}/{mode}/{scene}.npz',seeds,'gate',sc,student_seed=student,checkpoint=checkpoint,deterministic=mode=='D',action_seeds=None if mode=='D' else m['seeds'][str(student)]['actions']['gate'][int(mode[1:])],kind='gate_student')

def B_evaluate(student,arm,step):
 require_A_gate();m=verify();checkpoint=HERE/f'students/{student}/A3/checkpoint.pt' if arm=='supervised' else HERE/f'B/{student}/{arm}/milestones/steps_{step}/student.pt'
 scenes=m['scenarios'] if step==1000000 or arm=='supervised' else {k:v for k,v in m['scenarios'].items() if k.startswith('fixed_')}
 for mode in (['D','S0','S1','S2'] if step==1000000 and arm!='supervised' else ['D']):
  for scene,sc in scenes.items():run_episode_batch(HERE/f'evaluation/{student}/{arm}_{step}/{mode}/{scene}.npz',m['validation'],'validation',sc,student_seed=student,checkpoint=checkpoint,deterministic=mode=='D',action_seeds=None if mode=='D' else m['seeds'][str(student)]['actions']['final_B'][int(mode[1:])],kind='B_evaluation')

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--student',type=int);p.add_argument('--gate',action='store_true');p.add_argument('--arm');p.add_argument('--step',type=int,default=1000000);a=p.parse_args();guard()
 if a.gate:gate_evaluate(a.student)
 else:B_evaluate(a.student,a.arm,a.step)
