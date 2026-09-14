"""Idempotent entry for this sealed, conditionally stopped execution."""
from study import *
from run_dagger import run,parallel

def main():
 m=verify();assert read(HERE/'preflight.json')['state']=='PASS'
 if read(HERE/'status.json')['state']=='COMPLETED_STOPPED_AFTER_A':
  run('finalize.py',[],'reproduce_complete_run');return
 run('run_dagger.py',['--train-only'],'stage_A_orchestrated')
 jobs=[('evaluate.py',['--gate'],'gate_teacher')]+[('evaluate.py',['--gate','--student',s],f'gate_{s}') for s in m['students']]
 parallel(jobs,3);run('audit_trajectories.py',['--scope','data'],'audit_data')
 parallel([('audit_trajectories.py',['--scope','gate'],'audit_gate_teacher')]+[('audit_trajectories.py',['--scope','gate','--student',s],f'audit_gate_{s}') for s in m['students']],3)
 run('gate_check.py',[],'gate_check');g=read(HERE/'gate_report.json')
 if g['state']=='STOP_AFTER_A':
  if not (HERE/'report/budget_threshold_diagnostics.json').exists():run('budget_diagnostics.py',[],'budget_diagnostics')
  run('finalize.py',[],'finalize');return
 # The present sealed run fails A. Do not silently characterize unexecuted B
 # integration as passed by manufacturing B_preflight.json.
 assert read(HERE/'B_preflight.json')['state']=='PASS', 'All students passed, but conditional integration must actually pass before RL; no B preflight is inferred from A.'
 for s in m['students']:run('warmup_critic.py',['--student',s],'warmup_'+str(s))
 for s in m['students']:
  for arm in ['B0','B1']:run('train_joint.py',['--student',s,'--arm',arm],f'B_{s}_{arm}')

if __name__=='__main__':main()
