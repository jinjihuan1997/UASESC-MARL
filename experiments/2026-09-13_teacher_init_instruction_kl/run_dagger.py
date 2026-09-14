"""Fixed A0-A3 budget; fit every student and gate all three together."""
import os,sys,json,subprocess,concurrent.futures
from pathlib import Path
HERE=Path(__file__).resolve().parent
os.environ['PYTHONDONTWRITEBYTECODE']='1'
def read(p):return json.loads(p.read_text())
def write(p,d):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2)+'\n');t.replace(p)
def run(script,args,label):
 cmd=[sys.executable,str(HERE/script),*map(str,args)]
 with (HERE/f'logs/{label}.log').open('a') as f:
  p=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
  write(HERE/f'logs/process_{label}.json',dict(pid=p.pid,command=cmd,state='running'));rc=p.wait()
 write(HERE/f'logs/process_{label}.json',dict(pid=p.pid,command=cmd,state='complete' if rc==0 else 'failed',returncode=rc))
 if rc:raise RuntimeError(f'{label} failed, see log')
def parallel(jobs,workers=3):
 with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
  for f in concurrent.futures.as_completed([pool.submit(run,*j) for j in jobs]):f.result()
def main():
 m=read(HERE/'manifest.json');r=read(HERE/'resource_settings.json');assert read(HERE/'preflight.json')['state']=='PASS'
 for stage in ['A0','A1','A2','A3']:
  write(HERE/'status.json',dict(state='stage_A_running',stage=stage,B_authorized_by_gate=False))
  if stage=='A0':run('collect_demos.py',['--stage','A0'],'collect_A0')
  else:parallel([('collect_demos.py',['--student',s,'--stage',stage],f'collect_{s}_{stage}') for s in m['students']])
  parallel([('train_imitation.py',['--student',s,'--stage',stage,'--device',r['fit_device']],f'fit_{s}_{stage}') for s in m['students']],r['maximum_fit_workers'])
 if '--train-only' in sys.argv:
  write(HERE/'status.json',dict(state='A3_trained_gate_pending',stage='A3',B_authorized_by_gate=False));return
 write(HERE/'status.json',dict(state='stage_A_gate_evaluation',stage='gate',B_authorized_by_gate=False))
 jobs=[('evaluate.py',['--gate'],'gate_teacher')]+[('evaluate.py',['--gate','--student',s],f'gate_{s}') for s in m['students']]
 parallel(jobs)
 run('gate_check.py',[],'gate_check');gate=read(HERE/'gate_report.json')
 write(HERE/'status.json',dict(state='A_passed_B_pending' if gate['B_permitted'] else 'A_gate_failed_report_pending',stage='A_complete',B_authorized_by_gate=gate['B_permitted'],gate_state=gate['state']))
if __name__=='__main__':main()
