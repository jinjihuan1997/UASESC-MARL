"""Resource-only amendment: run frozen experiment on four efficiency cores.

Keep the original scientific manifest, trainer, checkpoints and thermal guard.
The separate amendment hashes this launcher and records the changed placement.
"""
import argparse
import datetime
import fcntl
import subprocess
import time
import supervisor as original
from helpers import *
from runtime_control import process_info,temperature,gpu_temperature

def amended_manifest():
    manifest=verify()
    amendment=read(HERE/'execution_amendment.json')
    assert digest(HERE/'manifest.json')==amendment['scientific_manifest_sha256']
    for file,h in amendment['input_hashes'].items():assert digest(HERE/file)==h,file
    cores=amendment['cores']
    for index,job in enumerate(manifest['jobs']):job['core']=cores[index%len(cores)]
    manifest['resources'].update(cores=cores,training_cores=cores,
        maximum_active_workers=len(cores),maximum_training_workers=len(cores),
        layout='cpu4_efficiency_cores',seconds_per_update_estimate=3.0)
    return manifest

def run():
    m=amended_manifest()
    amendment=read(HERE/'execution_amendment.json')
    # Changed placement gets a controlled restart after >=3 min cooling and
    # 10 continuous seconds below 80C on both sensors. Subsequent automatic
    # pause/resume continues to use the original, unmodified ThermalControl.
    earliest=datetime.datetime.fromisoformat(amendment['restart_not_before_utc']).timestamp()
    stable=None
    while True:
        now=time.time();cpu=temperature();gpu=gpu_temperature()
        safe=cpu is not None and gpu is not None and cpu<80 and gpu<80
        stable=(now if stable is None else stable) if safe else None
        if now>=earliest and stable is not None and now-stable>=10:break
        write(HERE/'status.json',dict(state='cooling_before_resource_amended_resume',updated_utc=stamp(),
            supervisor_pid=os.getpid(),cpu_max_c=cpu,gpu_max_c=gpu,
            training_progress=original.progress(m)[0],execution_amendment='execution_amendment.json'))
        time.sleep(1)
    original.verify=amended_manifest
    return original.run(resume=True)

def start():
    m=amended_manifest()
    with (HERE/'.launcher.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not original.live_record(HERE/'current_execution.json'),'Queue already running'
        assert not any(original.recorded_process_alive(HERE/j['output']) for j in m['jobs']+m['evaluation_jobs'])
        with (HERE/'logs/execution_override.log').open('a') as log:
            p=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'run'],cwd=HERE,
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        identity=process_info(p.pid);assert identity
        write(HERE/'current_execution.json',dict(identity,utc=stamp(),manifest_sha256=digest(HERE/'manifest.json'),
            execution_amendment_sha256=digest(HERE/'execution_amendment.json')))
        print(json.dumps(dict(state='resume_started',pid=p.pid,cores=m['resources']['cores'],max_concurrent=4)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['start','run']);a=p.parse_args()
    if a.command=='start':start()
    else:raise SystemExit(run())
