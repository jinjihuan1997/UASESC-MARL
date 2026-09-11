"""Host temperature and identity guards, copied from the previously validated queue."""

import os,subprocess

from pathlib import Path

from helpers import read

def process_info(pid):
    try:
        path = Path('/proc')/str(pid)
        stat = (path/'stat').read_text().split(') ', 1)[1].split()
        command = (path/'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
        return dict(pid=pid, uid=path.stat().st_uid, state=stat[0], start_ticks=int(stat[19]), command=command)
    except (OSError, IndexError, ValueError):
        return None




def recorded_process_alive(folder):
    path = Path(folder)/'process.json'
    if not path.exists():
        return False
    record = read(path)
    current = process_info(record['pid'])
    return bool(current and current['state'] != 'Z' and current['uid'] == record['uid']
                and current['start_ticks'] == record['start_ticks'])




class ThermalControl:
    def __init__(self):
        self.cooling, self.hot_since, self.cool_since, self.paused_since = False, None, None, None

    def tick(self, cpu, gpu, now):
        if cpu is None or gpu is None or not (0 <= cpu <= 150 and 0 <= gpu <= 150):
            raise RuntimeError('A required temperature sensor is unavailable or invalid')
        if self.cooling:
            self.cool_since = (now if self.cool_since is None else self.cool_since) if cpu < 75 and gpu < 75 else None
            if now-self.paused_since >= 180 and self.cool_since is not None and now-self.cool_since >= 60:
                self.cooling, self.hot_since, self.cool_since = False, None, None
                return 'resume'
            return None
        self.hot_since = (now if self.hot_since is None else self.hot_since) if cpu >= 85 else None
        if cpu >= 95 or gpu >= 83 or (self.hot_since is not None and now-self.hot_since >= 5):
            self.cooling, self.paused_since, self.cool_since = True, now, None
            return 'pause'
        return None


def gpu_temperature():
    result = subprocess.run(['nvidia-smi', '--query-gpu=temperature.gpu',
        '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
    return max(float(value) for value in result.stdout.splitlines())




def temperature():
    values=[]
    for hw in Path('/sys/class/hwmon').glob('hwmon*'):
        try:
            if (hw/'name').read_text().strip()!='coretemp': continue
            values.extend(float(p.read_text())/1000 for p in hw.glob('temp*_input'))
        except OSError: continue
    return max(values,default=None)


