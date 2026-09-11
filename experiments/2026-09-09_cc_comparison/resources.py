from pathlib import Path


def temperature():
    values=[]
    for hw in Path('/sys/class/hwmon').glob('hwmon*'):
        try:
            if (hw/'name').read_text().strip()!='coretemp': continue
            values.extend(float(p.read_text())/1000 for p in hw.glob('temp*_input'))
        except OSError: continue
    return max(values,default=None)
