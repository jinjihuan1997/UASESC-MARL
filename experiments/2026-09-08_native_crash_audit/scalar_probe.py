"""Exercise observed scalar/index/interpolation sites without training or IPC."""
import argparse
import faulthandler
import json
import os
from pathlib import Path
import time
import sys

faulthandler.enable(all_threads=True)
p=argparse.ArgumentParser()
p.add_argument('--output',type=Path,required=True)
p.add_argument('--iterations',type=int,default=1000000)
p.add_argument('--cpu',type=int,default=15)
args=p.parse_args()
os.sched_setaffinity(0,{args.cpu}); os.nice(15)
import numpy as np
lookup=np.tile(np.arange(16,dtype=np.int64),(4,1))
x=np.arange(21,dtype=float); y=x*2
start=time.monotonic(); checksum=0
for i in range(args.iterations):
    bucket=int(np.clip(i%7-1,0,lookup.shape[0]-1))
    mode=int(np.clip(i%20-2,0,lookup.shape[1]-1))
    result=int(np.clip(lookup[bucket,mode],0,15))
    if result!=min(15,max(0,i%20-2)):
        raise AssertionError(('scalar corruption',i,result))
    value=float(np.interp(float(i%21),x,y))
    if value!=2*(i%21):
        raise AssertionError(('interpolation corruption',i,value))
    checksum+=result
    if i%200000==199999:
        print(json.dumps(dict(iterations=i+1,seconds=time.monotonic()-start)),flush=True)
args.output.write_text(json.dumps(dict(python=sys.version,executable=sys.executable,numpy=np.__version__,
    cpu=args.cpu,iterations=args.iterations,checksum=checksum,elapsed_seconds=time.monotonic()-start,
    pythonmalloc=os.getenv('PYTHONMALLOC'),torch_imported='torch' in sys.modules,crash=False),indent=2)+'\n')
