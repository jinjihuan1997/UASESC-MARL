"""No NumPy or training dependencies: repeated verified Python function calls."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

p = argparse.ArgumentParser()
p.add_argument('--cpu', type=int, required=True)
p.add_argument('--iterations', type=int, default=20000000)
p.add_argument('--output', type=Path, required=True)
args = p.parse_args()
os.sched_setaffinity(0, {args.cpu})
os.nice(15)


def clamp(value, lower, upper):
    return lower if value < lower else upper if value > upper else value


def nested(value):
    return clamp(value, 0, 15)


start = time.monotonic()
checksum = 0
for i in range(args.iterations):
    value = nested(i % 20 - 2)
    assert value == min(15, max(0, i % 20 - 2))
    checksum += value
    if i % 2000000 == 1999999:
        print(json.dumps(dict(iterations=i+1, seconds=time.monotonic()-start)), flush=True)
args.output.write_text(json.dumps(dict(python=sys.version, cpu=args.cpu,
    iterations=args.iterations, checksum=checksum, elapsed_seconds=time.monotonic()-start,
    numpy_imported='numpy' in sys.modules, torch_imported='torch' in sys.modules), indent=2)+'\n')
