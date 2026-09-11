"""Run an unchanged frozen training job with fresh spawned environment workers."""
import argparse
import faulthandler
import multiprocessing as mp
from pathlib import Path
import signal
import sys

faulthandler.enable(all_threads=True)
if hasattr(signal, "SIGUSR1"):
    faulthandler.register(signal.SIGUSR1, all_threads=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--attempt", type=int, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    mp.set_start_method("spawn", force=True)
    sys.path.insert(0, str(out / "frozen"))
    import train
    if Path(train.__file__).resolve() != out / "frozen/train.py":
        raise RuntimeError("Training must use this run's frozen source")
    train.train(out, args.method, args.attempt)


if __name__ == "__main__":
    main()
