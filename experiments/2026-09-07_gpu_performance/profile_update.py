"""Profile the original GPU update in an isolated one-update run."""
from pathlib import Path
import cProfile
import pstats
import sys
import probe

out = Path(__file__).resolve().parent
original = probe.ProbeRunner.train
profile = cProfile.Profile()
def train(self):
    return profile.runcall(original, self)
probe.ProbeRunner.train = train
sys.argv = [str(out / 'probe.py'), '--mode', 'gpu', '--seed', '902', '--updates', '2']
try:
    probe.main()
finally:
    profile.dump_stats(out / 'original_gpu_update.prof')
    with (out / 'original_gpu_update_profile.txt').open('w') as fp:
        stats = pstats.Stats(profile, stream=fp)
        stats.strip_dirs().sort_stats('cumulative').print_stats(35)
        stats.sort_stats('tottime').print_stats(25)
