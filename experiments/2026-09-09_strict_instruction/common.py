"""Frozen strict-instruction interface; physical model inherited unchanged."""
from pathlib import Path
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'reference'))
from protocol import configurations, activate_runtime, TABLE_SHA, write, stamp, network_hash
activate_runtime()


def verify_reference():
    manifest = json.loads((ROOT/'reference_manifest.json').read_text())
    for name, expected in manifest['input_hashes'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'Frozen reference changed: {name}')
    assert hashlib.sha256((ROOT/'reference/inputs/profile.npz').read_bytes()).hexdigest() == TABLE_SHA
    return manifest


def configuration(method='IC_HAPPO', seed=1, steps=16000):
    return configurations(seed=seed, steps=steps, device='cuda')[method]
