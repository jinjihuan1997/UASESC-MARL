"""Experiment harness only; this module is never passed into online policies."""
import os
import sys
from pathlib import Path
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
sys.dont_write_bytecode = True
OUT = Path(__file__).resolve().parent
REFERENCE = OUT.parent / '2026-09-10_resource_selector_training'
sys.path.insert(0, str(REFERENCE))
import helpers as h
from evaluation import FIELDS, summarize
from aggregate import audit_trace
from rule_tools import predict_reward
import numpy as np
import torch


def verify():
    manifest = h.read(OUT / 'protocol.json')
    for file, digest in manifest['source_hashes'].items():
        assert h.digest(file) == digest, file
    h.verify()
    return manifest
