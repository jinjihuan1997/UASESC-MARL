"""CC configurations derived from the active frozen SC study."""
from pathlib import Path
import hashlib
import json
import sys
import copy

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


def configuration(method='CC_IC_HAPPO', seed=85, steps=16000):
    base = {'CC_IC_HAPPO': 'IC_HAPPO', 'CC_IC_MAPPO': 'IC_MAPPO'}[method]
    config = json.loads((ROOT/'inputs'/f'parent_{base}.json').read_text())
    if steps < 8000 or steps % 4000:
        raise ValueError('Use at least two complete 4000-step PPO batches')
    config['main_args'].update(env='uav_escs_cc', exp_name=method)
    config['algo_args']['seed'].update(seed=seed)
    config['algo_args']['device'].update(cuda=False, cuda_deterministic=False)
    config['algo_args']['train'].update(num_env_steps=steps, model_dir=None)
    config['env_args'].update(env='uav_escs_cc', n_semantic_modes=20,
        communication_model='cc_rgb8_average_single_attempt_v1',
        semantic_model_set='h264_ldpc_rgb8_20modes_v1',
        semantic_registry_path=str(ROOT/'inputs/mode_registry.json'),
        semantic_profile_path=str(ROOT/'inputs/profile.npz'),
        semantic_profile_quality_metric='psnr_rgb_expected',
        cc_delivery_moments_path=str(ROOT/'inputs/delivery_moments.npz'))
    return config
