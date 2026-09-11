"""Resume must preserve PPO/optimizer/RNG state across an episode reset."""
import copy
import os
from pathlib import Path
import sys
import tempfile
import unittest

import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import configuration
from tensor_train import TensorTrainer
from training_checkpoint import capture, load_checkpoint, restore, save_checkpoint


def advance(trainer, start, end):
    for update in range(start+1, end+1):
        for actor in trainer.actors:
            actor.lr_decay(update, trainer.total_updates)
        trainer.critic.lr_decay(update, trainer.total_updates)
        trainer.collect()
        trainer.update()
        trainer.buffer.after_update()


def equal_tree(case, left, right, path='state'):
    case.assertEqual(type(left), type(right), path)
    if isinstance(left, torch.Tensor):
        case.assertTrue(torch.equal(left, right), path)
    elif isinstance(left, dict):
        case.assertEqual(left.keys(), right.keys(), path)
        for key in left:
            equal_tree(case, left[key], right[key], path+'/'+str(key))
    elif isinstance(left, (list, tuple)):
        case.assertEqual(len(left), len(right), path)
        for index, (a, b) in enumerate(zip(left, right)):
            equal_tree(case, a, b, path+'/'+str(index))
    else:
        case.assertEqual(left, right, path)


class FullCheckpoint(unittest.TestCase):
    def test_resume_crosses_future_episode_reset_exactly(self):
        device = os.environ.get('GPU_ENV_TEST_DEVICE', 'cpu')
        methods = ('IC_HAPPO', 'IC_MAPPO', 'HAPPO_equal_resources') if device == 'cpu' else ('HAPPO_equal_resources',)
        for method in methods:
            with self.subTest(method=method, device=device), tempfile.TemporaryDirectory() as temporary:
                config = configuration(method, seed=379)
                # Test buffer is small; production remains 10 environments x 400 slots.
                config['algo_args']['train'].update(n_rollout_threads=2, episode_length=400, num_env_steps=3200)
                identity = dict(test=method, device=device)
                uninterrupted = TensorTrainer(copy.deepcopy(config), device)
                advance(uninterrupted, 0, 4)
                expected = capture(uninterrupted, 4, identity)
                split = TensorTrainer(copy.deepcopy(config), device)
                advance(split, 0, 2)
                self.assertEqual(split.env.step_index, 200)
                save_checkpoint(temporary, capture(split, 2, identity))
                saved, record = load_checkpoint(temporary)
                self.assertEqual(record['selected'], 'current')
                resumed = TensorTrainer(copy.deepcopy(config), device)
                restored = restore(resumed, saved, identity)
                self.assertEqual(restored, 2)
                self.assertEqual(resumed.env.step_index, 200)
                advance(resumed, restored, 4)
                self.assertEqual(resumed.env.episode_indices, uninterrupted.env.episode_indices)
                equal_tree(self, expected, capture(resumed, 4, identity))

    def test_hash_failure_uses_previous_committed_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            for update in range(3):
                current = save_checkpoint(temporary, dict(update=update, schema=1, tensor=torch.tensor([update])))
            self.assertEqual(len(list(Path(temporary).glob('*.pt'))), 2)
            (Path(temporary)/current['file']).write_bytes(b'interrupted or damaged file')
            state, record = load_checkpoint(temporary)
            self.assertEqual(state['update'], 1)
            self.assertEqual(record['selected'], 'previous')
            self.assertTrue(record['rejected'])

    def test_wrong_identity_is_rejected_before_loading(self):
        config = configuration(seed=29)
        config['algo_args']['train'].update(n_rollout_threads=1, episode_length=8, num_env_steps=16)
        trainer = TensorTrainer(config, 'cpu')
        state = capture(trainer, 0, dict(run='a'))
        with self.assertRaises(ValueError):
            restore(trainer, state, dict(run='b'))
        state['runtime']['torch'] = 'changed-version'
        with self.assertRaises(ValueError):
            restore(trainer, state, dict(run='a'))


if __name__ == '__main__':
    unittest.main()
