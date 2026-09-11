"""Real subprocess tests: preserve ordered results and bound abnormal exits."""
import os
from pathlib import Path
import signal
import sys
import time
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from protocol import activate_runtime, configurations
activate_runtime()
from harl.envs.env_wrappers import ShareSubprocVecEnv, ShareDummyVecEnv, WorkerError


class ProbeEnv:
    def __init__(self, rank=0, failure=None):
        from gymnasium.spaces import Box
        self.n_agents = 1
        self.rank, self.failure, self.t = rank, failure, 0
        self.observation_space = self.share_observation_space = self.action_space = [Box(-1, 1, (1,))]
        if failure == 'startup':
            raise RuntimeError('Intentional initialization failure')

    def reset(self):
        self.t = 0
        return [[float(self.rank)]], [[float(self.rank)]], [[1.]]

    def step(self, action):
        if self.failure == 'exit':
            os.kill(os.getpid(), signal.SIGKILL)
        if self.failure == 'exception':
            raise RuntimeError('Intentional environment failure')
        if self.failure == 'stall':
            time.sleep(60)
        self.t += 1
        value = float(self.rank * 100 + self.t)
        return [[value]], [[value]], [[float(action[0][0])]], [self.t == 3], [{}], [[1.]]

    def rank_value(self):
        return self.rank

    def close(self):
        if self.failure == 'close':
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            time.sleep(60)


def equal_tree(a, b):
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            equal_tree(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            equal_tree(x, y)
    else:
        np.testing.assert_array_equal(a, b)


class WorkerFailureTests(unittest.TestCase):
    def stopped(self, env):
        self.assertTrue(env.closed)
        self.assertFalse(env.waiting)
        self.assertTrue(all(not process.is_alive() for process in env.ps))

    def test_healthy_order_autoreset_and_call(self):
        factories = [lambda rank=rank: ProbeEnv(rank) for rank in range(2)]
        parallel, serial = ShareSubprocVecEnv(factories), ShareDummyVecEnv(factories)
        try:
            equal_tree(parallel.reset(), serial.reset())
            for _ in range(8):
                equal_tree(parallel.step([[[1.]], [[2.]]]), serial.step([[[1.]], [[2.]]]))
            self.assertEqual(parallel.call_at(1, 'rank_value'), 1)
        finally:
            parallel.close()
            serial.close()
        self.stopped(parallel)
        parallel.close()  # An outer runner may also clean up after a failure.

    def test_death_after_a_sibling_response_and_ordinary_exception(self):
        for rank, failure in [(0, 'exit'), (1, 'exit'), (1, 'exception')]:
            with self.subTest(rank=rank, failure=failure):
                env = ShareSubprocVecEnv([lambda i=i: ProbeEnv(i, failure if i == rank else None)
                                         for i in range(2)], close_timeout=.2)
                try:
                    env.reset()
                    started = time.monotonic()
                    with self.assertRaises(WorkerError):
                        env.step([[[0.]], [[0.]]])
                    self.assertLess(time.monotonic()-started, 4)
                    self.stopped(env)
                finally:
                    env.close()

    def test_stalled_worker_times_out(self):
        env = ShareSubprocVecEnv([lambda: ProbeEnv(0), lambda: ProbeEnv(1, 'stall')],
                                response_timeout=.5, close_timeout=.2)
        try:
            env.reset()
            started = time.monotonic()
            with self.assertRaises(TimeoutError):
                env.step([[[0.]], [[0.]]])
            self.assertLess(time.monotonic()-started, 4)
            self.stopped(env)
        finally:
            env.close()

    def test_close_pending_step_does_not_receive_again(self):
        env = ShareSubprocVecEnv([lambda: ProbeEnv(0, 'stall')], close_timeout=.2)
        env.reset()
        env.step_async([[[0.]]])
        started = time.monotonic()
        env.close()
        self.assertLess(time.monotonic()-started, 4)
        self.stopped(env)

    def test_close_escalates_when_cleanup_ignores_terminate(self):
        env = ShareSubprocVecEnv([lambda: ProbeEnv(0, 'close')], close_timeout=.2)
        env.reset()
        started = time.monotonic()
        env.close()
        self.assertLess(time.monotonic()-started, 4)
        self.stopped(env)

    def test_initialization_failure_exits_promptly(self):
        started = time.monotonic()
        with self.assertRaises(WorkerError):
            ShareSubprocVecEnv([lambda: ProbeEnv(0, 'startup')],
                              startup_timeout=10, close_timeout=.2)
        self.assertLess(time.monotonic()-started, 10)

    def test_real_sc_environment_matches_serial_fixed_actions(self):
        from harl.envs.uav_escs.SC.uav_escs_env_sc import SCUAVEnv
        config = configurations(steps=8000)['IC_HAPPO']['env_args']
        def factory(rank):
            def make():
                env = SCUAVEnv(config)
                env.seed(1 + rank * 1000)
                return env
            return make
        factories = [factory(rank) for rank in range(2)]
        parallel, serial = ShareSubprocVecEnv(factories), ShareDummyVecEnv(factories)
        try:
            equal_tree(parallel.reset(), serial.reset())
            rng = np.random.default_rng(691)
            for _ in range(20):
                actions = [[np.eye(size)[rng.integers(size)] for size in [3, 16, 16, 16]]
                           for _ in range(2)]
                equal_tree(parallel.step(actions), serial.step(actions))
        finally:
            parallel.close()
            serial.close()
        self.stopped(parallel)


if __name__ == '__main__':
    unittest.main()
