"""Budget boundary, action tie, and missing cache timestamp regressions."""
from pathlib import Path
import os
import sys
import unittest
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import configuration
from tensor_env import TensorSCEnv
from harl.envs.uav_escs.SC.uav_escs_env_sc import SCUAVEnv


class BudgetBoundaries(unittest.TestCase):
    def test_repeated_subtraction_and_low_index_ties(self):
        device=os.environ.get('GPU_ENV_TEST_DEVICE','cpu')
        for offset in (-1e-7,0.,1e-7):
            args=configuration()['env_args']
            gpu=TensorSCEnv(args,count=1,seed=1,device=device);gpu.reset()
            cpu=SCUAVEnv(args);cpu.seed(1);cpu.reset()
            # Synthetic fixture only; original profile file remains untouched.
            # Make every mode feasible, retain real measured load rows.
            cpu.semantic_library.profile['q_hat_mean'][:]=100.
            gpu.profiles[0].fill_(100.)
            cpu._update_semantic_tables();gpu.update_tables()
            budget=float(cpu.Lambda_sem[0,0,0])*2+offset
            for env in (cpu,gpu.p):
                env.B_sut_sat=budget*cpu.n_uav
                env.B_uav_sut=1e9
            # A cached stream with an invalid timestamp follows the CPU fallback.
            cpu.tau_cache[0,0]=-1;gpu.tau[0,0,0]=-1
            actions=[torch.zeros((1,s.shape[0]),device=device) for s in gpu.action_space]
            actual=gpu.step(actions,auto_reset=False)
            expected=cpu.step([a[0].cpu().numpy() for a in actions])
            np.testing.assert_array_equal(gpu.last['mode'][0].cpu(),cpu.last_selected_modes)
            np.testing.assert_array_equal(gpu.last['served'][0].reshape(-1).cpu(),cpu.last_y[cpu.owner_uav,np.arange(cpu.n_ds)])
            np.testing.assert_array_equal(gpu.aoi[0].reshape(-1).cpu(),cpu.A_rcc)
            np.testing.assert_allclose(actual[2][0].cpu(),expected[2],rtol=1e-6,atol=1e-6)
            # Equal AoI/action scores must serve lower DS IDs first.
            served=np.flatnonzero(cpu.last_y[0])
            np.testing.assert_array_equal(served,np.arange(len(served)))


if __name__=='__main__': unittest.main()
