"""Reuse SC's episode setup and spaces, with a complete CC row lookup."""
import numpy as np
from common import activate_runtime
from harl.envs.uav_escs.SC.uav_escs_env_sc import SCUAVEnv


class CCSetupEnv(SCUAVEnv):
    def _build_semantic_mode_lookup(self):
        if self.n_semantic_modes!=20 or not self.semantic_library.profile:
            raise ValueError('CC requires the frozen 20-row table')
        return np.tile(np.arange(20,dtype=int),(4,1))

    def _update_semantic_tables(self):
        self.side_info_bits=0.
        return super()._update_semantic_tables()

    def step(self,*args,**kwargs):
        raise RuntimeError('CCSetupEnv only generates episodes; TensorCCEnv implements packet delivery')
