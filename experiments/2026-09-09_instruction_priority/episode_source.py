"""CPU episode setup only: preserve the reference's independent random streams.

Potential exogenous draws are uploaded once per 600-slot episode. Policies only
receive current observations; future draws are private environment state.
No reference step() is used during GPU training.
"""
import copy
import numpy as np
from common import activate_runtime
from harl.envs.uav_escs.SC.uav_escs_env_sc import SCUAVEnv


class EpisodeSource:
    def __init__(self, args, count, seed):
        self.envs = [SCUAVEnv(copy.deepcopy(args)) for _ in range(count)]
        for i, env in enumerate(self.envs):
            env.seed(seed + 1000*i)

    def next(self):
        episodes = []
        for env in self.envs:
            env.reset()
            u, d, steps = env.n_uav, env.n_ds, env.max_steps
            # Generate channel draws in exactly the CPU order: US, satellite, DU.
            # Recreate the channel stream at its start, including slot zero.
            stream = np.random.default_rng(np.random.SeedSequence(
                [env._base_seed, env._episode_index]).spawn(5)[1])
            sigmas = [env.sigma_uav_sut_db, env.sigma_sut_sat_db, env.sigma_ds_uav_db]
            widths = [u, 1, u*d]
            normals = stream.standard_normal((steps+1, sum(w for w,s in zip(widths,sigmas) if s>0)))
            offset, noise = 0, []
            for width, sigma in zip(widths, sigmas):
                if sigma > 0:
                    noise.append(normals[:, offset:offset+width]*sigma)
                    offset += width
                else:
                    noise.append(np.zeros((steps+1, width)))
            content_rng = copy.deepcopy(env.rng_content)
            potential = content_rng.random((steps, u, d, env.content_feature_dim))
            schedule = np.asarray([env._get_instruction_id_for_slot(t) for t in range(steps+1)])
            episodes.append(dict(pos_uav=env.pos_uav.copy(), pos_ds=env.pos_ds.copy(),
                q_cache=env.q_cache.copy(), tau_cache=env.tau_cache.copy(), A_rcc=env.A_rcc.copy(),
                psi=env.psi.copy(), noise_us=noise[0], noise_sat=noise[1][:,0],
                noise_du=noise[2].reshape(steps+1,u,d), potential_content=potential,
                instructions=schedule, episode_index=env._episode_index))
        return episodes
