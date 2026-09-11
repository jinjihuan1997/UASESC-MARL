"""Diagnostic only: change instruction at matched physical states; do not tune policy."""
import numpy as np
from common import ROOT,configuration,write,verify_reference
from harl.envs.uav_escs.SC.uav_escs_env_sc import SCUAVEnv

def main():
    verify_reference();records=[]
    cfg=configuration('HAPPO_hidden_instruction',85,8000)
    for seed in [20260911,20260912,20260913]:
        env=SCUAVEnv(cfg['env_args']);env.seed(seed);env.reset()
        for sample in range(100):
            # Initial preloaded caches; independently evolving channel samples.
            # Compare the exact same cache/AoI/channel state for all instructions.
            observations=[];masks=[]
            for gid in range(3):
                env.current_instruction_id=gid;env._refresh_instruction_context_from_current()
                observations.append(env._build_obs_multi().copy())
                masks.append(env._build_action_available_masks().copy())
            sut_changed=any(not np.array_equal(observations[0][0],x[0]) for x in observations[1:])
            uav_obs_equal=all(np.array_equal(observations[0][1:],x[1:]) for x in observations[1:])
            assert uav_obs_equal
            unique_masks=len({x[1:].tobytes() for x in masks})
            records.append(dict(seed=seed,sample=sample,hidden_sut_demand_feature_changes=sut_changed,
                hidden_uav_explicit_observations_identical=uav_obs_equal,distinct_joint_uav_masks=unique_masks))
            env._update_channels();env._update_semantic_tables()
    write(ROOT/'instruction_visibility.json',dict(passed=True,physical_states=len(records),
        design='300 preloaded-cache channel states, 3 development seeds, same-state instruction swaps; diagnostic only',
        hidden_sut_feature_changed_fraction=float(np.mean([r['hidden_sut_demand_feature_changes'] for r in records])),
        hidden_uav_observation_equal_fraction=1.0,
        all_three_joint_masks_distinct_fraction=float(np.mean([r['distinct_joint_uav_masks']==3 for r in records])),records=records))
    print({k:v for k,v in __import__('json').loads((ROOT/'instruction_visibility.json').read_text()).items() if k!='records'})

if __name__=='__main__':main()
