"""Matched-state instruction visibility and unchanged critic/actor structure."""
import copy
import numpy as np
from common import ROOT,configuration,write,verify_reference,stamp
from harl.envs.uav_escs.SC.uav_escs_env_sc import SCUAVEnv

def main():
    verify_reference();records=[]
    for seed in [20260911,20260912,20260913]:
        env=SCUAVEnv(configuration()['env_args']);env.seed(seed);env.reset()
        rng=np.random.default_rng(seed)
        assert env.n_mu_modes==env.n_semantic_modes==16
        np.testing.assert_array_equal(env.semantic_mode_lookup_by_bucket_mu,np.tile(np.arange(16),(4,1)))
        for sample in range(30):
            for cache in ['full','partial','empty']:
                env.q_cache[:]=0
                q=np.ones(30,np.int8) if cache=='full' else (rng.random(30)<.5).astype(np.int8) if cache=='partial' else np.zeros(30,np.int8)
                env.q_cache[env.owner_uav,np.arange(30)]=q
                env.tau_cache[:]=-1;env.tau_cache[env.owner_uav,np.arange(30)]=np.where(q,0,-1)
                env.current_step=10;env.A_rcc=rng.integers(2,12,size=30).astype(float)
                env.beta_sut_sat=.05+.85*rng.dirichlet(np.ones(3));env._update_backhaul_budget()
                hidden=[];explicit=[];masks=[];shared=[]
                for gid in range(3):
                    env.current_instruction_id=gid;env._refresh_instruction_context_from_current()
                    env.actor_observe_instruction=True
                    o=env._build_obs_multi();s=env._build_share_obs_multi();a=env._build_action_available_masks()
                    env.actor_observe_instruction=False
                    h=env._build_obs_multi();hs=env._build_share_obs_multi();ha=env._build_action_available_masks()
                    np.testing.assert_array_equal(s,hs);np.testing.assert_array_equal(a,ha)
                    # Instruction-independent SUT prefix is exactly the same in both methods.
                    np.testing.assert_array_equal(o[0,:10],h[0,:10])
                    # Independent minimum-load formula, one copy per occupied source.
                    expected=np.array([sum(float(np.min(env.Lambda_sem[n,k,:])) for k in env.ds_by_uav[n] if env.q_cache[n,k]) for n in range(3)])/env.Lambda_ref
                    np.testing.assert_allclose(h[0,4:7],expected,rtol=1e-6,atol=1e-7)
                    hidden.append(h);explicit.append(o);masks.append(a);shared.append(s)
                for g in [1,2]:
                    np.testing.assert_array_equal(hidden[0],hidden[g])
                    np.testing.assert_array_equal(masks[0],masks[g])
                    assert not np.array_equal(explicit[0],explicit[g])
                    assert not np.array_equal(shared[0],shared[g])
                records.append({'seed':seed,'sample':sample,'cache':cache,'hidden_all_actor_inputs_invariant':True,
                    'all_actor_masks_invariant':True,'explicit_context_retained':True,'critic_identical_between_methods':True})
            env._update_channels();env._update_semantic_tables()
    write(ROOT/'interface_checks.json',dict(passed=True,utc=stamp(),physical_states=len(records),instruction_variants=3,
        table_modes=16,hidden_input_invariant_fraction=1.,mask_invariant_fraction=1.,records=records))
    print('PASS',len(records),'matched physical states with 3 instructions each')
if __name__=='__main__':main()
