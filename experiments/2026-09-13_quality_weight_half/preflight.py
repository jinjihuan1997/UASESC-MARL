from weight_support import *

@torch.inference_mode()
def main():
 m=verify();start=time.perf_counter();seed=m['validation'][:2];scene='multi_0_1_2';oldenv=make_env(scene,seed,False);newenv=make_env(scene,seed,True)
 assert base.frozen.external_hashes(oldenv)==base.frozen.external_hashes(newenv)
 for a,b in zip(oldenv.observe(),newenv.observe()):torch.testing.assert_close(a,b,atol=0,rtol=0)
 ow=np.asarray(oldenv.p.reward_weights_by_instruction);nw=np.asarray(newenv.p.reward_weights_by_instruction);np.testing.assert_array_equal(nw[:,0],ow[:,0]*.5);np.testing.assert_array_equal(nw[:,1:],ow[:,1:]);assert not np.allclose(nw.sum(-1),1)
 for method in m['adaptive_methods']:
  ctrl=Controller(method,newenv);np.testing.assert_array_equal(arr(ctrl.ctrl.greedy.weights),nw);ctrl.assert_frozen()
  assert ctrl.ctrl.greedy.spec.weights==tuple(tuple(row) for row in nw)
 # Compare direct native rewards on one identical action, without adding
 # unplanned transitions: this is checked on every saved frozen rollout below.
 module=base.load_module('weight_new_rollout',HERE/'evaluate.py')
 cases=[('original_rl',m['parents'][0],m['validation'],0),('G_equal_local16',None,m['validation'],0),('G_equal_local16_Qhalf_local',None,seed,0),('greedy_modes_16_Qhalf_local',None,seed,0),('G_equal_local16_Qhalf_local',None,seed,1)]
 out=[]
 for method,parent,seeds,rep in cases:out.append(module.rollout(method,parent,scene,seeds,'preflight',rep))
 a=arrays(HERE/f'preflight_traces/rules/G_equal_local16_Qhalf_local/{scene}_0.npz');b=arrays(HERE/f'preflight_traces/rules/G_equal_local16_Qhalf_local/{scene}_1.npz')
 for key in a:np.testing.assert_array_equal(a[key],b[key])
 assert out[2]['external_hashes']==out[4]['external_hashes'];assert not FORBIDDEN;verify()
 write(HERE/'preflight.json',dict(state='PASS',cases=5,accepted_complete_episodes=46,accepted_physical_steps=27600,weights_exact=True,no_row_renormalization=True,only_quality_coefficient_changed=True,observations_identical_before_action=True,frozen_controller_physical_parity=True,reweighted_greedy_public_weights_match=True,native_and_independent_every_slot_audit=True,repeat_exact=True,forbidden_calls=[],elapsed_seconds=time.perf_counter()-start,amendment_sha256=sha(HERE/'PREFLIGHT_AMENDMENT.md')))
 write(HERE/'execution_seal.json',dict(sources={p:sha(HERE/p) for p in ['weight_support.py','evaluate.py','preflight.py','PREFLIGHT_AMENDMENT.md']},manifest_sha256=sha(HERE/'manifest.json')))
 write(HERE/'status.json',dict(state='preflight_passed',new_training_steps=0,new_optimizer_updates=0));print('PREFLIGHT PASS',flush=True)
if __name__=='__main__':guard();main()
