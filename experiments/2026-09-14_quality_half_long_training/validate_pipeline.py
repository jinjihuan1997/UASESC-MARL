"""Read-only policy replay on the inherited 1M state and real new training audit."""
from train_support import *
SHORT=HERE.parent/'2026-09-14_quality_half_retraining'
@torch.inference_mode()
def main():
 guard();m=verify();ev=base.load_module('long_fixed_evaluation',HERE/'evaluate.py');base.load_module('long_statistics_import_check',HERE/'aggregate.py');checks={}
 for seed in m['seeds']:
  env=ws.make_env('fixed_0');net=ev.Policy(seed,1000000,env);path=SHORT/f'evaluation/seed_{seed}/steps_1000000/fixed_0.npz';z=arrays(path);meta=read(path.with_suffix('.json'));assert meta['trace_sha256']==sha(path)
  for t in range(600):
   resource=net.call(0,torch.from_numpy(z['sut_obs'][t]),env.observe()[2][:,0]);np.testing.assert_array_equal(arr(resource),z['raw_resource_action'][t])
   for u in range(3):
    action=net.call(u+1,torch.from_numpy(z['post_uav_obs'][t,:,u]),torch.from_numpy(z['uav_masks'][t,:,u]));np.testing.assert_array_equal(arr(action.argmax(-1)),z['requested_modes'][t,:,u])
  net.assert_frozen();checks[str(seed)]=dict(inherited_fixed_0_actions_exact=True,read_only_replay_decisions=12000,new_physical_steps=0,source_sha256=sha(path))
 seed=m['seeds'][0];selected=sorted((HERE/f'jobs/seed_{seed}/trajectory_audits').glob('episode_*.npz'));assert selected
 path=selected[0];z=arrays(path);d=read(path.with_suffix('.json'));env=TrainingEnv(config(seed)['env_args'],10,seed,'cpu')
 for e in env.source.envs:e._episode_index=d['episode']-1
 env.reset();physics=ws.independent_check(z,ws.PHYS.tables(env));verify();write(HERE/'pipeline_validation.json',dict(state='PASS',policy_replays=checks,new_training_offline_reconstruction=physics,new_physical_steps=0,new_optimizer_steps=0));print('PIPELINE VALIDATION PASS',flush=True)
if __name__=='__main__':main()
