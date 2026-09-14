"""Local-interface, ordering, physical and frozen-reference preflight gate."""
from light_support import *
from lightweight_policies import *
from reference_policies import policy,policy_hash
from independent_physics import tables,check
import inspect,ast

@torch.inference_mode()
def unit():
 obs=torch.zeros(8,76);mask=torch.ones(8,16);obs[:,0]=.2;obs[:,1:11]=1;obs[:,11:21]=.25;obs[:,21:31]=.375;obs[:,31:47]=.01;obs[:,47:63]=25/33;obs[:,74]=23/33
 gid=torch.arange(8)%3;obs[:,67:70]=torch.nn.functional.one_hot(gid,3).float();mask[1]=0;mask[1,[3,6]]=1;obs[1,31+6]=.009;obs[1,47+3]=29/33
 obs[2,1:11]=0;obs[3,0]=0;obs[4,31+5]=.019;obs[4,47+5]=30/33;obs[5,47+7]=29/33;mask[6,3]=0;obs[6,31+3]=.0001;obs[6,47+3]=1
 sut=torch.zeros(8,76);sut[:,23:26]=torch.nn.functional.one_hot(gid,3).float();sut[:,13:16]=torch.tensor([[0,0,0],[1,2,3],[3,0,0],[0,3,0],[0,0,3],[1,1,1],[4,2,1],[2,0,1]])/80
 dummy=types.SimpleNamespace(dtype=torch.float64,M=16,U=3)
 torch.testing.assert_close(equal_resource(sut),frozen.simple_rule_actions(dummy,'R_equal_single',sut[:,None])[0],atol=0,rtol=0)
 torch.testing.assert_close(urgency_resource(sut),frozen.simple_rule_actions(dummy,'R_single',sut[:,None])[0],atol=0,rtol=0)
 original_constructor=ONLINE.TreeScorePredictor.__init__;calls=[]
 def forbid_predictor(*a,**k):calls.append('predictor');raise AssertionError('New methods cannot load a predictor')
 ONLINE.TreeScorePredictor.__init__=forbid_predictor
 try:
  for method in METHODS:
   ctrl=Lightweight(method);before=obs.clone();bm=mask.clone();action=ctrl.mode(0,obs,mask);resource=ctrl.resource(sut);torch.testing.assert_close(obs,before,atol=0,rtol=0);torch.testing.assert_close(mask,bm,atol=0,rtol=0)
   if method.startswith('G_'):torch.testing.assert_close(action,ONLINE.UAVGreedy(list(range(16))).act(obs,mask),atol=0,rtol=0)
   else:
    load,quality,valid=allowed_local(obs,mask);expected=[]
    for i in range(8):
     ids=np.flatnonzero(arr(valid[i]));expected.append(min(ids,key=lambda k:(float(load[i,k]),-float(quality[i,k]),int(k)) if method=='R_equal_minload' else (-float(quality[i,k]),float(load[i,k]),int(k))) if len(ids) else 0)
    assert action.argmax(-1).tolist()==list(map(int,expected));assert int(action[0].argmax())==0 and int(action[2].argmax())==0 and int(action[3].argmax())==0
   for i in range(8):torch.testing.assert_close(ctrl.mode(0,obs[i:i+1],mask[i:i+1]),action[i:i+1],atol=0,rtol=0)
   assert not calls;ctrl.assert_frozen()
 finally:ONLINE.TreeScorePredictor.__init__=original_constructor
 env=make_env([[0,0]],manifest()['validation'][:2]);pre,_,am=env.observe();before={k:getattr(env,k).clone() for k in ['q','tau','aoi']};action=equal_resource(pre[:,0]);post,_,ma=env.allocate_resources(action)
 for k,v in before.items():assert torch.equal(getattr(env,k),v)
 assert env.step_index==0;np.testing.assert_allclose(arr(env.beta),1/3,atol=1e-12,rtol=0);np.testing.assert_allclose(arr(env.budget),20000,atol=1e-8,rtol=0)
 assert torch.equal(post[:,1:,0],(env.budget/100000).float())
 for f in [Lightweight.resource,Lightweight.mode,allowed_local,equal_resource,urgency_resource]:
  source=inspect.getsource(f);tree=ast.parse(__import__('textwrap').dedent(source));identifiers={x.id for x in ast.walk(tree) if isinstance(x,ast.Name)}
  assert not identifiers&{'env','critic','instructions','noise_us','noise_sat','future','step','commit_modes'}
 return dict(state='PASS',original_UAVGreedy_exact=True,predictor_construction_calls=0,fixed_tie_ordering=True,mask_and_empty_boundaries=True,original_helpers_resource_exact=True,mixed_instruction_batch_and_rowwise=True,input_tensors_unmodified=True,no_hidden_or_future_interface=True,allocation_no_physical_transition=True,lower_bound_once=True)

@torch.inference_mode()
def reference_replay():
 m=manifest();refs=read(HERE/'reference_index.json')['reused'];scene='multi_0_1_2';out={}
 cases=[(method,None) for method in ['greedy_modes_16','greedy_modes_3','R_instruction','R_equal_instruction']]+[(method,s) for method in ['original_rl','C1','C3'] for s in m['parents']]
 for method,parent in cases:
  key=f'{parent if parent else "rules"}/{method}/{scene}';item=refs[key];path=ROOT/item['path'];assert sha(path)==item['sha256'];z=arrays(path);meta=read(path.with_suffix('.json'));env=make_env(m['scenarios'][scene]);assert frozen.external_hashes(env)==meta['external_hashes'];ctrl=policy(method,env,parent);h=policy_hash(ctrl)
  physics=check(z,tables(env))
  for t in range(600):
   obs,_,mask=state_before(env,z,t);resource=ctrl.resource(obs[:,0],mask[:,0]);np.testing.assert_array_equal(arr(resource),z['raw_resource_action'][t]);post,_,am=env.allocate_resources(resource)
   np.testing.assert_array_equal(arr(am[:,1:]).astype(bool),z['uav_masks'][t])
   for u in range(3):np.testing.assert_array_equal(arr(ctrl.mode(u,post[:,u+1],am[:,u+1]).argmax(-1)),z['requested_modes'][t,:,u])
  ctrl.assert_frozen();assert policy_hash(ctrl)==h;out[key]=dict(state='PASS',policy_hash=h,trace_sha256=sha(path),physical_audit=physics);cost('reference_preflight_replay',method=method,parent=parent,scene=scene,physical_steps=0,replayed_sample_decisions=12000,seeds=m['validation']);print('reference replay',key,'PASS',flush=True)
 return out

def main():
 m=verify();start=time.perf_counter();write(HERE/'status.json',dict(state='preflight_running',new_training_steps=0,new_optimizer_updates=0))
 try:
  u=unit();write(HERE/'preflight_unit.json',u);refs=reference_replay();write(HERE/'preflight_reference_replay.json',refs)
  ev=load_module('lightweight_new_evaluation',HERE/'evaluate.py');repeats={}
  for rep,methods in [(0,METHODS),(1,list(reversed(METHODS)))]:
   for method in methods:repeats[f'{method}/{rep}']=ev.rollout(method,'multi_0_1_2','preflight',m['validation'][:2],rep)
  for method in METHODS:
   a=arrays(HERE/f'preflight_traces/{method}/multi_0_1_2_0.npz');b=arrays(HERE/f'preflight_traces/{method}/multi_0_1_2_1.npz')
   for k in a:np.testing.assert_array_equal(a[k],b[k],err_msg=method+':'+k)
   assert repeats[f'{method}/0']['external_hashes']==repeats[f'{method}/1']['external_hashes']
  for method in ['R_equal_single','R_single']:ev.rollout(method,'multi_0_1_2','preflight_prefix',repeat=0,length=30)
  assert not FORBIDDEN;verify()
  source=['light_support.py','lightweight_policies.py','reference_policies.py','evaluate.py','independent_physics.py','validate.py']
  write(HERE/'execution_seal.json',dict(source_sha256={p:sha(HERE/p) for p in source},manifest_sha256=sha(HERE/'manifest.json')))
  write(HERE/'preflight.json',dict(state='PASS',unit=u,reference_replay_cases=len(refs),reference_replay_samples=156000,repeat_complete_episodes=16,prefix_steps=1200,total_physical_steps=10800,reference_replay_physical_steps=0,full_new_physics_and_exact_repeat=True,forbidden_calls=[],elapsed_seconds=time.perf_counter()-start,source_sha256=sha(HERE/'validate.py')))
  write(HERE/'status.json',dict(state='preflight_passed',new_training_steps=0,new_optimizer_updates=0));print('PREFLIGHT PASS',flush=True)
 except BaseException as error:
  write(HERE/f'preflight_failures/{time.time_ns()}.json',dict(error=repr(error),traceback=traceback.format_exc()));raise
if __name__=='__main__':guard();main()
