"""Exact preflight relationships, mixed batches, frozen-state and routing audit."""
from support import *
from composition_policy import Composition
spec=importlib.util.spec_from_file_location('recomposition_evaluate',HERE/'evaluate.py')
evaluation=importlib.util.module_from_spec(spec);spec.loader.exec_module(evaluation)
rollout=evaluation.rollout;code_identity=evaluation.code_identity
import argparse,traceback
SCENES=['fixed_0','fixed_1','fixed_2','switch300_0_to_1','switch300_1_to_2']

def newpath(seed,c,s,repeat=False):return HERE/('preflight_repeat' if repeat else 'preflight_traces')/f'seed_{seed}'/c/f'{s}.npz'
def refpath(seed,method,s):return WORKSPACE/read(HERE/'reference_index.json')[f'{seed}/{method}/{s}']['path']
def same(a,b,ignore=()):
 with np.load(a,allow_pickle=False) as x,np.load(b,allow_pickle=False) as y:
  common=set(x.files)&set(y.files)-set(ignore)
  for k in common:np.testing.assert_array_equal(x[k],y[k],err_msg=f'{a}:{b}:{k}')
 return dict(left=str(a.relative_to(WORKSPACE)),right=str(b.relative_to(WORKSPACE)),fields=sorted(common),left_sha256=sha(a),right_sha256=sha(b))

@torch.inference_mode()
def mixed_batch(seed):
 m=manifest();envs=[make_env(seed,[[0,g]]) for g in range(3)]
 pre=[];post=[];masks=[];equal=[]
 for env in envs:
  obs,_,mask=env.observe();eq=frozen.simple_rule_actions(env,'R_equal_instruction',obs)[0]
  before={k:getattr(env,k).clone() for k in ('q','tau','aoi')};slot=env.step_index
  p,_,am=env.allocate_resources(eq)
  assert env.step_index==slot==0
  for k,v in before.items():assert torch.equal(v,getattr(env,k))
  pre.append(obs);post.append(p);masks.append(am);equal.append(eq)
 gids=torch.arange(20)%3;rows=torch.arange(20)
 obs=torch.stack(pre)[gids,rows];loc=torch.stack(post)[gids,rows];am=torch.stack(masks)[gids,rows];eq=torch.stack(equal)[gids,rows]
 assertions=0
 for c in m['controllers']:
  policy=Composition(seed,envs[0],c);resource,rs=policy.resource(obs[:,0],am[:,0],eq)
  expected=((gids==2)&policy.Q);assert torch.equal(rs.bool(),expected)
  orig=policy.call(policy.base[0],obs[:,0],am[:,0]);torch.testing.assert_close(resource,torch.where(expected[:,None],eq,orig) if policy.Q else orig,rtol=0,atol=0)
  for i in range(3):
   action,ps=policy.mode(i,loc[:,i+1],am[:,i+1]);want=torch.where(gids==2,2,torch.where((gids==0)&policy.B,1,0))
   assert torch.equal(ps,want)
   for source in (0,1,2):
    candidate=policy.call(policy.uav[source][i],loc[:,i+1],am[:,i+1]);torch.testing.assert_close(action[want==source],candidate[want==source],atol=0,rtol=0)
    assertions+=1
  policy.assert_frozen()
 return dict(state='PASS',mixed_gids=gids.tolist(),exact_source_action_assertions=assertions,resource_stage_physical_steps=0)

@torch.inference_mode()
def replay_reference(seed,scene):
 """Replay C0 on old before-state, without advancing or splicing a new episode."""
 env=make_env(seed,manifest()['scenarios'][scene]);policy=Composition(seed,env,'C0');p=refpath(seed,'C0',scene)
 with np.load(p,allow_pickle=False) as z:
  for slot in range(600):
   env.step_index=slot;env.allocation_pending=False
   env.q=torch.as_tensor(z['initial_q'] if slot==0 else z['cache_after'][slot-1],dtype=torch.bool)
   env.tau=torch.as_tensor(z['initial_tau'] if slot==0 else z['tau_after'][slot-1],dtype=torch.int64)
   env.aoi=torch.as_tensor(z['initial_aoi'] if slot==0 else z['aoi_after'][slot-1],dtype=torch.float64)
   env.beta=torch.full((20,3),1/3,dtype=torch.float64) if slot==0 else torch.as_tensor(z['resource_fractions'][slot-1])
   env.update_channels(slot);env.update_tables();obs,_,mask=env.observe()
   eq=frozen.simple_rule_actions(env,'R_equal_instruction',obs)[0];a,_=policy.resource(obs[:,0],mask[:,0],eq)
   np.testing.assert_array_equal(arr(a),z['raw_resource_action'][slot])
   post,_,am=env.allocate_resources(a);np.testing.assert_array_equal(arr(am[:,1:]).astype(bool),z['uav_masks'][slot])
   for i in range(3):
    act,_=policy.mode(i,post[:,i+1],am[:,i+1]);np.testing.assert_array_equal(arr(act.argmax(-1)),z['requested_modes'][slot,:,i])
   np.testing.assert_array_equal(arr(env.beta),z['resource_fractions'][slot])
  policy.assert_frozen()
 return dict(state='PASS',scenario=scene,reference_sha256=sha(p),inference_reconstructed_slots=12000,physical_steps=0)

def parent_worker(seed):
 m=verify_inputs();matches=[];references=[]
 for c in m['controllers']:
  folder=newpath(seed,c,'fixed_0').parent
  for s in SCENES:rollout(seed,c,s,folder)
 rollout(seed,'C0','multi_2_1_0',newpath(seed,'C0','multi_2_1_0').parent)
 for s in SCENES+['multi_2_1_0']:matches.append(same(newpath(seed,'C0',s),refpath(seed,'C0',s)))
 for c in m['controllers']:
  matches.append(same(newpath(seed,c,'fixed_1'),refpath(seed,'original_rl','fixed_1'),ignore=['adapter_enabled']))
  matches.append(same(newpath(seed,c,'fixed_0'),refpath(seed,'residual_all' if c in ('C1','C3') else 'original_rl','fixed_0'),ignore=['adapter_enabled']))
 for c in ('C0','C1'):matches.append(same(newpath(seed,c,'fixed_2'),refpath(seed,'C0','fixed_2')))
 matches.append(same(newpath(seed,'C2','fixed_2'),newpath(seed,'C3','fixed_2')))
 # Source annotations differ in controller identity only; used routes are identical.
 for s,pairs in [('switch300_0_to_1',[('C0','C2'),('C1','C3')]),('switch300_1_to_2',[('C0','C1'),('C2','C3')])]:
  for a,b in pairs:matches.append(same(newpath(seed,a,s),newpath(seed,b,s)))
 for c in ('C2','C3'):
  with np.load(newpath(seed,c,'fixed_2')) as z:
   np.testing.assert_allclose(z['resource_fractions'],np.full_like(z['resource_fractions'],1/3),atol=1e-12,rtol=0)
   np.testing.assert_array_equal(z['resource_source'],np.ones((600,20),dtype=np.int8))
 for scene in m['scenarios']:
  if scene not in SCENES+['multi_2_1_0']:references.append(replay_reference(seed,scene))
 if seed==m['parents'][0]:
  rollout(seed,'C3','switch300_1_to_2',newpath(seed,'C3','switch300_1_to_2',True).parent)
  matches.append(same(newpath(seed,'C3','switch300_1_to_2'),newpath(seed,'C3','switch300_1_to_2',True)))
 result=dict(state='PASS',parent=seed,matches=matches,remaining_C0_reference_inference=references,mixed_batch=mixed_batch(seed),environment_creations=ENV_CREATIONS,forbidden_training_calls=FORBIDDEN_CALLS,code_sha256=code_identity())
 write(HERE/f'preflight/seed_{seed}.json',result);print(json.dumps(dict(preflight_parent=seed,state='PASS')),flush=True)

def finish():
 m=verify_inputs();items=[read(HERE/f'preflight/seed_{s}.json') for s in m['parents']]
 assert all(x['state']=='PASS' and x['code_sha256']==code_identity() and not x['forbidden_training_calls'] for x in items)
 refs=read(HERE/'reference_index.json');allmeta=[]
 for p in (HERE/'preflight_traces').rglob('*.json'):
  d=read(p)
  if 'external_hashes' in d:
   i=d['identity'];expected=refs[f'{i["parent"]}/C0/{i["scenario"]}']['metadata']['external_hashes'];assert d['external_hashes']==expected
   allmeta.append(d)
 ledger=[read(p) for p in (HERE/'compute_ledger').glob('*.json')]
 assert sum(x['episodes'] for x in ledger if x['kind']=='preflight')==1280
 assert sum(x['physical_steps'] for x in ledger)==768000
 assert all(x['state']=='complete' for x in ledger)
 write(HERE/'preflight.json',dict(state='PASS',parent_checks=items,complete_episodes=1280,physical_steps=768000,reference_inference_only_slots=sum(y['inference_reconstructed_slots'] for x in items for y in x['remaining_C0_reference_inference']),reference_supplements=[],tolerance_changes=False,external_hash_pairing='PASS_all_preflight_against_previous_reference',protected_files=len(m['protected_input_sha256']),all_creations_development_only=True,new_training_steps=0,new_optimizer_updates=0,forbidden_training_calls=[]))
 write(HERE/'execution_seal.json',dict(protocol_sha256=sha(HERE/'PROTOCOL.md'),manifest_sha256=sha(HERE/'manifest.json'),code_sha256={p:sha(HERE/p) for p in ('support.py','composition_policy.py','evaluate.py','validate.py')},preflight_sha256=sha(HERE/'preflight.json')))
 write(HERE/'resource_settings.json',dict(device='cpu',workers=3,torch_threads=1,blas_threads=1,preflight_single_worker_measured_steps_per_second=ledger[0]['physical_steps_per_second'],reason='Exact frozen CPU execution reproduced; small batch inference and native checks measured fast; 3 single-thread workers within current 20 CPU capacity; GPU untouched'))
 print('ALL PREFLIGHT PASS; execution sealed',flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int);p.add_argument('--finish',action='store_true');a=p.parse_args();guard()
 try:finish() if a.finish else parent_worker(a.seed)
 except BaseException as e:
  write(HERE/f'preflight_failures/{time.time_ns()}.json',dict(error=repr(e),traceback=traceback.format_exc(),argv=sys.argv));raise
