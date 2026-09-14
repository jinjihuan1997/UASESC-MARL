"""Independent profile/physics reconstruction and exact saved-input policy replay."""
from study import *
from student_policy import Student,physical_shares,target_alpha
from teacher_interface import Teacher
import argparse

class Arrays(dict):
 @property
 def files(self):return list(self)

def load_arrays(path):
 with np.load(path,allow_pickle=False) as z:return Arrays({k:z[k] for k in z.files})

def independent_tables(env):
 p=env.p;diff=arr(env.pos_uav)-arr(env.sut_pos)
 distance=np.maximum(np.linalg.norm(diff,axis=-1),1.);horizontal=np.linalg.norm(diff[...,:2],axis=-1)
 angle=np.arctan2(np.abs(diff[...,2]),horizontal+1e-9)*180/np.pi
 los=1/(1+p.atg_los_a*np.exp(-p.atg_los_b*(angle-p.atg_los_a)))
 path=20*np.log10(distance)+20*np.log10(p.f_uav_sut)+20*np.log10(4*np.pi/p.c_light)+los*p.atg_eta_los_db+(1-los)*p.atg_eta_nlos_db
 def snr(loss,power,bandwidth,nf,gain):
  noise=p.N0*bandwidth*10**(nf/10)
  return np.maximum(power*10**((gain-loss)/10)/max(noise,1e-30),1e-12)
 us=snr(path[None]+arr(env.noise_us)[:600],p.p_uav_sut_w,p.B_uav_sut,p.nf_uav_sut_db,p.g_uav_sut_tx_db+p.g_uav_sut_rx_db)
 sl=20*np.log10(max(p.satellite_slant_range_m,1))+20*np.log10(p.f_sut_sat)+20*np.log10(4*np.pi/p.c_light)
 sat=snr(sl+arr(env.noise_sat)[:600],p.p_sut_sat_w,p.B_sut_sat,p.nf_sut_sat_db,p.g_sut_sat_tx_db+p.g_sut_sat_rx_db)
 gamma=np.maximum(us*sat[...,None]/(us+sat[...,None]+1),1e-12);db=10*np.log10(gamma)
 with np.load(FROZEN/'source/reference/inputs/profile.npz',allow_pickle=False) as z:
  grid=z['snr_grid_db'];x=np.clip(db,grid[0],grid[-1]);hi=np.clip(np.searchsorted(grid,x,side='right'),1,len(grid)-1);lo=hi-1;frac=(x-grid[lo])/(grid[hi]-grid[lo])
  def lookup(key):
   a=np.moveaxis(z[key][:,lo],0,-1);b=np.moveaxis(z[key][:,hi],0,-1);return a+(b-a)*frac[...,None]
  quality=lookup('q_hat_mean');load=lookup('bar_ls_main_mean')+p.side_info_bits/np.log2(1+gamma[...,None])
 gids=arr(env.instructions)[:600].astype(int);buckets=(db[...,None]>=np.asarray([5,10,15])).sum(-1)
 req=np.asarray(p.Q_req_by_instruction_snr_bucket)[gids[:,:,None],buckets]
 return dict(quality=quality,load=load,req=req)

def equivalent_map():
 with np.load(FROZEN/'source/reference/inputs/profile.npz',allow_pickle=False) as z:
  mapping=[];groups=[]
  for mode in range(16):
   found=next((g for g,items in enumerate(groups) if all(np.array_equal(z[k][mode],z[k][items[0]]) for k in ['q_hat_mean','bar_ls_main_mean'])),None)
   if found is None:found=len(groups);groups.append([])
   groups[found].append(mode);mapping.append(found)
 return np.array(mapping),groups

@torch.inference_mode()
def audit(path,inputs=True,replay=True,diagnose=True):
 path=Path(path);meta=read(path.with_suffix('.json'));identity=meta['identity'];m=manifest();relative=str(path.relative_to(HERE))
 outpath=HERE/'audits'/path.relative_to(HERE).with_suffix('.json')
 audit_identity=dict(trace_sha256=sha(path),source_sha256=sha(HERE/'audit_trajectories.py'),manifest_sha256=sha(HERE/'manifest.json'),inputs=inputs,replay=replay,diagnose=diagnose)
 if outpath.exists():
  old=read(outpath)
  if old.get('identity')==audit_identity and old.get('state')=='PASS':return old
 assert meta['state']=='complete' and meta['trace_sha256']==audit_identity['trace_sha256'] and identity['manifest_sha256']==sha(HERE/'manifest.json')
 z=load_arrays(path);seeds=z['seeds'].tolist();assert seeds==identity['seeds'];E=len(seeds)
 env=make_env(seeds,identity['purpose'],identity['schedule']);assert frozen.external_hashes(env)==meta['external_hashes']
 for k in ('q','tau','aoi'):np.testing.assert_array_equal(arr(getattr(env,k)),z['initial_'+k])
 gid=z['trace'][:,:,FIELDS.index('instruction_id')].astype(int);np.testing.assert_array_equal(gid,arr(env.instructions)[:600])
 tab=independent_tables(env);physical=prior_auditor().physical_audit(z,tab,cfg())
 student=None;streams=None;forwards=0;teacher=Teacher() if diagnose or identity['student_seed'] is None else None
 if identity['student_seed'] is not None:
  s=identity['student_seed'];stage='A3' if identity['purpose']=='gate' else 'A'+str(int(path.parent.name[1:])-1)
  ckpt=HERE/f'students/{s}/{stage}/checkpoint.pt';assert sha(ckpt)==identity['checkpoint_sha256']
  student=Student(s,env);student.load(ckpt);student.set_mode(False);hashes=student.hashes()
  assert hashes==meta['initial_actor_hashes']==meta['final_actor_hashes']
  if not identity['deterministic']:streams=ActionStreams(identity['action_seeds'],'cpu')
 for t in range(600):
  if inputs:
   for k in ('q','tau','aoi'):
    src=z['initial_'+k] if t==0 else z[{'q':'cache_after','tau':'tau_after','aoi':'aoi_after'}[k]][t-1]
    setattr(env,k,env.tensor(src,getattr(env,k).dtype))
   env.step_index=t;env.allocation_pending=False
   env.beta=env.tensor(np.full((E,3),1/3) if t==0 else z['resource_fractions'][t-1]);env.update_channels(t);env.update_tables()
   pre,_,ma=env.observe();np.testing.assert_array_equal(arr(pre[:,0]),z['sut_obs'][t])
   post,_,am=env.allocate_resources(torch.from_numpy(z['raw_resource_action'][t]))
   np.testing.assert_array_equal(arr(post[:,1:]),z['post_uav_obs'][t]);np.testing.assert_array_equal(arr(am[:,1:]).astype(bool),z['uav_masks'][t])
   # No physical transition, reward or teacher-controlled action during replay.
   assert env.step_index==t
  if replay and student:
   for i in range(4):
    obs=torch.from_numpy(z['sut_obs'][t] if i==0 else z['post_uav_obs'][t,:,i-1]);mask=torch.ones(E,3) if i==0 else torch.from_numpy(z['uav_masks'][t,:,i-1])
    with streams.use(i) if streams else contextlib.nullcontext():act,lp=student.forward(i,obs,mask,identity['deterministic'])
    wanted=z['raw_resource_action'][t] if i==0 else z['mode_actions'][t,:,i-1];logp=z['sut_logp'][t] if i==0 else z['uav_logp'][t,:,i-1]
    np.testing.assert_array_equal(arr(act),wanted);np.testing.assert_array_equal(arr(lp),logp)
    elp=student.evaluate_actions(i,obs,act,mask)[0];np.testing.assert_array_equal(arr(elp),logp)
    forwards+=2*E
 if student:assert student.hashes()==hashes
 details={};mapping,groups=equivalent_map()
 if teacher:
  # All teacher calls are after trajectory completion; only original local inputs.
  ta=[];labels=[]
  for start in range(0,600*E,2048):
   stop=min(start+2048,600*E);obs=torch.from_numpy(z['sut_obs'].reshape(-1,76)[start:stop]);ta.append(arr(teacher.resource(obs)))
   part=[]
   for u in range(3):
    lo=torch.from_numpy(z['post_uav_obs'][:,:,u].reshape(-1,76)[start:stop]);ma=torch.from_numpy(z['uav_masks'][:,:,u].reshape(-1,16)[start:stop])
    part.append([arr(x) for x in teacher.mode_label(u,lo,ma)])
   labels.append([np.stack([part[u][j] for u in range(3)],-1) for j in range(4)])
  ta=np.concatenate(ta).reshape(600,E,3);requested,effective,active,reason=[np.concatenate([row[j] for row in labels]).reshape(600,E,3) for j in range(4)]
  if identity['labels']:
   for key,value in [('teacher_resource',ta),('teacher_requested',requested),('teacher_effective',effective),('teacher_active',active),('teacher_skip_reason',reason)]:np.testing.assert_array_equal(z[key],value)
  elif student is None:
   np.testing.assert_array_equal(z['raw_resource_action'],ta);np.testing.assert_array_equal(z['requested_modes'],requested)
  before_q=np.concatenate([z['initial_q'][None],z['cache_after'][:-1]])
  valid=(tab['quality']>=tab['req'][...,None]-1e-9)&(tab['load']<=z['budgets'][...,None]+1e-9)&before_q.any(-1)[...,None]
  assert np.take_along_axis(valid,effective[...,None],-1).squeeze(-1)[active].all()
  executed=np.where(np.take_along_axis(valid,requested[...,None],-1).squeeze(-1),requested,valid.argmax(-1));np.testing.assert_array_equal(executed[active],effective[active])
  stshares=z['resource_fractions'];tshares=arr(physical_shares(torch.from_numpy(ta)));smooth=target_alpha(torch.from_numpy(ta))/1000;smshares=arr(.05+.85*smooth)
  # Student probabilities were recorded at the same executed post-allocation inputs.
  pred=z['uav_probabilities'].argmax(-1);prob=z['uav_probabilities'].astype(np.float64);rawentropy=-(prob*np.log(np.maximum(prob,1e-300))).sum(-1)
  grouped=np.stack([prob[...,g].sum(-1) for g in groups],-1);groupentropy=-(grouped*np.log(np.maximum(grouped,1e-300))).sum(-1)
  for g in range(3):
   take=gid==g;n=int(take.sum())
   if not n:continue
   ge=np.abs(stshares[take]-tshares[take]);bias=np.abs(smshares[take]-tshares[take]);uavs=[]
   for u in range(3):
    use=take&active[:,:,u];target=effective[:,:,u];px=pred[:,:,u];nv=int(use.sum())
    selected=np.take_along_axis(prob[:,:,u],target[...,None],-1).squeeze(-1)
    uavs.append(dict(valid_samples=nv,skip_reason_counts=np.bincount(reason[:,:,u][take],minlength=4).tolist(),original_mode_accuracy=float((px[use]==target[use]).mean()) if nv else None,equivalent_mode_accuracy=float((mapping[px[use]]==mapping[target[use]]).mean()) if nv else None,teacher_cross_entropy=float(-np.log(np.maximum(selected[use],1e-300)).mean()) if nv else None,original_entropy=float(rawentropy[:,:,u][take].mean()),equivalence_entropy=float(groupentropy[:,:,u][take].mean())))
   details[str(g)]=dict(slots=n,share_absolute_error_mean=ge.mean(0).tolist(),share_absolute_error_p95=np.quantile(ge,.95,axis=0).tolist(),budget_absolute_error_mean=(60000*ge).mean(0).tolist(),smoothing_share_bias_mean=bias.mean(0).tolist(),smoothing_budget_bias_mean=(60000*bias).mean(0).tolist(),student_total_concentration_mean=float(z['sut_concentration'][take].sum(-1).mean()),student_total_concentration_p05_p95=np.quantile(z['sut_concentration'][take].sum(-1),[.05,.95]).tolist(),uavs=uavs)
  teacher.assert_frozen()
 result=dict(state='PASS',identity=audit_identity,physical=physical,observation_reconstruction='ALL_SLOTS_EXACT' if inputs else 'NOT_RUN',policy_action_logp_replay='ALL_SLOTS_EXACT' if student and replay else 'TEACHER_LOCAL_QUERY_REPRODUCTION',teacher_label_reproduction='ALL_SLOTS_EXACT' if identity['labels'] else 'posthoc_readonly_diagnostic',model_hashes_unchanged=student.hashes()==hashes if student else True,diagnostics=details,equivalence_groups=groups,seeds=seeds,external_hashes=meta['external_hashes'],new_physical_steps=0,forward_example_evaluations=forwards,teacher_queries=teacher.queries if teacher else [0]*4)
 write(outpath,result);record_cost('offline_trajectory_audit',relative,physical_steps=0,forward_example_evaluations=forwards,teacher_queries=teacher.queries if teacher else [0]*4,seeds=seeds,purpose=identity['purpose']);print(relative,'PASS',flush=True);return result

def run(scope,student=None):
 verify()
 if scope=='data':paths=sorted((HERE/'data').rglob('batch_*.npz'))
 else:paths=sorted((HERE/'gate'/('teacher' if student is None else str(student))).rglob('*.npz'))
 for path in paths:audit(path)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--scope',choices=['data','gate'],required=True);p.add_argument('--student',type=int);a=p.parse_args();guard();run(a.scope,a.student)
