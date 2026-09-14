"""Independent NumPy propagation/profile interpolation plus all-trajectory audit.

No native update_channels/update_tables output is used for this reconstruction.
Uses authenticated episode initial positions and frozen exogenous noise tapes.
"""
from support import *

def rebuild(scene):
 m=manifest();env=make_env(m['parents'][0],m['scenarios'][scene]);p=env.p
 air=arr(env.pos_uav);ground=arr(env.sut_pos);diff=air-ground
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
   table=z[key];a=np.moveaxis(table[:,lo],0,-1);b=np.moveaxis(table[:,hi],0,-1)
   return a+(b-a)*frac[...,None]
  quality=lookup('q_hat_mean');load=lookup('bar_ls_main_mean')+p.side_info_bits/np.log2(1+gamma[...,None])
 gid=np.zeros(600,dtype=int)
 for start,g in m['scenarios'][scene]:gid[start:]=g
 buckets=(db[...,None]>=np.asarray([5,10,15])).sum(-1)
 req=np.asarray(p.Q_req_by_instruction_snr_bucket)[gid[:,None,None],buckets]
 return dict(quality=quality,load=load,req=req,external=frozen.external_hashes(env))

def main():
 m=verify_inputs();refs=read(HERE/'reference_index.json');out={};checks=prior_audit_module()
 for scene in m['scenarios']:
  tab=rebuild(scene);items=[]
  for key,r in refs.items():
   if key.endswith('/'+scene):items.append((key,WORKSPACE/r['path'],r['metadata']['external_hashes']))
  for seed in m['parents']:
   for c in m['new_controllers']:
    path=HERE/f'evaluation/seed_{seed}/{c}/{scene}.npz';items.append((f'{seed}/{c}/{scene}',path,read(path.with_suffix('.json'))['external_hashes']))
  for key,path,external in items:
   assert external==tab['external']
   with np.load(path,allow_pickle=False) as z:out[key]=checks.physical_audit(z,tab,cfg_for(m['parents'][0]))
  print(scene,len(out),flush=True)
 assert len(out)==325 and not FORBIDDEN_CALLS
 write(HERE/'report/independent_profile_audit.json',dict(state='PASS',source_sha256=sha(HERE/'independent_profile_check.py'),profile_sha256=sha(FROZEN/'source/reference/inputs/profile.npz'),traces=325,physical_reward_checks=out,propagation='independent numpy ATG/SAT SNR and backhaul coupling from initial positions and exogenous noise tapes',profile_lookup='independent numpy raw profile interpolation, side information load and instruction SNR bucket requirements',native_channel_or_table_output_used=False,new_physical_steps=0,new_gradient_steps=0,new_optimizer_updates=0,environment_creations=ENV_CREATIONS,tolerance_changes=False))
if __name__=='__main__':guard();main()
