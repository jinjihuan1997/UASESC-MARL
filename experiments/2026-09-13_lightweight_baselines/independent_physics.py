"""Independent NumPy propagation/profile and original NumPy executor audit."""
from light_support import *

def tables(env):
 p=env.p;diff=arr(env.pos_uav)-arr(env.sut_pos);distance=np.maximum(np.linalg.norm(diff,axis=-1),1.);horizontal=np.linalg.norm(diff[...,:2],axis=-1)
 angle=np.arctan2(np.abs(diff[...,2]),horizontal+1e-9)*180/np.pi;los=1/(1+p.atg_los_a*np.exp(-p.atg_los_b*(angle-p.atg_los_a)))
 path=20*np.log10(distance)+20*np.log10(p.f_uav_sut)+20*np.log10(4*np.pi/p.c_light)+los*p.atg_eta_los_db+(1-los)*p.atg_eta_nlos_db
 def snr(loss,power,bandwidth,nf,gain):return np.maximum(power*10**((gain-loss)/10)/max(p.N0*bandwidth*10**(nf/10),1e-30),1e-12)
 us=snr(path[None]+arr(env.noise_us)[:600],p.p_uav_sut_w,p.B_uav_sut,p.nf_uav_sut_db,p.g_uav_sut_tx_db+p.g_uav_sut_rx_db)
 sl=20*np.log10(max(p.satellite_slant_range_m,1))+20*np.log10(p.f_sut_sat)+20*np.log10(4*np.pi/p.c_light)
 sat=snr(sl+arr(env.noise_sat)[:600],p.p_sut_sat_w,p.B_sut_sat,p.nf_sut_sat_db,p.g_sut_sat_tx_db+p.g_sut_sat_rx_db)
 gamma=np.maximum(us*sat[...,None]/(us+sat[...,None]+1),1e-12);db=10*np.log10(gamma)
 with np.load(FROZEN/'source/reference/inputs/profile.npz',allow_pickle=False) as z:
  grid=z['snr_grid_db'];x=np.clip(db,grid[0],grid[-1]);hi=np.clip(np.searchsorted(grid,x,side='right'),1,len(grid)-1);lo=hi-1;frac=(x-grid[lo])/(grid[hi]-grid[lo])
  def lookup(k):
   a=np.moveaxis(z[k][:,lo],0,-1);b=np.moveaxis(z[k][:,hi],0,-1);return a+(b-a)*frac[...,None]
  quality=lookup('q_hat_mean');load=lookup('bar_ls_main_mean')+p.side_info_bits/np.log2(1+gamma[...,None])
 gids=arr(env.instructions)[:600].astype(int);buckets=(db[...,None]>=np.array([5,10,15])).sum(-1);req=np.asarray(p.Q_req_by_instruction_snr_bucket)[gids[:,:,None],buckets]
 return dict(quality=quality,load=load,req=req,external=frozen.external_hashes(env))

def check(z,tab):return auditor().physical_audit(z,tab,cfg())
