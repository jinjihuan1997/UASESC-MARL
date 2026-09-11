"""Batched Torch implementation of the frozen final-average-table SC environment.

State transitions and observations stay on device. Episode setup and exogenous
RNG preparation run on CPU once per episode. Float64 preserves physical-model
thresholds; network observations retain the reference's float32 interface.
"""
import numpy as np
import torch
import torch.nn.functional as F
from episode_source import EpisodeSource


class TensorSCEnv:
    def __init__(self, args, count=10, seed=1, device='cuda:0'):
        self.device = torch.device(device)
        self.dtype = torch.float64
        self.count = count
        self.source = EpisodeSource(args, count, seed)
        self.p = self.source.envs[0]
        p = self.p
        if not p.semantic_library.profile or p.semantic_mode_selection != 'all_modes':
            raise ValueError('GPU environment requires the frozen average table and all_modes')
        self.U, self.K, self.D, self.M = p.n_uav, p.ds_per_uav, p.n_ds, p.n_semantic_modes
        self.active_agent_ids = p.active_agent_ids
        self.n_agents = p.n_agents
        self.observation_space, self.action_space = p.observation_space, p.action_space
        self.share_observation_space = p.share_observation_space
        self.local_ds = torch.arange(self.D, device=self.device).reshape(self.U,self.K)
        self.owner = torch.arange(self.U, device=self.device)[:,None]
        self.owner_mask = F.one_hot(torch.arange(self.D,device=self.device)//self.K,self.U).T.bool()
        self.eye_u = torch.eye(self.U, dtype=self.dtype, device=self.device)
        self.sut_pos = self.tensor(p.pos_sut)
        self.bucket_edges = self.tensor([5., 10., 15.])
        profile = p.semantic_library.profile
        self.grid = self.tensor(profile['snr_grid_db'])
        self.profiles = self.tensor(np.stack([profile['q_hat_mean'],profile['l_z_mean'],profile['n_z_mean']]))
        self.requirements = self.tensor(p.Q_req_by_instruction_snr_bucket)
        self.limits = self.tensor(p.A_limit_by_instruction)
        self.penalties = self.tensor(p.constraint_penalty_A_by_instruction)
        self.step_index = 0

    def tensor(self, value, dtype=None):
        return torch.as_tensor(value, dtype=dtype or self.dtype, device=self.device)

    @torch.no_grad()
    def reset(self):
        episodes = self.source.next()
        self.episode_indices = [e['episode_index'] for e in episodes]
        for name in ('pos_uav','pos_ds','psi'):
            setattr(self,name,self.tensor(np.stack([e[name] for e in episodes])))
        self.q = self.tensor(np.stack([e['q_cache'] for e in episodes]),torch.bool)[:,self.owner,self.local_ds]
        self.tau = self.tensor(np.stack([e['tau_cache'] for e in episodes]),torch.int64)[:,self.owner,self.local_ds]
        self.aoi = self.tensor(np.stack([e['A_rcc'] for e in episodes])).reshape(self.count,self.U,self.K)
        for name in ('noise_us','noise_sat','noise_du','potential_content','instructions'):
            value=np.stack([e[name] for e in episodes],axis=1)
            setattr(self,name,self.tensor(value,torch.int64 if name=='instructions' else None))
        self.step_index=0
        self.rr_cursor=torch.zeros((self.count,self.U),dtype=torch.int64,device=self.device)
        self.beta=torch.full((self.count,self.U),1/self.U,dtype=self.dtype,device=self.device)
        self.path_us = self.atg(self.pos_uav, self.sut_pos, self.p.f_uav_sut)
        self.path_du = self.atg(self.pos_uav[:,:,None,:], self.pos_ds[:,None,:,:], self.p.f_ds_uav)
        self.update_channels(0)
        self.update_tables()
        return self.observe()

    def atg(self, air, ground, frequency):
        p=self.p
        diff=air-ground
        d3=torch.linalg.vector_norm(diff,dim=-1).clamp_min(1.)
        d2=torch.linalg.vector_norm(diff[...,:2],dim=-1)
        angle=torch.atan2(diff[...,2].abs(),d2+1e-9)*(180/np.pi)
        los=1/(1+p.atg_los_a*torch.exp(-p.atg_los_b*(angle-p.atg_los_a)))
        fspl=20*torch.log10(d3)+20*np.log10(frequency)+20*np.log10(4*np.pi/p.c_light)
        return fspl+los*p.atg_eta_los_db+(1-los)*p.atg_eta_nlos_db

    def snr(self, loss, power, bandwidth, nf, gain):
        noise=self.p.N0*bandwidth*10**(nf/10)
        return (power*torch.pow(10.,(gain-loss)/10)/max(noise,1e-30)).clamp_min(1e-12)

    def update_channels(self, slot):
        p=self.p
        self.gamma_us=self.snr(self.path_us+self.noise_us[slot],
            p.p_uav_sut_w,p.B_uav_sut,p.nf_uav_sut_db,p.g_uav_sut_tx_db+p.g_uav_sut_rx_db)
        sat_loss=20*np.log10(max(p.satellite_slant_range_m,1))+20*np.log10(p.f_sut_sat)+20*np.log10(4*np.pi/p.c_light)
        self.gamma_sat=self.snr(sat_loss+self.noise_sat[slot],p.p_sut_sat_w,p.B_sut_sat,p.nf_sut_sat_db,p.g_sut_sat_tx_db+p.g_sut_sat_rx_db)
        self.gamma_du=self.snr(self.path_du+self.noise_du[slot],
            p.p_ds_uav_w,p.B_ds_uav,p.nf_ds_uav_db,p.g_ds_uav_tx_db+p.g_ds_uav_rx_db)
        self.rate_du=p.B_ds_uav*torch.log2(1+self.gamma_du)
        self.gamma_bh=(self.gamma_us*self.gamma_sat[:,None]/(self.gamma_us+self.gamma_sat[:,None]+1)).clamp_min(1e-12)
        self.update_budget()

    def update_budget(self):
        self.bandwidth=torch.minimum(torch.full_like(self.beta,self.p.B_uav_sut),self.beta*self.p.B_sut_sat)
        self.budget=self.p.delta_T*self.bandwidth

    def update_tables(self):
        snr=10*torch.log10(self.gamma_bh.clamp_min(1e-12))
        self.snr_db=snr
        self.buckets=(snr[...,None]>=self.bucket_edges).sum(-1)
        x=snr.clamp(self.grid[0],self.grid[-1])
        if self.grid.numel()==1:
            value=self.profiles[:,:,0][:,None,None,:].expand(-1,self.count,self.U,-1)
        else:
            hi=torch.searchsorted(self.grid,x.contiguous(),right=True).clamp(1,self.grid.numel()-1)
            lo=hi-1
            f=(x-self.grid[lo])/(self.grid[hi]-self.grid[lo])
            # [quantity, mode, env, UAV] -> [quantity, env, UAV, mode]
            a=self.profiles[:,:,lo].permute(0,2,3,1)
            b=self.profiles[:,:,hi].permute(0,2,3,1)
            value=a+(b-a)*f[None,:,:,None]
        self.quality,self.lz,self.nz=value.unbind(0)
        self.load=self.lz+self.p.side_info_bits/torch.log2(1+self.gamma_bh[:,:,None])

    def context(self):
        gid=self.instructions[self.step_index]
        return gid,self.limits[gid],self.requirements[gid[:,None],self.buckets]

    def observe(self):
        p=self.p
        E,U,K,M=self.count,self.U,self.K,self.M
        gid,limit,req=self.context()
        onehot=F.one_hot(gid,p.num_instructions).to(self.dtype)
        bucket_hot=F.one_hot(self.buckets,4).to(self.dtype)
        sut_context=[];uav_context=[];share_context=[]
        if p.use_instruction_constraints:
            if p.include_instruction_id_in_obs:
                sut_context.append(onehot if p.actor_observe_instruction else torch.zeros_like(onehot))
                uav_context.append(onehot[:,None,:].expand(-1,U,-1) if p.actor_observe_instruction else torch.zeros((E,U,p.num_instructions),device=self.device,dtype=self.dtype))
                share_context.append(onehot)
            if p.include_instruction_constraints_in_obs:
                l=limit[:,None]/p.A_limit_context_ref
                sut_context.append(l if p.actor_observe_instruction else torch.zeros_like(l))
                task=torch.stack([req/p.Q_max,(limit/p.A_limit_context_ref)[:,None].expand(-1,U)],-1)
                uav_context.extend([bucket_hot,task if p.actor_observe_instruction else torch.zeros_like(task)])
                share_context.extend([l,req/p.Q_max])
        feasible=self.quality>=req[:,:,None]-1e-9
        minimum=torch.where(feasible,self.load,torch.inf).amin(-1)
        pending=torch.where(feasible.any(-1),minimum,0)*self.q.sum(-1)
        sut=torch.cat([self.gamma_us/100,self.gamma_sat[:,None]/100,pending/p.Lambda_ref,
                       self.aoi.amax(-1)/p.A_max,*sut_context],-1)
        age=torch.where(self.tau>=0,(self.step_index-self.tau).clamp_min(0)/p.A_max,0.)
        quality=self.quality[:,:,None,:].expand(-1,-1,K,-1)
        uav=torch.cat([self.budget[:,:,None]/p.Lambda_ref,self.q.to(self.dtype),age,self.aoi/p.A_max,
            self.load/p.Lambda_ref,quality.reshape(E,U,K*M)/p.Q_max,self.eye_u[None].expand(E,-1,-1),*uav_context],-1)
        obs=torch.cat([F.pad(sut,(0,p.obs_dim_common-sut.shape[-1]))[:,None,:],F.pad(uav,(0,p.obs_dim_common-uav.shape[-1]))],1)
        state=torch.cat([self.gamma_us/100,self.gamma_sat[:,None]/100,self.beta,self.q.reshape(E,-1).to(self.dtype),
            age.reshape(E,-1),self.aoi.reshape(E,-1)/p.A_max,self.load.reshape(E,-1)/p.Lambda_ref,
            quality.reshape(E,-1)/p.Q_max,*share_context],-1).float()
        masks=torch.zeros((E,U+1,p.common_act_dim),dtype=torch.float32,device=self.device)
        masks[:,0,:U]=1
        valid=feasible & self.q.any(-1)[:,:,None]
        valid=valid | ~valid.any(-1,keepdim=True)
        masks[:,1:,:M]=valid
        return obs[:,self.active_agent_ids].float(),state[:,None,:].expand(-1,self.n_agents,-1),masks[:,self.active_agent_ids]

    def fixed_mode(self):
        # Same observable cheapest-feasible rule; ties: higher quality, lower ID.
        _,_,req=self.context()
        valid=(self.quality>=req[:,:,None]-1e-5)&self.q.any(-1)[:,:,None]
        order=torch.argsort(-self.quality,dim=-1,stable=True)
        order=order.gather(-1,torch.argsort(self.load.gather(-1,order),dim=-1,stable=True))
        flags=valid.gather(-1,order)
        mode=order.gather(-1,flags.long().argmax(-1,keepdim=True)).squeeze(-1)
        mode=torch.where(valid.any(-1),mode,0)
        return F.one_hot(mode,self.M).to(self.dtype)

    @torch.no_grad()
    def step(self, actions, auto_reset=True):
        p=self.p;E,U,K=self.count,self.U,self.K
        if len(actions)!=self.n_agents: raise ValueError('One batched action tensor per active agent required')
        sut=torch.zeros_like(self.beta)
        mu=self.fixed_mode() if p.fixed_mode_rule else torch.zeros((E,U,self.M),dtype=self.dtype,device=self.device)
        for value,physical,space in zip(actions,self.active_agent_ids,self.action_space):
            if value.shape!=(E,space.shape[0]) or value.device!=self.device:
                raise ValueError('Action shape/device mismatch')
            if physical==0: sut=value.to(self.dtype)
            else: mu[:,physical-1]=value
        z=sut-sut.amax(-1,keepdim=True)
        exp=z.clamp(-60,60).exp();beta=exp/exp.sum(-1,keepdim=True)
        beta=(p.beta_sat_lower_bound+(1-p.beta_sat_lower_bound*U)*beta).clamp(p.beta_sat_lower_bound,1.)
        self.beta=beta/beta.sum(-1,keepdim=True).clamp_min(1e-12)
        self.update_budget()
        gid,limit,req=self.context()
        valid=(self.quality>=req[:,:,None]-1e-9)&(self.load<=self.budget[:,:,None]+1e-9)&self.q.any(-1)[:,:,None]
        order=torch.argsort(-mu,dim=-1,stable=True)
        mode=order.gather(-1,valid.gather(-1,order).long().argmax(-1,keepdim=True)).squeeze(-1)
        has_mode=valid.any(-1)
        chosen_load=self.load.gather(-1,mode[:,:,None]).squeeze(-1)
        chosen_quality=self.quality.gather(-1,mode[:,:,None]).squeeze(-1)
        scores=self.aoi if p.scheduler=='aoi' else -((torch.arange(K,device=self.device)-self.rr_cursor[:,:,None])%K).to(self.dtype)
        ranks=torch.argsort(torch.where(self.q,scores,-torch.inf),dim=-1,descending=True,stable=True)
        cached=self.q.gather(-1,ranks)
        left=self.budget.clone()
        takes=[]
        # Across all environments/UAVs concurrently; retain sequential budget subtraction.
        for j in range(K):
            take=has_mode & cached[:,:,j] & (chosen_load<=left+1e-9)
            left=left-torch.where(take,chosen_load,0.)
            takes.append(take)
        ordered=torch.stack(takes,-1)
        served=torch.zeros_like(self.q).scatter(-1,ranks,ordered)
        count=served.sum(-1)
        mode=torch.where(count>0,mode,-1)
        if p.scheduler=='round_robin':
            last=torch.where(ordered,torch.arange(K,device=self.device),-1).amax(-1)
            local=ranks.gather(-1,last.clamp_min(0)[:,:,None]).squeeze(-1)
            self.rr_cursor=torch.where(last>=0,(local+1)%K,self.rr_cursor)
        old_aoi=self.aoi
        generated=torch.where(self.tau<0,self.step_index,self.tau)
        next_aoi=torch.where(served,(self.step_index-generated+1).to(self.dtype),old_aoi+1).clamp_max(p.A_max)
        admission=~self.q
        self.tau=torch.where(admission,self.step_index,torch.where(self.q & ~served,self.tau,-1))
        self.q=(self.q & ~served)|admission
        qgain=((chosen_quality-p.Q_min_eval)/max(p.Q_max-p.Q_min_eval,1e-9)).clamp_min(0)*count
        usage=chosen_load*count
        quality_term=qgain.sum(-1)/self.D
        load_term=(usage/p.Lambda_ref).sum(-1)/U
        norm=next_aoi/p.A_max
        aoi_term=p.aoi_mean_weight*norm.mean((-1,-2))+p.aoi_max_weight*norm.amax((-1,-2))+p.aoi_tail_weight*((next_aoi-p.aoi_tail_threshold)/p.A_max).clamp_min(0).mean((-1,-2))
        base=p.omega_Q*quality_term-p.omega_Lambda*load_term-p.omega_A*aoi_term
        violation=((next_aoi.amax((-1,-2))-limit)/limit.clamp_min(1e-9)).clamp_min(0)
        bonus=p.eta_recv_aoi_bonus*(old_aoi-next_aoi).clamp_min(0).mean((-1,-2))/limit.clamp_min(1e-9)
        auxiliary=-self.penalties[gid]*violation+bonus
        reward=base+auxiliary if p.use_task_auxiliary_reward else base
        self.last=dict(mode=mode,served=served,admission=admission,usage=usage,quality=chosen_quality,
            quality_term=quality_term,load_term=load_term,aoi_term=aoi_term,base_reward=base,
            auxiliary=auxiliary,common_reward=base+auxiliary,reward=reward,req=req,gid=gid,
            budget=self.budget,quality_table=self.quality,load_table=self.load)
        self.aoi=next_aoi
        full_admission=torch.zeros((E,U,self.D),dtype=torch.bool,device=self.device)
        full_admission[:,self.owner,self.local_ds]=admission
        self.psi=torch.where(full_admission[:,:,:,None],self.potential_content[self.step_index],self.psi)
        self.step_index+=1
        self.update_channels(self.step_index)
        self.update_tables()
        obs,state,masks=self.observe()
        done=self.step_index>=p.max_steps
        info=self.last.copy()
        if done and auto_reset:
            info.update(terminal_observation=obs,terminal_state=state,terminal_available=masks)
            obs,state,masks=self.reset()
        return obs,state,reward[:,None,None].expand(-1,self.n_agents,-1).float(),torch.full((E,self.n_agents),done,dtype=torch.bool,device=self.device),info,masks
