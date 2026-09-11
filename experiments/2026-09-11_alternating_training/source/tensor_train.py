"""GPU-resident rollouts with the reference HARL actor/critic update kernels.

Supports the frozen feed-forward EP HAPPO/MAPPO comparison only. Versioned
parameters are fixed within each experiment; no old weights or live formal jobs are modified.
"""
import copy
import random
import numpy as np
import torch
from common import network_hash
from tensor_env import TensorSCEnv
from harl.algorithms.actors.happo import HAPPO
from harl.algorithms.actors.mappo import MAPPO
from harl.algorithms.critics.v_critic import VCritic
from harl.common.valuenorm import ValueNorm
from harl.utils.ratio_tools import aggregate_action_ratio


class TensorValueNorm(ValueNorm):
    def denormalize(self, x):
        mean,var=self.running_mean_var()
        return x*var.sqrt()+mean


class TensorRollout:
    def __init__(self, env, length, args):
        self.length,self.count,self.agents=length,env.count,env.n_agents
        self.device=env.device
        def zeros(*shape): return torch.zeros(shape,device=self.device)
        self.obs=zeros(length+1,env.count,env.n_agents,env.obs_dim_common)
        self.state=zeros(length+1,env.count,env.share_obs_dim)
        self.available=zeros(length+1,env.count,env.n_agents,env.p.common_act_dim)
        self.actions=[zeros(length,env.count,s.shape[0]) for s in env.action_space]
        self.log_probs=[torch.zeros_like(a) for a in self.actions]
        self.values=zeros(length+1,env.count,1)
        self.rewards=zeros(length,env.count,1)
        self.masks=torch.ones_like(self.values)
        self.rnn=zeros(env.count,args['recurrent_n'],args['hidden_sizes'][-1])
        self.flat_rnn=zeros(length*env.count,args['recurrent_n'],args['hidden_sizes'][-1])

    def store_observation(self, step, obs, state, available):
        self.obs[step].copy_(obs);self.state[step].copy_(state[:,0]);self.available[step].copy_(available)

    def after_update(self):
        for tensor in (self.obs,self.state,self.available,self.values,self.masks): tensor[0].copy_(tensor[-1])

    @torch.no_grad()
    def returns(self, normalizer, gamma, gae_lambda):
        values=normalizer.denormalize(self.values)
        result=torch.zeros_like(values[:-1]);gae=torch.zeros_like(values[0])
        for t in reversed(range(self.length)):
            delta=self.rewards[t]+gamma*values[t+1]*self.masks[t+1]-values[t]
            gae=delta+gamma*gae_lambda*self.masks[t+1]*gae
            result[t]=gae+values[t]
        return result

    def actor_inputs(self, agent):
        return (self.obs[:-1,:,agent].flatten(0,1),self.flat_rnn,self.actions[agent].flatten(0,1),
                self.masks[:-1].flatten(0,1),self.available[:-1,:,agent].flatten(0,1),
                torch.ones_like(self.masks[:-1].flatten(0,1)))


class TensorTrainer:
    def __init__(self, config, device='cuda:0'):
        self.config=copy.deepcopy(config)
        config=self.config
        config['algo_args']['device'].update(cuda=device!='cpu',cuda_deterministic=device!='cpu')
        algo=config['algo_args'];self.algo=algo
        if (algo['model']['use_recurrent_policy'] or algo['model']['use_naive_recurrent_policy']
            or algo['algo']['share_param'] or algo['algo']['use_instruction_adv_norm']
            or config['env_args'].get('state_type','EP')!='EP'
            or not algo['train']['use_valuenorm'] or not algo['algo']['use_gae']
            or algo['train']['model_dir'] is not None):
            raise ValueError('Only frozen feed-forward EP configuration without prior weights is supported')
        if algo['train']['use_proper_time_limits'] or config['training_design']['termination']!='finite_600_slot_task':
            raise ValueError('This version uses true finite-task termination, not time-limit truncation')
        self.device=torch.device(device)
        seed=algo['seed']['seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
        if self.device.type=='cuda':
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        torch.set_num_threads(1)
        self.env=TensorSCEnv(config['env_args'],count=algo['train']['n_rollout_threads'],seed=seed,device=device)
        self.algorithm=config['main_args']['algo']
        if self.algorithm not in ('happo','mappo'): raise ValueError(self.algorithm)
        factory=HAPPO if self.algorithm=='happo' else MAPPO
        params={**algo['model'],**algo['algo']}
        self.actors=[factory(params,o,a,self.device) for o,a in zip(self.env.observation_space,self.env.action_space)]
        self.critic=VCritic(params,self.env.share_observation_space[0],self.device)
        self.normalizer=TensorValueNorm(1,device=self.device)
        self.buffer=TensorRollout(self.env,algo['train']['episode_length'],algo['model'])
        self.batch=self.buffer.length*self.env.count
        self.total_updates=algo['train']['num_env_steps']//self.batch
        self.initial_actors=[network_hash(a.actor) for a in self.actors]
        self.initial_critic=network_hash(self.critic.critic)
        self.buffer.store_observation(0,*self.env.reset())
        self.set_training_update(1)

    def set_training_update(self, update):
        design=self.config['training_design']
        self.training_update=update
        if design['arm']=='joint':
            self.stage='joint'
            self.fixed_resources=False
            self.trainable_agent_ids=list(range(len(self.actors)))
        elif design['arm']=='alternating':
            steps=min((update-1)*self.batch, self.total_updates*self.batch-1)
            phase=next(x for x in design['phases'] if steps < x['end_step'])
            self.stage=phase['name']
            self.fixed_resources=phase['resource_source']=='warmup_mixture'
            self.trainable_agent_ids=list(phase['trainable_actor_ids'])
        else:
            raise ValueError('This experiment supports joint and alternating only')
        for i,actor in enumerate(self.actors):
            actor.actor.requires_grad_(i in self.trainable_agent_ids)

    @torch.no_grad()
    def sample_warmup_resources(self):
        # Dirichlet(1,1,1) plus equal split. The environment applies the existing
        # 0.05 lower-bound transform exactly once. Uses checkpointed Torch RNG.
        random_share=torch.distributions.Dirichlet(
            torch.ones((self.env.count,self.env.U),device=self.device)).sample()
        equal=torch.rand((self.env.count,1),device=self.device)<0.5
        share=torch.where(equal,torch.full_like(random_share,1/self.env.U),random_share)
        return 2*share-1,equal.squeeze(-1)

    @torch.no_grad()
    def collect(self):
        b=self.buffer
        for a in self.actors: a.prep_rollout()
        self.critic.prep_rollout()
        self.reward_component_sums={}
        self.instruction_counts=torch.zeros(3,device=self.device)
        self.instruction_rewards=torch.zeros(3,device=self.device)
        self.warmup_equal_draws=torch.zeros((),device=self.device)
        self.mode_budget_counts=torch.zeros(3*self.env.U*6*17,device=self.device)
        self.resource_share_sum=torch.zeros(self.env.U,device=self.device)
        self.unused_budget_sum=torch.zeros(self.env.U,device=self.device)
        self.delivery_sum=torch.zeros(self.env.U,device=self.device)
        edges=torch.tensor([.15,.25,.35,.50,.70],device=self.device)
        for t in range(b.length):
            value,_=self.critic.get_values(b.state[t],b.rnn,b.masks[t]);b.values[t].copy_(value)
            if self.fixed_resources:
                action,equal_draw=self.sample_warmup_resources()
                self.warmup_equal_draws+=equal_draw.sum()
                logp=torch.zeros_like(action)
            else:
                action,logp,_=self.actors[0].get_actions(b.obs[t,:,0],b.rnn,b.masks[t],b.available[t,:,0])
            b.actions[0][t].copy_(action);b.log_probs[0][t].copy_(logp)
            post,_,post_masks=self.env.allocate_resources(action)
            shares=self.env.beta.clone()
            # Keep pre-allocation critic/SUT inputs; replay exactly the UAV inputs
            # conditioned on this sampled SUT action, without resampling resources.
            b.obs[t,:,1:].copy_(post[:,1:]);b.available[t,:,1:].copy_(post_masks[:,1:])
            modes=[]
            for i in range(1,len(self.actors)):
                action,logp,_=self.actors[i].get_actions(b.obs[t,:,i],b.rnn,b.masks[t],b.available[t,:,i])
                b.actions[i][t].copy_(action);b.log_probs[i][t].copy_(logp);modes.append(action)
            obs,state,reward,done,info,available=self.env.commit_modes(modes)
            budget_bin=torch.bucketize(shares,edges)
            executed=(info['mode']+1).long()
            uav_id=torch.arange(self.env.U,device=self.device)[None,:]
            index=(((info['gid'][:,None]*self.env.U+uav_id)*6+budget_bin)*17+executed).reshape(-1)
            self.mode_budget_counts.scatter_add_(0,index,torch.ones_like(index,dtype=torch.float32))
            self.resource_share_sum+=shares.sum(0)
            self.unused_budget_sum+=(info['budget']-info['usage']).sum(0)
            self.delivery_sum+=info['served'].sum(-1).sum(0)
            self.instruction_counts.scatter_add_(0,info['gid'],torch.ones_like(reward[:,0,0]))
            self.instruction_rewards.scatter_add_(0,info['gid'],reward[:,0,0])
            for key in ('quality_credit','age_mean_cost','age_max_cost','age_tail_cost','resource_cost','service_violation_cost','reception_credit','objective_reward'):
                value=info[key].mean()
                self.reward_component_sums[key]=self.reward_component_sums.get(key,0)+value
            b.store_observation(t+1,obs,state,available)
            b.rewards[t].copy_(reward[:,0]);b.masks[t+1].copy_((~done[:,0,None]).float())
        value,_=self.critic.get_values(b.state[-1],b.rnn,b.masks[-1]);b.values[-1].copy_(value)

    def update(self, permutations=None, agent_order=None):
        """Optional fixed permutations enable comparison against the CPU runner."""
        frozen={i:network_hash(actor.actor) for i,actor in enumerate(self.actors)
                if i not in self.trainable_agent_ids}
        b=self.buffer;args=self.algo['algo']
        returns=b.returns(self.normalizer,args['gamma'],args['gae_lambda'])
        advantages=returns-self.normalizer.denormalize(b.values[:-1])
        normalized=(advantages-advantages.mean())/(advantages.std(unbiased=False)+1e-5)
        factor=torch.ones_like(advantages).flatten(0,1)
        if agent_order is None:
            ids=self.trainable_agent_ids
            agent_order=([ids[j] for j in torch.randperm(len(ids)).tolist()]
                if self.algorithm=='happo' and not args['fixed_order'] else ids)
        if sorted(agent_order)!=self.trainable_agent_ids: raise ValueError('Actor update order must exclude frozen SUT')
        # Ordering is a few CPU integers per update, not rollout data movement.
        permutation_iter=iter(permutations) if permutations is not None else None
        def batches(count):
            order=next(permutation_iter) if permutation_iter is not None else torch.randperm(self.batch,device=self.device)
            order=torch.as_tensor(order,device=self.device)
            return order.chunk(count)
        actor_metrics=[]
        for i in agent_order:
            actor=self.actors[i];actor.prep_training()
            obs,rnn,action,mask,available,active=b.actor_inputs(i)
            with torch.no_grad(): old=actor.evaluate_actions(obs,rnn,action,mask,available,active)[0]
            for epoch in range(args['ppo_epoch']):
                for indices in batches(args['actor_num_mini_batch']):
                    sample=(obs[indices],rnn[indices],action[indices],mask[indices],active[indices],
                        b.log_probs[i].flatten(0,1)[indices],normalized.flatten(0,1)[indices],available[indices])
                    if self.algorithm=='happo': sample=(*sample,factor[indices])
                    metrics=actor.update(sample)
                    actor_metrics.append(torch.stack([torch.as_tensor(x,device=self.device).detach().mean() for x in metrics]))
            if self.algorithm=='happo':
                with torch.no_grad():
                    new=actor.evaluate_actions(obs,rnn,action,mask,available,active)[0]
                    ratio=aggregate_action_ratio(new-old,args['action_aggregation'],clip=20.)
                    ratio=torch.nan_to_num(ratio,nan=1.,posinf=1e6,neginf=0.).clamp(0,1e3)
                    factor=torch.nan_to_num(factor*ratio,nan=1.,posinf=1e6,neginf=0.).clamp(0,1e3)
        self.critic.prep_training();critic_metrics=[]
        for epoch in range(args['critic_epoch']):
            for indices in batches(args['critic_num_mini_batch']):
                sample=(b.state[:-1].flatten(0,1)[indices],b.flat_rnn[indices],b.values[:-1].flatten(0,1)[indices],
                    returns.flatten(0,1)[indices],b.masks[:-1].flatten(0,1)[indices])
                loss,grad,_=self.critic.update(sample,self.normalizer)
                critic_metrics.append(torch.stack([loss.detach(),grad.detach()]))
        for i,before in frozen.items():
            assert network_hash(self.actors[i].actor)==before, f'Frozen actor {i} changed'
        return dict(actor=torch.stack(actor_metrics).mean(0),critic=torch.stack(critic_metrics).mean(0),returns=returns)

    def synchronize(self):
        if self.device.type=='cuda': torch.cuda.synchronize(self.device)
