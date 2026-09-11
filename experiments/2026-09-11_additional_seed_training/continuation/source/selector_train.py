"""Versioned 10M -> 2M transfer and a learned binary resource selector.

Only selector log likelihood enters its PPO loss. Frozen controller outputs
are deterministic candidates, not stochastic actions of the new policy.
"""
import copy
import torch
import numpy as np
from gym.spaces import Discrete
from tensor_train import TensorTrainer
from harl.algorithms.actors.mappo import MAPPO
from common import network_hash
from training_checkpoint import capture,restore,digest,runtime_signature


def build_selector(config,env,device):
    args={**config['algo_args']['model'],**config['algo_args']['algo']}
    args['hidden_sizes']=config['continuation']['selector_hidden_sizes']
    return MAPPO(args,env.observation_space[0],Discrete(2),torch.device(device))


class ContinuationTrainer(TensorTrainer):
    def __init__(self,config,device='cpu'):
        super().__init__(config,device)
        self.arm=config['continuation']['arm']
        self.selector=None
        if self.arm=='selector':
            self.selector=build_selector(config,self.env,self.device)
            self.initial_selector=network_hash(self.selector.actor)
            for actor in self.actors:
                actor.actor.requires_grad_(False)
            b=self.buffer
            b.selector_actions=torch.zeros((b.length,b.count,1),device=self.device)
            b.selector_log_probs=torch.zeros_like(b.selector_actions)
        self.stage='selector_only' if self.selector else 'joint_continuation'
        self.trainable_agent_ids=['selector'] if self.selector else list(range(4))

    def initialize_transfer(self,identity):
        spec=self.config['continuation'];path=spec['parent_checkpoint']
        assert digest(path)==spec['parent_sha256']
        state=torch.load(path,map_location='cpu',weights_only=True)
        assert state['update']==2500 and state['total_updates']==2500
        assert state['runtime']==runtime_signature()
        # Explicit versioned migration. Do not loosen exact-resume validation.
        source_cfg=state['config']
        old_env=copy.deepcopy(source_cfg['env_args']);new_env=copy.deepcopy(self.config['env_args'])
        for field in ('semantic_profile_path','semantic_registry_path'):
            assert digest(old_env[field])==digest(new_env[field])
            old_env.pop(field);new_env.pop(field)
        assert old_env==new_env,'Physical environment changed during transfer'
        assert source_cfg['algo_args']['seed']==self.config['algo_args']['seed']
        assert source_cfg['algo_args']['algo']==self.config['algo_args']['algo']
        assert source_cfg['algo_args']['train']['n_rollout_threads']==self.env.count
        assert source_cfg['algo_args']['train']['episode_length']==self.buffer.length
        old_model=copy.deepcopy(source_cfg['algo_args']['model']);new_model=copy.deepcopy(self.config['algo_args']['model'])
        for field in ('lr','critic_lr'):old_model.pop(field);new_model.pop(field)
        assert old_model==new_model
        selector_buffers={k:getattr(self.buffer,k).clone() for k in ('selector_actions','selector_log_probs')} if self.selector else {}
        state['buffer'].update(selector_buffers)
        state.update(config=self.config,identity=identity,device=str(self.device),total_updates=self.total_updates,update=0)
        if self.device.type=='cuda':state['rng']['cuda']=torch.cuda.get_rng_state(self.device)
        restore(self,state,identity)
        self.initial_actors=[network_hash(a.actor) for a in self.actors]
        self.initial_critic=network_hash(self.critic.critic)
        assert self.initial_actors==[network_hash_from_state(x) for x in state['actors']]
        self.stage='selector_only' if self.selector else 'joint_continuation'
        self.trainable_agent_ids=['selector'] if self.selector else list(range(4))

    def set_training_update(self,update):
        # TensorTrainer invokes this once before the selector exists.
        super().set_training_update(update)
        if not hasattr(self,'arm'):return
        self.stage='selector_only' if self.selector else 'joint_continuation'
        self.trainable_agent_ids=['selector'] if self.selector else list(range(4))
        spec=self.config['continuation'];fraction=(update-1)/max(1,self.total_updates-1)
        interpolate=lambda a,b:a+(b-a)*fraction
        actor_lr=interpolate(spec['actor_lr_start'],spec['actor_lr_end'])
        critic_lr=interpolate(spec['critic_lr_start'],spec['critic_lr_end'])
        entropy=interpolate(spec['entropy_start'],spec['entropy_end'])
        active=[self.selector] if self.selector else self.actors
        for actor in active:
            for group in actor.actor_optimizer.param_groups:group['lr']=actor_lr
            actor.entropy_coef=entropy
        for group in self.critic.critic_optimizer.param_groups:group['lr']=critic_lr

    @torch.no_grad()
    def collect(self):
        if not self.selector:return super().collect()
        b=self.buffer
        for actor in self.actors:actor.prep_rollout()
        self.selector.prep_rollout();self.critic.prep_rollout()
        self.reward_component_sums={}
        self.instruction_counts=torch.zeros(3,device=self.device)
        self.instruction_rewards=torch.zeros(3,device=self.device)
        self.selector_counts=torch.zeros((3,2),device=self.device)
        available=torch.ones((b.count,2),device=self.device)
        for t in range(b.length):
            value,_=self.critic.get_values(b.state[t],b.rnn,b.masks[t]);b.values[t].copy_(value)
            choice,logp,_=self.selector.get_actions(b.obs[t,:,0],b.rnn,b.masks[t],available)
            b.selector_actions[t].copy_(choice);b.selector_log_probs[t].copy_(logp)
            proposal=self.actors[0].act(b.obs[t,:,0],b.rnn,b.masks[t],b.available[t,:,0],deterministic=True)[0]
            action=torch.where(choice.bool(),torch.full_like(proposal,2/self.env.U-1),proposal)
            b.actions[0][t].copy_(action);b.log_probs[0][t].zero_()
            post,_,post_masks=self.env.allocate_resources(action)
            b.obs[t,:,1:].copy_(post[:,1:]);b.available[t,:,1:].copy_(post_masks[:,1:])
            modes=[]
            for i in range(1,len(self.actors)):
                action=self.actors[i].act(b.obs[t,:,i],b.rnn,b.masks[t],b.available[t,:,i],deterministic=True)[0]
                b.actions[i][t].copy_(action);b.log_probs[i][t].zero_();modes.append(action)
            obs,state,reward,done,info,masks=self.env.commit_modes(modes)
            self.instruction_counts.scatter_add_(0,info['gid'],torch.ones(b.count,device=self.device))
            self.instruction_rewards.scatter_add_(0,info['gid'],reward[:,0,0])
            self.selector_counts.view(-1).scatter_add_(0,2*info['gid']+choice.squeeze(-1),torch.ones(b.count,device=self.device))
            for key in ('quality_credit','age_mean_cost','age_max_cost','age_tail_cost','resource_cost','service_violation_cost','reception_credit','objective_reward'):
                self.reward_component_sums[key]=self.reward_component_sums.get(key,0)+info[key].mean()
            b.store_observation(t+1,obs,state,masks);b.rewards[t].copy_(reward[:,0]);b.masks[t+1].copy_((~done[:,0,None]).float())
        value,_=self.critic.get_values(b.state[-1],b.rnn,b.masks[-1]);b.values[-1].copy_(value)

    def update(self,permutations=None,agent_order=None):
        if not self.selector:return super().update(permutations,agent_order)
        assert permutations is None and agent_order is None
        b=self.buffer;args=self.algo['algo'];self.selector.prep_training()
        returns=b.returns(self.normalizer,args['gamma'],args['gae_lambda'])
        adv=returns-self.normalizer.denormalize(b.values[:-1]);adv=(adv-adv.mean())/(adv.std(unbiased=False)+1e-5)
        obs=b.obs[:-1,:,0].flatten(0,1);actions=b.selector_actions.flatten(0,1)
        mask=b.masks[:-1].flatten(0,1);active=torch.ones_like(mask);available=torch.ones((self.batch,2),device=self.device)
        actor_metrics=[]
        for epoch in range(args['ppo_epoch']):
            for indices in torch.randperm(self.batch,device=self.device).chunk(args['actor_num_mini_batch']):
                sample=(obs[indices],b.flat_rnn[indices],actions[indices],mask[indices],active[indices],
                        b.selector_log_probs.flatten(0,1)[indices],adv.flatten(0,1)[indices],available[indices])
                values=self.selector.update(sample)
                actor_metrics.append(torch.stack([torch.as_tensor(x,device=self.device).detach().mean() for x in values]))
        self.critic.prep_training();critic_metrics=[]
        for epoch in range(args['critic_epoch']):
            for indices in torch.randperm(self.batch,device=self.device).chunk(args['critic_num_mini_batch']):
                sample=(b.state[:-1].flatten(0,1)[indices],b.flat_rnn[indices],b.values[:-1].flatten(0,1)[indices],
                        returns.flatten(0,1)[indices],mask[indices])
                loss,grad,_=self.critic.update(sample,self.normalizer)
                critic_metrics.append(torch.stack([loss.detach(),grad.detach()]))
        assert [network_hash(a.actor) for a in self.actors]==self.initial_actors,'Frozen controllers changed'
        return dict(actor=torch.stack(actor_metrics).mean(0),critic=torch.stack(critic_metrics).mean(0),returns=returns)


def network_hash_from_state(state):
    import hashlib
    h=hashlib.sha256()
    for key,value in sorted(state.items()):
        h.update(key.encode());h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def capture_fork(trainer,update,identity):
    state=capture(trainer,update,identity)
    if trainer.selector:
        from training_checkpoint import cpu_tree
        state['selector']=cpu_tree(trainer.selector.actor.state_dict())
        state['selector_optimizer']=cpu_tree(trainer.selector.actor_optimizer.state_dict())
        state['initial_selector']=trainer.initial_selector
    return state


def restore_fork(trainer,state,identity):
    update=restore(trainer,state,identity)
    if trainer.selector:
        trainer.selector.actor.load_state_dict(state['selector'])
        trainer.selector.actor_optimizer.load_state_dict(state['selector_optimizer'])
        trainer.initial_selector=state['initial_selector']
    trainer.set_training_update(update+1)
    return update
