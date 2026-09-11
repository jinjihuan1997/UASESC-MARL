"""GPU-resident rollouts with the reference HARL actor/critic update kernels.

Supports the frozen feed-forward EP HAPPO/MAPPO comparison only. Scientific
parameters stay fixed; no old weights or live formal jobs are modified.
"""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import os
import random
import shutil
import time
import numpy as np
import torch
from common import ROOT, configuration, verify_reference, write, stamp, network_hash
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
        self.obs=zeros(length+1,env.count,env.n_agents,env.p.obs_dim_common)
        self.state=zeros(length+1,env.count,env.p.share_obs_dim)
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

    @torch.no_grad()
    def collect(self):
        b=self.buffer
        for a in self.actors: a.prep_rollout()
        self.critic.prep_rollout()
        for t in range(b.length):
            actions=[]
            for i,a in enumerate(self.actors):
                action,logp,_=a.get_actions(b.obs[t,:,i],b.rnn,b.masks[t],b.available[t,:,i])
                b.actions[i][t].copy_(action);b.log_probs[i][t].copy_(logp);actions.append(action)
            value,_=self.critic.get_values(b.state[t],b.rnn,b.masks[t]);b.values[t].copy_(value)
            obs,state,reward,done,info,available=self.env.step(actions)
            b.store_observation(t+1,obs,state,available)
            b.rewards[t].copy_(reward[:,0]);b.masks[t+1].copy_((~done[:,0,None]).float())
        value,_=self.critic.get_values(b.state[-1],b.rnn,b.masks[-1]);b.values[-1].copy_(value)

    def update(self, permutations=None, agent_order=None):
        """Optional fixed permutations enable comparison against the CPU runner."""
        b=self.buffer;args=self.algo['algo']
        returns=b.returns(self.normalizer,args['gamma'],args['gae_lambda'])
        advantages=returns-self.normalizer.denormalize(b.values[:-1])
        normalized=(advantages-advantages.mean())/(advantages.std(unbiased=False)+1e-5)
        factor=torch.ones_like(advantages).flatten(0,1)
        if agent_order is None:
            agent_order=(list(torch.randperm(len(self.actors)).tolist())
                if self.algorithm=='happo' and not args['fixed_order'] else list(range(len(self.actors))))
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
        return dict(actor=torch.stack(actor_metrics).mean(0),critic=torch.stack(critic_metrics).mean(0),returns=returns)

    def synchronize(self):
        if self.device.type=='cuda': torch.cuda.synchronize(self.device)

    def train(self, output):
        output=Path(output);output.mkdir(parents=True,exist_ok=False)
        started=time.perf_counter();timings=[]
        write(output/'config.json',self.config)
        hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                for p in ROOT.glob('*.py')}
        source=output/'source';source.mkdir()
        for name in hashes: shutil.copy2(ROOT/name,source/name)
        shutil.copy2(ROOT/'reference_manifest.json',source/'reference_manifest.json')
        shutil.copytree(ROOT/'reference',source/'reference',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        shutil.copytree(ROOT/'tests',source/'tests',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        write(output/'manifest.json',dict(created_utc=stamp(),purpose='GPU migration validation, not formal paper results',
            device=str(self.device),env_device=str(self.env.device),environment_dtype=str(self.env.dtype),
            reference=verify_reference(),sources=hashes,seed=self.algo['seed']['seed'],batch=self.batch,
            active_agent_ids=self.env.active_agent_ids,weights_only_checkpoints=True,
            initial_actor_hashes=self.initial_actors,initial_critic_hash=self.initial_critic,
            random_source='Per-episode CPU exogenous tape, same reference streams; current slot only exposed',
            action_sampling='Torch RNG on selected model device; CPU and CUDA samples are not assumed bit-identical'))
        try:
            for update in range(1,self.total_updates+1):
                if self.algo['train']['use_linear_lr_decay']:
                    for actor in self.actors: actor.lr_decay(update,self.total_updates)
                    self.critic.lr_decay(update,self.total_updates)
                self.synchronize();t0=time.perf_counter();self.collect();self.synchronize();t1=time.perf_counter()
                metrics=self.update();self.synchronize();t2=time.perf_counter()
                finite=torch.stack([torch.isfinite(x).all() for x in metrics.values()]).all()
                if not bool(finite): raise FloatingPointError('Non-finite rollout returns or training metrics')
                timings.append(dict(update=update,steps=update*self.batch,collect_seconds=t1-t0,update_seconds=t2-t1,total_seconds=t2-t0))
                with (output/'timing.jsonl').open('a') as f: f.write(json.dumps(timings[-1])+'\n')
                with (output/'training_metrics.jsonl').open('a') as f:
                    f.write(json.dumps(dict(update=update,steps=update*self.batch,
                        mean_training_reward=float(self.buffer.rewards.mean()),
                        mean_return=float(metrics['returns'].mean()),
                        actor_loss_entropy_grad_ratio=metrics['actor'].detach().cpu().tolist(),
                        critic_loss_grad=metrics['critic'].detach().cpu().tolist()))+'\n')
                self.buffer.after_update()
                write(output/'status.json',dict(state='training',updated_utc=stamp(),pid=os.getpid(),completed_steps=update*self.batch,
                    target_steps=self.total_updates*self.batch,device=str(self.device),last_timing=timings[-1]))
            for i,a in enumerate(self.actors): torch.save(a.actor.state_dict(),output/f'actor_agent{i}.pt')
            torch.save(self.critic.critic.state_dict(),output/'critic_agent.pt')
            torch.save(self.normalizer.state_dict(),output/'value_normalizer.pt')
            final=[network_hash(a.actor) for a in self.actors]
            assert all(a!=b for a,b in zip(final,self.initial_actors)), 'Actor parameters did not update'
            assert network_hash(self.critic.critic)!=self.initial_critic,'Critic parameters did not update'
            for path in output.glob('*.pt'):
                state=torch.load(path,map_location='cpu',weights_only=True)
                assert all(torch.isfinite(v).all() for v in state.values())
            write(output/'status.json',dict(state='complete',updated_utc=stamp(),pid=os.getpid(),completed_steps=self.total_updates*self.batch,
                wall_seconds=time.perf_counter()-started,device=str(self.device),environment_device=str(self.env.device),
                steady_steps_per_second=(len(timings)-1)*self.batch/sum(t['total_seconds'] for t in timings[1:]),
                final_actor_hashes=final,final_critic_hash=network_hash(self.critic.critic),
                checkpoint_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in output.glob('*.pt')}))
        except BaseException as exc:
            write(output/'status.json',dict(state='failed',error=repr(exc),updated_utc=stamp(),pid=os.getpid()))
            raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method',default='IC_HAPPO')
    parser.add_argument('--device',default='cuda:0',choices=['cpu','cuda:0'])
    parser.add_argument('--seed',type=int,default=1)
    parser.add_argument('--steps',type=int,default=16000)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();verify_reference()
    if args.device.startswith('cuda') and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
    trainer=TensorTrainer(configuration(args.method,args.seed,args.steps),args.device)
    trainer.train(args.output)
