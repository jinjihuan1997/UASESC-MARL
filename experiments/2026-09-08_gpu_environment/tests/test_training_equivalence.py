"""Use identical rollouts/minibatches to compare against original HARL updates."""
from pathlib import Path
import copy
import os
import sys
import unittest
from unittest.mock import patch
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import ROOT, configuration, write
from tensor_train import TensorTrainer
from harl.runners.on_policy_ha_runner import OnPolicyHARunner
from harl.runners.on_policy_ma_runner import OnPolicyMARunner
from harl.common.buffers.on_policy_actor_buffer import OnPolicyActorBuffer
from harl.common.buffers.on_policy_critic_buffer_ep import OnPolicyCriticBufferEP
from harl.common.valuenorm import ValueNorm


class UpdateEquivalence(unittest.TestCase):
    def test_happo_and_mappo_full_update(self):
        device=os.environ.get('GPU_ENV_TEST_DEVICE','cpu')
        results=[]
        for method in ('IC_HAPPO','IC_MAPPO'):
            config=configuration(method)
            config['algo_args']['train'].update(episode_length=8,n_rollout_threads=2)
            trainer=TensorTrainer(config,device)
            trainer.collect();b=trainer.buffer
            cls=OnPolicyHARunner if method=='IC_HAPPO' else OnPolicyMARunner
            ref=cls.__new__(cls);ref.algo_args=copy.deepcopy(config['algo_args'])
            ref.state_type='EP';ref.num_agents=trainer.env.n_agents
            ref.fixed_order=False;ref.share_param=False;ref.action_aggregation='prod'
            ref.actor=copy.deepcopy(trainer.actors);ref.critic=copy.deepcopy(trainer.critic)
            ref.value_normalizer=ValueNorm(1,device=torch.device(device))
            ref.value_normalizer.load_state_dict(trainer.normalizer.state_dict())
            args={**ref.algo_args['train'],**ref.algo_args['model'],**ref.algo_args['algo']}
            ref.actor_buffer=[]
            for i in range(ref.num_agents):
                buf=OnPolicyActorBuffer(args,trainer.env.observation_space[i],trainer.env.action_space[i])
                for name,data in [('obs',b.obs[:,:,i]),('actions',b.actions[i]),('action_log_probs',b.log_probs[i]),
                                  ('masks',b.masks),('available_actions',b.available[:,:,i])]:
                    getattr(buf,name)[:]=data.cpu().numpy()
                ref.actor_buffer.append(buf)
            ref.critic_buffer=OnPolicyCriticBufferEP(args,trainer.env.share_observation_space[0])
            for name,data in [('share_obs',b.state),('value_preds',b.values),('rewards',b.rewards),('masks',b.masks)]:
                getattr(ref.critic_buffer,name)[:]=data.cpu().numpy()
            ref.critic_buffer.compute_returns(b.values[-1].cpu().numpy(),ref.value_normalizer)
            returns=b.returns(trainer.normalizer,args['gamma'],args['gae_lambda'])
            torch.testing.assert_close(returns.cpu(),torch.from_numpy(ref.critic_buffer.returns[:-1]),rtol=1e-6,atol=1e-6)
            gen=torch.Generator().manual_seed(889)
            permutations=[torch.randperm(trainer.batch,generator=gen) for _ in range(ref.num_agents*args['ppo_epoch']+args['critic_epoch'])]
            order=list(reversed(range(ref.num_agents))) if method=='IC_HAPPO' else list(range(ref.num_agents))
            draws=([torch.tensor(order)] if method=='IC_HAPPO' else [])+[p.clone() for p in permutations]
            with patch.object(torch,'randperm',side_effect=draws) as draw:
                ref.train()
                self.assertEqual(draw.call_count,len(draws))
            trainer.update(permutations=permutations,agent_order=order)
            largest=0.
            for left,right in [(x.actor,y.actor) for x,y in zip(trainer.actors,ref.actor)]+[(trainer.critic.critic,ref.critic.critic),(trainer.normalizer,ref.value_normalizer)]:
                for name,value in left.state_dict().items():
                    expected=right.state_dict()[name]
                    largest=max(largest,float((value-expected).abs().max()))
                    torch.testing.assert_close(value,expected,rtol=2e-5,atol=2e-6,msg=lambda msg:f'{method}/{name}: {msg}')
            results.append(dict(method=method,device=device,max_parameter_absolute_difference=largest,passed=True))
        path=ROOT/'results'/f'update_equivalence_{device.replace(":","_")}.json'
        write(path,dict(passed=True,results=results))


if __name__=='__main__': unittest.main()
