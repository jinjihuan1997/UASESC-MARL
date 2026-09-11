"""Read-only reward audit over frozen inputs, with parameter-free replay."""
import os
import sys
from pathlib import Path
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
sys.dont_write_bytecode = True
OUT = Path(__file__).resolve().parent
LONG = OUT.parent / '2026-09-10_preference_long_training'
GAP = OUT.parent / '2026-09-10_preference_rule_gap'
sys.path.insert(0, str(LONG))
from helpers import *
from tensor_train import TensorTrainer
from rule_tools import predict_reward


def distribution(x):
    x = np.asarray(x, dtype=float).ravel()
    assert np.isfinite(x).all()
    return dict(mean=float(x.mean()), sd=float(x.std()), minimum=float(x.min()),
                p01=float(np.quantile(x, .01)), median=float(np.median(x)),
                p99=float(np.quantile(x, .99)), maximum=float(x.max()))


def decompose(data, cfg):
    e = cfg['env_args']
    x = data.reshape(-1, data.shape[-1])
    weights = np.asarray(e['reward_weights_by_instruction'])[x[:, 5].astype(int)]
    # Validated delivered qualities are all above Q_min_eval.
    q = (x[:, 2] - e['Q_min_eval'] * x[:, 3]) / (e['Q_max'] - e['Q_min_eval']) / e['n_ds']
    load = x[:, 4] / e['Lambda_ref'] / e['n_uav']
    qc = weights[:, 0] * q
    lc = -weights[:, 2] * load
    ac = x[:, 0] - qc - lc
    return dict(reward=distribution(x[:, 0]), quality_contribution=distribution(qc),
                aoi_contribution=distribution(ac), load_contribution=distribution(lc),
                reward_float32_abs_error=float(np.abs(x[:, 0].astype(np.float32).astype(float)-x[:, 0]).max()),
                normalized_quality=distribution(q), normalized_load=distribution(load),
                normalized_aoi_inferred=distribution(-ac/weights[:, 1]))


def replay(path, schedule, seeds):
    with np.load(path) as z:
        data, modes, fractions = z['trace'], z['modes'], z['resource_fractions']
    env, _, _, _ = make_env(seeds, schedule)
    records = []
    max_error = 0.0
    for slot in range(600):
        shares = (fractions[slot] - env.p.beta_sat_lower_bound) / (1-env.p.beta_sat_lower_bound*env.U)
        mu = np.eye(env.M)[modes[slot].clip(min=0)]
        actions = [env.tensor(2*shares-1)] + [env.tensor(mu[:, i]) for i in range(env.U)]
        _, _, _, info, checked = checked_step(env, actions)
        actual = np.column_stack([checked[k] for k in ('common_reward','mean_aoi','predicted_quality_sum','deliveries','channel_uses')])
        max_error = max(max_error, float(np.abs(actual-data[slot, :, :5]).max()))
        np.testing.assert_allclose(actual, data[slot, :, :5], atol=1e-8, rtol=1e-10)
        np.testing.assert_array_equal(arr(info['mode']), modes[slot])
        np.testing.assert_allclose(arr(env.beta), fractions[slot], atol=1e-12, rtol=0)
        a = arr(env.aoi).reshape(env.count, -1)
        count = arr(info['served']).sum((-1,-2))
        quality = arr(info['quality'])
        assert np.all(quality[arr(info['served']).any(-1)] >= env.p.Q_min_eval)
        g = arr(info['gid']).astype(int)
        w = np.asarray(env.p.reward_weights_by_instruction)[g]
        row = dict(reward=checked['common_reward'], mean_aoi=a.mean(-1), max_aoi=a.max(-1),
                   mean_part=.4*a.mean(-1)/10, max_part=.3*a.max(-1)/10,
                   tail_part=.3*np.maximum(a-10,0).mean(-1)/10,
                   quality_term=arr(info['quality_term']), load_term=arr(info['load_term']),
                   quality_contribution=w[:,0]*arr(info['quality_term']),
                   aoi_contribution=-w[:,1]*arr(info['aoi_term']),
                   load_contribution=-w[:,2]*arr(info['load_term']),
                   count=count, normalized_budget_use=checked['channel_uses']/60000,
                   quality_violation=checked['quality_violations'],
                   no_delivery=(count==0).astype(float))
        for threshold in (3,4,6,10):
            row[f'any_above_{threshold}']=(a>threshold).any(-1).astype(float)
            row[f'fraction_above_{threshold}']=(a>threshold).mean(-1)
            row[f'mean_excess_{threshold}']=np.maximum(a-threshold,0).mean(-1)/threshold
            row[f'max_excess_{threshold}']=np.maximum(a.max(-1)-threshold,0)/threshold
        # Counterfactual scoring only: unchanged trajectories and weights.
        row['reward_load_budget_norm'] = row['reward'] - w[:,2]*(row['normalized_budget_use']-row['load_term'])
        for t in (4,6,10):
            row[f'reward_minus_point1_mean_excess_{t}']=row['reward']-.1*row[f'mean_excess_{t}']
            row[f'reward_minus_point1_max_excess_{t}']=row['reward']-.1*row[f'max_excess_{t}']
        records.append(row)
    combined = {k:np.stack([r[k] for r in records]) for k in records[0]}
    return dict(metrics={k:distribution(v) for k,v in combined.items()},
                max_original_trace_abs_error=max_error, episodes=len(seeds), slots=len(seeds)*600)


def training_probe(item, cfg):
    trainer = TensorTrainer(cfg, device='cpu')
    folder = LONG/'jobs'/item
    before = {}
    for p in folder.glob('*.pt'):
        before[str(p)] = digest(p)
    for i, actor in enumerate(trainer.actors):
        actor.actor.load_state_dict(torch.load(folder/f'actor_agent{i}.pt', map_location='cpu', weights_only=True))
    trainer.critic.critic.load_state_dict(torch.load(folder/'critic_agent.pt',map_location='cpu',weights_only=True))
    trainer.normalizer.load_state_dict(torch.load(folder/'value_normalizer.pt',map_location='cpu',weights_only=True))
    trainer.collect()
    b=trainer.buffer
    with torch.no_grad():
        returns=b.returns(trainer.normalizer,.99,.95)
        values=trainer.normalizer.denormalize(b.values[:-1])
        advantages=returns-values
        norm=(advantages-advantages.mean())/(advantages.std(unbiased=False)+1e-5)
        scaled=(100*advantages-(100*advantages).mean())/((100*advantages).std(unbiased=False)+1e-5)
        mean,var=trainer.normalizer.running_mean_var()
    logs=[json.loads(s) for s in (folder/'training_metrics.jsonl').read_text().splitlines()]
    tail=logs[-100:]
    assert all(np.isfinite(r['actor_loss_entropy_grad_ratio']).all() and np.isfinite(r['critic_loss_grad']).all() for r in logs)
    result=dict(sampled_steps=trainer.batch, parameter_updates=0,
                reward=distribution(arr(b.rewards)), returns=distribution(arr(returns)),
                advantages=distribution(arr(advantages)),normalized_advantages=distribution(arr(norm)),
                scale100_normalized_adv_max_abs_difference=float((norm-scaled).abs().max()),
                value_norm_mean=float(mean.item()),value_norm_variance=float(var.item()),
                critic_explained_variance=float(1-(returns-values).var(unbiased=False)/returns.var(unbiased=False)),
                log_updates=len(logs),log_all_finite=True,
                last100_actor_grad_norm=distribution([r['actor_loss_entropy_grad_ratio'][2] for r in tail]),
                last100_critic_grad_norm=distribution([r['critic_loss_grad'][1] for r in tail]),
                last100_entropy=distribution([r['actor_loss_entropy_grad_ratio'][1] for r in tail]),
                last100_training_reward=distribution([r['mean_training_reward'] for r in tail]))
    assert all(digest(Path(p))==h for p,h in before.items())
    return result


def penalty_routing_probe():
    """Synthetic reachable-shape state; test latent routing, not real performance."""
    out={}
    for enabled in (False,True):
        args=config()['env_args']
        args.update(constraint_penalty_A_by_instruction=[.1,.1,.1],
                    use_task_auxiliary_reward=enabled,
                    instruction_mode_strategy='explicit_evaluation',
                    explicit_instruction_schedule=[[0,0]])
        env=TensorSCEnv(args,count=1,seed=20261901,device='cpu')
        env.reset()
        env.aoi.fill_(12)
        env.q.zero_()
        actions=rule_actions(env,'m0_equal')
        predicted=arr(predict_reward(env,actions))
        _,_,reward,_,info,_=env.step(actions,auto_reset=False)
        base=float(info['base_reward'].item())
        common=float(info['common_reward'].item())
        actual=float(reward[0,0,0].item())
        np.testing.assert_allclose(base-common,.1*(13-6)/6,atol=1e-12,rtol=0)
        np.testing.assert_allclose(actual,common if enabled else base,atol=1e-7,rtol=0)
        np.testing.assert_allclose(predicted[0],base,atol=1e-12,rtol=0)
        out[str(enabled)]=dict(training_reward=actual,common_evaluation_reward=common,
                              myopic_predicted_reward=float(predicted[0]),
                              penalty=base-common)
    return out


def main():
    manifest=verify_parent()
    cfg=read(LONG/'configs/seed_85/IC_HAPPO.json')
    assert cfg['env_args']['constraint_penalty_A_by_instruction']==[0,0,0]
    items=[j['id'] for j in manifest['jobs']]+['rules/'+r for r in manifest['rules']]
    input_hashes={str(LONG/p):h for p,h in manifest['input_hashes'].items()}
    parts_by_item={}; component_results={}; replays={}; probes={}
    for item in items:
        folder=LONG/'evaluation'/item; summary=read(folder/'summary.json')
        parts=[]; component_results[item]={}
        for scenario,schedule in manifest['scenarios'].items():
            path=folder/f'{scenario}.npz'; h=digest(path)
            assert h==summary['traces'][scenario]; input_hashes[str(path)]=h
            with np.load(path) as z: data=z['trace']
            assert data.shape==(600,20,9) and not np.any(data[:,:,6:])
            parts.append(data)
            component_results[item][scenario]=decompose(data,cfg)
        component_results[item]['overall']=decompose(np.concatenate(parts),cfg)
        parts_by_item[item]=parts
    write(OUT/'components.json',component_results)
    print('All 117 formal traces read and hashes verified.',flush=True)
    for item in items:
        replays[item]={}
        for g in range(3):
            scenario=f'fixed_{g}'
            replays[item][scenario]=replay(LONG/'evaluation'/item/f'{scenario}.npz',manifest['scenarios'][scenario],manifest['evaluation_seeds'])
        write(OUT/'replay_partial.json',replays)
        print('Replay complete:',item,flush=True)
    for seed in manifest['seeds']:
        audit=read(GAP/f'seed_{seed}/audit.json')
        for branch in ('equal_resources','rule_mode'):
            path=GAP/f'seed_{seed}'/branch/'fixed_2.npz'
            h=digest(path); assert h==audit['trace_hashes'][str(path.relative_to(GAP))]
            input_hashes[str(path)]=h
            key=f'gap/seed_{seed}/{branch}'
            replays[key]={'fixed_2':replay(path,[[0,2]],manifest['evaluation_seeds'])}
        print('Quality interventions replayed:',seed,flush=True)
    write(OUT/'replays.json',replays)
    for job in manifest['jobs']:
        item=job['id']; job_cfg=read(LONG/'configs'/f'{item}.json')
        status=read(LONG/'jobs'/item/'status.json')
        for name,h in status['checkpoint_hashes'].items():
            path=LONG/'jobs'/item/name; assert digest(path)==h; input_hashes[str(path)]=h
        probes[item]=training_probe(item,job_cfg)
        write(OUT/'training_probes.json',probes)
        print('No-update training probe:',item,flush=True)
    write(OUT/'penalty_routing_probe.json',penalty_routing_probe())
    assert all(digest(Path(p))==h for p,h in input_hashes.items())
    write(OUT/'input_hashes.json',input_hashes)
    write(OUT/'audit.json',dict(state='PASS',formal_trace_files=117,formal_slots=1404000,
          physical_replay_episodes=660,physical_replay_slots=396000,
          stochastic_probe_slots=24000,parameter_updates=0,all_source_hashes_unchanged=True))
    print('PASS reward audit',flush=True)


if __name__=='__main__':
    torch.set_num_threads(1)
    main()
