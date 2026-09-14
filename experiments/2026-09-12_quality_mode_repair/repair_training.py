"""Independent weight-warm-start trainer; reuses unchanged HAPPO/critic kernels."""
from repair_support import *
from adapter_policy import ResidualPolicy
import argparse
import signal
import time

class RepairTrainer:
    def __init__(self, seed, arm, device='cpu', preflight=False, length=None):
        self.seed, self.arm, self.device = seed, arm, torch.device(device)
        self.config = read(HERE / f'configs/seed_{seed}/{arm}.json')
        self.algo = self.config['algo_args']
        self.seeds = self.config['training_design']['seeds']
        torch.manual_seed(self.seeds['global_initialization']); np.random.seed(self.seeds['global_initialization']); random.seed(self.seeds['global_initialization'])
        if self.device.type == 'cuda':
            torch.cuda.manual_seed_all(self.seeds['global_initialization'])
            torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
        envseeds = manifest()['preflight_seeds'] * 2 if preflight else self.seeds['environment']
        self.env = make_env(self.config, envseeds, device, purpose='preflight' if preflight else 'training')
        self.actors = bases(self.config, self.env, seed, self.device)
        self.base_hashes = [network_hash(a.actor) for a in self.actors]
        for i in range(1, 4):
            torch.manual_seed(self.seeds['initialization'][i-1])
            if self.device.type == 'cuda': torch.cuda.manual_seed_all(self.seeds['initialization'][i-1])
            self.actors[i].actor = ResidualPolicy(self.actors[i].actor, self.env.obs_dim_common, arm, self.device)
            self.actors[i].actor_optimizer = torch.optim.Adam(self.actors[i].actor.adapter.parameters(), lr=1e-4, eps=1e-5, weight_decay=0)
        params = {**self.algo['model'], **self.algo['algo']}
        self.critic = VCritic(params, self.env.share_observation_space[0], self.device)
        self.critic.critic.load_state_dict(torch.load(model_dir(seed)/'critic_agent.pt', map_location=self.device, weights_only=True), strict=True)
        self.normalizer = TensorValueNorm(1, device=self.device)
        self.normalizer.load_state_dict(torch.load(model_dir(seed)/'value_normalizer.pt', map_location=self.device, weights_only=True), strict=True)
        assert network_hash(self.critic.critic) == read(model_dir(seed)/'status.json')['critic_hash']
        self.buffer = TensorRollout(self.env, length or self.algo['train']['episode_length'], self.algo['model'])
        self.batch = self.buffer.length*self.env.count
        self.total_updates = 250 if not preflight else 12
        self.initial_actors = [network_hash(a.actor) for a in self.actors]
        self.initial_critic = network_hash(self.critic.critic)
        self.initial_normalizer = network_hash(self.normalizer)
        self.initial_adapters = [network_hash(a.actor.adapter) for a in self.actors[1:]]
        self.streams = ActionStreams(self.seeds['actions'], self.device)
        self.update_generator = torch.Generator(device=self.device).manual_seed(self.seeds['updates'])
        self.buffer.store_observation(0,*self.env.observe())
        self.counters = dict(instruction_samples=[0,0,0],actor_updates=[0,0,0],quality_sample_exposures=[0,0,0],skipped_no_quality=[0,0,0],physical_steps=0)
        self.episode_ledger = [dict(episodes=list(self.env.episode_indices),hashes=exogenous(self.env))]
        self.completed_update = 0
        self.set_training_update(1)

    def synchronize(self):
        if self.device.type == 'cuda': torch.cuda.synchronize(self.device)

    def set_training_update(self, update):
        assert 1 <= update <= self.total_updates+1
        self.training_update = update
        self.assert_frozen()

    def assert_frozen(self):
        for i, actor in enumerate(self.actors):
            base = actor.actor if i == 0 else actor.actor.base_policy
            assert network_hash(base) == self.base_hashes[i]
            assert not any(p.requires_grad for p in base.parameters())
            assert not base.training
            if i:
                opt = {id(p) for g in actor.actor_optimizer.param_groups for p in g['params']}
                expected = {id(p) for p in actor.actor.adapter.parameters()}
                assert opt == expected and not opt & {id(p) for p in base.parameters()}

    @torch.no_grad()
    def collect(self):
        b = self.buffer
        for a in self.actors: a.prep_rollout()
        self.critic.prep_rollout()
        stats = np.zeros((3,len(FIELDS)), dtype=np.float64)
        modes = np.zeros((3,3,17),dtype=np.int64)
        requested = np.zeros((3,3,16),dtype=np.int64)
        resources = {k:np.zeros((3,3),dtype=np.float64) for k in ('shares','budget','usage','unused','deliveries')}
        ins = np.zeros(3,dtype=np.int64)
        for t in range(b.length):
            value,_ = self.critic.get_values(b.state[t], b.rnn, b.masks[t]); b.values[t].copy_(value)
            resource = self.actors[0].act(b.obs[t,:,0],b.rnn,b.masks[t],b.available[t,:,0],deterministic=True)[0]
            b.actions[0][t].copy_(resource); b.log_probs[0][t].zero_() # Never a stochastic SUT density.
            post,_,available = self.env.allocate_resources(resource)
            b.obs[t,:,1:].copy_(post[:,1:]); b.available[t,:,1:].copy_(available[:,1:])
            actions = [resource]
            for i in range(1,4):
                with self.streams.use(i-1):
                    action,logp,_ = self.actors[i].get_actions(b.obs[t,:,i],b.rnn,b.masks[t],b.available[t,:,i],deterministic=False)
                b.actions[i][t].copy_(action); b.log_probs[i][t].copy_(logp); actions.append(action)
            obs,state,available,info,values = checked_step(self.env,actions)
            gid = arr(info['gid']).astype(int); values['instruction_id'] = gid
            np.add.at(ins,gid,1)
            np.add.at(stats,gid,np.column_stack([values[f] for f in FIELDS]))
            ex = arr(info['mode']).astype(int)+1
            req = np.column_stack([arr(a.argmax(-1)) for a in actions[1:]])
            for u in range(3):
                np.add.at(modes[:,u],(gid,ex[:,u]),1)
                np.add.at(requested[:,u],(gid,req[:,u]),1)
            for k, v in [('shares',self.env.beta),('budget',info['budget']),('usage',info['usage']),
                         ('unused',info['budget']-info['usage']),('deliveries',info['served'].sum(-1))]:
                np.add.at(resources[k],gid,arr(v))
            reward = torch.as_tensor(values['training_reward'], device=self.device,dtype=torch.float32).unsqueeze(-1)
            # check_tensor verified this equals the actually returned float32 reward.
            b.rewards[t].copy_(reward)
            done = self.env.step_index == 600
            b.masks[t+1].fill_(0 if done else 1)
            if done:
                obs,state,available = self.env.reset()
                self.episode_ledger.append(dict(episodes=list(self.env.episode_indices),hashes=exogenous(self.env)))
            b.store_observation(t+1,obs,state,available)
        value,_ = self.critic.get_values(b.state[-1],b.rnn,b.masks[-1]); b.values[-1].copy_(value)
        self.counters['physical_steps'] += self.batch
        self.counters['instruction_samples'] = (np.asarray(self.counters['instruction_samples'])+ins).tolist()
        return dict(instruction_samples=ins.tolist(),fields=FIELDS,reward_and_physical_sums=stats.tolist(),
                    executed_mode_counts=modes.tolist(),requested_mode_counts=requested.tolist(),
                    resources={k:v.tolist() for k,v in resources.items()})

    def update(self):
        b = self.buffer; args = self.algo['algo']
        returns = b.returns(self.normalizer,args['gamma'],args['gae_lambda'])
        old_value = self.normalizer.denormalize(b.values[:-1])
        advantages = returns-old_value
        normalized = (advantages-advantages.mean())/(advantages.std(unbiased=False)+1e-5)
        factor = torch.ones_like(advantages).flatten(0,1)
        agent_order = (torch.randperm(3,generator=self.update_generator,device=self.device)+1).tolist()
        def batches(count): return torch.randperm(self.batch,generator=self.update_generator,device=self.device).chunk(count)
        actor_metrics = []
        for i in agent_order:
            actor = self.actors[i]; actor.prep_training()
            obs,rnn,action,mask,available,active = b.actor_inputs(i)
            quality = obs[:,67:70].argmax(-1) == 2
            with torch.no_grad(): old = actor.evaluate_actions(obs,rnn,action,mask,available,active)[0]
            first_ratio = aggregate_action_ratio(old-b.log_probs[i].flatten(0,1),'prod',clip=20.)
            # Batched GEMM may round differently from rollout-sized GEMM. Log it;
            # preflight checks an identical-size evaluation exactly before update.
            if not torch.allclose(first_ratio,torch.ones_like(first_ratio),atol=3e-6,rtol=0):
                raise AssertionError(('rollout replay ratio',float((first_ratio-1).abs().max())))
            rows = []; skipped = 0; quality_exposures = 0
            for epoch in range(args['ppo_epoch']):
                for indices in batches(args['actor_num_mini_batch']):
                    qcount = int(quality[indices].sum()); quality_exposures += qcount
                    if self.arm == 'residual_quality' and qcount == 0:
                        skipped += 1; continue
                    sample = (obs[indices],rnn[indices],action[indices],mask[indices],active[indices],
                        b.log_probs[i].flatten(0,1)[indices],normalized.flatten(0,1)[indices],available[indices],factor[indices])
                    loss,entropy,grad,ratio = actor.update(sample)
                    rows.append(dict(policy_loss=float(loss.detach()),entropy=float(entropy.detach()),gradient_norm=float(grad),
                        ratio_mean=float(ratio.detach().mean()),ratio_min=float(ratio.detach().min()),ratio_max=float(ratio.detach().max()),
                        clip_fraction=float(((ratio.detach()-1).abs()>.1).float().mean()),quality_samples=qcount,batch_samples=len(indices)))
            with torch.no_grad():
                new = actor.evaluate_actions(obs,rnn,action,mask,available,active)[0]
                ratio = aggregate_action_ratio(new-old,'prod',clip=20.)
                if self.arm == 'residual_quality' and (~quality).any():
                    assert torch.equal(ratio[~quality],torch.ones_like(ratio[~quality]))
                factor = torch.nan_to_num(factor*torch.nan_to_num(ratio,nan=1.,posinf=1e6,neginf=0.).clamp(0,1e3),nan=1.,posinf=1e6,neginf=0.).clamp(0,1e3)
            self.counters['actor_updates'][i-1] += len(rows)
            self.counters['quality_sample_exposures'][i-1] += quality_exposures
            self.counters['skipped_no_quality'][i-1] += skipped
            actor_metrics.append(dict(agent=i,updates=len(rows),skipped_no_quality=skipped,quality_samples=int(quality.sum()),
                                      preupdate_ratio_max_error=float((first_ratio-1).abs().max()),minibatches=rows))
        self.critic.prep_training(); critic_metrics = []
        for epoch in range(args['critic_epoch']):
            for indices in batches(args['critic_num_mini_batch']):
                sample = (b.state[:-1].flatten(0,1)[indices],b.flat_rnn[indices],b.values[:-1].flatten(0,1)[indices],
                          returns.flatten(0,1)[indices],b.masks[:-1].flatten(0,1)[indices])
                loss,grad,_ = self.critic.update(sample,self.normalizer)
                critic_metrics.append(dict(loss=float(loss.detach()),gradient_norm=float(grad)))
        with torch.no_grad():
            self.critic.prep_rollout()
            v,_ = self.critic.get_values(b.state[:-1].flatten(0,1),b.flat_rnn,b.masks[:-1].flatten(0,1))
            v = self.normalizer.denormalize(v).reshape_as(returns)
            diag = dict(gae_target_mean=float(returns.mean()),gae_target_variance=float(returns.var(unbiased=False)),
                value_mse_before=float(((returns-old_value)**2).mean()),value_mse_after=float(((returns-v)**2).mean()),
                explained_variance_after=float(1-(returns-v).var(unbiased=False)/returns.var(unbiased=False).clamp_min(1e-12)))
        self.assert_frozen()
        return dict(actor_order=agent_order,actors=actor_metrics,critic_minibatches=critic_metrics,value_diagnostic=diag)

    def state(self, update, identity):
        state = capture(self,update,identity)
        state['repair'] = dict(action_rng=[x.clone() for x in self.streams.states],update_rng=self.update_generator.get_state().clone(),
            counters=copy.deepcopy(self.counters),episode_ledger=copy.deepcopy(self.episode_ledger),
            initial_adapters=self.initial_adapters,initial_normalizer=self.initial_normalizer,base_hashes=self.base_hashes)
        return state

    def restore(self,state,identity):
        update = restore(self,state,identity)
        r = state['repair']; self.streams.states = [x.clone() for x in r['action_rng']]; self.update_generator.set_state(r['update_rng'])
        for key in ('counters','episode_ledger','initial_adapters','initial_normalizer','base_hashes'): setattr(self,key,copy.deepcopy(r[key]))
        self.completed_update = update; self.assert_frozen()
        return update

    def snapshot(self,folder,update,identity):
        assert not folder.exists(), 'Immutable milestone already exists'
        folder.mkdir(parents=True)
        for i,actor in enumerate(self.actors[1:],1): torch.save(actor.actor.adapter.state_dict(),folder/f'adapter_agent{i}.pt')
        torch.save(self.critic.critic.state_dict(),folder/'critic_agent.pt')
        torch.save(self.normalizer.state_dict(),folder/'value_normalizer.pt')
        hashes = {p.name:sha(p) for p in folder.iterdir()}
        write(folder/'status.json',dict(state='complete',identity=identity,update=update,completed_steps=update*self.batch,
            parent_steps=1000000,base_hashes=self.base_hashes,adapter_hashes=[network_hash(a.actor.adapter) for a in self.actors[1:]],
            initial_adapter_hashes=self.initial_adapters,initial_critic_hash=self.initial_critic,initial_normalizer_hash=self.initial_normalizer,
            critic_hash=network_hash(self.critic.critic),normalizer_hash=network_hash(self.normalizer),
            checkpoint_hashes=hashes,counters=self.counters))

def main():
    p = argparse.ArgumentParser(); p.add_argument('--seed',type=int,required=True);p.add_argument('--arm',required=True,choices=['residual_all','residual_quality']);p.add_argument('--device',default='cpu');p.add_argument('--resume',action='store_true')
    args = p.parse_args(); guard(); m = verify_inputs()
    assert read(HERE/'preflight.json')['state'] == 'PASS'
    folder = HERE / f'jobs/seed_{args.seed}/{args.arm}'; folder.mkdir(parents=True,exist_ok=True)
    identity = dict(manifest_sha256=sha(HERE/'manifest.json'),config_sha256=sha(HERE/f'configs/seed_{args.seed}/{args.arm}.json'),device=args.device)
    stop = []; signal.signal(signal.SIGTERM,lambda n,f:stop.append(n));signal.signal(signal.SIGINT,lambda n,f:stop.append(n))
    trainer = RepairTrainer(args.seed,args.arm,args.device)
    start = 0
    if (folder/'checkpoints/index.json').exists():
        assert args.resume, 'Existing run requires explicit same-protocol resume'
        state, provenance = load_checkpoint(folder/'checkpoints'); start = trainer.restore(state,identity)
        reconcile_logs(folder,start); append(folder/'resume.jsonl',dict(utc=stamp(),provenance=provenance))
    else:
        assert not (folder/'training_metrics.jsonl').exists()
        write(folder/'initialization.json',dict(identity=identity,base_hashes=trainer.base_hashes,adapter_hashes=trainer.initial_adapters,
            critic_hash=trainer.initial_critic,normalizer_hash=trainer.initial_normalizer,fresh_optimizers=True,seeds=trainer.seeds))
    for update in range(start+1,251):
        t0 = time.monotonic(); trainer.set_training_update(update)
        collect = trainer.collect(); t1 = time.monotonic()
        metrics = trainer.update(); trainer.buffer.after_update(); trainer.completed_update = update
        trainer.synchronize(); t2 = time.monotonic()
        append(folder/'training_metrics.jsonl',dict(update=update,steps=update*4000,collection=collect,optimization=metrics))
        append(folder/'timing.jsonl',dict(update=update,collect_seconds=t1-t0,update_seconds=t2-t1,steps_per_second=4000/(t2-t0)))
        if update%5 == 0 or stop:
            save_checkpoint(folder/'checkpoints',trainer.state(update,identity))
        if update%50 == 0:
            snap = folder/f'milestones/steps_{update*4000}'
            if snap.exists():
                assert read(snap/'status.json')['identity'] == identity
                assert read(snap/'status.json')['adapter_hashes'] == [network_hash(a.actor.adapter) for a in trainer.actors[1:]]
            else: trainer.snapshot(snap,update,identity)
        write(folder/'status.json',dict(state='complete' if update==250 else 'paused' if stop else 'training',update=update,
              completed_steps=update*4000,target_steps=1000000,identity=identity,counters=trainer.counters,updated_utc=stamp()))
        if update%10 == 0: print(json.dumps(dict(seed=args.seed,arm=args.arm,steps=update*4000,steps_per_second=4000/(t2-t0))),flush=True)
        if stop: return 75
    write(folder/'environment_ledger.json',dict(seeds=trainer.seeds['environment'],episodes=trainer.episode_ledger))
    trainer.assert_frozen(); verify_inputs()
    return 0

if __name__ == '__main__': raise SystemExit(main())
