"""New paired numeric traces; no calibration, fitting, or checkpoint selection."""
from repair_support import *
from adapter_policy import ResidualPolicy
import argparse
import importlib.util

def greedy_policy(method):
    spec = importlib.util.spec_from_file_location('frozen_observation_only_policy', GREEDY/'online_policy.py')
    module = importlib.util.module_from_spec(spec); sys.modules[spec.name]=module; spec.loader.exec_module(module)
    variant = method.removeprefix('greedy_')
    assert variant in ('modes_3','modes_16')
    path = GREEDY/variant/'predictor.npz'
    assert sha(path) == read(GREEDY/variant/'fit.json')['predictor_sha256']
    candidates = [0,5,10] if variant == 'modes_3' else list(range(16))
    return module.SUTGreedy(module.TreeScorePredictor(path)), module.UAVGreedy(candidates)

def load_policy(seed,method,env,steps=None):
    cfg = cfg_for(seed)
    actors = bases(cfg,env,seed,'cpu')
    if method.startswith('residual_'):
        folder = HERE / f'jobs/seed_{seed}/{method}/milestones/steps_{steps}'
        status = read(folder/'status.json')
        assert status['state']=='complete' and status['completed_steps']==steps
        for f,h in status['checkpoint_hashes'].items(): assert sha(folder/f)==h
        for i in range(1,4):
            actors[i].actor = ResidualPolicy(actors[i].actor,76,method,'cpu')
            actors[i].actor.adapter.load_state_dict(torch.load(folder/f'adapter_agent{i}.pt',weights_only=True),strict=True)
            actors[i].actor.requires_grad_(False);actors[i].prep_rollout()
            assert network_hash(actors[i].actor.adapter)==status['adapter_hashes'][i-1]
    return actors

@torch.no_grad()
def actions_for(env,obs,masks,method,actors=None,greedy=None):
    rnn = torch.zeros(env.count,1,256); mask = torch.ones(env.count,1)
    if method in ('R_instruction','R_equal_instruction'):
        acts = frozen.simple_rule_actions(env,method,obs)
        post,_,postmask = env.allocate_resources(acts[0])
        return acts,post,postmask,np.zeros((env.count,3),dtype=bool)
    if method.startswith('greedy_'):
        resource,_ = greedy[0].act(obs[:,0])
        post,_,postmask = env.allocate_resources(resource)
        modes = [greedy[1].act(post[:,i],postmask[:,i]) for i in range(1,4)]
        return [resource]+modes,post,postmask,np.zeros((env.count,3),dtype=bool)
    resource = actors[0].act(obs[:,0],rnn,mask,masks[:,0],deterministic=True)[0]
    post,_,postmask = env.allocate_resources(resource)
    modes = [actors[i].act(post[:,i],rnn,mask,postmask[:,i],deterministic=True)[0] for i in range(1,4)]
    gates = np.zeros((env.count,3),dtype=bool)
    if method == 'quality_rule_hybrid':
        rule_modes = frozen.simple_rule_actions(env,'R_instruction',obs)[1:]
        modes = [torch.where((post[:,i,67:70].argmax(-1)==2)[:,None],rule_modes[i-1],modes[i-1]) for i in range(1,4)]
    elif method.startswith('residual_'):
        gates = np.column_stack([arr(actors[i].actor.gate(post[:,i])) for i in range(1,4)])
    return [resource]+modes,post,postmask,gates

@torch.no_grad()
def evaluate(seed,method,steps=None):
    m=verify_inputs()
    rules=method in m['rules']
    assert rules or seed in m['parents']
    if method.startswith('residual_'): assert steps in m['milestones']
    label=method+(f'_at_{steps}' if steps else '')
    item=f'rules/{label}' if rules else f'seed_{seed}/{label}'
    folder=HERE/'evaluation'/item;folder.mkdir(parents=True,exist_ok=True)
    cfg=cfg_for(seed or m['parents'][0]);scenes=m['scenarios']
    if steps and steps<1000000: scenes={k:v for k,v in scenes.items() if k.startswith('fixed_')}
    actors=None;greedy=greedy_policy(method) if method.startswith('greedy_') else None
    identities=dict(manifest_sha256=sha(HERE/'manifest.json'),seed=seed,method=method,steps=steps)
    results={}
    for name,schedule in scenes.items():
        file=folder/f'{name}.npz';meta=folder/f'{name}.json'
        if meta.exists():
            old=read(meta);assert old['identity']==identities and sha(file)==old['trace_sha256']
            results[name]=old;continue
        env=make_env(cfg,m['validation'],'cpu',schedule,'evaluation')
        obs,_,masks=env.observe();external=exogenous(env)
        if not rules and actors is None: actors=load_policy(seed,method,env,steps)
        initial={k:arr(getattr(env,k)).copy() for k in ('aoi','q','tau')}
        records={k:[] for k in ('trace','modes','requested_modes','resource_fractions','budgets','usage','unused_budgets',
            'aoi_after','cache_after','tau_after','served','predicted_quality','quality_requirement','adapter_enabled','uav_masks','raw_resource_action')}
        for slot in range(600):
            acts,post,postmask,gates=actions_for(env,obs,masks,method,actors,greedy)
            assert env.step_index==slot
            obs,_,masks,info,values=checked_step(env,acts)
            values['instruction_id']=arr(info['gid'])
            records['trace'].append(np.column_stack([values[f] for f in FIELDS]))
            records['modes'].append(arr(info['mode']).astype(np.int8))
            records['requested_modes'].append(np.column_stack([arr(a.argmax(-1)) for a in acts[1:]]).astype(np.int8))
            for key,value in [('resource_fractions',env.beta),('budgets',info['budget']),('usage',info['usage']),
                ('unused_budgets',info['budget']-info['usage']),('predicted_quality',info['quality']),('quality_requirement',info['req']),
                ('raw_resource_action',acts[0])]:records[key].append(arr(value).copy())
            records['adapter_enabled'].append(gates)
            records['uav_masks'].append(arr(postmask[:,1:]).astype(bool))
            records['aoi_after'].append(arr(env.aoi).astype(np.uint16));records['cache_after'].append(arr(env.q).astype(bool))
            records['tau_after'].append(arr(env.tau).astype(np.int16));records['served'].append(arr(info['served']).astype(bool))
        records={k:np.stack(v) for k,v in records.items()}
        records.update(fields=np.asarray(FIELDS),seeds=np.asarray(m['validation']),initial_aoi=initial['aoi'],initial_q=initial['q'],initial_tau=initial['tau'])
        with file.with_suffix('.tmp').open('wb') as stream:np.savez_compressed(stream,**records)
        file.with_suffix('.tmp').replace(file)
        data=records['trace'];scores=data[:,:,FIELDS.index('common_reward')].mean(0)*100
        result=dict(identity=identities,scenario=name,schedule=schedule,external_hashes=external,trace_sha256=sha(file),
            score_x100=float(scores.mean()),scores_by_environment=scores.tolist(),episodes=len(m['validation']),slots_each=600,
            physical_reward_audit='PASS_every_slot_original_independent_checker',adapter_enabled_samples=int(records['adapter_enabled'].sum()))
        write(meta,result);results[name]=result
        write(folder/'status.json',dict(state='evaluating',completed_episodes=len(results)*20,target_episodes=len(scenes)*20))
        print(json.dumps(dict(item=item,scenario=name,score_x100=result['score_x100'])),flush=True)
    write(folder/'summary.json',dict(identity=identities,split='development_validation',scenarios=results))
    write(folder/'status.json',dict(state='complete',completed_episodes=len(results)*20,target_episodes=len(scenes)*20,summary_sha256=sha(folder/'summary.json')))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int);p.add_argument('--method',required=True);p.add_argument('--steps',type=int)
    args=p.parse_args();guard();evaluate(args.seed,args.method,args.steps)
