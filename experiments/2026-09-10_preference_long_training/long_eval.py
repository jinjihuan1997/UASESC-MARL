"""Resumable paired final-model evaluation, one item per process."""
import argparse
import signal
from helpers import *
from rule_tools import myopic_actions,summarize

FIELDS=['common_reward','mean_aoi','predicted_quality_sum','deliveries','channel_uses','instruction_id','quality_violations','budget_violations','cache_violations']


def evaluate_item(item,stopping):
    manifest=read(HERE/'manifest.json');selection=read(HERE/'rule_selection.json')
    target_episodes=20*len(manifest['scenarios'])
    folder=HERE/'evaluation'/item;folder.mkdir(parents=True,exist_ok=True)
    learned=item.startswith('seed_');method=item.split('/')[-1]
    if learned:
        model=read(HERE/'jobs'/item/'status.json');assert model['state']=='complete' and model['completed_steps']==10_000_000
        for f,h in model['checkpoint_hashes'].items():assert digest(HERE/'jobs'/item/f)==h
        cfg=read(HERE/'configs'/f'{item}.json')
    actors=None;scenarios={};pairing={};traces={}
    for scenario,schedule in manifest['scenarios'].items():
        output=folder/f'{scenario}.json';file=folder/f'{scenario}.npz'
        if output.exists():
            v=read(output);assert digest(file)==v['trace_sha256'];scenarios[scenario]=v;pairing[scenario]=v['external_hashes'];traces[scenario]=v['trace_sha256']
            continue
        env,obs,_,masks=make_env(manifest['evaluation_seeds'],schedule,method=='HAPPO_hidden_instruction')
        external=external_hashes(env)
        if learned and actors is None:actors=load_actors(cfg,env,HERE/'jobs'/item)
        rows=[];modes=[];fractions=[]
        for slot in range(600):
            predicted=None
            if learned:actions=actions_for(actors,obs,masks)
            elif method=='R_single':actions=rule_actions(env,selection['single'])
            elif method=='R_instruction':actions=rule_actions(env,selection['by_instruction'][int(env.context()[0][0])])
            elif method=='R_myopic':actions,predicted=myopic_actions(env)
            else:raise ValueError(item)
            obs,_,masks,info,metrics=checked_step(env,actions)
            if predicted is not None:np.testing.assert_allclose(arr(predicted),metrics['common_reward'],atol=1e-9,rtol=0)
            v=dict(metrics,instruction_id=arr(info['gid']))
            rows.append(np.column_stack([v[f] for f in FIELDS]));modes.append(arr(info['mode']));fractions.append(arr(env.beta))
        trace=np.stack(rows);assert not np.any(trace[:,:,6:])
        with file.with_suffix('.tmp').open('wb') as stream:
            np.savez_compressed(stream,trace=trace,fields=np.asarray(FIELDS),seeds=np.asarray(manifest['evaluation_seeds']),modes=np.stack(modes),resource_fractions=np.stack(fractions))
        file.with_suffix('.tmp').replace(file)
        value=dict(overall=summarize(trace),by_evaluation_seed=[summarize(trace[:,i]) for i in range(20)],
            by_instruction={str(g):summarize(trace[trace[:,:,5]==g]) for g in range(3) if np.any(trace[:,:,5]==g)},
            external_hashes=external,trace_sha256=digest(file))
        write(output,value);scenarios[scenario]=value;pairing[scenario]=external;traces[scenario]=value['trace_sha256']
        write(folder/'status.json',dict(state='evaluating',completed_episodes=len(scenarios)*20,total_episodes=target_episodes,updated_utc=stamp()))
        if stopping:
            write(folder/'status.json',dict(state='paused',completed_episodes=len(scenarios)*20,updated_utc=stamp()));return 75
    write(folder/'summary.json',dict(item=item,scenarios=scenarios,pairing=pairing,traces=traces))
    write(folder/'status.json',dict(state='complete',completed_episodes=target_episodes,summary_sha256=digest(folder/'summary.json'),updated_utc=stamp()))
    return 0


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--item',required=True);args=parser.parse_args()
    stopping=[]
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda signum,frame:stopping.append(signum))
    torch.set_num_threads(1)
    raise SystemExit(evaluate_item(args.item,stopping))
