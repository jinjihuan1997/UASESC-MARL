"""Paired evaluation with both fixed scoring standards and physical traces."""
import argparse
import signal
from helpers import *
from rule_tools import myopic_actions

FIELDS=['common_reward','mean_aoi','predicted_quality_sum','deliveries','channel_uses','instruction_id',
        'quality_violations','budget_violations','cache_violations','objective_ref10','objective_ref8',
        'fraction_above4','fraction_above6','quality_credit','age_mean_cost','age_max_cost','age_tail_cost',
        'resource_cost','service_violation_cost','recv_aoi_bonus','max_aoi']


def summarize(data):
    x=data.reshape(-1,len(FIELDS));v={f:x[:,i] for i,f in enumerate(FIELDS)}
    keys=['common_reward','mean_aoi','objective_ref10','objective_ref8','fraction_above4','fraction_above6',
          'quality_credit','age_mean_cost','age_max_cost','age_tail_cost','resource_cost','service_violation_cost','recv_aoi_bonus','max_aoi']
    d=v['deliveries'].sum()
    return dict(slots=len(x),**{k:float(v[k].mean()) for k in keys},
                delivered_predicted_psnr=float(v['predicted_quality_sum'].sum()/d) if d else None,
                deliveries_per_slot=float(d/len(x)),channel_uses_per_slot=float(v['channel_uses'].mean()))


def evaluate_item(item,stopping):
    manifest=read(HERE/'manifest.json');category,method=item.split('/')
    learned=category.startswith('seed_')
    if learned:
        arm,step_text=method.rsplit('_at_',1);steps=int(step_text)
        assert arm in METHODS and steps in manifest['milestones']
        model_dir=HERE/'jobs'/category/arm/'milestones'/f'steps_{steps}'
    else: arm='joint_continue'
    cfg=config(arm,int(category[5:]) if learned else 810974)
    actors=None
    if learned:
        status=read(model_dir/'status.json')
        assert status['state']=='complete' and status['artifact']=='inference_snapshot' and status['completed_steps']==steps
        for f,h in status['checkpoint_hashes'].items():assert digest(model_dir/f)==h
    selection=read(HERE/'rule_selection.json')
    folder=HERE/'evaluation'/item;folder.mkdir(parents=True,exist_ok=True)
    summaries={};pairing={};hashes={}
    for scenario,schedule in manifest['scenarios'].items():
        file=folder/f'{scenario}.npz';meta=folder/f'{scenario}.json'
        if meta.exists():
            old=read(meta);assert digest(file)==old['trace_sha256']
            summaries[scenario]=old;pairing[scenario]=old['external_hashes'];hashes[scenario]=old['trace_sha256']
            continue
        env,obs,_,masks=make_env(cfg,manifest['evaluation_seeds'],schedule)
        external=external_hashes(env)
        if learned and actors is None:actors=load_actors(cfg,env,model_dir)
        records=[];modes=[];fractions=[];ages=[];choices=[]
        for slot in range(600):
            predicted=None
            if learned:actions=learned_actions(env,actors,obs,masks,arm)
            elif method=='R_single':actions=rule_actions(env,selection['single'])
            elif method=='R_instruction':actions=rule_actions(env,selection['by_instruction'][int(env.context()[0][0])])
            elif method=='R_myopic':actions,predicted=myopic_actions(env)
            else:raise ValueError(method)
            choices.append(arr(env.last_selector_choice).copy() if learned and arm=='selector' else np.full(env.count,-1))
            obs,_,masks,info,values=checked_step(env,actions)
            if predicted is not None:np.testing.assert_allclose(arr(predicted),values['common_reward'],atol=1e-9,rtol=0)
            values['instruction_id']=arr(info['gid'])
            records.append(np.column_stack([values[f] for f in FIELDS]))
            modes.append(arr(info['mode']));fractions.append(arr(env.beta));ages.append(arr(env.aoi).astype(np.uint16))
        data=np.stack(records)
        with file.with_suffix('.tmp').open('wb') as stream:
            np.savez_compressed(stream,trace=data,fields=np.asarray(FIELDS),seeds=np.asarray(manifest['evaluation_seeds']),
                                modes=np.stack(modes),resource_fractions=np.stack(fractions),aoi_after=np.stack(ages),selector_choices=np.stack(choices))
        file.with_suffix('.tmp').replace(file)
        result=dict(overall=summarize(data),by_evaluation_seed=[summarize(data[:,i]) for i in range(env.count)],
                    external_hashes=external,trace_sha256=digest(file))
        write(meta,result);summaries[scenario]=result;pairing[scenario]=external;hashes[scenario]=result['trace_sha256']
        write(folder/'status.json',dict(state='evaluating',completed_episodes=len(summaries)*env.count,updated_utc=stamp()))
        if stopping:
            write(folder/'status.json',dict(state='paused',completed_episodes=len(summaries)*env.count,updated_utc=stamp()));return 75
    write(folder/'summary.json',dict(item=item,scenarios=summaries,pairing=pairing,traces=hashes))
    write(folder/'status.json',dict(state='complete',completed_episodes=len(summaries)*len(manifest['evaluation_seeds']),
                                  summary_sha256=digest(folder/'summary.json'),updated_utc=stamp()))
    return 0


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--item',required=True);args=parser.parse_args()
    stopping=[]
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda n,f:stopping.append(n))
    raise SystemExit(evaluate_item(args.item,stopping))
