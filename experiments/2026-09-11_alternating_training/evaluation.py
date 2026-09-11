"""Full paired validation episodes, using only the four requested simple rules."""
import argparse
import signal
from helpers import *

FIELDS=['common_reward','mean_aoi','predicted_quality_sum','deliveries','channel_uses','instruction_id',
    'quality_violations','budget_violations','cache_violations','fraction_above4','fraction_above6',
    'quality_credit','age_mean_cost','age_max_cost','age_tail_cost','resource_cost',
    'service_violation_cost','recv_aoi_bonus','max_aoi']

def summarize(data):
    x=data.reshape(-1,len(FIELDS));v={f:x[:,i] for i,f in enumerate(FIELDS)}
    delivery=v['deliveries'].sum()
    keys=['common_reward','mean_aoi','fraction_above4','fraction_above6','quality_credit','age_mean_cost',
        'age_max_cost','age_tail_cost','resource_cost','service_violation_cost','recv_aoi_bonus','max_aoi']
    return dict(slots=len(x),**{k:float(v[k].mean()) for k in keys},
        delivered_predicted_psnr=float(v['predicted_quality_sum'].sum()/delivery) if delivery else None,
        deliveries_per_slot=float(delivery/len(x)),channel_uses_per_slot=float(v['channel_uses'].mean()))

@torch.no_grad()
def evaluate_item(item,stopping):
    manifest=verify();category,method=item.split('/')
    learned=category.startswith('seed_');actors=None;diagnostic=False
    if learned:
        arm,step_text=method.rsplit('_at_',1);steps=int(step_text)
        assert arm in METHODS and steps in manifest['milestones']
        cfg=config(arm,int(category[5:]))
        model_dir=HERE/'jobs'/category/arm/'milestones'/f'steps_{steps}'
        model_status=read(model_dir/'status.json')
        assert model_status['state']=='complete' and model_status['completed_steps']==steps
        for f,h in model_status['checkpoint_hashes'].items():assert digest(model_dir/f)==h
        diagnostic=arm=='alternating' and steps<=400000
    else:
        assert method in RULE_METHODS
        cfg=config('joint',manifest['seeds'][0]);model_status=None
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
        records=[];modes=[];fractions=[];ages=[];budgets=[];unused=[]
        for slot in range(600):
            if diagnostic:
                resource=torch.full((env.count,env.U),2/env.U-1,dtype=torch.float32)
                post,_,post_masks=env.allocate_resources(resource)
                rnn=torch.zeros((env.count,1,256));mask=torch.ones((env.count,1))
                choices=[actors[i].act(post[:,i],rnn,mask,post_masks[:,i],deterministic=True)[0]
                         for i in range(1,4)]
                actions=[resource]+choices
            elif learned:actions=learned_actions(env,actors,obs,masks)
            else:actions=simple_rule_actions(env,method,obs)
            obs,_,masks,info,values=checked_step(env,actions)
            values['instruction_id']=arr(info['gid'])
            records.append(np.column_stack([values[f] for f in FIELDS]))
            modes.append(arr(info['mode']));fractions.append(arr(env.beta));ages.append(arr(env.aoi).astype(np.uint16))
            budgets.append(arr(info['budget']));unused.append(arr(info['budget']-info['usage']))
        data=np.stack(records)
        with file.with_suffix('.tmp').open('wb') as stream:
            np.savez_compressed(stream,trace=data,fields=np.asarray(FIELDS),seeds=np.asarray(manifest['evaluation_seeds']),
                modes=np.stack(modes),resource_fractions=np.stack(fractions),aoi_after=np.stack(ages),
                budgets=np.stack(budgets),unused_budgets=np.stack(unused))
        file.with_suffix('.tmp').replace(file)
        result=dict(overall=summarize(data),by_evaluation_seed=[summarize(data[:,i]) for i in range(env.count)],
            external_hashes=external,trace_sha256=digest(file),diagnostic_equal_resources=diagnostic)
        write(meta,result);summaries[scenario]=result;pairing[scenario]=external;hashes[scenario]=result['trace_sha256']
        write(folder/'status.json',dict(state='evaluating',completed_episodes=len(summaries)*env.count,updated_utc=stamp()))
        if stopping:
            write(folder/'status.json',dict(state='paused',completed_episodes=len(summaries)*env.count,updated_utc=stamp()))
            return 75
    write(folder/'summary.json',dict(item=item,split='new_validation',diagnostic_equal_resources=diagnostic,
        model_status=model_status,scenarios=summaries,pairing=pairing,traces=hashes))
    write(folder/'status.json',dict(state='complete',completed_episodes=len(summaries)*len(manifest['evaluation_seeds']),
        summary_sha256=digest(folder/'summary.json'),updated_utc=stamp()))
    return 0

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--item',required=True);args=p.parse_args();stopping=[]
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda n,f:stopping.append(n))
    raise SystemExit(evaluate_item(args.item,stopping))
