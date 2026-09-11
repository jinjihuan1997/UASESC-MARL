"""Frozen-model closed-loop ablations for mode and resource decisions."""
import argparse
import json
import sys
from pathlib import Path
OUT=Path(__file__).resolve().parent
LONG=OUT.parent/'2026-09-10_preference_long_training'
sys.path.insert(0,str(LONG))
from helpers import *

FIELDS=['common_reward','mean_aoi','predicted_quality_sum','deliveries','channel_uses','instruction_id','quality_term','aoi_term','load_term']
CONDITIONS=['original','equal_resources','rule_mode']


def aggregate(data):
    x=data.reshape(-1,len(FIELDS));d=x[:,3].sum()
    return dict(steps=len(x),common_reward=float(x[:,0].mean()),mean_aoi=float(x[:,1].mean()),
        delivered_predicted_psnr=float(x[:,2].sum()/d),deliveries_per_slot=float(d/len(x)),
        channel_uses_per_slot=float(x[:,4].mean()),quality_term=float(x[:,6].mean()),
        aoi_term=float(x[:,7].mean()),load_term=float(x[:,8].mean()))


def run(seed):
    manifest=verify_parent();method='IC_HAPPO';item=f'seed_{seed}/{method}'
    cfg=read(LONG/'configs'/f'{item}.json');model=read(LONG/'jobs'/item/'status.json')
    assert model['state']=='complete' and model['completed_steps']==10000000
    for f,h in model['checkpoint_hashes'].items():assert digest(LONG/'jobs'/item/f)==h
    folder=OUT/f'seed_{seed}';folder.mkdir(parents=True,exist_ok=True)
    identity=dict(script_sha256=digest(__file__),protocol_sha256=digest(OUT/'PROTOCOL.md'),parent_manifest_sha256=digest(LONG/'manifest.json'),checkpoint_hashes=model['checkpoint_hashes'])
    assert not (folder/'identity.json').exists(),'Preserve prior diagnostic'
    write(folder/'identity.json',identity)
    selection=read(LONG/'rule_selection.json');actors=None;result={};all_data={};pairing={};hashes={}
    formal=read(LONG/'evaluation'/item/'summary.json')
    for condition in CONDITIONS:
        result[condition]={};all_data[condition]=[]
        for scenario,schedule in manifest['scenarios'].items():
            env,obs,_,masks=make_env(manifest['evaluation_seeds'],schedule)
            external=external_hashes(env);assert external==formal['pairing'][scenario]
            if actors is None:actors=load_actors(cfg,env,LONG/'jobs'/item)
            records=[];modes=[];fractions=[]
            for slot in range(600):
                actions=actions_for(actors,obs,masks)
                if condition=='equal_resources':actions[0]=torch.full_like(actions[0],2/env.U-1)
                elif condition=='rule_mode':
                    gid=int(env.context()[0][0]);actions[1:]=rule_actions(env,selection['by_instruction'][gid])[1:]
                obs,_,masks,info,metrics=checked_step(env,actions)
                values=dict(metrics,instruction_id=arr(info['gid']),**{k:arr(info[k]) for k in FIELDS[6:]})
                records.append(np.column_stack([values[k] for k in FIELDS]));modes.append(arr(info['mode']));fractions.append(arr(env.beta))
            data=np.stack(records);mode=np.stack(modes);beta=np.stack(fractions)
            if condition=='original':
                reference=LONG/'evaluation'/item/f'{scenario}.npz';assert digest(reference)==formal['traces'][scenario]
                with np.load(reference) as z:
                    np.testing.assert_allclose(data[:,:,:6],z['trace'][:,:,:6],atol=1e-8,rtol=1e-10)
                    np.testing.assert_array_equal(mode,z['modes']);np.testing.assert_allclose(beta,z['resource_fractions'],atol=1e-10,rtol=0)
            path=folder/condition/f'{scenario}.npz';path.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(path,trace=data,modes=mode,resource_fractions=beta,fields=np.asarray(FIELDS))
            with np.load(path) as z:
                np.testing.assert_array_equal(data,z['trace']);stats=aggregate(z['trace'])
            result[condition][scenario]=stats;all_data[condition].append(data);hashes[str(path.relative_to(OUT))]=digest(path)
            print(f'{seed} {condition} {scenario}',flush=True)
        merged=np.concatenate(all_data[condition]);result[condition]['overall']=aggregate(merged)
        for g in range(3):result[condition][f'true_instruction_{g}']=aggregate(merged[merged[:,:,5]==g])
    write(folder/'results.json',result)
    write(folder/'audit.json',dict(state='PASS',episodes=780,slots=468000,original_reproduces_formal=True,external_pairing=True,trace_hashes=hashes))
    for f,h in model['checkpoint_hashes'].items():assert digest(LONG/'jobs'/item/f)==h


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int,required=True);parser.add_argument('--core',type=int,required=True);args=parser.parse_args()
    os.sched_setaffinity(0,{args.core});torch.set_num_threads(1);run(args.seed)
