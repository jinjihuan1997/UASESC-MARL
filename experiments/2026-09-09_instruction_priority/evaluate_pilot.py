"""Final-checkpoint paired pilot evaluation; independent CPU physical metrics."""
from pathlib import Path
import argparse,csv,hashlib,json
import numpy as np
import torch
from common import write,stamp,verify_reference
from evaluate import aggregate,external_metrics,load_actors
from harl.envs.uav_escs.SC.uav_escs_env_sc import SCUAVEnv

def main(run,method):
    verify_reference();torch.set_num_threads(1)
    manifest=json.loads((run/'manifest.json').read_text())
    cfg=json.loads((run/'configs'/f'{method}.json').read_text())
    model=run/'jobs'/method;status=json.loads((model/'status.json').read_text())
    assert status['state']=='complete' and status['completed_steps']==manifest['steps']
    for name,digest in status['checkpoint_hashes'].items():
        assert hashlib.sha256((model/name).read_bytes()).hexdigest()==digest
    output=run/'evaluation'/method;output.mkdir(parents=True,exist_ok=False)
    args=cfg['env_args'].copy();args['explicit_instruction_schedule']=[[0,0]]
    env=SCUAVEnv(args);actors=load_actors(cfg,env,model);summaries=[]
    total=len(manifest['scenarios'])*len(manifest['evaluation_seeds'])
    for scenario,schedule in manifest['scenarios'].items():
        env.explicit_instruction_schedule=schedule
        for seed in manifest['evaluation_seeds']:
            env.seed(seed);obs,_,available=env.reset();rows=[];trace=hashlib.sha256()
            for a in [env.pos_uav,env.q_cache,env.tau_cache]:trace.update(a.tobytes())
            rnn=np.zeros((1,1,256),np.float32);masks=np.ones((1,1),np.float32)
            for t in range(600):
                trace.update(env.gamma_bh.tobytes());trace.update(np.asarray([env.current_instruction_id]).tobytes())
                with torch.no_grad():
                    actions=[actor.act(obs[i:i+1],rnn,masks,available[i:i+1],deterministic=True)[0].numpy().reshape(-1)
                             for i,actor in enumerate(actors)]
                obs,_,_,done,info,available=env.step(actions)
                expected_g=next(g for start,g in reversed(schedule) if t>=start)
                assert info[0]['instruction_id']==expected_g and bool(all(done))==(t==599)
                row=dict(slot=t,instruction_id=expected_g,**external_metrics(env,info[0]))
                row['fractions']=json.dumps(env.beta_sut_sat.tolist())
                row['modes']=json.dumps(env.last_selected_modes.tolist());rows.append(row)
            path=output/f'{scenario}_{seed}.csv'
            with path.open('x') as f:
                writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
            item=dict(method=method,scenario=scenario,seed=seed,external_sha256=trace.hexdigest(),
                      trace_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),**aggregate(rows))
            item['by_instruction']={str(g):aggregate(part) for g in range(3) if (part:=[r for r in rows if r['instruction_id']==g])}
            summaries.append(item)
            write(output/'status.json',dict(state='evaluating',completed=len(summaries),total=total,utc=stamp()))
    write(output/'summary.json',dict(method=method,training_seed=manifest['seed'],training_steps=manifest['steps'],episodes=summaries))
    write(output/'status.json',dict(state='complete',completed=total,total=total,utc=stamp()))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--method',required=True)
    a=p.parse_args();main(a.run,a.method)
