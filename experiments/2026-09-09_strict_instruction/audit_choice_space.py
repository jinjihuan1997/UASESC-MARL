"""Read-only choice audit of frozen profile, matched evaluation channels and budgets."""
from pathlib import Path
import sys,json,csv,hashlib
ROOT=Path(__file__).resolve().parent
RUN=ROOT/'runs/seed85_1m'
sys.path.insert(0,str(RUN/'source'))
from common import verify_reference
from harl.envs.uav_escs.SC.uav_escs_env_sc import SCUAVEnv
import numpy as np
verify_reference()
manifest=json.loads((RUN/'manifest.json').read_text());cfg=json.loads((RUN/'configs/IC_HAPPO.json').read_text())
env=SCUAVEnv(cfg['env_args']);env.seed(manifest['evaluation_seeds'][0]);env.reset()
p=env.semantic_library.profile;grid=np.asarray(p['snr_grid_db']);qprof=np.asarray(p['q_hat_mean']);lprof=np.asarray(p['l_z_mean'])
req=np.asarray(env.Q_req_by_instruction_snr_bucket)
def tables(db):
    q=np.stack([np.interp(db,grid,v) for v in qprof],axis=-1)
    load=np.stack([np.interp(db,grid,v) for v in lprof],axis=-1)+env.side_info_bits/np.log2(1+10**(db[...,None]/10))
    return q,load

def options(q,load,g,budget,snr):
    bucket=(snr>=np.array([5.,10.,15.])).sum();qual=q>=req[g,bucket]-1e-9;valid=qual&(load<=budget+1e-9)
    ids=np.flatnonzero(valid);pareto=[]
    for i in ids:
        if not any(load[j]<=load[i]+1e-9 and q[j]>=q[i]-1e-9 and (load[j]<load[i]-1e-9 or q[j]>q[i]+1e-9) for j in ids if j!=i):pareto.append(int(i))
    capacity=np.minimum(10,np.floor((budget+1e-9)/load[ids])).astype(int)
    return dict(quality_threshold=float(req[g,bucket]),quality_feasible=int(qual.sum()),budget_feasible=int(valid.sum()),pareto_modes=pareto,
        capacity_range=[int(capacity.min()),int(capacity.max())] if len(capacity) else None,
        predicted_psnr_range=[float(q[ids].min()),float(q[ids].max())] if len(ids) else None)
example=[]
for snr in [0.,5.,10.,15.,20.]:
    q,load=tables(np.array(snr))
    for g in range(3):example.append(dict(snr_db=snr,instruction=g,budget=20000.,**options(q,load,g,20000.,snr)))
# Regenerate only exogenous channels; action-dependent state is not inferred.
channels={};initial={}
for seed in manifest['evaluation_seeds']:
    env.seed(seed);env.reset();initial[seed]=[x.tobytes() for x in [env.pos_uav,env.q_cache,env.tau_cache]]
    gamma=[]
    for t in range(600):
        gamma.append(env.gamma_bh.copy());env._update_channels()
    channels[seed]=np.asarray(gamma)
all_results={}
for method in manifest['methods']:
    logs=json.loads((RUN/'evaluation'/method/'summary.json').read_text())['episodes'];parts={g:[] for g in range(3)};proposal_counts={g:{} for g in range(3)};selected_counts={g:{} for g in range(3)}
    for entry in logs:
        seed=entry['seed'];path=RUN/'evaluation'/method/f"{entry['scenario']}_{seed}.csv"
        assert hashlib.sha256(path.read_bytes()).hexdigest()==entry['trace_sha256']
        rows=list(csv.DictReader(path.open()));gamma=channels[seed];db=10*np.log10(gamma);q,load=tables(db)
        tape=hashlib.sha256()
        for x in initial[seed]:tape.update(x)
        for t,row in enumerate(rows):
            g=int(row['instruction_id']);tape.update(gamma[t].tobytes());tape.update(np.asarray([g]).tobytes())
            beta=np.asarray(json.loads(row['fractions']));budget=env.backhaul_availability*env.delta_T*np.minimum(env.B_uav_sut,beta*env.B_sut_sat)
            buckets=(db[t,:,None]>=np.array([5.,10.,15.])).sum(-1)
            qual=q[t]>=req[g,buckets,None]-1e-9;valid=qual&(load[t]<=budget[:,None]+1e-9)
            parts[g].append(np.stack([qual.sum(-1),valid.sum(-1)],axis=-1))
            for field,target in [('proposed_modes',proposal_counts),('modes',selected_counts)]:
                for m in json.loads(row[field]):target[g][str(m)]=target[g].get(str(m),0)+1
        assert tape.hexdigest()==entry['external_sha256'],'External channel mismatch'
    stats={}
    for g in range(3):
        a=np.concatenate(parts[g]);stats[str(g)]={}
        for i,label in enumerate(['quality_feasible','budget_feasible']):
            x=a[:,i];stats[str(g)][label]=dict(min=int(x.min()),median=float(np.median(x)),max=int(x.max()),mean=float(x.mean()),zero_fraction=float((x==0).mean()),at_least_two_fraction=float((x>=2).mean()))
        stats[str(g)]['proposals']=proposal_counts[g];stats[str(g)]['executed']=selected_counts[g]
    all_results[method]=stats
out=dict(examples_equal_allocation_full_cache=example,evaluation_channel_and_actual_budget_counts=all_results,
    caveat='Budget feasible counts assume a pending update. Full-cache capacity assumes 10 queued updates, not actual per-slot queue length. Pareto means lower load/higher predicted quality, not long-horizon optimality.',
    paired_external_channel_hashes_verified=True)
(RUN/'analysis/choice_space_audit.json').write_text(json.dumps(out,indent=2)+'\n')
for x in example: print(x)
print(json.dumps(all_results,indent=2))
