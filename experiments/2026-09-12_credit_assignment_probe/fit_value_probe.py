"""Fit only new offline diagnostic regressors. No original optimizer is used."""
from support import *
import argparse


class Probe(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.layers = torch.nn.Sequential(torch.nn.Linear(dim, 256), torch.nn.ReLU(),
            torch.nn.Linear(256, 256), torch.nn.ReLU(), torch.nn.Linear(256, 1))

    def forward(self, x): return self.layers(x).squeeze(-1)


def collect_dataset(parent, manifest):
    batches = []
    gamma = manifest['gamma_by_model'][str(parent)]
    for action in manifest['probe_action_seeds']:
        folder = HERE / 'probe' / f'seed_{parent}' / f'action{action}'
        meta = read(folder / 'complete.json'); assert sha(folder / 'trajectory.npz') == meta['trace_sha256']
        with np.load(folder / 'trajectory.npz') as f:
            data = {k: f[k] for k in f.files}
        assert data['done'].shape == (600, 12)
        assert not data['done'][:-1].any() and data['done'][-1].all()
        r = data['training_reward'].astype(np.float64)
        G = np.zeros_like(r); future = np.zeros(12, dtype=np.float64)
        for t in range(599, -1, -1):
            future = r[t] + gamma * future * (~data['done'][t]); G[t] = future
        np.testing.assert_array_equal(G[-1], r[-1])
        np.testing.assert_allclose(G[:-1], r[:-1] + gamma * G[1:], atol=1e-12, rtol=0)
        item = dict(state=data['critic_input_preallocation'], shares=data['resource_fractions'], target=G,
                    instruction=data['trace'][..., FIELDS.index('instruction_id')].astype(np.int8),
                    probabilities=data['mode_probabilities'], sampled_modes=data['sampled_modes'],
                    env_seed=np.broadcast_to(data['env_seeds'][None], (600,12)),
                    action_seed=np.full((600,12), action), remaining=np.broadcast_to(np.arange(600,0,-1)[:,None], (600,12)))
        batches.append({k: v.transpose(1,0,*range(2,v.ndim)).reshape(7200,*v.shape[2:]) for k,v in item.items()})
    data = {k: np.concatenate([b[k] for b in batches]) for k in batches[0]}
    fit = np.isin(data['env_seed'], manifest['probe_fit_seeds'])
    held = np.isin(data['env_seed'], manifest['probe_holdout_seeds'])
    assert np.all(fit ^ held) and fit.sum() == 19200 and held.sum() == 9600
    data['fit_mask'] = fit
    return data


def prediction_metrics(y, p):
    residual = y-p
    variance = float(np.var(y))
    return dict(n_slots=len(y), mse=float(np.mean(residual**2)), mae=float(np.mean(np.abs(residual))),
        explained_variance=float(1-np.var(residual)/variance) if variance else None,
        residual_mean=float(residual.mean()), residual_variance=float(np.var(residual)))


def bootstrap_prediction(y, p0, p1, env_seed):
    seeds = np.unique(env_seed)
    rng = np.random.default_rng(202609120911)
    draws = rng.integers(0,len(seeds),(2000,len(seeds)))
    records = []
    for s in seeds:
        ix = env_seed == s; r0 = y[ix]-p0[ix]; r1=y[ix]-p1[ix]
        records.append([np.mean(r1*r1)-np.mean(r0*r0),
                        np.mean(np.abs(r1))-np.mean(np.abs(r0)),
                        r0.mean(), (r0*r0).mean(), r1.mean(), (r1*r1).mean()])
    means = np.asarray(records)[draws].mean(1)
    var_diff=(means[:,5]-means[:,4]**2)-(means[:,3]-means[:,2]**2)
    return dict(cluster='held-out environment seed; all four action repeats and 600 slots kept together',
                clusters=len(seeds), bootstrap_samples=2000,
                mse_P1_minus_P0_ci95=np.quantile(means[:,0],[.025,.975]).tolist(),
                mae_P1_minus_P0_ci95=np.quantile(means[:,1],[.025,.975]).tolist(),
                residual_variance_P1_minus_P0_ci95=np.quantile(var_diff,[.025,.975]).tolist())


def logit_proxy(data, target, predictions):
    # Exact categorical score w.r.t. the 16 masked logits, not actor parameters.
    # The sampled mode is used even where environment fallback executes another ID.
    probability = data['probabilities'].astype(np.float64)
    score = np.eye(16)[data['sampled_modes']] - probability
    np.testing.assert_allclose(score.sum(-1), 0, atol=1e-6, rtol=0)
    residual = target-predictions
    gradients = score * residual[:,None,None]
    mean = gradients.mean(0)
    component_var = np.mean(gradients**2,0)-mean**2
    # Samples within a trajectory are correlated: also aggregate per 600-slot episode.
    episode_keys = sorted(set(zip(data['env_seed'].tolist(),data['action_seed'].tolist())))
    episode_means = np.stack([gradients[(data['env_seed']==s)&(data['action_seed']==a)].mean(0)
                              for s,a in episode_keys])
    return dict(kind='UAV categorical-logit score-gradient proxy; not full actor gradient or HAPPO update',
        baseline_detached=True, sampled_not_executed_mode=True, sut_not_evaluated=True,
        slot_gradient_variance_trace_per_uav=component_var.sum(-1).tolist(),
        episode_mean_gradient_variance_trace_per_uav=np.var(episode_means,axis=0).sum(-1).tolist(),
        episode_count=len(episode_keys), score_gradient_mean_norm_per_uav=np.linalg.norm(mean,axis=-1).tolist())


def grouped_errors(data, target, p):
    def group(indices):
        return prediction_metrics(target[indices],p[indices]) if indices.any() else None
    edges=np.asarray([0,.15,.25,.35,.50,.70,1.01]); shares=data['shares']
    return dict(by_instruction={str(g):group(data['instruction']==g) for g in range(3)},
        by_remaining_slots={label:group((data['remaining']>=low)&(data['remaining']<=high))
                            for label,low,high in [('1-100',1,100),('101-300',101,300),('301-600',301,600)]},
        by_uav_budget_share={str(u):{f'{edges[j]:.2f}-{edges[j+1]:.2f}':group((shares[:,u]>=edges[j])&(shares[:,u]<edges[j+1]))
                                        for j in range(len(edges)-1)} for u in range(3)})


def benchmark_fit(dim=205):
    result=[]
    for device in ['cpu'] + (['cuda:0'] if torch.cuda.is_available() else []):
        with torch.random.fork_rng():
            torch.manual_seed(112233)
            model=Probe(dim).to(device)
            x=torch.randn((1024,dim),device=device); y=torch.randn(1024,device=device)
            opt=torch.optim.Adam(model.parameters(),lr=1e-3)
            for _ in range(3):
                opt.zero_grad(set_to_none=True); ((model(x)-y)**2).mean().backward(); opt.step()
            if device!='cpu': torch.cuda.synchronize()
            start=time.monotonic()
            for _ in range(20):
                opt.zero_grad(set_to_none=True); ((model(x)-y)**2).mean().backward(); opt.step()
            if device!='cpu': torch.cuda.synchronize()
            result.append(dict(device=device, seconds=time.monotonic()-start, batches=20, rows_per_batch=1024))
    return result


def fit_pair(parent, initialization, data, m, device):
    out=HERE/'fits'/f'seed_{parent}'/f'init_{initialization}'; out.mkdir(parents=True,exist_ok=True)
    identity=dict(parent_seed=parent, initialization_seed=initialization, protocol_sha256=m['protocol_sha256'],
                  diagnostic_code_sha256=m['diagnostic_code_sha256'], dataset_numeric_sha256=digest_arrays(data),
                  device=device, epochs=50, batch_size=1024, learning_rate=1e-3)
    complete=out/'complete.json'
    if complete.exists():
        marker=read(complete); assert marker['identity']==identity
        for name,h in marker['output_hashes'].items(): assert sha(out/name)==h
        return read(out/'results.json')
    fit=data['fit_mask']; held=~fit
    mean=data['state'][fit].astype(np.float64).mean(0); std=data['state'][fit].astype(np.float64).std(0).clip(1e-6)
    bm=data['shares'][fit].mean(0); bs=data['shares'][fit].std(0).clip(1e-6)
    ym=float(data['target'][fit].mean()); ys=max(float(data['target'][fit].std()),1e-6)
    xs=((data['state']-mean)/std).astype(np.float32)
    xb=((data['shares']-bm)/bs).astype(np.float32)
    x0=np.concatenate([xs,np.zeros_like(xb)],1); x1=np.concatenate([xs,xb],1)
    assert x0.shape==x1.shape and np.array_equal(x0[:,:-3],x1[:,:-3])
    normal=dict(state_mean=mean,state_std=std,budget_mean=bm,budget_std=bs,target_mean=np.asarray(ym),target_std=np.asarray(ys))
    save_npz(out/'normalization.npz', normal)
    with torch.random.fork_rng():
        torch.manual_seed(initialization)
        models=[Probe(x0.shape[1]),Probe(x0.shape[1])]
        models[1].load_state_dict(models[0].state_dict())
    assert frozen.network_hash(models[0])==frozen.network_hash(models[1])
    init_hash=frozen.network_hash(models[0])
    models=[p.to(device) for p in models]
    opts=[torch.optim.Adam(p.parameters(),lr=1e-3) for p in models]
    x=[torch.as_tensor(z[fit],device=device) for z in (x0,x1)]
    target=torch.as_tensor(((data['target'][fit]-ym)/ys).astype(np.float32),device=device)
    generator=torch.Generator().manual_seed(initialization+1)
    losses=[]; epoch0=0; checkpoint=out/'resume.pt'
    if checkpoint.exists():
        saved=torch.load(checkpoint,map_location=device,weights_only=False)
        assert saved['identity']==identity
        for i in range(2): models[i].load_state_dict(saved['models'][i]); opts[i].load_state_dict(saved['optimizers'][i])
        generator.set_state(saved['batch_rng'].cpu()); epoch0=saved['epoch']; losses=saved['losses']
    start=time.monotonic()
    for epoch in range(epoch0,50):
        order=torch.randperm(len(target),generator=generator).to(device)
        total=[0.,0.]
        for indices in order.split(1024):
            for i in range(2):
                opts[i].zero_grad(set_to_none=True)
                loss=((models[i](x[i][indices])-target[indices])**2).mean()
                assert torch.isfinite(loss), 'Nonfinite probe loss; no automatic parameter changes'
                loss.backward(); opts[i].step()
                total[i]+=float(loss.detach())*len(indices)
        losses.append(dict(epoch=epoch+1,P0_normalized_mse=total[0]/len(target),P1_normalized_mse=total[1]/len(target)))
        if (epoch+1)%10==0:
            tmp=out/'resume.tmp'
            torch.save(dict(identity=identity,epoch=epoch+1,models=[p.state_dict() for p in models],
                optimizers=[o.state_dict() for o in opts],batch_rng=generator.get_state(),losses=losses),tmp)
            tmp.replace(checkpoint)
            write(out/'progress.json',dict(epoch=epoch+1,total_epochs=50,elapsed_seconds=time.monotonic()-start,updated_utc=stamp()))
            print(f'probe parent={parent} init={initialization} epoch={epoch+1}/50',flush=True)
    predictions=[]
    for model,values in zip(models,(x0,x1)):
        model.eval()
        with torch.no_grad():
            p=torch.cat([model(torch.as_tensor(z,device=device)).cpu() for z in np.array_split(values[held],10)]).numpy().astype(np.float64)*ys+ym
        predictions.append(p)
    target_held=data['target'][held]
    held_data={k:v[held] for k,v in data.items() if k!='fit_mask'}
    results=dict(parent_seed=parent,initialization_seed=initialization,epochs=50,device=device,
        parameter_count=sum(p.numel() for p in models[0].parameters()),matched_initialization_hash=init_hash,
        fit_environment_seeds=m['probe_fit_seeds'],holdout_environment_seeds=m['probe_holdout_seeds'],
        fit_slots=int(fit.sum()),holdout_slots=int(held.sum()),fit_episodes=32,holdout_episodes=16,
        gamma=m['gamma_by_model'][str(parent)],normalization_fit_only=True,
        metrics={name:prediction_metrics(target_held,p) for name,p in zip(['P0','P1'],predictions)},
        grouped={name:grouped_errors(held_data,target_held,p) for name,p in zip(['P0','P1'],predictions)},
        paired_cluster_bootstrap=bootstrap_prediction(target_held,*predictions,held_data['env_seed']),
        gradient_proxy={name:logit_proxy(held_data,target_held,p) for name,p in zip(['P0','P1'],predictions)},
        elapsed_fitting_seconds=time.monotonic()-start)
    for i,model in enumerate(models): torch.save(model.cpu().state_dict(),out/f'P{i}.pt')
    save_npz(out/'heldout_predictions.npz',dict(target=target_held,P0=predictions[0],P1=predictions[1],
         env_seed=held_data['env_seed'],action_seed=held_data['action_seed'],instruction=held_data['instruction'],
         remaining=held_data['remaining'],shares=held_data['shares']))
    write(out/'training_losses.json',losses); write(out/'results.json',results)
    output_files=['P0.pt','P1.pt','normalization.npz','heldout_predictions.npz','training_losses.json','results.json']
    write(complete,dict(state='complete',identity=identity,output_hashes={p:sha(out/p) for p in output_files},updated_utc=stamp()))
    return results


def main():
    m=verify_inputs(); pre=read(HERE/'preflight.json'); assert pre['state']=='PASS'
    assert read(HERE/'probe/complete.json')['episodes']==144
    count=0
    for parent in m['training_seeds']:
        dataset=collect_dataset(parent,m)
        for initialization in m['fit_initialization_seeds']:
            fit_pair(parent,initialization,dataset,m,pre['fitting_device'])
            count+=2
            save_status('probe_fitting',completed_execution_episodes=2280,completed_probe_episodes=144,completed_probe_fits=count,total_probe_fits=18)
    verify_inputs()
    write(HERE/'fits/complete.json',dict(state='complete',fits=count,updated_utc=stamp()))


if __name__=='__main__': main()
