"""Reload the saved offline probes and independently reproduce held-out predictions."""
from support import *
from fit_value_probe import collect_dataset, Probe
import importlib.util
spec=importlib.util.spec_from_file_location('credit_probe_aggregation',HERE/'aggregate.py')
aggregation=importlib.util.module_from_spec(spec);spec.loader.exec_module(aggregation)
profile_groups=aggregation.profile_groups


def main():
    m=verify_inputs(full=True); device=read(HERE/'preflight.json')['fitting_device']
    checks=[]
    for parent in m['training_seeds']:
        data=collect_dataset(parent,m); held=~data['fit_mask']
        # Independently check a full Monte Carlo return against explicit powers.
        rawfile=HERE/'probe'/f'seed_{parent}'/f"action{m['probe_action_seeds'][0]}"/'trajectory.npz'
        with np.load(rawfile) as z:
            r=z['training_reward'][:,0].astype(np.float64)
        expected=np.dot(np.power(m['gamma_by_model'][str(parent)],np.arange(600)),r)
        assert abs(expected-data['target'][0])<1e-10
        for initialization in m['fit_initialization_seeds']:
            folder=HERE/'fits'/f'seed_{parent}'/f'init_{initialization}'
            with np.load(folder/'normalization.npz') as z: normal={k:z[k] for k in z.files}
            np.testing.assert_allclose(normal['state_mean'],data['state'][~held].astype(np.float64).mean(0),atol=0,rtol=0)
            np.testing.assert_allclose(normal['budget_mean'],data['shares'][~held].mean(0),atol=0,rtol=0)
            np.testing.assert_allclose(normal['target_mean'],data['target'][~held].mean(),atol=0,rtol=0)
            state=((data['state'][held]-normal['state_mean'])/normal['state_std']).astype(np.float32)
            budget=((data['shares'][held]-normal['budget_mean'])/normal['budget_std']).astype(np.float32)
            with np.load(folder/'heldout_predictions.npz') as z: saved={k:z[k] for k in z.files}
            np.testing.assert_array_equal(saved['target'],data['target'][held])
            errors={}
            for name,b in [('P0',np.zeros_like(budget)),('P1',budget)]:
                x=np.concatenate([state,b],1)
                model=Probe(x.shape[1]).to(device)
                model.load_state_dict(torch.load(folder/f'{name}.pt',map_location=device,weights_only=True))
                model.eval(); model.requires_grad_(False)
                with torch.no_grad():
                    p=torch.cat([model(torch.as_tensor(chunk,device=device)).cpu() for chunk in np.array_split(x,10)]).numpy().astype(np.float64)
                p=p*float(normal['target_std'])+float(normal['target_mean'])
                errors[name]=float(np.max(np.abs(p-saved[name])))
                np.testing.assert_array_equal(p,saved[name])
            checks.append(dict(parent_seed=parent,initialization_seed=initialization,loaded_probe_prediction_max_abs_difference=errors))
    # Grouping is also checked against the original loader's in-memory aliases.
    env,_,_=make_env(m['training_seeds'][0],m['preflight_seeds'],'fixed_0','cpu')
    profile=env.p.semantic_library.profile; groups,_=profile_groups()
    rows=[np.concatenate([profile[k][i] for k in ('q_hat_mean','l_z_mean','n_z_mean')]) for i in range(16)]
    for i in range(16):
        for j in range(16):
            assert np.array_equal(rows[i],rows[j])==any(i in g and j in g for g in groups)
    verify_inputs(full=True)
    result=dict(state='PASS',probe_networks_reloaded=18,heldout_prediction_replay='bitwise identical on original fitting device',
                device=device,full_return_direct_sum_verified_parent_models=3,normalization_fit_only_verified=True,
                physical_equivalence_matches_original_runtime_loader=True,checks=checks,
                new_environment_steps=0,rl_training_steps=0,verification_script_sha256=sha(Path(__file__)))
    write(HERE/'independent_verification.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='checks'}),flush=True)


if __name__=='__main__':main()
