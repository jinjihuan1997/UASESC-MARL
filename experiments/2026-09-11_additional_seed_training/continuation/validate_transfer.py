"""Gate continuation on inherited models, physical state and trainable actors."""
from helpers import *
from selector_train import ContinuationTrainer

def main():
    assert len(SEEDS)==1;seed=SEEDS[0];models=[]
    for arm in METHODS:
        cfg=config(arm,seed);t=ContinuationTrainer(cfg,'cpu')
        t.initialize_transfer(dict(validation='supplementary_seed_transfer'))
        assert t.total_updates==500 and t.env.step_index==400
        models.append(dict(arm=arm,actors=t.initial_actors,critic=t.initial_critic,
            environment=external_hashes(t.env),initial_selector=getattr(t,'initial_selector',None)))
        t.set_training_update(1);t.collect()
        if t.selector:
            b=t.buffer;n=t.batch;active=torch.ones((n,1))
            logp,_,_=t.selector.evaluate_actions(b.obs[:-1,:,0].flatten(0,1),b.flat_rnn,
                b.selector_actions.flatten(0,1),b.masks[:-1].flatten(0,1),torch.ones((n,2)),active)
            np.testing.assert_allclose(arr(logp),arr(b.selector_log_probs.flatten(0,1)),atol=1e-6,rtol=0)
        metrics=t.update()
        assert all(torch.isfinite(v).all() for v in metrics.values())
        if t.selector:
            assert [network_hash(a.actor) for a in t.actors]==t.initial_actors
            assert network_hash(t.selector.actor)!=t.initial_selector
        del t
    for key in ['actors','critic','environment']:assert models[0][key]==models[1][key]
    write(HERE/'preflight_results.json',dict(state='PASS',seed=seed,transferred_initial_models=models,
        matched_pretrained_actors_and_critic=True,matched_inherited_environment=True,
        selector_log_likelihood_verified=True,frozen_controllers_and_trainable_selector_verified=True,
        finite_actual_update_both_arms=True,smoke_physical_steps=8000,
        smoke_results_used_for_model_selection=False))
    print('Supplementary transfer PASS',seed,flush=True)

if __name__=='__main__':main()
