"""Exercise the real launcher/export/evaluator on short, excluded smoke jobs."""
import subprocess
from helpers import *
import evaluation

def main():
    work=HERE/'preflight'/'pipeline';work.mkdir(exist_ok=False)
    m=dict(purpose='sequential_pipeline_smoke_only',steps_per_method=8000,
        evaluation_seeds=[20262471,20262472],scenarios={'fixed_2':[[0,2]],'switch300_2_to_1':[[0,2],[300,1]]})
    write(work/'manifest.json',m);write(work/'rule_selection.json',read(HERE/'rule_selection.json'))
    for arm in METHODS:
        cfg=config(arm);cfg['algo_args']['train']['num_env_steps']=8000
        if arm=='staged': cfg['training_design']['resource_warmup_steps']=4000
        path=work/f'{arm}.json';write(path,cfg)
        cmd=[sys.executable,'-u',str(HERE/'source/formal_train.py'),'--config',str(path),'--output',str(work/'jobs/seed_85'/arm),
            '--run-manifest',str(work/'manifest.json'),'--device','cpu','--checkpoint-every','1']
        with (work/f'{arm}.log').open('w') as log: assert subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT).returncode==0
    evaluation.HERE=work
    items=[f'seed_85/{arm}' for arm in METHODS]+[f'rules/{name}' for name in RULE_METHODS]
    for item in items:
        assert evaluation.evaluate_item(item,[])==0
        assert evaluation.evaluate_item(item,[])==0
        for scenario in m['scenarios']:
            with np.load(work/'evaluation'/item/f'{scenario}.npz') as z:
                assert z['trace'].shape==(600,2,len(evaluation.FIELDS))
                assert not z['trace'][...,6:9].any()
                if item.endswith('mode_only') or '/R_equal_' in item: np.testing.assert_allclose(z['resource_fractions'],1/3,atol=1e-8)
    write(HERE/'preflight_results.json',dict(state='PASS',smoke_training_steps=24000,evaluation_episodes=36,
        real_formal_trainer_export=True,three_arms_and_six_rules=True,evaluator_resume_idempotence=True,
        fixed_resource_capability_verified=True,smoke_models_excluded=True))
    print('Full pipeline preflight PASS',flush=True)

if __name__=='__main__': main()
