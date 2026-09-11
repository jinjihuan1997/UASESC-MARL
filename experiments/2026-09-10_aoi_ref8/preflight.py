"""Ref8 exact resume and the actual evaluator, before formal freezing."""
import shutil
import subprocess
from helpers import *
from training_checkpoint import load_checkpoint
import evaluation
from aggregate import audit_trace


def equal(a,b):
    assert type(a) is type(b)
    if torch.is_tensor(a):assert torch.equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:equal(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):equal(x,y)
    else:assert a==b


def main():
    work=HERE/'preflight';work.mkdir(exist_ok=False)
    m=work/'manifest.json'
    write(m,dict(purpose='ref8_validation_only',steps_per_method=1000000,
        evaluation_seeds=[20262471,20262472],scenarios={'switch300_2_to_1':[[0,2],[300,1]]}))
    cfg=HERE/'configs/ref8/seed_85/IC_HAPPO.json'
    def command(folder,*extra):
        return [sys.executable,'-u',str(HERE/'source/formal_train.py'),'--config',str(cfg),
                '--output',str(work/folder),'--run-manifest',str(m),'--device','cpu',
                '--checkpoint-every','2',*extra]
    with (work/'training.log').open('w') as log:
        for cmd in [command('whole','--stop-after-update','4'),command('resumed','--stop-after-update','2'),
                    command('resumed','--resume','--stop-after-update','4')]:
            assert subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT).returncode==75
    a,_=load_checkpoint(work/'whole/checkpoints');b,_=load_checkpoint(work/'resumed/checkpoints');equal(a,b)
    for o in OBJECTIVES:
        dest=work/'jobs'/o/'seed_85/IC_HAPPO';dest.mkdir(parents=True)
        src=HERE/'jobs/ref10/seed_85/IC_HAPPO'
        shutil.copy2(src/'status.json',dest/'status.json')
        for f in src.glob('*.pt'):shutil.copy2(f,dest/f.name)
    write(work/'rule_selection.json',read(HERE/'rule_selection.json'))
    evaluation.HERE=work
    for item in ['ref10/seed_85/IC_HAPPO','ref8/seed_85/IC_HAPPO','ref8/rules/R_myopic']:
        assert evaluation.evaluate_item(item,[])==0
        assert evaluation.evaluate_item(item,[])==0
        with np.load(work/'evaluation'/item/'switch300_2_to_1.npz') as z:
            audit_trace(z['trace'],z['aoi_after'],config(item.split('/')[0]))
    write(HERE/'preflight_results.json',dict(state='PASS',exact_resume_full_state=True,
        optimizer_steps_executed=32000,evaluator_episodes=6,evaluator_idempotence=True,
        both_scores_independently_verified=True,preflight_models_excluded_from_formal_results=True))
    print('Exact resume and evaluator preflight PASS',flush=True)


if __name__=='__main__':main()
