"""Exact CPU checkpoint continuation through one further PPO update."""
import tempfile
from pathlib import Path
import torch
from common import ROOT,configuration,write,network_hash
from tensor_train import TensorTrainer
from training_checkpoint import capture,restore,save_checkpoint,load_checkpoint

def advance(t):
    t.collect();m=t.update();t.buffer.after_update()
    assert all(torch.isfinite(v).all() for v in m.values())
    return [network_hash(a.actor) for a in t.actors],network_hash(t.critic.critic)

def main():
    results=[]
    for name in ['IC_HAPPO','HAPPO_hidden_instruction']:
        cfg=configuration(name,85,8000);a=TensorTrainer(cfg,'cpu');advance(a)
        ident={'test':name}
        with tempfile.TemporaryDirectory(prefix='instruction-priority-checkpoint-') as tmp:
            save_checkpoint(Path(tmp),capture(a,1,ident))
            expected=advance(a)
            b=TensorTrainer(cfg,'cpu');state,_=load_checkpoint(Path(tmp));restore(b,state,ident)
            actual=advance(b);assert actual==expected
        results.append({'method':name,'exact_continuation':True,'steps_checked':8000})
    write(ROOT/'resume_checks.json',{'passed':True,'results':results})
    print(results)

if __name__=='__main__':main()
