"""Atomic, immutable inference snapshots at predeclared training milestones."""
import json
import uuid
from pathlib import Path
import torch
from common import write,stamp,network_hash
from training_checkpoint import digest,cpu_tree

def export_snapshot(trainer,output,update,identity):
    steps=update*trainer.batch
    target=Path(output)/'milestones'/f'steps_{steps}'
    actor_hashes=[network_hash(a.actor) for a in trainer.actors]
    critic_hash=network_hash(trainer.critic.critic)
    if target.exists():
        status=json.loads((target/'status.json').read_text())
        assert status['state']=='complete' and status['completed_steps']==steps and status['identity']==identity
        assert status['actor_hashes']==actor_hashes and status['critic_hash']==critic_hash
        for f,h in status['checkpoint_hashes'].items(): assert digest(target/f)==h
        return
    target.parent.mkdir(parents=True,exist_ok=True)
    temporary=target.parent/f'.steps_{steps}_{uuid.uuid4().hex}.tmp'
    temporary.mkdir()
    for i,actor in enumerate(trainer.actors): torch.save(cpu_tree(actor.actor.state_dict()),temporary/f'actor_agent{i}.pt')
    torch.save(cpu_tree(trainer.critic.critic.state_dict()),temporary/'critic_agent.pt')
    torch.save(cpu_tree(trainer.normalizer.state_dict()),temporary/'value_normalizer.pt')
    for f in temporary.glob('*.pt'):
        assert all(torch.isfinite(v).all() for v in torch.load(f,map_location='cpu',weights_only=True).values())
    write(temporary/'status.json',dict(state='complete',artifact='inference_snapshot',completed_steps=steps,
        update=update,identity=identity,created_utc=stamp(),actor_hashes=actor_hashes,critic_hash=critic_hash,
        checkpoint_hashes={f.name:digest(f) for f in temporary.glob('*.pt')}))
    temporary.rename(target)
