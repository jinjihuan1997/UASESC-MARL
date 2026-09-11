"""Freeze selected resources, all six configurations and input hashes."""
import ast
from helpers import *
from training_checkpoint import runtime_signature


if __name__=='__main__':
    assert not (HERE/'manifest.json').exists()
    benchmark=read(HERE/'benchmark_results.json');assert benchmark['state']=='PASS'
    layout=benchmark['selected'];jobs=[]
    for core,(seed,method) in enumerate((s,m) for s in [85,218,966] for m in ['IC_HAPPO','HAPPO_hidden_instruction']):
        device='cuda:0' if layout=='cpu4_gpu2' and seed==966 else 'cpu'
        item=f'seed_{seed}/{method}';path=HERE/'configs'/f'{item}.json';c=read(path)
        c['algo_args']['device'].update(cuda=device!='cpu',cuda_deterministic=device!='cpu')
        write(path,c)
        jobs.append(dict(id=item,seed=seed,method=method,device=device,core=core,config=str(path.relative_to(HERE)),output='jobs/'+item))
    for seed in [85,218,966]:
        a=read(HERE/'configs'/f'seed_{seed}/IC_HAPPO.json');b=read(HERE/'configs'/f'seed_{seed}/HAPPO_hidden_instruction.json')
        assert a['algo_args']==b['algo_args']
        assert [k for k,v in a['env_args'].items() if v!=b['env_args'][k]]==['actor_observe_instruction']
        for root in ['cpu6','cpu4_gpu2']:
            folder=HERE/'benchmarks'/root/'jobs'/f'seed_{seed}'
            x=read(folder/'IC_HAPPO/manifest.json');y=read(folder/'HAPPO_hidden_instruction/manifest.json')
            assert x['initial_actor_hashes']==y['initial_actor_hashes'] and x['initial_critic_hash']==y['initial_critic_hash']
    for p in HERE.glob('*.py'):ast.parse(p.read_text())
    files=list(HERE.glob('*.py'))+[HERE/n for n in ['PROTOCOL.md','provenance.json','rule_selection.json','benchmark_results.json','validation.json']]
    for folder in [HERE/'source',HERE/'configs']:files.extend(p for p in folder.rglob('*') if p.is_file())
    original=read(HERE/'source/parent_provenance.json')
    # The scenario definitions are inherited from the fixed reference protocol.
    from protocol import SCENARIOS
    m=dict(schema=1,created_utc=stamp(),purpose='preference_only_formal_three_seed_10m',seeds=[85,218,966],
        methods=['IC_HAPPO','HAPPO_hidden_instruction'],rules=['R_single','R_instruction','R_myopic'],jobs=jobs,
        steps_per_method=10_000_000,total_training_steps=60_000_000,total_training_jobs=6,batch=4000,checkpoint_every_updates=25,
        runtime=runtime_signature(),evaluation_seeds=list(range(20261701,20261721)),scenarios=SCENARIOS,
        evaluation_episodes=2340,resources=dict(layout=layout,parallel_jobs=6,cores=list(range(6)),evaluation_device='cpu',
            estimated_training_seconds=benchmark['layouts'][layout]['predicted_training_seconds']),
        input_hashes={str(p.relative_to(HERE)):digest(p) for p in sorted(set(files))})
    assert len(m['scenarios'])==13
    write(HERE/'manifest.json',m);write(HERE/'status.json',dict(state='prepared',total_training_jobs=6,total_training_steps=60_000_000,updated_utc=stamp()))
    print('Frozen:',layout,'6 jobs, 60M steps, 2340 final evaluation episodes')
