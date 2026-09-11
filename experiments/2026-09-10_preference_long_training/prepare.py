"""Snapshot validated sources and create a separate long-run configuration."""
import ast
import hashlib
import json
from pathlib import Path
import shutil

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parent.parent
PILOT=PROJECT/'experiments/2026-09-10_preference_only'
PARENT=PROJECT/'experiments/2026-09-09_instruction_long_training/runs/three_seed_sc_20260909'


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n')


def functions(path,names):
    text=path.read_text();lines=text.splitlines(keepends=True);result=[]
    for node in ast.parse(text).body:
        if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name in names:
            start=min([node.lineno]+[d.lineno for d in node.decorator_list])
            result.append(''.join(lines[start-1:node.end_lineno]))
    assert len(result)==len(names)
    return '\n\n'.join(result)+'\n'


if __name__=='__main__':
    assert not (HERE/'source').exists(),'Preserve prepared experiment'
    assert read(PILOT/'independent_audit.json')['state']=='PASS'
    for f,h in read(PILOT/'artifact_hashes.json').items():assert sha(PILOT/f)==h,f
    for f,h in read(PARENT/'manifest.json')['input_hashes'].items():assert sha(PARENT/f)==h,f
    shutil.copytree(PARENT/'source',HERE/'source',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    prior=(PILOT/'probe_common.py').read_text()
    head=prior.split('spec = importlib.util.spec_from_file_location')[0]
    head=head.replace("PARENT = PROJECT/'experiments/2026-09-09_instruction_long_training/runs/three_seed_sc_20260909'",'PARENT = HERE')
    tail=prior[prior.index('RULES = '):]
    (HERE/'helpers.py').write_text(head+'from metrics import arr, check_tensor, explicit_columns, actions_for\nfrom evaluate import load_actors\n'+tail)
    path=PROJECT/'experiments/2026-09-10_instruction_closed_loop/evaluate.py'
    (HERE/'metrics.py').write_text('import numpy as np\nimport torch\n'+functions(path,{'arr','check_tensor','explicit_columns','actions_for'}))
    fields="FIELDS=['common_reward','mean_aoi','predicted_quality_sum','deliveries','channel_uses','instruction_id','quality_violations','budget_violations','cache_violations']\n"
    (HERE/'rule_tools.py').write_text('from helpers import *\n'+fields+functions(PILOT/'pilot_evaluate.py',{'predict_reward','myopic_actions','summarize'}))
    shutil.copy2(PILOT/'gate_selection.json',HERE/'rule_selection.json')
    for seed in [85,218,966]:
        for method in ['IC_HAPPO','HAPPO_hidden_instruction']:
            c=read(PILOT/'configs'/f'{method}.json')
            c['algo_args']['seed']['seed']=seed
            c['algo_args']['train']['num_env_steps']=10_000_000
            c['env_args'].update(semantic_registry_path=str(HERE/'source/reference/inputs/mode_registry.json'),semantic_profile_path=str(HERE/'source/reference/inputs/profile.npz'))
            write(HERE/'configs'/f'seed_{seed}'/f'{method}.json',c)
    for p in HERE.glob('*.py'):ast.parse(p.read_text())
    write(HERE/'provenance.json',dict(parent_manifest_sha256=sha(PARENT/'manifest.json'),pilot_audit_sha256=sha(PILOT/'independent_audit.json'),
        pilot_results_sha256=sha(PILOT/'pilot_results.json'),rule_selection_sha256=sha(HERE/'rule_selection.json'),
        source_snapshot_origin=str(PARENT/'source'),helper_origins={str(p):sha(p) for p in [PILOT/'probe_common.py',PILOT/'pilot_evaluate.py',path]}))
    print('Prepared six 10M configurations; validated sources snapshotted.')
