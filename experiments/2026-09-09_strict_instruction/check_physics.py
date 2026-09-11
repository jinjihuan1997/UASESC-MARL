from pathlib import Path
import argparse,json,hashlib,subprocess,sys,tempfile,ast
from common import ROOT,write,stamp

def main():
    parent=Path(json.loads((ROOT/'parent_evidence.json').read_text())['parent_run'])/'source'
    # Same supplied actions must induce bit-identical physical/reward traces.
    with tempfile.TemporaryDirectory(prefix='strict-physics-') as tmp:
        traces=[]
        for i,version in enumerate([parent,ROOT]):
            out=Path(tmp)/f'{i}.json'
            subprocess.run([sys.executable,str(ROOT/'physics_trace.py'),'--version',str(version),'--output',str(out)],check=True)
            traces.append(json.loads(out.read_text())['records'])
        assert traces[0]==traces[1]
        digest=hashlib.sha256(json.dumps(traces[0],sort_keys=True).encode()).hexdigest()
    unchanged={}
    for name in ['reference/runtime/harl/envs/uav_escs/SC/uav_escs_env_sc.py','tensor_env.py']:
        code=[]
        for base in [parent,ROOT]:
            text=(base/name).read_text();tree=ast.parse(text)
            code.append({f.name:ast.dump(f,include_attributes=False) for cls in tree.body if isinstance(cls,ast.ClassDef) for f in cls.body if isinstance(f,ast.FunctionDef)})
        changed=[k for k in code[0] if code[0][k]!=code[1][k]]
        assert set(changed)==({'_build_sut_obs','_build_action_available_masks'} if name.startswith('reference') else {'observe'}),changed
        unchanged[name]={'only_changed_methods':changed,'step_unchanged':code[0]['step']==code[1]['step']}
    write(ROOT/'physics_preservation.json',dict(passed=True,utc=stamp(),slots_per_version=len(traces[0]),physical_reward_trace_sha256=digest,ast_comparison=unchanged))
    print('PASS:',len(traces[0]),'identical physical/reward slots; only actor interface changed')
if __name__=='__main__':main()
