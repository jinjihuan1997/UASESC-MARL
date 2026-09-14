from study import *
from kl_control import summarize_kl

def main():
 m=verify();checks={name:read(HERE/f'preflight/{name}.json') for name in ['foundation','teacher_reproduction','physical_smoke','resume']}
 assert all(x['state']=='PASS' for x in checks.values());assert checks['teacher_reproduction']['episodes']==260
 values=torch.zeros(4000,dtype=torch.float64);gids=torch.zeros(4000,dtype=torch.int64);gids[3900:3964]=1;gids[3964:]=2
 values[gids==2]=.02;assert not summarize_kl(values,gids)['exceeds']
 values[gids==1]=.02;assert summarize_kl(values,gids)['exceeds']
 assert not summarize_kl(values,gids,limit=float('inf'))['exceeds']
 resource=read(HERE/'resource_settings.json');assert resource['fit_device']=='cuda:0'
 costs=[]
 for p in (HERE/'costs').glob('*.jsonl'):costs.extend(json.loads(x) for x in p.read_text().splitlines())
 assert all(not x.get('error') for x in costs)
 result=dict(state='PASS',checks=checks,KL_group_minimum64_test='PASS',infinite_limit_test='PASS',resources=resource,physical_steps=sum(x.get('physical_steps',0) for x in costs),preflight_optimizer_steps=sum(x.get('gradient_steps',0) for x in costs),protected_files=len(m['protected_sha256']),teacher_state_frozen=True,stage_A_authorized=True,stage_B_requires_all_three_gate=True)
 write(HERE/'preflight.json',result)
 files=['study.py','student_policy.py','teacher_interface.py','trajectory.py','collect_demos.py','train_imitation.py','run_dagger.py','happo_kernel.py','kl_control.py','checkpointing.py']
 write(HERE/'execution_seal.json',dict(stage='A_collection_and_fitting',manifest_sha256=sha(HERE/'manifest.json'),source_sha256={f:sha(HERE/f) for f in files},preflight_sha256=sha(HERE/'preflight.json')))
 print(dict(state='PASS',physical_steps=result['physical_steps'],prototype_optimizer_steps=result['preflight_optimizer_steps'],fit_device=resource['fit_device']))
if __name__=='__main__':guard();main()
