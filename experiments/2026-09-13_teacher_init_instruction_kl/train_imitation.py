"""Fixed-epoch Dirichlet-distribution supervision and masked 16-class CE."""
from study import *
from student_policy import Student,sut_loss,physical_shares,target_alpha
import argparse

def training_files(student,stage):
 out=sorted((HERE/'data/A0').glob('batch_*.npz'))
 assert len(out)==6
 for r in range(1,int(stage[1:])+1):
  items=sorted((HERE/f'data/{student}/A{r}').glob('batch_*.npz'));assert len(items)==4;out+=items
 return out

def train(student_seed,stage,device='cuda:0'):
 m=verify();assert read(HERE/'preflight.json')['state']=='PASS';assert student_seed in m['students']
 assert stage in ('A0','A1','A2','A3');target_epochs=20 if stage=='A0' else 10
 files=training_files(student_seed,stage);hashes={str(p.relative_to(HERE)):sha(p) for p in files}
 env=make_env(m['preflight'][:1],'preflight');student=Student(student_seed,env,device)
 student.set_mode(True)
 optim=[torch.optim.Adam(net.parameters(),lr=.001,eps=1e-8) for net in student.actors]
 rng=torch.Generator().manual_seed(m['seeds'][str(student_seed)]['fit'])
 folder=HERE/f'students/{student_seed}/{stage}';checkpoint=folder/'checkpoint.pt';resume=folder/'resume.pt'
 if checkpoint.exists():
  d=student.load(checkpoint);assert d['stage']==stage and d['completed_epochs']==target_epochs and d['data_sha256']==hashes
  return
 completed=0;counts=[0]*4
 if resume.exists():
  d=student.load(resume);assert d['data_sha256']==hashes and d['stage']==stage
  completed=d['completed_epochs'];counts=d['optimizer_steps']
 elif stage!='A0':d=student.load(HERE/f'students/{student_seed}/A{int(stage[1:])-1}/checkpoint.pt');counts=d['optimizer_steps']
 else:d=None
 if d:
  for o,state in zip(optim,d['optimizers']):o.load_state_dict(state)
  rng.set_state(d['shuffle_rng'].cpu())
 student.set_mode(True)
 buffers={k:[] for k in ('sut_obs','post_uav_obs','uav_masks','teacher_resource','teacher_effective','teacher_active')}
 for f in files:
  meta=read(f.with_suffix('.json'));assert meta['state']=='complete' and sha(f)==meta['trace_sha256']
  assert set(meta['identity']['seeds'])<=set(m['allowed_environment_seeds'][meta['identity']['purpose']])
  with np.load(f,allow_pickle=False) as z:
   for k in buffers:buffers[k].append(z[k].reshape(-1,*z[k].shape[2:]))
 data={k:torch.as_tensor(np.concatenate(v),device=device) for k,v in buffers.items()}
 n=len(data['sut_obs']);assert n==72000+48000*int(stage[1:])
 initial_counts=counts.copy();started=time.perf_counter()
 for epoch in range(completed,target_epochs):
  order=torch.randperm(n,generator=rng);sums=np.zeros((4,4));valid_counts=[0]*4;epochsteps=[0]*4
  for ids in order.split(1024):
   idx=ids.to(device);batch=len(ids)
   for i in range(4):
    active=torch.ones(batch,dtype=torch.bool,device=device) if i==0 else data['teacher_active'][idx,i-1].bool()
    if not active.any():continue
    if i==0:
     dist=student.distribution(i,data['sut_obs'][idx],torch.ones(batch,3,device=device))
     losses=sut_loss(dist,data['teacher_resource'][idx]);loss=losses.mean()
     entropy=dist.entropy().mean()
    else:
     obs=data['post_uav_obs'][idx,i-1];mask=data['uav_masks'][idx,i-1];target=data['teacher_effective'][idx,i-1].long()
     dist=student.distribution(i,obs,mask);losses=torch.nn.functional.cross_entropy(dist.logits.squeeze(1)[active],target[active],reduction='none');loss=losses.mean();entropy=dist.entropy()[active].mean()
    assert torch.isfinite(loss) and torch.isfinite(entropy)
    optim[i].zero_grad(set_to_none=True);loss.backward()
    norm=torch.nn.utils.clip_grad_norm_(student.actors[i].parameters(),1.0,error_if_nonfinite=True)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in student.actors[i].parameters())
    optim[i].step();counts[i]+=1;epochsteps[i]+=1
    sums[i,0]+=float(loss.detach())*int(active.sum());sums[i,1]+=float(norm);sums[i,2]+=float(entropy.detach());sums[i,3]+=1;valid_counts[i]+=int(active.sum())
  if device!='cpu':torch.cuda.synchronize()
  log=dict(student=student_seed,stage=stage,epoch=epoch+1,samples=n,valid_samples=valid_counts,optimizer_steps=epochsteps,cumulative_optimizer_steps=counts.copy(),loss=[float(sums[i,0]/max(valid_counts[i],1)) for i in range(4)],mean_preclip_gradient_norm=[float(sums[i,1]/max(sums[i,3],1)) for i in range(4)],mean_entropy=[float(sums[i,2]/max(sums[i,3],1)) for i in range(4)],elapsed_seconds=time.perf_counter()-started)
  append(folder/'fit.jsonl',log)
  record_cost('supervised_fit',f'{student_seed}/{stage}/epoch_{epoch+1}',optimizer_steps=epochsteps,examples_presented=n,valid_examples=valid_counts,gradient_steps=sum(epochsteps),physical_steps=0)
  state=dict(student_seed=student_seed,manifest_sha256=sha(HERE/'manifest.json'),stage=stage,completed_epochs=epoch+1,actors=student.state(),actor_hashes=student.hashes(),optimizers=[o.state_dict() for o in optim],optimizer_steps=counts.copy(),shuffle_rng=rng.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],data_sha256=hashes,device=device)
  save(resume,state);print(json.dumps(log),flush=True)
 # This is the fixed endpoint, not a selected validation checkpoint.
 save(checkpoint,state)
 write(folder/'status.json',dict(state='complete',stage=stage,epochs=target_epochs,samples=n,checkpoint_sha256=sha(checkpoint),actor_hashes=student.hashes(),optimizer_steps=counts,added_optimizer_steps=[x-y for x,y in zip(counts,initial_counts)]))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--student',type=int,required=True);p.add_argument('--stage',required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args();guard();train(a.student,a.stage,a.device)
