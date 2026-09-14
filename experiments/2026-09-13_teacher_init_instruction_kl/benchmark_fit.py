"""Resource choice from steady four-actor fitting throughput, not reward scores."""
from study import *
from student_policy import Student,sut_loss
from teacher_interface import Teacher

def main():
 m=verify();env=make_env(m['preflight'],'preflight');obs,_,am=env.observe();teacher=Teacher();a=teacher.resource(obs[:,0]);post,_,mask=env.allocate_resources(a)
 labels=[teacher.mode_label(i,post[:,i+1],mask[:,i+1]) for i in range(3)];out={}
 for device in ('cpu','cuda:0'):
  student=Student(m['preflight_model_seed'],env,device);student.set_mode(True);optim=[torch.optim.Adam(n.parameters(),lr=.001) for n in student.actors]
  x=[(obs[:,0] if i==0 else post[:,i]).repeat(103,1)[:1024].to(device) for i in range(4)]
  masks=[(am[:,0] if i==0 else mask[:,i]).repeat(103,1)[:1024].to(device) for i in range(4)]
  target=a.repeat(103,1)[:1024].to(device);targets=[e[1].repeat(103)[:1024].to(device) for e in labels]
  seconds=[]
  for step in range(30):
   start=time.perf_counter()
   for i in range(4):
    d=student.distribution(i,x[i],masks[i]);loss=sut_loss(d,target).mean() if i==0 else torch.nn.functional.cross_entropy(d.logits.squeeze(1),targets[i-1])
    assert torch.isfinite(loss);optim[i].zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(student.actors[i].parameters(),1.,error_if_nonfinite=True);optim[i].step()
   if device!='cpu':torch.cuda.synchronize()
   seconds.append(time.perf_counter()-start)
  out[device]=dict(warmup_batches=5,measured_batches=25,median_seconds_per_four_actor_batch=float(np.median(seconds[5:])),total_optimizer_steps=120)
  record_cost('preflight_gradient','steady_fit_benchmark_'+device,gradient_steps=120,optimizer_steps=120,physical_steps=0,teacher_queries=teacher.queries if device=='cpu' else [0]*4)
 choice=min(out,key=lambda d:out[d]['median_seconds_per_four_actor_batch'])
 write(HERE/'resource_settings.json',dict(fit_device=choice,maximum_fit_workers=1 if choice!='cpu' else 3,collection_workers=3,torch_threads=1,blas_threads=1,benchmark=out,selection_uses_performance_scores=False))
 print(out,'choice',choice,flush=True)
if __name__=='__main__':guard();main()
