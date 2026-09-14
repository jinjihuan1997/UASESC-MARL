from study import *
from student_policy import Student,sut_loss,physical_shares,target_alpha
from teacher_interface import Teacher
from trajectory import run_episode_batch
from kl_control import distribution_snapshot,analytic_kl,summarize_kl,Candidate
from happo_kernel import strict_actor_step
from harl.algorithms.actors.happo import HAPPO
from harl.utils.ratio_tools import aggregate_action_ratio
import argparse

def foundation():
 m=verify();env=make_env(m['preflight'],'preflight');obs,state,mask=env.observe();teacher=Teacher();ta=teacher.resource(obs[:,0]);post,_,am=env.allocate_resources(ta)
 student=Student(m['preflight_model_seed'],env);student.set_mode(True);checks={};torch.manual_seed(m['preflight_stat_seed'])
 actor_steps=0
 for i in range(4):
  x=obs[:,0] if i==0 else post[:,i];available=mask[:,0] if i==0 else am[:,i]
  with torch.no_grad():action,lp=student.forward(i,x,available,False)
  evaluated=student.evaluate_actions(i,x,action,available)[0]
  torch.testing.assert_close(lp,evaluated,atol=0,rtol=0)
  assert torch.equal(aggregate_action_ratio(evaluated-lp,'prod',clip=20.),torch.ones(len(x),1))
  dist=student.distribution(i,x,available)
  event=dist.log_probs(.5*(action+1)) if i==0 else dist.log_probs(action.argmax(-1).reshape(-1,1,1)).reshape(-1,1)
  torch.testing.assert_close(lp.sum(-1,keepdim=True),event,atol=1e-6,rtol=1e-6)
  if i==0:loss=sut_loss(dist,ta).mean()
  else:
   req,eff,active,why=teacher.mode_label(i-1,x,available)
   loss=torch.nn.functional.cross_entropy(dist.logits.squeeze(1)[active],eff[active])
  opt=torch.optim.Adam(student.actors[i].parameters(),lr=.001);opt.zero_grad();loss.backward()
  gn=torch.nn.utils.clip_grad_norm_(student.actors[i].parameters(),1.,error_if_nonfinite=True);assert float(gn)>0
  opt.step();actor_steps+=1
  old=distribution_snapshot(student,i,x,available);candidate=Candidate(student.actors[i],opt)
  before=student.hashes()[i];beforelog=student.evaluate_actions(i,x,action,available)[0].detach()
  opt.zero_grad();fake=sum(p.square().sum() for p in student.actors[i].parameters());fake.backward();opt.step();actor_steps+=1
  values=analytic_kl(old,student,i,x,available);assert float(values.mean())>0
  candidate.rollback();assert student.hashes()[i]==before and state_equal(opt.state_dict(),candidate.adam)
  afterlog=student.evaluate_actions(i,x,action,available)[0].detach()
  assert torch.equal(aggregate_action_ratio(afterlog-beforelog,'prod',clip=20.),torch.ones(len(x),1))
  checks[str(i)]=dict(logp_replay='exact',initial_ratio=1,supervised_gradient_norm=float(gn),candidate_KL=float(values.mean()),parameter_and_Adam_rollback='exact',factor_after_rollback=1)
 # Analytic KL checked against independent Monte Carlo log-density ratio.
 torch.manual_seed(m['preflight_stat_seed']);mc={}
 for kind in ('dirichlet','categorical'):
  if kind=='dirichlet':a=torch.distributions.Dirichlet(torch.tensor([.9,1.2,2.1],dtype=torch.float64));b=torch.distributions.Dirichlet(torch.tensor([1.0,1.1,2.0],dtype=torch.float64))
  else:a=torch.distributions.Categorical(probs=torch.tensor([.2,.3,.5],dtype=torch.float64));b=torch.distributions.Categorical(probs=torch.tensor([.25,.3,.45],dtype=torch.float64))
  samples=a.sample((100000,));ratio=a.log_prob(samples)-b.log_prob(samples);exact=float(torch.distributions.kl_divergence(a,b));mean=float(ratio.mean());se=float(ratio.std()/np.sqrt(100000));assert abs(mean-exact)<=6*se+1e-3
  mc[kind]=dict(analytic=exact,mc=mean,standard_error=se,samples=100000)
 # Native finite HAPPO kernel equivalence with KL disabled/infinite threshold.
 args={**cfg()['algo_args']['model'],**cfg()['algo_args']['algo']};equivalence={}
 for i in (0,1):
  native=HAPPO(args,env.observation_space[i],env.action_space[i],torch.device('cpu'));other=copy.deepcopy(native)
  x=obs[:,0] if i==0 else post[:,i];av=mask[:,0] if i==0 else am[:,i];rnn=torch.zeros(10,1,256);ones=torch.ones(10,1)
  with torch.no_grad():action,lp,_=native.get_actions(x,rnn,ones,av)
  sample=(x,rnn,action,ones,ones,lp,torch.linspace(-1,1,10).reshape(-1,1),av,ones)
  native.update(sample);strict_actor_step(other,sample);actor_steps+=2
  assert state_equal(native.actor.state_dict(),other.actor.state_dict()) and state_equal(native.actor_optimizer.state_dict(),other.actor_optimizer.state_dict())
  equivalence[str(i)]='exact parameters and Adam after native/new finite step'
 # No-cache and true no-budget fallback label skip, without using hidden state.
 no_cache=post[:,1].clone();no_cache[:,1:11]=0
 _,_,active,why=teacher.mode_label(0,no_cache,torch.ones_like(am[:,1]));assert not active.any() and (why==1).all()
 no_budget=post[:,1].clone();no_budget[:,0]=0
 _,_,active,why=teacher.mode_label(0,no_budget,torch.ones_like(am[:,1]));assert not active.any() and (why==2).all()
 teacher.assert_frozen()
 # CPU/CUDA loss/gradient finite and timing; never changes a formal student.
 benchmark={}
 for device in ('cpu','cuda:0'):
  net=Student(m['preflight_model_seed'],env,device);net.set_mode(True);o=torch.optim.Adam(net.actors[0].parameters(),lr=.001)
  x=obs[:,0].repeat(103,1)[:1024].to(device);a=ta.repeat(103,1)[:1024].to(device);av=torch.ones(1024,3,device=device)
  start=time.perf_counter()
  for _ in range(5):
   o.zero_grad();loss=sut_loss(net.distribution(0,x,av),a).mean();loss.backward();torch.nn.utils.clip_grad_norm_(net.actors[0].parameters(),1.,error_if_nonfinite=True);o.step();actor_steps+=1
  if device!='cpu':torch.cuda.synchronize()
  benchmark[device]=dict(five_batches_seconds=time.perf_counter()-start,loss=float(loss),finite=True)
 # Save-load exact initial controller and retain prototype for physical sampling smoke.
 proto=Student(m['preflight_model_seed'],env);snap=dict(student_seed=proto.seed,manifest_sha256=sha(HERE/'manifest.json'),actors=proto.state(),actor_hashes=proto.hashes())
 save(HERE/'preflight/prototype.pt',snap);other=Student(proto.seed,env);other.load(HERE/'preflight/prototype.pt')
 for i in range(4):
  x=obs[:,0] if i==0 else post[:,i];av=mask[:,0] if i==0 else am[:,i]
  for a,b in zip(proto.forward(i,x,av,True),other.forward(i,x,av,True)):torch.testing.assert_close(a,b,atol=0,rtol=0)
 record_cost('preflight_gradient','foundation',optimizer_steps=actor_steps,gradient_steps=actor_steps,physical_steps=0,teacher_queries=teacher.queries,mc_samples=200000)
 write(HERE/'preflight/foundation.json',dict(state='PASS',actors=checks,analytic_MC=mc,KL_disabled_native_equivalence=equivalence,empty_label_checks='PASS',save_load='exact',benchmark=benchmark,preflight_optimizer_steps=actor_steps))
 print('foundation PASS',flush=True)

def teacher_reproduction():
 m=verify();results={}
 for scene,schedule in m['scenarios'].items():
  path=HERE/f'preflight/teacher_reproduction/{scene}.npz'
  result=run_episode_batch(path,m['validation'],'validation',schedule,kind='preflight_teacher_reproduction')
  old=REPAIR/f'evaluation/rules/greedy_modes_16/{scene}.npz';meta=read(old.with_suffix('.json'));assert sha(old)==meta['trace_sha256'] and result['external_hashes']==meta['external_hashes']
  with np.load(path) as a,np.load(old) as b:
   for k in set(a.files)&set(b.files):np.testing.assert_array_equal(a[k],b[k],err_msg=scene+':'+k)
  results[scene]=dict(state='PASS',old_sha256=sha(old),new_sha256=sha(path),score=np.mean(result['score_x100_by_environment']))
 write(HERE/'preflight/teacher_reproduction.json',dict(state='PASS',scenarios=results,episodes=260))

def smoke():
 m=verify();proto=HERE/'preflight/prototype.pt'
 for deterministic in (True,False):
  run_episode_batch(HERE/f'preflight/student_{"D" if deterministic else "S"}.npz',m['preflight'],'preflight',student_seed=m['preflight_model_seed'],checkpoint=proto,deterministic=deterministic,action_seeds=None if deterministic else m['preflight_action_seeds'],labels=True,kind='preflight_student')
 write(HERE/'preflight/physical_smoke.json',dict(state='PASS',episodes=20,physical_steps=12000))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('task',choices=['foundation','teacher','smoke']);a=p.parse_args();guard()
 try:{'foundation':foundation,'teacher':teacher_reproduction,'smoke':smoke}[a.task]()
 except BaseException as e:write(HERE/f'preflight_failures/{time.time_ns()}.json',dict(error=repr(e),traceback=traceback.format_exc(),task=a.task));raise
