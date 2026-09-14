from diag_support import *
@torch.no_grad()
def main():
 guard();m=verify();result={};A=ws.make_env('fixed_0');Q=ws.make_env('fixed_2');a,sa,ma=A.observe();q,sq,mq=Q.observe();assert base.frozen.external_hashes(A)!=base.frozen.external_hashes(Q)
 for k in ['noise_us','noise_sat','noise_du','potential_content','pos_uav','q','aoi','tau']:assert torch.equal(getattr(A,k),getattr(Q,k)),k
 diff=(a[:,0]!=q[:,0]).any(0).nonzero().flatten().tolist();assert diff==[23,25];act=torch.full((20,3),-1/3,dtype=torch.float64);ua,_,_=A.allocate_resources(act);uq,_,_=Q.allocate_resources(act);udiff=(ua[:,1]!=uq[:,1]).any(0).nonzero().flatten().tolist();assert udiff==[67,69]
 result['instruction']=dict(names=A.p.instruction_names,actor_observe_instruction=A.p.actor_observe_instruction,sut_indices=[23,24,25],uav_indices=[67,68,69],critic_changed_indices=(sa[:,0]!=sq[:,0]).any(0).nonzero().flatten().tolist(),quality_id=2,state_aliasing=False)
 env=ws.make_env('fixed_2');obs,state,mask=env.observe();policy=Policy(m['parents'][0],6000000,env);acts=[];errors=[]
 for i in range(4):
  if i==1:obs,_,mask=env.allocate_resources(acts[0])
  action,lp=policy.call(i,obs[:,i],mask[:,i],False);acts.append(action);new,_,dist=policy.nets[i].evaluate_actions(obs[:,i],torch.zeros((20,1,256)),action,torch.ones((20,1)),mask[:,i],torch.ones((20,1)));torch.testing.assert_close(lp,new,rtol=0,atol=1e-6);errors.append(float((lp-new).abs().max()))
 policy.assert_frozen();result['sample_evaluate_logp_max_errors']=errors
 for rewards in [[1,1,1],[0,0,1],[.1,.2,.2,1],[-3,-2,0,1]]:
  x=np.array(rewards);t=tail_advantage(x);np.testing.assert_allclose(t,tail_advantage(x+5),atol=1e-14,rtol=0);assert abs(t.sum())<1e-12
  for u in set(rewards):assert np.ptp(t[x==u])<1e-12
 result['tail_weights_ties_shift_and_centering']='PASS';result['zero_physical_steps']=True;result.update(state='PASS',new_training_steps=0,new_optimizer_updates=0);write(HERE/'preflight.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':
 try:main()
 except BaseException as e:write(HERE/'preflight.json',dict(state='FAIL',error=repr(e),traceback=traceback.format_exc()));raise
