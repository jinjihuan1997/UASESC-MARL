"""Verify a completed supplementary base model; evaluation follows downstream."""
from helpers import *

def main():
    m=verify();assert len(m['jobs'])==len(m['seeds'])
    for item in m['jobs']:
        job=HERE/item['output'];s=read(job/'status.json')
        assert s['state']=='complete' and s['completed_steps']==10000000
        for f,h in s['checkpoint_hashes'].items():assert digest(job/f)==h
        rows=[json.loads(x) for x in (job/'training_metrics.jsonl').read_text().splitlines()]
        assert len(rows)==2500 and all(x['trained_actor_ids']==[0,1,2,3] for x in rows)
        for row in rows:np.testing.assert_allclose(row['actor_learning_rates'],1e-4*(1-(row['update']-1)/2500),rtol=1e-10,atol=1e-14)
        snapshot=job/'milestones/steps_10000000';ss=read(snapshot/'status.json')
        for f,h in ss['checkpoint_hashes'].items():assert digest(snapshot/f)==h
    for f,h in read(HERE/'provenance.json')['parent_input_hashes'].items():assert digest(f)==h,f
    write(HERE/'report/audit.json',dict(state='PASS',episodes=0,training_steps=m['total_training_steps'],
        final_model_and_snapshot_verified=True,learning_rate_and_budget_verified=True,evaluation='Deferred to continuation phase'))
    (HERE/'REPORT.md').write_text('三个新种子joint基础模型各完成1000万步；最终模型与快照已核验。下一阶段自动继承同一检查点，训练选择器和原方案续训对照。\n')
    print('Base model PASS',m['seeds'],flush=True)

if __name__=='__main__':main()
