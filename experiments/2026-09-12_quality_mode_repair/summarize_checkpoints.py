"""Derive cumulative per-milestone statistics without modifying snapshots."""
from repair_support import *

def main():
    guard();m=verify_inputs();outputs={}
    for seed in m['parents']:
        for arm in m['arms']:
            folder=HERE/f'jobs/seed_{seed}/{arm}'
            log=folder/'training_metrics.jsonl'
            rows=[json.loads(line) for line in log.read_text().splitlines()]
            assert len(rows)==250
            for step in m['milestones']:
                part=rows[:step//4000]
                status=read(folder/f'milestones/steps_{step}/status.json')
                instruction=np.sum([r['collection']['instruction_samples'] for r in part],0)
                assert instruction.tolist()==status['counters']['instruction_samples']
                mode=np.sum([r['collection']['executed_mode_counts'] for r in part],0)
                eq=np.stack([mode[:,:,0]]+[mode[:,:,[v+1 for v in group]].sum(-1) for group in m['equivalence_groups']],-1)
                actor=[]
                for agent in range(1,4):
                    updates=[a for r in part for a in r['optimization']['actors'] if a['agent']==agent]
                    mini=[b for a in updates for b in a['minibatches']]
                    values={k:dict(mean=float(np.mean([b[k] for b in mini])),minimum=float(min(b[k] for b in mini)),maximum=float(max(b[k] for b in mini))) for k in
                            ['policy_loss','entropy','gradient_norm','ratio_mean','ratio_min','ratio_max','clip_fraction']} if mini else {}
                    actor.append(dict(agent=agent,actual_optimizer_steps=len(mini),quality_sample_exposures=sum(b['quality_samples'] for b in mini),
                        skipped_no_quality=sum(a['skipped_no_quality'] for a in updates),metrics=values))
                rewards=np.sum([r['collection']['reward_and_physical_sums'] for r in part],0)
                resources={k:np.sum([r['collection']['resources'][k] for r in part],0) for k in part[0]['collection']['resources']}
                value=dict(parent_seed=seed,arm=arm,parent_steps=1000000,added_steps=step,update=step//4000,
                    milestone_status=status,training_log_sha256=sha(log),training_log_prefix_updates=step//4000,
                    instruction_samples=instruction.tolist(),actor_statistics=actor,
                    critic_loss_mean=float(np.mean([x['loss'] for r in part for x in r['optimization']['critic_minibatches']])),
                    last_update_value_diagnostic=part[-1]['optimization']['value_diagnostic'],
                    reward_and_physical_fields=FIELDS,reward_and_physical_sums=rewards.tolist(),
                    reward_and_physical_means_by_instruction=(rewards/np.maximum(instruction[:,None],1)).tolist(),
                    resource_sums={k:v.tolist() for k,v in resources.items()},
                    resource_means_by_instruction={k:(v/np.maximum(instruction[:,None],1)).tolist() for k,v in resources.items()},
                    requested_mode_counts=np.sum([r['collection']['requested_mode_counts'] for r in part],0).tolist(),
                    executed_mode_counts_minus1_then_0_to_15=mode.tolist(),equivalence_groups=m['equivalence_groups'],
                    equivalence_counts_no_delivery_then_groups=eq.tolist())
                path=HERE/f'checkpoint_statistics/seed_{seed}/{arm}/steps_{step}.json';write(path,value)
                outputs[str(path.relative_to(HERE))]=sha(path)
    write(HERE/'checkpoint_statistics/manifest.json',dict(derivation_script_sha256=sha(HERE/'summarize_checkpoints.py'),files=outputs,
          immutable_model_files_changed=False,meaning='Descriptive aggregation of exact training-log prefixes; no selection or retraining'))
    print(json.dumps(dict(statistic_files=len(outputs))))

if __name__=='__main__':main()
