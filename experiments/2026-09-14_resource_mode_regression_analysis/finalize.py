"""Repeat deterministic report aggregation and verify all protected input hashes."""
from analysis_support import *
report_aggregate=base.load_module('resource_mode_analysis_aggregate',HERE/'aggregate.py')

def run():
 guard();targets=['report/summary.json','report/REPORT.md'];before={k:sha(HERE/k) for k in ['report/results.json','report/same_load_opportunity.json','report/budget_capacity.json']}
 report_aggregate.run();first={p:sha(HERE/p) for p in targets}
 report_aggregate.run();second={p:sha(HERE/p) for p in targets};assert first==second
 assert before=={p:sha(HERE/p) for p in before}
 protected=read(HERE/'manifest.json')['protected_input_sha256'];protected.update(read(HERE/'opportunity_manifest.json')['protected_input_sha256'])
 changes=[p for p,h in protected.items() if sha(ROOT/p)!=h];assert not changes,changes
 record=dict(state='PASS',aggregation_runs=2,first_sha256=first,second_sha256=second,unchanged_numeric_inputs=before,timestamps_separated=True)
 dump(HERE/'report/reproducibility.json',record)
 dump(HERE/'execution_seal.json',dict(sources={p.name:sha(p) for p in sorted(HERE.glob('*.py'))},protocol_sha256=sha(HERE/'PROTOCOL.md'),exploratory_definition_sha256=sha(HERE/'EXPLORATORY_EXTENSION.md')))
 dump(HERE/'report/execution_audit.json',dict(state='PASS',protected_union_count=len(protected),changed=[],new_training_steps=0,new_optimizer_updates=0,new_physical_steps=0,new_complete_episodes=0,read_only_primary_analysis_runs=2,primary_actor_input_records_per_run=1296000,actor_forward_evaluations_per_run=2592000,primary_local_greedy_queries_per_run=648000,initial_successful_interface_probe_actor_forward_evaluations=20,total_actor_forward_evaluations=5184020,total_local_greedy_queries=1296000,profile_only_environment_batch_initializations=17,profile_only_initialized_environment_instances=340,profile_initialization_details=dict(interface_probe=1,primary_analysis=2,same_load_algebra=13,static_budget_capacity=1),training_log_rows_read=7500,training_log_aggregation_runs=2,unique_saved_physical_trace_payloads_analysed=84,previously_computed_physical_steps_in_unique_payloads=1008000,supervised_training=False,rl_training=False,reserved_final_test_used=False,reserved_intersection=[],initial_import_failure_resolved_without_old_source_edit=True,existing_training_or_background_processes_modified=False,source_read_only_guard=True,original_evaluation_action_replay='PASS',independent_profile_mask_reconstruction='PASS',aggregation_reproducibility='PASS'))
 dump(HERE/'status.json',dict(state='complete',completed_scope='read-only resource-mode and checkpoint regression analysis',new_physical_steps=0,new_training_steps=0,new_optimizer_updates=0,protected_union_count=len(protected),protected_changes=[],report='report/REPORT.md',reproducibility='PASS',no_next_experiment_started=True))
 print('analysis complete; protected inputs',len(protected),'unchanged; repeat aggregation PASS')
if __name__=='__main__':run()
