"""Compare only complete, seed-paired SC/CC final-budget evaluations."""
import argparse
from pathlib import Path
import statistics
from common import stamp,write
from multiseed_protocol import read,verify_run,verify_model
from training_checkpoint import digest


def compare(cc_run):
    cc_run=Path(cc_run).resolve();m=verify_run(cc_run);sc_run=Path(m['parent']['parent_run']);sc=read(sc_run/'manifest.json')
    if m['purpose'].startswith('smoke') or m['steps_per_method']!=sc['steps_per_method']:
        raise ValueError('Only matching formal budgets may be compared')
    if (m['seeds']!=sc['seeds'] or m['scenarios']!=sc['scenarios'] or m['evaluation_seeds']!=sc['evaluation_seeds']
        or digest(sc_run/'manifest.json')!=m['parent']['parent_manifest_sha256']):
        raise ValueError('SC/CC protocols are not paired')
    missing=[];pairs=[]
    for algorithm in ('IC_HAPPO','IC_MAPPO'):
        for seed in m['seeds']:
            row={}
            for label,run,method,manifest in [('SC',sc_run,algorithm,sc),('CC',cc_run,'CC_'+algorithm,m)]:
                job=next(j for j in manifest['jobs'] if j['seed']==seed and j['method']==method)
                path=run/'evaluation'/job['id'];status=read(path/'status.json') if (path/'status.json').exists() else {}
                if status.get('state')!='complete': missing.append(f'{label}/{job["id"]}');continue
                verify_model(run/job['output'],manifest['steps_per_method'])
                if digest(path/'summary.json')!=status['summary_sha256']: raise ValueError('Evaluation summary changed')
                row[label]=read(path/'summary.json')
            if len(row)==2:
                if row['SC']['pairing']!=row['CC']['pairing']: raise AssertionError('SC/CC external trajectory hashes differ')
                pairs.append(dict(algorithm=algorithm,seed=seed,SC=row['SC']['overall'],CC=row['CC']['overall']))
    output=cc_run/'sc_comparison'
    if missing:
        result=dict(execution_status='WAITING_FOR_EVALUATIONS',science_status='PENDING',updated_utc=stamp(),
                    missing=missing,paired_differences=None)
        write(output/'status.json',result);return result
    metrics=['common_reward','mean_aoi','delivered_predicted_psnr','deliveries_per_slot','channel_uses_per_slot']
    values={}
    for algorithm in ('IC_HAPPO','IC_MAPPO'):
        selected=[r for r in pairs if r['algorithm']==algorithm];values[algorithm]={}
        for key in metrics:
            delta=[r['SC'][key]-r['CC'][key] if r['SC'][key] is not None and r['CC'][key] is not None else None for r in selected]
            defined=all(v is not None for v in delta)
            values[algorithm][key]=dict(per_seed=dict(zip(map(str,m['seeds']),delta)),mean=statistics.mean(delta) if defined else None,
                sample_std=statistics.stdev(delta) if defined else None,n_training_seeds=3)
    result=dict(execution_status='COMPLETE',science_status='DESCRIPTIVE_RESULTS',updated_utc=stamp(),
        difference_direction='SC minus CC; positive is better for reward/quality/deliveries, worse for AoI/load',
        paired_differences=values,paired_results=pairs,
        note='Three training seeds; no automatic significance claim. Fixed SC average model vs measured/analytic CC average model.')
    write(output/'comparison.json',result);write(output/'status.json',dict(execution_status='COMPLETE',science_status='DESCRIPTIVE_RESULTS',updated_utc=stamp(),comparison_sha256=digest(output/'comparison.json')))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--cc-run',type=Path,required=True)
    args=parser.parse_args();print(compare(args.cc_run))
