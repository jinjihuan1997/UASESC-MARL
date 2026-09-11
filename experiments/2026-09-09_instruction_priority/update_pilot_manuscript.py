"""Insert verified pilot results into the single new TeX working copy."""
from pathlib import Path
import argparse,json,re

def main(run,tex):
    d=json.loads((run/'analysis/summary.json').read_text());assert d['training_seed']==85 and d['paired_episodes']==65
    m=d['metrics'];ic=m['IC_HAPPO']['overall'];h=m['HAPPO_hidden_instruction']['overall'];delta=ic['mean_reward']-h['mean_reward']
    rows=[]
    for key,label in [('overall','All'),('0','Balance'),('1','Freshness'),('2','Quality')]:
        for method,short in [('IC_HAPPO','IC'),('HAPPO_hidden_instruction','Hidden')]:
            v=m[method]['overall'] if key=='overall' else m[method]['by_instruction'][key]
            rows.append(f"{label} / {short} & {v['mean_reward']:.5f} & {v['mean_aoi']:.3f} & {v['predicted_psnr']:.2f}"+r'\\')
    verdict=('has higher' if delta>0 else 'does not improve')
    text=(r'''The frozen pilot completes $10^6$ steps for each method and all 65 paired evaluation episodes. Table~\ref{tab:pilot} reports the final-checkpoint results. Conditional PSNR is weighted by delivered-update count, whereas reward and age are averaged over the corresponding slots. The overall row weights the 13 schedules equally.
\begin{table}[t]
\centering\caption{Single-seed pilot: explicit versus hidden actor inputs.}
\label{tab:pilot}
\begin{tabular}{lrrr}\hline
Instruction / method & Reward & AoI & PSNR\\\hline
'''+ '\n'.join(rows)+r'''
\hline\end{tabular}
\end{table}
'''+f"Explicit conditioning {verdict} overall mean reward in this pilot: {ic['mean_reward']:.6f} versus {h['mean_reward']:.6f}, a paired difference of {delta:+.6f}. ")
    for method,label in [('IC_HAPPO','IC-HAPPO'),('HAPPO_hidden_instruction','hidden-input HAPPO')]:
        v=m[method]['overall'];text+=f"The {label} policy delivers {v['deliveries_per_slot']:.3f} updates per slot with an age-target exceedance fraction of {100*v['aoi_exceedance_fraction']:.3f}\\%. "
    violations={k:sum(m[x]['overall'][k] for x in m) for k in ['quality_violations','budget_violations','cache_violations']}
    if all(v==0 for v in violations.values()):text+='No quality-threshold, budget, or cache-feasibility violations are recorded in these evaluation traces. '
    else:text+='Observed feasibility violations must be resolved before treating these results as valid control evidence. '
    text+='These are descriptive results for one training seed and a short budget. Differences across five exogenous evaluation seeds do not substitute for independent training-seed replication. Given the indirect task signals demonstrated above, any advantage is incremental to the shared feasibility mechanisms. '
    if delta<=0:text+='The present pilot does not confirm an overall benefit from adding explicit instruction fields; it therefore does not justify a claim of algorithmic superiority.'
    p=Path(tex);s=p.read_text();new,n=re.subn(r'% PILOT_RESULTS_BEGIN.*?% PILOT_RESULTS_END',lambda _: '% PILOT_RESULTS_BEGIN\n'+text+'\n% PILOT_RESULTS_END',s,flags=re.S);assert n==1;p.write_text(new)
    print('Updated',p,'delta',delta)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--tex',type=Path,required=True);a=p.parse_args();main(a.run,a.tex)
