"""Read frozen final evaluation traces; report paired results without seed inflation."""
from pathlib import Path
import argparse,csv,hashlib,json,statistics

def main(run):
    manifest=json.loads((run/'manifest.json').read_text())
    assert json.loads((run/'status.json').read_text())['state']=='complete'
    for name,digest in manifest['input_hashes'].items():
        assert hashlib.sha256((run/name).read_bytes()).hexdigest()==digest,name
    methods=manifest['methods']; data={}; summaries={}
    for method in methods:
        p=run/'evaluation'/method/'summary.json';s=json.loads(p.read_text());summaries[method]=s['episodes']
        rows=[]
        for episode in s['episodes']:
            path=p.parent/f"{episode['scenario']}_{episode['seed']}.csv"
            assert hashlib.sha256(path.read_bytes()).hexdigest()==episode['trace_sha256']
            with path.open() as f:
                for row in csv.DictReader(f):
                    for key in row:
                        if key not in ('fractions','modes'):row[key]=float(row[key])
                    rows.append(row)
        data[method]=rows
    for x,y in zip(summaries[methods[0]],summaries[methods[1]]):
        assert (x['scenario'],x['seed'],x['external_sha256'])==(y['scenario'],y['seed'],y['external_sha256'])
    def metrics(rows):
        deliveries=sum(r['deliveries'] for r in rows)
        return dict(slots=len(rows),mean_reward=statistics.mean(r['common_reward'] for r in rows),
            mean_aoi=statistics.mean(r['mean_aoi'] for r in rows),
            predicted_psnr=sum(r['predicted_quality_sum'] for r in rows)/deliveries if deliveries else None,
            deliveries_per_slot=deliveries/len(rows),channel_uses_per_slot=sum(r['channel_uses'] for r in rows)/len(rows),
            aoi_exceedance_fraction=statistics.mean(r['aoi_exceedance_fraction'] for r in rows),
            quality_violations=sum(r['quality_violations'] for r in rows),
            budget_violations=sum(r['budget_violations'] for r in rows),
            cache_violations=sum(r['cache_violations'] for r in rows))
    result={m:{'overall':metrics(rows),'by_instruction':{str(g):metrics([r for r in rows if int(r['instruction_id'])==g]) for g in range(3)}} for m,rows in data.items()}
    paired=[]
    for a,b in zip(summaries[methods[0]],summaries[methods[1]]):
        paired.append(dict(scenario=a['scenario'],seed=a['seed'],reward_delta=a['common_reward']-b['common_reward'],aoi_delta=a['mean_aoi']-b['mean_aoi']))
    deltas={str(seed):statistics.mean(p['reward_delta'] for p in paired if p['seed']==seed) for seed in manifest['evaluation_seeds']}
    report=dict(training_seed=manifest['seed'],training_steps_per_method=manifest['steps'],paired_episodes=len(paired),
        metrics=result,paired_differences=paired,reward_delta_by_exogenous_seed=deltas,
        note='One training seed; descriptive pilot only. Exogenous seeds are not training-seed replications.',
        confidence_interval=None)
    out=run/'analysis';out.mkdir(exist_ok=True)
    (out/'summary.json').write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    lines=['单种子预实验结果','',f"种子85，两方法各{manifest['steps']:,}步；65组配对episode，5个外部种子，每episode600槽。",'',
        '|方法/指令|平均奖励|平均AoI|交付预测PSNR|每槽交付|AoI超限比例|', '|---|---:|---:|---:|---:|---:|']
    for m in methods:
        for key,value in [('overall',result[m]['overall']),*result[m]['by_instruction'].items()]:
            lines.append(f"|{m}/{key}|{value['mean_reward']:.6f}|{value['mean_aoi']:.4f}|{value['predicted_psnr']:.3f}|{value['deliveries_per_slot']:.3f}|{value['aoi_exceedance_fraction']:.4%}|")
    lines+=['','指令0/1/2分别为balance/aoi/quality。PSNR按交付数加权；其他均值按相应槽加权。','',
        f"IC减hidden的整体奖励差：{result[methods[0]]['overall']['mean_reward']-result[methods[1]]['overall']['mean_reward']:+.6f}。",'',
        '5个外部种子的配对奖励差（每个种子先平均13场景）：'+json.dumps(deltas),'',
        '这是一个训练种子、短预算预实验，不给出跨训练种子显著性结论，也不能仅凭不同指令的AoI差异证明显式输入有价值。',
        '没有修改已冻结配置、没有按测试结果选择checkpoint、没有覆盖旧正式结果。原始逐槽数据与外生过程哈希保存在evaluation。']
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(report['metrics'],indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);main(p.parse_args().run)
