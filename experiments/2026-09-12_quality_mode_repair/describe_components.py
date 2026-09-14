"""Descriptive fixed-quality breakdown, using already audited measurements only."""
from repair_support import *

def main():
    guard();m=verify_inputs()
    assert sha(HERE/'describe_components.py')==read(HERE/'descriptive_manifest.json')['script_sha256']
    stats=read(HERE/'report/physical_statistics.json')
    labels={'original_rl':'原RL','quality_rule_hybrid':'质量规则混合','residual_all':'全指令修正','residual_quality':'质量门控修正'}
    lines=['# 固定质量任务：奖励与模式明细','',
      '全部取固定100万步检查点、同样20个开发验证种子和完整600槽。下表每行先合并交付数再计算PSNR；它是平均表的预测值。奖励各项均×100，可直接核算“质量收益−AoI代价−资源代价=得分”（本配置其他奖励项为0）。','',
      '| 父模型 | 方法 | 预测PSNR/交付 | 每槽交付数 | 平均AoI | 质量收益 | AoI代价 | 资源代价 | 得分 |',
      '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    entries=[]
    for seed in m['parents']:
        for method in labels:
            label=method+('_at_1000000' if method in m['arms'] else '')
            entries.append((str(seed),labels[method],stats[f'seed_{seed}/{label}/fixed_2']['2']))
    for method in m['rules']:entries.append(('共同基线',method,stats[f'rules/{method}/fixed_2']['2']))
    for seed,method,v in entries:
        x=v['physical_reward_means'];age=sum(x[k] for k in ('age_mean_cost','age_max_cost','age_tail_cost'))
        lines.append(f"| {seed} | {method} | {v['predicted_psnr_per_delivery']:.4f} | {x['deliveries']:.4f} | {x['mean_aoi']:.4f} | {100*x['quality_credit']:.4f} | {100*age:.4f} | {100*x['resource_cost']:.4f} | {v['score_x100']:.4f} |")
    lines+=['','模式保持原编号，不合并动作空间。下面仅在统计时合并profile逐值精确相等的模式，比例以全部UAV×时隙为分母（包含未交付时隙）。原始16类分布和每UAV明细均在 physical_statistics.json。','',
      '| 父模型 | 方法 | 未交付UAV时隙比例 | 最常用的三个精确等价组及比例 |','|---|---|---:|---|']
    for seed,method,v in entries:
        counts=np.asarray(v['equivalence_counts_no_delivery_then_groups']).sum(0);total=counts.sum()
        order=np.argsort(-counts[1:],kind='stable')[:3]
        top='；'.join(f"{m['equivalence_groups'][i]}: {counts[i+1]/total:.2%}" for i in order)
        lines.append(f"| {seed} | {method} | {counts[0]/total:.2%} | {top} |")
    (HERE/'report/MODE_AND_REWARD_DETAILS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(path='report/MODE_AND_REWARD_DETAILS.md',sha256=sha(HERE/'report/MODE_AND_REWARD_DETAILS.md'))))

if __name__=='__main__':main()
