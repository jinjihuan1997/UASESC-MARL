"""Historical fixed trajectories: score change only, not new training evidence."""
from helpers import *


def main():
    manifest=read(PARENT/'manifest.json');per={};hashes={};episodes=0;slots=0
    for method in METHODS+RULE_METHODS:
        cats=[f'seed_{s}' for s in SEEDS] if method in METHODS else ['rules']
        for cat in cats:
            item=f'candidate_tail4/{cat}/{method}';folder=PARENT/'evaluation'/item
            meta=read(folder/'summary.json');parts=[]
            for scenario in manifest['scenarios']:
                f=folder/f'{scenario}.npz';h=digest(f);assert h==meta['traces'][scenario];hashes[str(f)]=h
                with np.load(f) as z:
                    data=z['trace'];fs=list(z['fields']);age=z['aoi_after'].reshape(600,20,30).astype(float)
                v={k:data[...,i] for i,k in enumerate(fs)}
                w=np.asarray(config('ref10')['env_args']['reward_weights_by_instruction'])[v['instruction_id'].astype(int)]
                q=(v['predicted_quality_sum']-21*v['deliveries'])/360
                fage=.4*age.mean(-1)+.3*age.max(-1)+.3*np.maximum(age-4,0).mean(-1)
                load=.02*v['channel_uses']/60000
                old=w[...,0]*q-w[...,1]*fage/10-load
                new=w[...,0]*q-w[...,1]*fage/8-load
                np.testing.assert_allclose(old,v['objective_tail4'],atol=1e-9,rtol=0)
                np.testing.assert_allclose(new-old,-.25*(v['age_mean_cost']+v['age_max_cost']+v['age_tail_cost']),atol=1e-9,rtol=0)
                parts.append([old.mean(),new.mean(),v['mean_aoi'].mean()]);episodes+=20;slots+=12000
            per[f'{cat}/{method}']=dict(zip(['score_ref10','score_ref8','mean_aoi'],map(float,np.mean(parts,axis=0))))
    grouped={}
    for method in METHODS+RULE_METHODS:
        items=[f'seed_{s}/{method}' for s in SEEDS] if method in METHODS else [f'rules/{method}']
        grouped[method]={k:float(np.mean([per[i][k] for i in items])) for k in per[items[0]]}
    write(HERE/'quick_rescore.json',dict(state='PASS',interpretation='historical_frozen_actions_only',
        per_item=per,grouped=grouped,input_hashes=hashes,episodes=episodes,slots=slots,
        physical_behavior_changed=False,new_training=False,rule_recalibration_included=False))
    lines=['# 参考值8：旧轨迹重新计分','',
        '以下沿用上一轮测试种子20262201—20262220、冻结模型和冻结规则动作。只展示计分变化，不是新训练，也不是规则重新优化后的正式对比。',
        '', '| 方法 | 参考10分数×100 | 参考8分数×100 |', '|---|---:|---:|']
    for method,v in grouped.items():lines.append(f'| {method} | {100*v["score_ref10"]:.4f} | {100*v["score_ref8"]:.4f} |')
    lines+=['',f'共重新核算{episodes}回合、{slots}时隙；每时隙新分数=旧分数−旧AoI扣分×0.25。物理指标没有因重新打分改变。',
        '', '正式新训练与基线重新校准的独立测试见本目录REPORT.md（完成后生成）。']
    (HERE/'QUICK_RESCORE.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(grouped,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
