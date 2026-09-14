"""One-time recorded editorial enhancement; no execution or measurement edits."""
from pathlib import Path
import json,hashlib,ast
p=Path(__file__).resolve().parent;f=p/'report_writer.py';t=f.read_text().replace('def render(r,physical,p,gaps,audit):','def render(r,physical,p,gaps,audit,focus):')
needle=" lines+=['','physical_statistics.json完整"
insert=''' lines+=['','### 全部13场景的真实质量时隙：160441552的UAV3','',
 '这张表包含切换后的质量阶段，各控制器用自身真实闭环轨迹。分母为45,000个质量UAV时隙/父模型，不能与固定质量12,000时隙表混用。','',
 '|控制器|预算均值|组5预算可行率|组5最终可行率|可行时请求组5比例|执行组0比例|执行组5比例|交付总数|预测PSNR|平均AoI|',
 '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
 for method in ['C0','C1','C2','C3']:
  d=focus['by_parent']['160441552'][method][2];n=d['quality_decision_slots'];lo=eq(d,0);mid=eq(d,5)
  lines.append(f"|{method}|{d['budget_mean']:.3f}|{mid['budget_feasible_count']/n:.2%}|{mid['fully_feasible_count']/n:.2%}|{mid['requested_given_fully_feasible_fraction']:.2%}|{lo['executed_fraction_of_all_quality_uav_slots']:.2%}|{mid['executed_fraction_of_all_quality_uav_slots']:.2%}|{d['deliveries_total']}|{d['predicted_psnr_per_delivery']:.4f}|{d['aoi_mean']:.4f}|")
 lines+=['','完整三父模型及三UAV的全部质量时隙统计保存在focus_case.json，C2−C0与C3−C1可按对应原始量相减；没有用固定任务代替切换后质量表现。']
'''
assert needle in t;t=t.replace(needle,insert+needle)
t=t.replace('两次完整聚合分别重新核验325条轨迹','初次完整聚合后补全解释性报告；最终两次完整聚合分别重新核验325条轨迹')
f.write_text(t)
for file in p.glob('*.py'):ast.parse(file.read_text())
def sha(f):return hashlib.sha256(f.read_bytes()).hexdigest()
(p/'analysis_manifest.json').write_text(json.dumps(dict(code_sha256={f:sha(p/f) for f in ['aggregate.py','report_writer.py','focus_statistics.py','run_evaluation.py','run_evaluation.sh']},protocol_sha256=sha(p/'PROTOCOL.md'),execution_seal_sha256=sha(p/'execution_seal.json'),enhancement='Add explicit evidence conclusions and all-13-scenario quality focus; no reward, route, dataset, checkpoint, or evaluation changes; initial full pass retained in aggregation_runs/initial_pass.json'),indent=2)+'\n')
(p/'analysis_enhancement.json').write_text(json.dumps(dict(changes=['Explicit answers to all eight report questions','All-13-scene quality-slot UAV statistics and fixed-quality resource tradeoff explanation'],unchanged=['Protocol','Evaluation implementation','All model weights','Raw trajectories','Seeds and metrics'],editing_failure='One code-edit assertion failed before adding report section; corrected before execution; no physical or gradient steps.'),indent=2)+'\n')
