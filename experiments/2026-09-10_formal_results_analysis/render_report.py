"""Render the audited analysis; no training, experiment selection, or input edits."""
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/tmc_formal_analysis_mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent.parent
data = json.loads((HERE/'analysis.json').read_text())
audit = json.loads((HERE/'audit.json').read_text())
assert audit['state'] == 'PASS'


def part(method, scope='overall'):
    value = data['methods'][method]
    return value['overall'] if scope == 'overall' else value['by_scenario'][scope]


def mean(method, metric, scope='overall'):
    return part(method, scope)[metric]['mean']


def number(method, metric, scope='overall', digits=3, uncertainty=False):
    x = part(method, scope)[metric]
    text = f"{x['mean']:.{digits}f}"
    if uncertainty and x['sd'] is not None:
        text += f" ± {x['sd']:.{digits}f}"
    return text


def link(label, rel, line=None):
    path = PROJECT/rel
    return f'[{label}]({path}{":"+str(line) if line else ""})'


plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10, 'axes.spines.top':False,
    'axes.spines.right':False, 'axes.grid':True, 'grid.alpha':.18, 'axes.axisbelow':True,
    'savefig.dpi':170})
fig, axes = plt.subplots(1,2,figsize=(10.6,4.2), layout='constrained')
names = [('IC_HAPPO','SC IC-HAPPO','#116cba','o'),
         ('HAPPO_hidden_instruction','SC hidden instruction','#cc6d00','s'),
         ('R_fixed','SC fixed rule','#16836f','^')]
for ax, key, ylabel in zip(axes, ['mean_aoi','delivered_predicted_psnr'],
                          ['Mean AoI (slots; lower is fresher)','Delivered predicted PSNR (dB)']):
    for method, label, color, marker in names:
        values = [part(method,f'fixed_{g}')[key] for g in [1,0,2]]
        ax.errorbar(range(3),[v['mean'] for v in values],yerr=[v['sd'] or 0 for v in values],
                    label=label,color=color,marker=marker,capsize=4,linewidth=1.6,markersize=5)
    ax.set_xticks(range(3),['AoI priority','Balance','Quality priority'])
    ax.set_ylabel(ylabel)
axes[0].legend(loc='upper left',fontsize=8.5)
fig.suptitle('Instruction response is clear; explicit-input benefit is not established',fontsize=12)
fig.savefig(HERE/'fixed_instruction.png')
fig.savefig(HERE/'fixed_instruction.pdf')
plt.close(fig)

fig, axes = plt.subplots(1,3,figsize=(12.5,4.7),layout='constrained')
methods=['IC_HAPPO','IC_MAPPO','R_fixed','CC_IC_HAPPO','CC_IC_MAPPO','CC_R_fixed_diagnostic']
labels=['SC\nHAPPO','SC\nMAPPO','SC\nrule','CC\nHAPPO','CC\nMAPPO','CC\nrule*']
colors=['#176ead','#509acb','#85b8d8','#b85b21','#da915b','#e7b990']
for ax,key,ylabel in zip(axes,['mean_aoi','delivered_predicted_psnr','deliveries_per_slot'],
        ['Mean AoI (slots)','Delivered predicted PSNR (dB)','Successful updates / slot']):
    means=[mean(m,key) for m in methods]
    bars=ax.bar(range(len(methods)),means,color=colors,yerr=[part(m)[key]['sd'] or 0 for m in methods],capsize=3)
    ax.set_xticks(range(len(methods)),labels,fontsize=8.5)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0,max(means)*1.3)
    ax.bar_label(bars,fmt='%.2f',padding=4,fontsize=8)
fig.suptitle('All 13 scenarios; means ± training-seed SD (n=3 for learned methods)',fontsize=12)
fig.supxlabel('* CC rule is a post-hoc diagnostic. SC / CC use different average-profile abstractions.',fontsize=9)
fig.savefig(HERE/'sc_cc_comparison.png')
fig.savefig(HERE/'sc_cc_comparison.pdf')
plt.close(fig)

scsrc='experiments/2026-09-09_instruction_long_training/runs/three_seed_sc_20260909/source/'
ccsrc='experiments/2026-09-09_cc_comparison/runs/three_seed_cc_20260909/source/'
lines=[
'本轮完整分析已经完成。当前证据支持“SC 系统能按指令在更新速度与质量之间切换”，但不支持“显式指令 RL 已经稳定优于隐藏指令策略”。另一个重要发现是：补测的 CC 简单规则明显优于本轮两个 CC 学习策略，所以原 SC–CC 差距不能全部归功于通信方式。',
'',
'这不是只看总表得出的判断。已核验 27 个最终模型（每个 1000 万步，合计 2.7 亿训练步）、8060 个正式评估回合、483.6 万槽；核对全部正式轨迹哈希、414 个输入/报告/模型文件哈希和 260 组共同外生轨迹。固定指令下的 111.6 万槽从 CSV 独立重新聚合。所有已记录的资源、缓存、期望质量可行性检查通过，聚合结果与原报告一致。另完成 CC 规则的 260 回合、15.6 万槽事后诊断，逐槽核算状态与奖励，并验证外生轨迹和丢包随机数与正式 CC 一致。这些检查支持结果完整性，不是对论文全部假设或所有潜在程序错误的无限保证。',
'',
'学习方法均先在每个训练种子内汇总，再报告 3 个种子的均值和样本标准差；规则没有训练种子。成功交付质量按“质量总和/交付总数”计算，不是所有时刻显示画面的质量。不同指令的奖励权重不同，跨指令用物理指标解释；共同奖励只在相同评估条件的方法之间比较。本次不宣称统计显著，不把复用的数千回合当作数千个独立训练模型。',
'',
'**指令有效，但有效的是整个系统，不能直接等同于显式指令学习的增益。**',
'',
'| 方法 | AoI 优先：AoI | AoI 优先：交付 PSNR | 质量优先：AoI | 质量优先：交付 PSNR |',
'|---|---:|---:|---:|---:|']
for m in ['IC_HAPPO','IC_MAPPO','HAPPO_hidden_instruction','MAPPO_hidden_instruction','R_fixed']:
    lines.append(f'| {m} | {number(m,"mean_aoi","fixed_1")} | {number(m,"delivered_predicted_psnr","fixed_1")} | {number(m,"mean_aoi","fixed_2")} | {number(m,"delivered_predicted_psnr","fixed_2")} |')
lines += [
'',
'以 IC_HAPPO 为例，质量优先相对 AoI 优先：交付质量提高 **6.657 dB**（3 种子范围 6.529–6.795），AoI 增加 **2.818 槽**（范围 2.815–2.822），成功更新量从 **14.991 降到 7.493 次/槽**。三个种子的方向一致。说明此前“AoI 指令没有效果”的现象在这一版设置下已经解决；质量指令也有清楚的响应。',
'',
f'![固定指令比较]({HERE}/fixed_instruction.png)',
'',
'但隐藏指令组几乎得到同样的变化，简单规则也能做到。其原因在代码中可追溯：隐藏的是 actor 观察里的指令编号和显式门限；质量可行动作掩码仍按真实指令生成，并进入分类动作头；SUT 的待传载荷观察也由该指令下的最低可行载荷计算；训练奖励和集中式 critic 仍使用任务上下文。它是“去掉显式 actor 指令输入”的消融，不是完全没有任务信息的策略。',
'',
f'证据：{link("观察、待传载荷与掩码",scsrc+"tensor_env.py",119)}；{link("动作头实际应用掩码",scsrc+"reference/runtime/harl/models/base/act.py",30)}。质量约束是系统设计的合法组成部分，不应为了让基线变差随意去掉；应据此限定消融结论。',
'',
'这里还需要纠正容易引起误解的“系统把动作换掉”：本轮检查的固定指令轨迹中，SC 的 7 种学习方法在实际执行时的提出/执行模式不一致比例为 **0**。记录的提出动作已经经过策略动作头的掩码。因此这批 SC 学习策略的响应不能解释成执行阶段不断强行替换动作；策略可选范围和部分观察本来就随任务变化。不能用“0 次执行替换”反过来证明不存在掩码影响。该零替换结论不包括 G_local_greedy 和 Random 两个规则基线，它们确有执行替换。',
'',
'**显式指令输入、模式学习和奖励附加项的独立贡献，目前没有得到稳定证据。**',
'',
'下表为 13 场景共同奖励的逐训练种子配对差值，正值表示左侧更好。这里只给描述性结果。',
'',
'| 配对比较 | 平均差值 | 3 种子最小–最大 | 正差种子数 |',
'|---|---:|---:|---:|']
for pair in ['IC_HAPPO - HAPPO_hidden_instruction','IC_MAPPO - MAPPO_hidden_instruction',
             'IC_HAPPO - HAPPO_fixed_mode_rule','IC_HAPPO - HAPPO_equal_resources',
             'IC_HAPPO - HAPPO_no_task_aux_reward','IC_HAPPO - R_fixed']:
    v=data['paired_differences'][pair]['overall']['common_reward']
    lines.append(f"| {pair} | {v['mean']:+.6f} | {v['min']:+.6f} – {v['max']:+.6f} | {v['positive_count']}/3 |")
lines += [
'',
'IC_HAPPO 比 R_fixed 的整体 AoI 仅降低 **0.0347 槽，约 0.93%**；质量优先场景下降 0.2739 槽，但交付质量也低 0.1342 dB。均衡场景反而是 R_fixed 的 AoI 和质量都略好。与均分资源消融相比有小幅、三种子同向的奖励收益，可以保留为有限的资源分配证据；与隐藏指令、固定模式规则、去掉辅助奖励相比，尚不能归因出稳定的相应模块增益。HAPPO 与 MAPPO 的整体奖励均值只差约 0.000038，也不支持明显优越性的表述。',
'',
'**为什么 AoI 优先场景几乎没有继续拉开差距的空间。**',
'',
'当前缓存规则要求：发送后释放缓存，下一槽采集新的块，再下一槽才能发送。按长期连续成功更新的理想状态，一个 DS 最快每两槽更新一次；30 个 DS 对应约 15 次更新/槽，AoI 通常在 2、3 间交替，长期均值约 2.5。有限回合的初始缓存和初始 AoI 会使均值略低于 2.5。IC_HAPPO 在 AoI 指令下达到 14.991 次/槽和 2.499975，已经贴近这套时序允许的理想水平。这个“天花板”来自缓存时序、资源和载荷共同作用，不是把 AoI 上限再降低就能消除的。',
'',
f'证据：{link("缓存与 AoI 更新时序",scsrc+"tensor_env.py",208)}。目前固定指令的 SC IC_HAPPO 超限比例为零；13 场景整体约 0.0247%，主要是切换过程。应同时报告物理 AoI 和超限率，不能只看奖励惩罚。',
'',
'SC 表还存在动作冗余：0/4/8/12 的质量与载荷完全相同，5/9/13 完全相同；按全 SNR 网格的质量不低且载荷不高标准，16 行只有 **3 类不同的非支配质量—载荷选择**，可用 0、5、10 表示。此结论仅针对当前固定平均模型。AoI 优先的 IC_HAPPO 全部选择低载荷同类模式，均衡全部选择中间同类模式。不同模式编号的变化不能直接算作不同编码行为。质量优先时约 31.8% 的执行选择为 14，而 10 在该表中支配 14，也提示模式学习仍有不足。',
'',
'**指令切换确实发生，且隐藏指令组也同样能恢复。**',
'',
'以第 300 槽切换为例，窗口是切换前 50 槽、切换后 50 槽和末尾 50 槽；以下为 IC_HAPPO 的 3 种子均值。',
'',
'| 切换 | 窗口 | AoI | 交付 PSNR |',
'|---|---|---:|---:|']
vv=[v for k,v in data['switch_windows'].items() if k.endswith('/IC_HAPPO')]
for s,label in [('switch300_1_to_2','AoI → 质量'),('switch300_2_to_1','质量 → AoI')]:
    for w,wl in [('before','切换前'),('after','切换后'),('late','末尾')]:
        a=np.mean([v[s+'/'+w]['mean_aoi'] for v in vv]);q=np.mean([v[s+'/'+w]['delivered_predicted_psnr'] for v in vv])
        lines.append(f'| {label} | {wl} | {a:.3f} | {q:.3f} |')
lines += [
'',
'末尾 AoI 与相同槽位置的固定目标指令相比，差约 0.0194 槽（转质量）和 0.00014 槽（转 AoI）。隐藏指令组对应差约 0.0200 和 0.00007 槽，没有表现出显式指令组特有的恢复优势。这是窗口统计，不表示精确在 50 槽内收敛，也不是相同物理状态下的因果干预。',
'',
'**SC 更新更快，CC 成功交付质量更高；CC 已训练策略偏弱。**',
'',
'| 方法 | AoI（槽） | 交付 PSNR（dB） | 更新/槽 | 使用的复信道次数/槽 |',
'|---|---:|---:|---:|---:|']
for m in methods:
    label=m+('（事后规则）' if m=='CC_R_fixed_diagnostic' else '')
    lines.append(f'| {label} | {number(m,"mean_aoi",uncertainty=True)} | {number(m,"delivered_predicted_psnr")} | {number(m,"deliveries_per_slot")} | {number(m,"channel_uses_per_slot",digits=0)} |')
lines += [
'',
f'![SC 与 CC 比较]({HERE}/sc_cc_comparison.png)',
'',
'原配对中，SC-HAPPO 比 CC-HAPPO 的 AoI 低 6.998 槽，CC 质量高 5.283 dB；HAPPO/MAPPO 合计六个配对种子的方向均相同。因此可以说当前模型下 SC 更侧重及时更新，不能说 SC 在质量上也全面优于 CC。SC-HAPPO 实际使用约 47989 次/槽，高于 CC-HAPPO 的 41664；它的每次成功更新平均消耗约 4206，低于 CC 的 10670。总资源消耗与每次更新消耗是两种指标，不能混写“SC 总消耗更低”。',
'',
'新增 CC 规则采用均分资源、最低载荷可行模式和原 AoI 调度，不训练、不改表，完整跑了同样的 260 组条件。它把 CC-HAPPO 的平均 AoI 从 **10.689 降到 7.709，改善约 27.9%**，交付质量从 33.128 变为 33.120 dB。规则的共同奖励和 AoI 优于全部六个 CC 学习模型。这个结果证明当前 CC 策略存在可避免的损失，但规则同时改变模式和资源分配，不能仅凭本次诊断把全部差异归因于某一个动作头。',
'',
'CC 包失败占尝试次数的比例约 1.1%–1.3%；规则约 1.44%，稍高却仍得到更低 AoI。低成功率不足以单独解释学习策略的低更新量，但少量失败仍可能影响 AoI 尾部，不能说丢包完全不重要。主要线索是每个块较大，以及资源分配和模式选择造成的整包容量损失。',
'',
'一个可计算的例子：QP42、码率 5/6 的平均块载荷为 9840，发两块需 19680。均分每 UAV 预算为 20000，只多 320；资源份额从 1/3 略降到 0.328 以下，就从能发两块变为最多一块。固定指令轨迹中，CC-HAPPO 在执行该模式时约 49.3% 的资源份额落在这个阈值下。较高冗余的其他码率进一步减少可容纳的块数。并非所有未用满预算都能转化为可发整包，不能用 GPU 利用率或训练步数来解释这项损失。',
'',
f'另一个需要分离检查的实现选择是：训练采样 Dirichlet 资源份额，确定性评估使用其均值，见 {link("Simplex 确定性动作",ccsrc+"reference/runtime/harl/models/base/distributions.py",150)}。在整包数量取整的环境里，平均动作的收益未必等于采样动作的平均收益；这是待验证的原因，不是本次已证明的程序错误。',
'',
'CC 的最低压缩质量档 QP42 本身成功质量已约 33.11 dB；本轮三个指令下，CC-HAPPO 约 99% 的执行模式都在该 QP 档。当前 CC 模式库没有充分覆盖 SC 在 AoI 指令下约 24.87 dB 的低质量、低载荷区域。因此两个通信分支并没有形成同样宽的质量选择区间。它不证明 H.264 必然需要这么高质量，应在后续有真实编码测量依据地扩展压缩档位，而不是人为缩放载荷表。',
'',
'与更强的 CC 规则相比，SC-HAPPO 的 AoI 仍低约 4.019 槽，但 SC 规则的 AoI 也已经是 3.725。说明剩余大差距有很强的平均表、协议和系统设定成分，不能作为指令学习本身创新的证据。CC 规则也不是最优 CC，不应把这个对比写成通信方式的理论极限。',
'',
f'公平性范围继续受 {link("已冻结的 CC 协议","experiments/2026-09-09_cc_comparison/PROTOCOL.md")} 约束：SC 为用户指定的固定平均表，真实编解码器/历史视频/载荷绑定尚不足；CC 为真实 H.264 测量加 LDPC/整包概率模型。两者平均块尺寸和载荷单位已对齐，传输成功抽象仍不同。CC 验证集质量 MAE 1.585 dB、载荷 MAPE 17.84%，这些平均模型的不确定性没有由本次 RL 的三种子误差条覆盖。',
'',
'**对论文与下一步工作的判断。**',
'',
'1. 当前先保留全部正式结果，不需要因为本次统计分析而作废或重训。没有发现能使这一批评估失效的聚合、配对或约束核算错误。论文可以保留“任务条件下的系统质量—时效取舍”，并如实给出上述消融和 CC 诊断结果。',
'2. 不应把主要贡献写成“显式指令输入显著提升性能”“HAPPO 明显胜过 MAPPO”或“SC 在所有指标上全面胜过 CC”。当前结果不足以证明这些话。这也不等于已经证明算法完全没有创新；它说明新增设计的独立经验收益还没有被隔离出来。',
'3. 下一项最值得做的短诊断是冻结现有模型，分别测试 CC 的“只均分资源”“只固定最低载荷模式”，再在预先固定的采样种子下比较确定性与随机动作评估，定位损失来自哪一环。之后才决定是否修动作参数化、部署决策或重训，不能直接认定加倍训练就会解决。',
'4. 若要证明显式指令学习的价值，应另设一项明确的“只改变任务偏好”实验：保持物理信道、缓存、安全底线和可行动作范围相同，让指令主要改变奖励偏好；带指令 actor 接收偏好，不带指令 actor 不接收且观察特征不间接编码偏好。当前“指令改变硬质量要求”的任务保留为另一项系统实验。两者研究问题不同，不能混为一次消融或为了制造差距剥夺合法安全信息。',
'5. 在新设计中先根据载荷/缓存推导可达更新率，检查是否有多个非支配动作及真实的奖励取舍；若 AoI 已贴近 2.5，则降低软上限只能增加惩罚，不能创造物理空间。是否需要调带宽、可用回传比例或业务负载，要以预先声明的工作点和短诊断为依据，不根据最终排名事后挑参数。',
'6. CC 后续应补齐有真实测量支持的低质量压缩档位，并保留本轮表作为历史版本；SC 表若继续按用户要求固定，就必须把结论限定在该平均模型。只有确定新问题、动作设计或表确实改变之后，才需要从头进行新的三种子训练。',
'',
'本轮已经完成证据核验、正式结果分析和唯一预先写入本轮协议的 CC 规则诊断；上面的后续实验尚未执行。本次没有修改原稿、冻结训练源代码、表或模型。',
'',
f'完整数值：{link("analysis.json","experiments/2026-09-10_formal_results_analysis/analysis.json")}；{link("逐种子独立聚合","experiments/2026-09-10_formal_results_analysis/independent_per_seed_results.json")}；{link("核验记录","experiments/2026-09-10_formal_results_analysis/audit.json")}；{link("本轮事后分析协议","experiments/2026-09-10_formal_results_analysis/PROTOCOL.md")}。图的 PDF 版本与 PNG 同目录。',
'',
'复现分析（读取现有结果；CC 规则会校验并复用已完成的回合）：',
'',
'```bash',
f'cd {PROJECT}',
'export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1',
'TMC_PY=/home/king/miniconda3/envs/harl_sionna/bin/python',
'TMC_ANALYSIS=experiments/2026-09-10_formal_results_analysis',
'"$TMC_PY" "$TMC_ANALYSIS/cc_rule_diagnostic.py"',
'"$TMC_PY" "$TMC_ANALYSIS/analyze.py"',
'"$TMC_PY" "$TMC_ANALYSIS/render_report.py"',
'```',
]
(HERE/'REPORT.md').write_text('\n'.join(lines)+'\n')
files=['PROTOCOL.md','analyze.py','cc_rule_diagnostic.py','render_report.py','audit.json','analysis.json',
       'independent_per_seed_results.json','metrics.csv','REPORT.md','fixed_instruction.png','fixed_instruction.pdf',
       'sc_cc_comparison.png','sc_cc_comparison.pdf']
manifest={name:hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in files}
(HERE/'artifact_hashes.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(HERE/'REPORT.md')
