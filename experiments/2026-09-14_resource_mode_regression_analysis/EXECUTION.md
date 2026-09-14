# 只读诊断执行索引

工作目录：`/home/king/Downloads/Projects/2_th_paper_TMC`。

环境：Python `/home/king/miniconda3/envs/harl_sionna/bin/python`；Python 3.10.19，Torch 2.4.1+cu121，NumPy 1.26.4。实际使用CPU，Torch/BLAS各1线程，主要分析固定逻辑CPU0；未终止其他进程，未启动RL工作进程。

```bash
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
taskset -c 0 /home/king/miniconda3/envs/harl_sionna/bin/python experiments/2026-09-14_resource_mode_regression_analysis/analyze.py
taskset -c 0 /home/king/miniconda3/envs/harl_sionna/bin/python experiments/2026-09-14_resource_mode_regression_analysis/opportunity.py
taskset -c 0 /home/king/miniconda3/envs/harl_sionna/bin/python experiments/2026-09-14_resource_mode_regression_analysis/budget_capacity.py
taskset -c 0 /home/king/miniconda3/envs/harl_sionna/bin/python experiments/2026-09-14_resource_mode_regression_analysis/finalize.py
```

analyze.py实际执行两次，日志为analysis.log、analysis_repeat.log。第一次完成后清理新分析脚本的无用变量，明确区分输入记录数与两次网络前向开销，再次执行；没有改算法、数据、检查容差。每次处理54份固定任务历史轨迹、18个冻结模型检查点和9组同输入网络比较。每个模型输入调用一次deterministic forward和一次evaluate_actions核对logp，故1,296,000条actor输入记录对应2,592,000次样本网络前向。初始接口探针额外20条evaluate_actions前向。全部是重复推理，物理环境步为0。

opportunity.py执行一次，保存同载荷质量差的探索性统计及独立输入索引。它为13场景构造外生profile表但不调用step；不将反事实代数结果列成新的控制器成绩。budget_capacity.py执行一次，仅在已保存质量状态计算中档可服务块数；没有实际替换SUT。

总计17次20环境的仅初始化操作：成功接口探针1、主分析2、等载荷检查13、预算容量检查1。没有任何环境推进，没有训练/拟合/优化器更新。84份不同的旧轨迹包含1,008,000个已完成物理步；这里是只读分析覆盖，不是新增计算回合。7,500条已有训练日志共聚合两次。各阶段的原始耗时见日志/主audit，报告汇总不混入运行时间戳。

最初新模块曾命名support.py，与冻结policy_recomposition模块重名，导致导入repair_support失败；在新目录重命名为analysis_support.py后通过接口预检。该失败没有创建/推进环境，没有修改冻结代码，记录在failures.jsonl。

报告汇总也遇到同名aggregate模块被旧目录优先导入的问题，已在新finalize入口改为按绝对路径、唯一模块名加载。失败没有运行聚合或环境。aggregate.py先独立成功运行一次，之后finalize内连续成功运行两次作确定性核验，共三次成功报告聚合。

输出：PROTOCOL.md与EXPLORATORY_EXTENSION.md分别区分初始诊断和读取profile后追加的代数分析；manifest.json、opportunity_manifest.json保存输入SHA；numerics/保存模型浓度、模式概率和确定性动作；report/results.json保存主统计；same_load_opportunity.json、budget_capacity.json保存补充统计；summary.json与REPORT.md由aggregate.py生成。finalize.py连续执行两次aggregate，检查确定性结果一致，重新核对全部输入哈希，写出reproducibility.json、execution_audit.json和完成状态。

当前目录不是可用Git根目录，HARL/.git也无可用提交。冻结实验源版本以已核验manifest与逐文件SHA为准；没有借用邻近名称相似的另一仓库。不创建提交、不推送、不改论文，也不执行新的控制器评估或训练。
