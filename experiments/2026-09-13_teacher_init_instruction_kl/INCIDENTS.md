# 本轮异常及处理

1. A1监督从A0热启动：`TypeError: RNG state must be a torch.ByteTensor`，train_imitation.py原第33行`rng.set_state(d['shuffle_rng'])`。map_location=cuda:0使CPU Generator状态被装到GPU。三个学生均在A1第一步梯度更新前停止。原日志保留在logs/fit_<seed>_A1.log、logs/stage_A.log。将字节张量送回CPU，逐值核对保存随机状态后继续。科学协议、种子、数据、模型和优化器状态不变；修复前后源码哈希见AMENDMENT_001.md。

2. 新离线审核第一次烟测命令误用`data/A0/batch_000.npz`；实际编号是`batch_00.npz`。`FileNotFoundError: .../data/A0/batch_000.json`发生在读取元数据时，未创建环境、未推进物理步、未更新参数。后续正确路径的教师与采样学生审核通过，见logs/audit_smoke.log。

3. 为完成审核采用三个CPU工作进程；原串行队列和后启动的分学生队列在结尾发生17份离线轨迹审核重叠。两次都只读同一完成轨迹，报告数值相同；实际增加408,000次单actor样本级教师查询和816,000个策略前向样本，全部写入costs账本。不增加任何物理采集、正式评估回合或梯度更新。当前完成入口不会扩大计算预算。

4. 阶段A工程准入0/3，不是实现异常。按预先协议停止critic预热和B0/B1，禁止追加训练修到过关。
