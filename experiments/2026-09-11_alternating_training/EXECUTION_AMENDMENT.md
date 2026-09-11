# 执行资源调整，科学协议不变

最初6个CPU训练加2个评估并行，在2026-09-11T07:42:08Z因CPU持续85°C触发温控。六个训练任务均成功保存断点：三个joint各20000步、三个alternating各24000步，共132000步，没有失败任务。

仅调整执行资源：从性能核0–7改为能效核8–11，同时运行最多4个任务，六个训练任务排队；评估也计入4任务总上限。保留全部六个任务、训练预算、种子、参数、平均表、奖励、优化器状态、模型权重与Torch随机状态。

科学manifest.json、原训练器、原supervisor.py和runtime_control.py都保持原哈希。execution_override.py在加载并核验原manifest后，仅在内存中覆盖核绑定和并行数；训练仍使用原manifest做断点身份验证。execution_amendment.json记录本次执行调整及新启动器哈希。

调整后的首次恢复要求距离原热暂停至少3分钟，CPU与GPU均连续10秒低于80°C。之后仍使用原温控：CPU持续85°C或达到95°C、GPU达到83°C触发暂停；自动恢复使用原冷却标准。未修改传感器或其他项目进程。

当前恢复入口：`/home/king/miniconda3/envs/harl_sionna/bin/python execution_override.py start`。暂停仍为`/home/king/miniconda3/envs/harl_sionna/bin/python supervisor.py pause`。命令须在本目录执行，或将脚本替换为完整路径。旧run_training.sh保留用于复现最初执行布局；继续本次任务请用新的恢复入口。
