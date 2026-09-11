严格即时指令接口：本轮执行记录

新稿仅新增在Manuscript/main_tccn_strict_20260909.tex，原稿和上一轮修订稿保留；Manuscript没有新子目录。

两组共同变化：SUT待发送载荷改为不使用指令的全部16模式最小载荷汇总；actor侧16模式始终可提议。IC_HAPPO保留显式指令，HAPPO_hidden_instruction隐藏显式字段。执行器仍使用当前质量/资源/缓存规则；critic相同，历史状态仍可能反映过去指令。因此这是更严格的即时输入对照，不叫完全无指令系统。

修改只涉及CPU环境的_build_sut_obs、_build_action_available_masks及tensor环境的observe。奖励、状态递推、动作执行器、平均质量表、所有物理参数和训练超参数保持。AST检查确认step完全一致；2480步相同动作下的物理和奖励轨迹逐字节一致。270个匹配状态×3指令，覆盖满、部分、空缓存：隐藏actor的全部观测和掩码不变。每项CPU/GPU对照均9080槽、72640离散检查。两方法各8000步GPU短训练、CPU/GPU精确恢复、3600槽评价与汇总冒烟检查通过。

必须更正上一轮的写作错误：实际配置一直是all_modes，所有SNR下都允许全部16个模式，表的16行是SCI×rate组合。不是每个SNR桶4个模式；也没有“跨不允许SNR模式的SUT载荷汇总”问题。上一轮代码与运行结果不用因此重算。新稿按mode_mapping_audit.json及实例化环境的16×4映射证据修正。旧稿保留可追溯，后续以本轮新tex为准。

启动的诊断：seed85，IC_HAPPO、HAPPO_hidden_instruction各1M步，共2M步，新初始化，两GPU任务同时运行，CPU辅助核心16/18。按上一轮实测训练约16分钟，65组配对评价约3.5分钟，合计约20分钟；温控或机器负载可能延长。没有启动21项全量重训。局部温控阈值和恢复条件在PROTOCOL.md中。

本轮复用上一轮13场景和5个外部种子，以便诊断性对照；不作为新盲测，也不做多训练种子显著性结论。记录actor提议与实际模式的差异，帮助检查共同执行器的作用。程序完成后会自动在runs/seed85_1m/analysis/report.md和summary.json汇总结果。

运行入口（项目根目录，使用harl_sionna的Python，线程环境变量均设为1）：

```bash
python experiments/2026-09-09_strict_instruction/pilot.py run --run experiments/2026-09-09_strict_instruction/runs/seed85_1m
```

已有进程运行时不要再次执行；恢复时会由同设备完整checkpoint续训。新独立实验应先用pilot.py prepare准备新run路径。冻结输入不能修改。
