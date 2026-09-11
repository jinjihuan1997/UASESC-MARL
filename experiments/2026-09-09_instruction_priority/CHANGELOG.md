本轮修订与验证

原稿：Manuscript/main.tex，保留。新稿：Manuscript/main_tccn_revision_20260909.tex。Manuscript没有新增子目录、图片、bib或编译产物；参考文献内嵌在新tex中，临时编译输出在/tmp/tccn_revision_build。

|问题|性质|本轮处理|是否重新训练|
|---|---|---|---|
|Dirichlet输出再次softmax，收窄资源范围|动作接口实现与预期模型不一致|新版本直接映射剩余份额，单UAV可逼近.05至.90|是|
|AoI均值用600归一化，正常变化对基础奖励很弱|奖励设计，不是AoI递推错误|独立奖励参考10，状态上限保留600|是|
|三种指令基本奖励权重相同|目标设计|质量/AoI/开销权重随指令变化|是|
|原环境接近缓存流水线下限|资源与服务门限设置|保留物理B/P，eta=.6；调整AoI质量门限及三个年龄软目标|是|
|缓存、时间戳、1秒/8帧及AoI公式不清|论文口径与代码对应|按当前阻塞缓存递推重写，解释2.4967槽有限时域下限|没有修改缓存递推|
|把平均服务表写成逐视频实测或特定codec推导|证据范围/写作|保留最终表；明确是预设平均质量—载荷模型|没有重新标定表|
|模式提议、实际调度和同步策略决策混淆|论文与实现对应|明确同时提议、规则执行；没有学习DS调度头|沿用实现|
|AF、带宽/功率、元数据及信道损耗假设缺失|科学建模范围|补充等效AF、固定PSD和理想元数据假设，区分channel uses与bits|未声称加入新物理仿真|
|隐藏指令被理解成系统不接收指令|实验解释|仅隐藏actor显式字段；共同约束及间接信号仍在|保留同口径对比|
|CC32帧与SC8帧不对齐|对比口径|保留CC名称，推迟到对齐后补训|本轮不启动CC|

实现文件以reference/runtime/harl/envs/uav_escs/SC/uav_escs_env_sc.py、tensor_env.py、reference/evaluate.py、reference/runtime/harl/envs/uav_escs/SC/rules.py、reference/config/base_env.json为主。代码、配置和最终表都复制到独立版本；旧2026-09-08_gpu_environment没有修改。

完成的检查：
- CPU与张量环境、实际CUDA与CPU环境：每项9080槽、72640离散值检查；新版本首次CPU对照发现未同步的AoI归一化，修复后v2通过，失败记录保留。
- 独立资源映射、eta不改SNR、预算比例、奖励算式与状态上限检查通过。
- 参数开发集36个600槽episode通过；完整保留eta四点，未按学习优势选参数。
- 两方法GPU短训练各8000步通过，参数有限且已更新。
- CPU和实际CUDA完整checkpoint恢复检查通过，两方法续训均与不间断参考得到相同网络哈希。

已完成pilot由pilot.py执行：IC_HAPPO和HAPPO_hidden_instruction各1M，seed85，两GPU任务并行，并已完成65组配对episode评估。pilot源码及配置已经冻结在runs/seed85_1m/source和configs。后续分析脚本summarize_pilot.py只读这些冻结输入与输出。最终结果尚不能替代原计划三种子正式实验。

复现入口（在本版本目录，使用harl_sionna的Python）：

```bash
python validate.py --help
python check_model.py
python check_resume.py
python pilot.py prepare --run /path/to/a/new/run
python pilot.py run --run /path/to/a/new/run
python summarize_pilot.py --run /path/to/completed/run
```

训练过程中不可重建该run的manifest或修改冻结输入。已有run可在相同设备/运行库条件下恢复；不要以新的总训练步数续跑旧checkpoint。

补充边界检查：instruction_visibility.json在300个满缓存匹配信道状态下，隐藏组UAV观测不变，但SUT待发送载荷特征100%变化，三种指令联合UAV动作掩码100%互异。这是隐藏显式字段的对照，不是移除所有指令信息。SUT载荷特征是跨所有质量合格模式的乐观汇总，实际调度还受当前SNR模式桶约束；不应把这个特征写成严格可行的最小需求。此检查没有修改训练输入。

最终结果：两方法各100万步、250次更新完成；65组配对评价共78000槽完成，外生过程哈希匹配。单种子总体奖励IC=-0.024927，hidden=-0.022443，差值=-0.002485；5个外部种子上的场景平均差值均为负。IC在balance指令较好，aoi和quality组较差。质量、预算、缓存违规均为0。两方法GPU并行训练约16.2分钟，评价约3.5分钟，总计约20分钟；训练CPU峰值76C/GPU61C，没有暂停。

本轮结论：环境已形成服务取舍，显式指令总体收益尚未确认。下一步应先评估一个进一步移除间接指令提示的对照设计，同时保留本轮结果，不直接把现有对照称为完全无指令。该诊断不能证明间接提示是IC未获益的唯一原因，也不能用1M单种子结果断言10M多种子必然失败。本轮没有启动全量21项重训。
