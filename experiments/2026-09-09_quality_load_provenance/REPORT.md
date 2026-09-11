**质量—载荷表来源复核，2026-09-09**

结论：当前SC训练使用的是“以历史SCI验证PSNR为基准、用确定性规则构造的平均质量—载荷模型”。旧CC表也是公式模型。项目确实另有一张可从逐次视频测量记录重建的SC表，但它不是当前使用的表。不能把几张表的来源、模式含义和实验结果混在一起。

本次检查覆盖：正式训练配置和冻结输入、SC/CC表读取路径、原始NPZ全部字段、历史SCI TensorBoard日志、现有SC/CC导表脚本、9月7日校准原始记录及权重哈希、数据索引与实际加载器输出、主稿中相关描述。没有重新执行1540次视频推理，没有重新训练编解码器，也没有审计项目内与表无关的所有第三方代码。全部原有表、权重和运行中实验保持原样。

**1. 先分清四张表**

|表|来源证据与内容|当前正式训练是否使用|
|---|---|---|
|`author_sci_profile_snr0_20.npz`|16×5；四个历史SCI质量基准减去共同SNR损失，各SCI下四个码率质量相同；载荷为固定档位|否|
|`derived_coupled_profile_snr0_20.npz`|16×5；在上一张表上加入码率质量限制，25个质量单元改变，所有载荷数组完全相同|是，字节完全一致的冻结副本|
|`cc_h264_ldpc_profile_snr0_20.npz`|5×5；`cc_synth_v1`，可用现有CC近似公式重建|否，CC待补训|
|9月7日`calibration/profile.npz`|5×7；由1120次校准视频重建记录取均值，另有420次诊断记录|否，历史初筛使用|

前三张位于[profiles目录](/home/king/Downloads/Projects/2_th_paper_TMC/HARL/HARL/harl/envs/uav_escs/semantic_models/profiles)，第四张位于[实测校准目录](/home/king/Downloads/Projects/2_th_paper_TMC/experiments/2026-09-07_happo_instruction_ab/calibration)。文件修改时间显示旧SC两张表为6月17日，CC为6月18日；修改时间仅是文件属性，不单独作为历史生成证据。

正式训练的`semantic_profile_path`实际指向：

`experiments/2026-09-09_instruction_long_training/runs/three_seed_sc_20260909/source/reference/inputs/profile.npz`

它与旧派生SC表的SHA-256均为`a6f721686b28d6c28ccf811ab8b63e80ae8c135a08b9d7e1b4bbfe7ca13d91cc`。本次完整哈希和字段导出见[evidence.json](/home/king/Downloads/Projects/2_th_paper_TMC/experiments/2026-09-09_quality_load_provenance/evidence.json)。

**2. SC质量基准可以追溯到原始训练日志**

表内旧来源路径指向`CRL-SemCom-VidCI/logs/24-03-08/24-03-08-MST/MST_fixed/v_1`至`v_4`。我读取了这四份TensorBoard事件文件的全部`val/psnr`记录：

|SCI日志|最后记录步数|最后验证PSNR|四舍五入后|表内20 dB质量基准|
|---|---:|---:|---:|---:|
|v_1|94000|25.7988147736|25.80|25.80|
|v_2|82000|30.7224311829|30.72|30.72|
|v_3|83000|32.8124465942|32.81|32.81|
|v_4|100000|32.3668441772|32.37|32.37|

四个数全部一致，且不是这四段训练记录中的最高PSNR。这是表内质量基准来自这些历史最终验证值的强证据；本地没有找到当时生成旧表的原始命令，所以不能把数值一致进一步说成已经找回完整生成日志。历史验证数据的具体划分、每段原始视频质量和当时完整运行代码，也不能仅从这四个最终标量恢复。

第一张SC表的80个质量值可以零误差重建：

`Q_author(sci, rate, snr) = C_sci - D_snr`

其中`C=[25.80,30.72,32.81,32.37]`，SNR网格为`[0,5,10,15,20] dB`，`D=[4.09,1.61,0.52,0.14,0] dB`。四个码率不影响这张表的质量，只改变载荷。

这组共同SNR损失的独立测量来源没有找到。因此可确认其数值和作用，不能声称它已由当前SCE/SCD逐模式传输测试验证。

**3. 当前派生表如何让码率影响质量**

派生表保留相同SNR网格与所有载荷，只调整质量。以下等价表达可零误差复现全部80个质量值：

`Q_derived(sci, rate, snr) = min(C_sci, R_rate) - D_snr`

`R=[25.80,30.72,32.81,32.81]`是一组能复现表值的单调码率质量上限。第3、4档取不低于32.81的值时，在现有SCI基准下无法区分，因此原生成程序的上限写法不能唯一反推。这里报告的是经过验证的等价构造，不是宣称找到了原始实现。

20 dB下的完整质量关系如下；其他SNR统一减去对应的`D`：

|表内SCI档位|rate 1|rate 2|rate 3|rate 4|
|---|---:|---:|---:|---:|
|sci0 / meas1|25.80|25.80|25.80|25.80|
|sci1 / meas2|25.80|30.72|30.72|30.72|
|sci2 / meas4|25.80|30.72|32.81|32.81|
|sci3 / meas8|25.80|30.72|32.37|32.37|

因此，“16行”不等于“16个分别实测的SCE/RAN/SCD模型”。旧表的`checkpoint_paths`全部为`derived`，`mu_comm`和`trained_snr_db`全部为NaN；当前冻结注册表也明确清空检查点映射，只把它当作固定表。旧集合名中的`snr0_15_mu_grid_16modes`保留是为了格式匹配，不能用于解释当前模式含义。

一个实际影响：只看该表的质量和载荷、将相同值合并后，各SNR下只有3种非劣权衡，代表行为是低质量低载荷、中质量中载荷、高质量较高载荷。对应代表行是0、5、10；其余行要么与它们重复，要么存在质量不低且载荷更小的替代行。精确的非劣行与重复组保存在证据JSON中。这说明有效通信选择比16个名称少，但不等于整个资源调度或RL问题只有3种策略。

**4. SC载荷是怎样得到的**

两张旧SC表的主载荷均为：

|码率档位|有效实坐标数`n_z`|主载荷：复信道使用次数`L_z`|
|---|---:|---:|
|1|4096|2048|
|2|8192|4096|
|3|12288|6144|
|4|16384|8192|

它们与SCI档位、SNR和具体视频内容无关。表内侧信息开销为`2048/log2(1+gamma)`，表内总载荷为主载荷加该开销。所有这些值均已逐单元精确复算。

实际RL读取主载荷后重新计算侧信息：`2052/log2(1+gamma)`，其中多4 bit表示16模式的标识。RL不直接读取表内`bar_ls_total_mean`，所以没有把侧信息重复加两次。SNR在网格之间线性插值，网格外取端点质量/主载荷；侧信息仍用当前实际SNR计算。[读取实现](/home/king/Downloads/Projects/2_th_paper_TMC/HARL/HARL/harl/envs/uav_escs/semantic_models/semantic_registry.py:242)。

例如10 dB、rate 1：表内总载荷为`2640.004764`，当前RL实际使用约`2641.161023`次复信道使用。两者差异来自4 bit模式信息。

另一个需要说明的口径：在32×32 latent网格、4码率均匀分组的假设下，`L_z=2048*rate`相当于16个latent通道；48通道实现对应的是`6144*rate`。项目确实有`SemCom_joint_modes_lat16`训练目录，但仅有该目录不能证明这张表是该模型的实测结果；派生表没有相应检查点哈希或逐次测量记录。不能一边使用这张表的载荷，一边声称它是已登记48通道模型直接测出来的载荷。

**5. CC表的生成关系也已重建**

旧CC表声明`schema_version=cc_synth_v1`、`dataset_name=synthetic_h264_ldpc`。使用[CC环境公式](/home/king/Downloads/Projects/2_th_paper_TMC/HARL/HARL/harl/envs/uav_escs/CC/uav_escs_env_cc.py:163)和[CC配置](/home/king/Downloads/Projects/2_th_paper_TMC/HARL/HARL/harl/configs/envs_cfgs/uav_escs_cc.yaml:57)，它可以如下重建：

`b_source = (256*256*32)*0.010*2^((42-QP)/6)*(1+0.60*psi_mean) + 512`

`b_coded = b_source / code_rate`

`L_CC = b_coded / log2(1+10^(SNR_dB/10))`

`Q_CC = 28 + 0.55*(42-QP) - 1.20*psi_mean - 2*max(SNR_required-SNR_dB,0)`

QP为`[42,38,34,30,26]`，码率为`[0.5,0.6666667,0.75,0.8333333,0.8333333]`，SNR门限为`[-1,2,5,8,8]`。由一格质量值反推出`psi_mean=0.16626688527057176`，再独立检查全部质量与载荷：25个质量单元最大误差约`1.07e-14 dB`，25个载荷单元最大误差约`4.37e-11`。反推的平均内容值来自何种原始样本总体尚未查到。

这说明它是现有近似公式在一个固定平均内容值下的离线导出，而不是当前真实编码测试脚本的产物。旧表把`b_coded`放在名为`avg_kept_real_symbols_mean`的字段内；这个字段在SC中表示实坐标，在旧CC中实际表示编码后bit数，不能直接横向比较。

[真实CC导表脚本](/home/king/Downloads/Projects/2_th_paper_TMC/HARL/HARL/scripts/build_cc_h264_ldpc_profile.py:190)是另一条路径：读视频→ffmpeg H.264编码/解码→Sionna LDPC/QAM/AWGN得到BLER→估计交付概率、质量与载荷。它保存`bler`、`deliver_prob`、`h264_coded_bits_mean`、样本数等字段，固定码率/调制时载荷不随SNR改变；现有旧CC表没有这些字段，且载荷按容量随SNR下降。脚本存在不等于这张旧表由它生成。

还应注意[CC配置第69行附近](/home/king/Downloads/Projects/2_th_paper_TMC/HARL/HARL/harl/configs/envs_cfgs/uav_escs_cc.yaml:69)的注释：`h264_ref_bpp`被描述为按round-robin服务数与SC接近、同时维持CC较低质量载荷效率来标定。该注释确实存在；本次没有找回当时完整调参记录。后续正式比较需要独立解释或标定这个系数，不能用预设SC优势的系数再证明SC本身更高效。

**6. 项目中真正有原始测量记录的是哪张表**

[calibrate_instruction_pilot.py](/home/king/Downloads/Projects/2_th_paper_TMC/HARL/HARL/scripts/calibrate_instruction_pilot.py:52)按原登记集查找本地可用检查点，选取5个，另外11个排除。使用20个测试视频的固定分割：0—7校准、8—13诊断、14—19保留。

校准选16个片段，诊断选6个片段；5模型×7 SNR点×22片段×2次随机噪声，共1540次重建。只有1120次校准记录进入5×7表，每格32次；420次诊断记录不用于填表。

本次重新聚合了全部原始记录。质量均值、质量标准差、主载荷和实坐标数与保存表的差异全部为0；原始记录哈希、校准计划哈希、数据索引哈希和5份权重哈希全部匹配。诊断PSNR预测MAE为3.298870 dB、RMSE为3.841124 dB。这里验证的是已有测量证据的完整性与聚合，不是本次重新运行1540次网络推理。

因此，不能说项目从来没有真实视频测量；正确说法是：曾经建立过一张有测量证据的小规模平均表，随后当前训练按用户决定使用原来的16行最终派生表。两者的模式数、输入业务定义、载荷和质量范围不同，不能直接替换后沿用现有训练结果。

**7. 数据集、8帧、16帧和32帧到底是什么关系**

当前本地NFS索引有80个训练视频/5984个片段、20个测试视频/1511个片段，每个索引项列出32张RGB图像路径。我实际读取了一个样本确认：

|对象|实际形状或配置|
|---|---|
|目录名|`nfs_block_rgb_256_8f`|
|索引项|32张RGB图像路径|
|模型输入与重建目标|各16×256×256，灰度|
|参考输入|16×64×64，灰度|
|当前RL业务元数据|8×256×256，RGB|
|旧CC公式中的源图像量|32×256×256像素，公式没有单独颜色通道因子|

代码先使用前16张图片形成主输入/目标，后16张形成低分辨率参考，内部还有8帧分段调换；默认`color=False`。依据：[dataio.py](/home/king/Downloads/Projects/2_th_paper_TMC/CRL-SemCom-VidCI/src/dataio.py:115)。

此前“SC8帧、CC32帧”的说法准确描述了当前RL元数据与旧CC配置的差异，但不足以说明实际视频编解码口径。目录中的`8f`不是可直接使用的帧数证明。CC配置注释还提到VIDI，但当前实际检查到的输入索引指向NFS，这个注释也不能代替数据来源证据。

**8. 对论文和下一步CC训练的意义**

当前SC训练是在固定平均模型下学习指令、资源和调度策略。它能回答“在这个模型环境中，哪些策略表现更好、指令是否改变行为”，不能单凭这些结果证明真实SCE/SCD视频传输性能，也不能证明旧CC公式准确代表H.264/LDPC实际性能。[main.tex第365行](/home/king/Downloads/Projects/2_th_paper_TMC/Manuscript/main.tex:365)和[第552行](/home/king/Downloads/Projects/2_th_paper_TMC/Manuscript/main.tex:552)关于逐配置离线验证的表述需要与实际证据对应。

用户已确定保留当前平均表，本次没有更换它。后续补CC需要先明确比较属于“固定平均模型下的系统控制仿真”还是“真实编解码性能对比”。前一种可以继续使用平均模型，但CC参数来源与公平业务量必须独立说明，不能按所期望的SC优势反调；后一种需要相同实际目标视频、参考信息口径、指标和信道设置下的SC/CC测量证据。无论哪条路线，都不能仅将旧CC载荷除以4、保留其它假设后直接声称完成公平对齐。

**9. 复现与证据边界**

本次新建[audit_profiles.py](/home/king/Downloads/Projects/2_th_paper_TMC/experiments/2026-09-09_quality_load_provenance/audit_profiles.py)，只读已有数据，输出新证据文件。可用以下命令复现，输出文件必须尚未存在：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
/home/king/miniconda3/envs/harl_sionna/bin/python \
/home/king/Downloads/Projects/2_th_paper_TMC/experiments/2026-09-09_quality_load_provenance/audit_profiles.py \
--output /tmp/quality_load_provenance_recheck.json
```

审计验证通过。旧表原始生成命令、共同SNR损失的测量来源、CC平均内容值所用样本总体、旧表与特定端到端检查点的完整绑定仍是UNKNOWN。不能通过数值拟合把这些缺失证据补写成已证实的历史事实。
