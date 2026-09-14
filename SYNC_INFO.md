# 2026-09-14同步说明

这是对main分支的正常追加提交，保留此前7cc2372及其历史，不使用force push。所有科学源码、原始报告、数值和模型文件均来自本地已完成实验，不在同步时修改算法或结果。README与本说明为发布索引。

实际上传清单在发布任务本地审计目录中逐文件保存SHA-256、Git blob ID和模式，推送后重新从GitHub取回提交树核对。原始工作区和其未提交修改不由发布任务写入或清理。

不上传大于10MiB的模型文件、完整训练状态、原始数据集、教师/DAgger训练数据NPZ以及本地安装/缓存/私有配置。小型策略权重按上次已授权范围保留。不存在按得分删除实验或替换种子。

## 唯一文件格式转换

- 原路径：`experiments/2026-09-12_credit_assignment_probe/logs/HARL_original_worktree.diff`
- 发布路径：`experiments/2026-09-12_credit_assignment_probe/logs/HARL_original_worktree.diff.gz`
- 原始字节数：132364514
- 原始SHA-256：`7a31d84de7ae09675bf6a774cee3cfa3dfb7eb22418cbbfe736233b43956fdd2`
- gzip SHA-256：`43b95cb8d43c8b221b121788188468c51cc607d5f0f40e63ba4d71e4b2c2e90f`

该日志原大小超过100MiB。gzip采用无损压缩，解压后内容已按SHA-256核对；本地原文件没有改变。若需要在新检出副本还原，可在目标路径尚不存在时运行：

```python
from pathlib import Path
import gzip, hashlib
src = Path("experiments/2026-09-12_credit_assignment_probe/logs/HARL_original_worktree.diff.gz")
dst = src.with_suffix("")
assert not dst.exists()
with gzip.open(src, "rb") as reader, dst.open("xb") as writer:
    while chunk := reader.read(1024 * 1024):
        writer.write(chunk)
```

数值结果、CSV/JSON/NPZ轨迹没有采用有损变换。模拟器生成的评估轨迹属于结果记录；教师监督训练用的数据包留在本地，不能把排除训练数据解释为该阶段没有训练成本。
