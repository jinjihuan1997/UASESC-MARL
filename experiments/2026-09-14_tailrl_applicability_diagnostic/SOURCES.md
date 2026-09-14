# 方法依据

- [Tail-Likelihood Reinforcement Learning，arXiv:2609.02987v1](https://arxiv.org/html/2609.02987v1)，2026-09-02。查阅§4、有限阶排序间隔估计以及附录C/D；本轮实现有限N、中心化排序优势，不把它称作精确无限尾部梯度。
- [官方仓库 Zanette-Labs/TailRL](https://github.com/Zanette-Labs/TailRL)。本轮没有安装或运行其训练流水线，也没有将生成任务的Best-of-N选择当成本系统可部署动作规则。
- 本地科学输入：manifest.json、history_index.json、training_evidence_index.json；模型、profile、配置和具体源码均以实际路径及SHA-256为准。
