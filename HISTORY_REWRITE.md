# 2026-09-14 Git历史清理

用户追加要求清理历史中此前上传的模型和大体量数据。远程只有main；本次重写整个main可达历史，不创建保留旧产物的远程备份分支。

使用已有git-filter-repo，按上一轮PUBLICATION_MANIFEST已审查的3411个路径保留历史，并移除超过8MiB的blob。权重、拟合参数、数据集、原轨迹、密集数组和大型日志的路径均不在清单中；逐个历史版本检查路径、大小与物理查表类型。只含被排除产物的15个空提交被移除，24个原提交留下9个有内容的历史提交，再追加本说明。

过滤前后当前文件树完全一致：`a76636f70ec98a2d71ab9a0912ec62a699d164a4`。本次随后仅更新发布说明与清单；科学源码、配置、物理表和数值报告不变。原作者与有意义的代码/结果历史保留，提交SHA因历史过滤而改变。

[完整提交映射](HISTORY_REWRITE.json)保留原始SHA，以及过滤后对应提交。`pruned_after_filter=true`表示该提交只剩空修改；`filtered_equivalent`为其过滤后内容对应的保留祖先。历史科学报告中的旧SHA保持原样作为实验原始出处，不假装实验当时使用了改写后的SHA。完整原始Git历史及科研文件在本地备份中保留。

更新采用绑定原main SHA的`--force-with-lease`，防止覆盖并发提交。推送后执行不带depth或filter的独立完整克隆，核验全部历史和对象，检查已排除权重不存在于任何可达版本。发布任务的详细运行日志和恢复备份仅保留本地。

旧克隆请重新克隆当前仓库；保留旧文件作本地备份时，不要将旧历史合并回main。不能把当前分支历史已清理等同于GitHub服务器缓存已即时回收；缓存、其他人的旧克隆或fork不受此次操作控制。[GitHub说明](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/removing-sensitive-data-from-a-repository)
