"""Freeze all decisions before looking at diagnostic scores."""
from support import *
import subprocess


def git(root, *args):
    p = subprocess.run(['git', '-C', str(root), *args], text=True, capture_output=True)
    return dict(returncode=p.returncode, stdout=p.stdout, stderr=p.stderr)


def main():
    assert not (HERE / 'manifest.json').exists(), 'Do not overwrite an existing manifest'
    original = frozen.verify(); selected = read(FROZEN / 'seed_selection.json')
    assert selected['training'] == [104948945, 111868397, 160441552]
    excluded_seeds = selected['training'] + selected['validation'] + selected['reserved_final_test'] + [selected['preflight_seed']]
    # Single deterministic draw made before any new execution. Never score-filtered.
    seed_rng = np.random.default_rng(202609120731)
    seeds = []
    while len(seeds) < 12:
        s = int(seed_rng.integers(400000000, 1900000000))
        if s not in excluded_seeds + seeds:
            seeds.append(s)
    action_seeds = [710921, 710929, 710933]
    probe_actions = [820921, 820929, 820933, 820939]
    fit_seeds = [930921, 930929, 930941]
    protocol = f'''# 冻结策略归因诊断与预算条件价值探针

本次只诊断 2026-09-11 实验的三个 joint、固定 1,000,000 步模型。原源码、模型、critic、配置、表、报告和论文只读；所有输出在本目录；不提交或推送、不启动 RL 训练。Python 写入审计禁止向本目录之外写文件；actor 参数 requires_grad=False，原 actor 优化器 step 被只读包装阻止，原 critic 不参与优化。

## 阶段一

训练种子：{selected['training']}。20 个环境种子按原 validation 列表原顺序整批运行，均作为开发诊断。三个场景 fixed_0/fixed_1/fixed_2，每回合600物理槽。

A：确定资源/确定模式；B：采样资源/确定模式；C：确定资源/采样模式；D：两者采样；E：均分资源/原确定模式；F：原确定资源/原 R_instruction 模式。B/C/D 的动作种子为 {action_seeds}，每个动作种子独立运行一个20环境批次；A/E/F各一次。共2160个新回合、1,296,000物理步。另在同样环境重新运行两个原规则，每规则60回合，共120回合、72,000步，基线不当作训练种子重复。

各槽按原实现：观察→一次 allocate_resources→新 UAV 观测/掩码→一次 commit_modes(auto_reset=False)→原独立物理与奖励检查。Dirichlet 确定性动作保留原 mean 实现；下限只由环境应用一次。F直接调用原 simple_rule_actions 的模式部分，不改变映射和回退。

环境独立 EpisodeSource 流不使用 Torch 动作 RNG。每个批次分别保管 SUT/UAV1/UAV2/UAV3 的 Torch 状态，子种子由 SeedSequence([动作种子, agent_id,12092026]) 生成；同一动作种子不同方法保留相同初始流。固定批次布局，完整记录种子、初末状态、初始拓扑/缓存/内容/信道/指令哈希。读取动作头 forward 的分布输出以获取概率和浓度，不额外抽样。

每步调用原 check_tensor，原 AoI 容差1e-12、用量1e-8、common_reward1e-9；浮点训练奖励与共同奖励沿用1e-6相对/绝对容差。A与原历史完整轨迹对比。CPU/CUDA计时只决定设备；不按分数挑设备。预检全部通过后才运行正式批次。

## 阶段二

一次确定的新拟合环境种子：{seeds[:8]}；留出环境种子：{seeds[8:]}。来自固定生成器种子202609120731，与原训练/验证/预检/预留最终测试种子不重叠。每环境动作重复 {probe_actions}。每重复按这12个种子的顺序整批运行；拟合/留出只按环境种子划分。3模型×12环境×4重复×600槽=86,400步。沿用原 random_switch_once 生成机制，SUT与UAV均按原策略采样。

标签为完整600槽真实 float32 训练奖励计算的折扣 Monte Carlo 回报，gamma读取原配置（当前0.99）。真实终止处不bootstrap。P0输入原分配前完整critic状态和3个零；P1输入相同状态和3个本槽实际份额。原state的第4:7列是上槽份额，双方保留；新增预算不包含模式、当步奖励或未来。

每父模型分别拟合P0/P1：2层256 ReLU，Adam lr=1e-3，batch1024，50轮，初始化种子 {fit_seeds}。同一初始化双方复制同一初始权重，使用同一训练样本和每轮顺序。输入均值方差只用拟合集：公共状态共用统计，新增3槽采用真实份额的拟合集统计；P0追加槽在归一化后仍设零。标签归一化也只由拟合集决定。固定最后一轮评分，留出数据不选超参数、轮数或检查点。名义参数量严格相同。

留出报告MSE/MAE/解释方差/残差均值和方差，按指令、实际份额区间[0,.15,.25,.35,.50,.70,1.01]及剩余槽(1–100,101–300,301–600)分组。另计算UAV分类logit score梯度代理 (onehot(采样模式)-概率)*(G-V)，只针对UAV，不计算SUT动作依赖基准。它不是全参数梯度或HAPPO更新方差；不把预测改善等同RL得分提升。

## 统计与资源

原始奖励和奖励×100并列，回合总回报另列；主要分数为完整回合平均奖励×100。只汇总实际运行的3固定场景，不伪造13场景总分。模式既报告原0–15编号，也按profile三数组逐值精确相等分组；保持16动作空间。

动作重复先在模型/环境/场景内平均；报告全部训练种子。阶段一95%区间使用2000次环境种子成组配对bootstrap，共同抽样环境索引保留模型与场景配对，明确以当前冻结模型为条件；规则仅独立运行一次。阶段二以留出环境种子为cluster（保留4动作回合及全部时隙），2000次配对bootstrap；只有4个留出环境，区间仅作小规模诊断。bootstrap种子202609120911。不用时隙数作独立样本量。

默认1个进程、Torch/BLAS各1线程，CPU/GPU预检实测后在preflight.json冻结设备选择，最大并发1。不终止其他项目。可恢复单位为完整600槽批次，trace内容和协议/输入/代码哈希一致才复用，部分失败文件不算完成。每次拟合可从本诊断自己的完整epoch断点恢复，不触及原模型。

本次固定预算，不补种子、不挑赢家。预留最终测试严格禁止进入构造环境的白名单。完成后停止。
'''
    (HERE / 'PROTOCOL.md').write_text(protocol)
    paths = set()
    execution_paths = set()
    for rel in original['input_hashes']:
        p = FROZEN / rel; paths.add(p); execution_paths.add(p)
    paths.update([FROZEN / 'manifest.json', FROZEN / 'seed_selection.json'])
    for seed in selected['training']:
        paths.update(model_dir(seed).iterdir())
        execution_paths.update(model_dir(seed).iterdir())
        cfg = adapted_config(seed)
        write(HERE / 'configs' / f'seed_{seed}.json', cfg)
    for p in FROZEN.rglob('*'):
        if p.is_file() and (p.suffix in ('.md', '.json') or p.parent == FROZEN / 'report'):
            paths.add(p)
    for category in [f'seed_{s}/joint_at_1000000' for s in selected['training']] + ['rules/R_instruction', 'rules/R_equal_instruction']:
        for g in range(3):
            paths.add(FROZEN / 'evaluation' / category / f'fixed_{g}.npz')
            paths.add(FROZEN / 'evaluation' / category / f'fixed_{g}.json')
    paths.update(p for p in (WORKSPACE / 'Manuscript').rglob('*') if p.is_file())
    paths = sorted(paths)
    hashes = {str(p.relative_to(WORKSPACE)): sha(p) for p in paths}
    tree = git(REFERENCE_GIT, 'ls-tree', '-rz', REFERENCE_COMMIT, 'experiments/2026-09-11_alternating_training')
    assert tree['returncode'] == 0
    entries = {}
    for line in tree['stdout'].split('\0'):
        if line:
            spec, path = line.split('\t'); entries[path] = spec.split()[-1]
    differences = []
    for p in paths:
        rel = str(p.relative_to(WORKSPACE))
        if not rel.startswith('experiments/'): continue
        if rel not in entries:
            differences.append(dict(path=rel, kind='not_in_reference_tree')); continue
        data = p.read_bytes()
        blob = hashlib.sha1(f'blob {len(data)}\0'.encode() + data).hexdigest()
        if blob != entries[rel]: differences.append(dict(path=rel, kind='content_changed'))
    assert not differences, differences
    git_info = {str(p): dict(head=git(p, 'rev-parse', 'HEAD'), status=git(p, 'status', '--short'))
                for p in [WORKSPACE, REFERENCE_GIT, WORKSPACE / 'HARL/HARL', WORKSPACE / 'CRL-SemCom-VidCI', WORKSPACE / 'UASESE-MARL']}
    for p in [WORKSPACE / 'HARL/HARL', WORKSPACE / 'CRL-SemCom-VidCI']:
        (HERE / 'logs' / (p.name + '_original_worktree.diff')).write_text(git(p, 'diff', '--no-ext-diff')['stdout'])
    write(HERE / 'git_inspection.json', dict(reference_commit=REFERENCE_COMMIT, git=git_info,
          reference_differences=differences, note='Workspace is a project root without .git; published checkout supplies the reference Git identity. Frozen inputs match its committed blobs.'))
    resource = subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.total,memory.used,utilization.gpu,temperature.gpu', '--format=csv'], capture_output=True, text=True)
    write(HERE / 'resource_initial.json', dict(python=sys.executable, version=sys.version, torch=torch.__version__,
          numpy=np.__version__, cuda_available=torch.cuda.is_available(), cpu_count=os.cpu_count(),
          affinity=sorted(os.sched_getaffinity(0)), load=os.getloadavg(), gpu=resource.stdout, torch_threads=1, max_workers=1))
    relocations = []
    for seed in selected['training']:
        old = frozen.config('joint', seed); new = adapted_config(seed)
        for key in ['semantic_registry_path', 'semantic_profile_path']:
            relocations.append(dict(seed=seed, field=key, original=old['env_args'][key], resolved=new['env_args'][key]))
    write(HERE / 'manifest.json', dict(schema=1, created_utc=stamp(), reference_commit=REFERENCE_COMMIT,
          workspace=str(WORKSPACE), frozen_experiment=str(FROZEN), reference_git=str(REFERENCE_GIT),
          training_seeds=selected['training'], validation_seeds=selected['validation'],
          reserved_final_test_seeds=selected['reserved_final_test'], preflight_seeds=[selected['preflight_seed']],
          probe_fit_seeds=seeds[:8], probe_holdout_seeds=seeds[8:], action_seeds=action_seeds,
          probe_action_seeds=probe_actions, fit_initialization_seeds=fit_seeds,
          scenarios=['fixed_0', 'fixed_1', 'fixed_2'], gamma_by_model={str(s): adapted_config(s)['algo_args']['algo']['gamma'] for s in selected['training']},
          environment_steps=dict(execution=1296000, rules=72000, probe=86400), stage_one_episodes=2280,
          probe_episodes=144, final_test_run=False, policy_training=False, max_workers=1,
          path_relocations=relocations, protected_input_sha256=hashes,
          execution_input_paths=[str(p.relative_to(WORKSPACE)) for p in sorted(execution_paths)],
          protocol_sha256=sha(HERE / 'PROTOCOL.md'), diagnostic_code_sha256={},
          python=sys.executable, torch=torch.__version__, numpy=np.__version__))
    save_status('prepared', completed_execution_episodes=0, completed_probe_episodes=0, completed_probe_fits=0)
    print(json.dumps(dict(state='PREPARED', path=str(HERE), protected_files=len(paths), probe_seeds=seeds)))


if __name__ == '__main__': main()
