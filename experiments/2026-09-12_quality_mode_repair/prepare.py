from repair_support import *
import subprocess

def main():
    assert not (HERE / 'manifest.json').exists()
    frozen.verify()
    selected = read(FROZEN / 'seed_selection.json')
    assert selected['training'] == [104948945, 111868397, 160441552]
    old_probe = read(HERE.parent / '2026-09-12_credit_assignment_probe/manifest.json')
    paths = {WORKSPACE / p for p in old_probe['protected_input_sha256']}
    paths.update(p for p in (HERE.parent / '2026-09-12_credit_assignment_probe').rglob('*') if p.is_file() and p.suffix in ('.py', '.json', '.md'))
    paths.update(p for p in GREEDY.rglob('*') if p.is_file() and (p.suffix in ('.py', '.json', '.md') or p.name in ('predictor.npz', 'predictor_validation.npz')))
    rng = np.random.default_rng(202609122121)
    excluded = set(selected['training'] + selected['validation'] + selected['reserved_final_test'] + [selected['preflight_seed']])
    def new():
        while True:
            s = int(rng.integers(400000000, 1900000000))
            if s not in excluded: excluded.add(s); return s
    seeds = {str(s): dict(environment=[new() for _ in range(10)], initialization=[new() for _ in range(3)],
                          actions=[new() for _ in range(3)], updates=new(), global_initialization=new()) for s in selected['training']}
    configs = {}
    for seed in selected['training']:
        cfg = cfg_for(seed)
        a = cfg['algo_args']; mod, alg, tr = a['model'], a['algo'], a['train']
        expected = dict(lr=1e-4, critic_lr=4e-4)
        assert all(mod[k] == v for k, v in expected.items())
        assert (tr['n_rollout_threads'], tr['episode_length'], tr['num_env_steps']) == (10, 400, 1000000)
        assert (alg['ppo_epoch'], alg['clip_param'], alg['entropy_coef'], alg['gamma'], alg['gae_lambda']) == (5, .1, .008, .99, .95)
        assert not tr['use_linear_lr_decay'] and cfg['env_args']['T'] == 600
        status = read(model_dir(seed) / 'status.json')
        assert status['state'] == 'complete' and status['completed_steps'] == 1000000
        assert status['identity']['config_sha256'] == sha(FROZEN / f'configs/seed_{seed}/joint.json')
        for f, h in status['checkpoint_hashes'].items(): assert sha(model_dir(seed) / f) == h
        paths.update(model_dir(seed).iterdir())
        for arm in ('residual_all', 'residual_quality'):
            target = HERE / f'configs/seed_{seed}/{arm}.json'
            cfg_new = copy.deepcopy(cfg)
            cfg_new['training_design'] = dict(arm=arm, termination='finite_600_slot_task', version='quality_mode_repair_v1',
                parent_steps=1000000, added_steps=1000000, frozen_sut_deterministic=True,
                frozen_base_uav=True, critic_current_budget_added=False, adapter_hidden=[64,64],
                seeds=seeds[str(seed)], checkpoint_steps=[200000,400000,600000,800000,1000000])
            write(target, cfg_new); configs[str(target.relative_to(HERE))] = sha(target)
    profile = np.load(FROZEN / 'source/reference/inputs/profile.npz', allow_pickle=False)
    groups = []
    for mode in range(16):
        match = next((g for g in groups if all(np.array_equal(profile[k][mode], profile[k][g[0]]) for k in ('q_hat_mean','bar_ls_main_mean','avg_kept_real_symbols_mean'))), None)
        if match is None: groups.append([mode])
        else: match.append(mode)
    gitref = WORKSPACE.parent / '.github-sync/UASESC-MARL_20260911/checkout'
    def git(root, *args):
        p = subprocess.run(['git','-C',str(root),*args], capture_output=True, text=True)
        return dict(returncode=p.returncode, stdout=p.stdout, stderr=p.stderr)
    commit = '7cc2372d19f36ed7d91ed289f83dcc79fe0f3e9d'
    assert git(gitref, 'rev-parse','HEAD')['stdout'].strip() == commit
    tree = git(gitref,'ls-tree','-rz',commit,'experiments/2026-09-11_alternating_training')
    assert tree['returncode'] == 0
    entries = {row.split('\t')[1]: row.split('\t')[0].split()[-1] for row in tree['stdout'].split('\0') if row}
    differences = []
    for p in paths:
        rel = str(p.relative_to(WORKSPACE))
        if p.is_relative_to(FROZEN):
            data = p.read_bytes(); blob = hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest()
            if entries.get(rel) != blob: differences.append(rel)
    assert not differences, differences
    write(HERE / 'git_inspection.json', dict(reference_commit=commit, verified_frozen_blob_differences=differences,
        repositories={str(r): dict(head=git(r,'rev-parse','HEAD'),status=git(r,'status','--short')) for r in
          [WORKSPACE, gitref, WORKSPACE/'HARL/HARL', WORKSPACE/'CRL-SemCom-VidCI', WORKSPACE/'UASESE-MARL']}))
    write(HERE / 'seed_selection.json', dict(parent_models=selected['training'],validation=selected['validation'],
          reserved_final_test=selected['reserved_final_test'],preflight=[selected['preflight_seed']],new_seeds=seeds,
          generation_seed=202609122121,selection='one draw before outcomes; no replacements'))
    m = dict(schema=1,created_utc=stamp(),reference_commit=commit,parent_manifest_sha256=sha(FROZEN/'manifest.json'),
        parents=selected['training'],arms=['residual_all','residual_quality'],validation=selected['validation'],
        reserved_final_test=selected['reserved_final_test'],preflight_seeds=[selected['preflight_seed']],new_seeds=seeds,
        scenarios=read(FROZEN/'manifest.json')['scenarios'],milestones=[200000,400000,600000,800000,1000000],
        added_steps_each=1000000,total_added_steps=6000000,parent_steps_each=1000000,
        rules=['R_instruction','R_equal_instruction','greedy_modes_3','greedy_modes_16'],
        equivalence_groups=groups,protocol_sha256=sha(HERE/'PROTOCOL.md'),config_sha256=configs,
        execution_code_sha256={},protected_input_sha256={str(p.relative_to(WORKSPACE)):sha(p) for p in sorted(paths)},
        runtime=runtime_signature(),bootstrap_seed=202609122122,bootstrap_replicates=4000,
        tolerances=dict(physics_aoi=1e-12,usage=1e-8,common_reward=1e-9,float32_reward_atol_rtol=1e-6,
                        frozen_distribution=0.0,fixed_nonquality_trajectory=0.0,resume=0.0),
        initial_resource=dict(cpu_count=os.cpu_count(),affinity=sorted(os.sched_getaffinity(0)),load=os.getloadavg(),
            gpu=subprocess.run(['nvidia-smi','--query-gpu=index,name,memory.total,memory.used,utilization.gpu,temperature.gpu','--format=csv'],capture_output=True,text=True).stdout))
    write(HERE/'manifest.json',m)
    write(HERE/'status.json',dict(state='prepared',completed_training_steps=0,final_test_used=False))
    print(json.dumps(dict(prepared=str(HERE),protected_files=len(paths),seeds=seeds)))

if __name__ == '__main__': main()
