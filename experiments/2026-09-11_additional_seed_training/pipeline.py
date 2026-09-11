"""One predeclared supplementary seed: base -> paired continuation -> reporting."""
import argparse,datetime,fcntl,hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path

ROOT=Path(__file__).resolve().parent
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def write(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
def immutable(p,value):
    if Path(p).exists():assert read(p)==value,p
    else:write(p,value)
def verify():
    for f,h in read(ROOT/'code_manifest.json')['input_hashes'].items():assert sha(ROOT/f)==h,f
    return read(ROOT/'PLAN.json')
def alive(record):
    try:
        p=Path('/proc')/str(record['pid']);fields=(p/'stat').read_text().split(') ',1)[1].split()
        return p.stat().st_uid==record['uid'] and int(fields[19])==record['start_ticks'] and fields[0]!='Z'
    except (FileNotFoundError,KeyError,IndexError):return False
def process_record(pid):
    p=Path('/proc')/str(pid);fields=(p/'stat').read_text().split(') ',1)[1].split()
    return dict(pid=pid,uid=p.stat().st_uid,start_ticks=int(fields[19]))

def make_manifest(phase):
    plan=verify();p=ROOT/phase;seed=plan['seed'];base=phase=='base'
    if (p/'manifest.json').exists():
        m=read(p/'manifest.json')
        for f,h in m['input_hashes'].items():assert sha(p/f)==h,f
        return m
    methods=['joint'] if base else plan['arms'];steps=10000000 if base else 2000000
    milestones=[10000000] if base else [1000000,2000000]
    jobs=[dict(kind='training',id=f'seed_{seed}/{a}',seed=seed,method=a,device='cpu',core=i,
        config=f'configs/seed_{seed}/{a}.json',output=f'jobs/seed_{seed}/{a}') for i,a in enumerate(methods)]
    ev=[]
    if not base:
        ev=[dict(kind='evaluating',id=f'rules/{r}',output=f'evaluation/rules/{r}') for r in ['R_single','R_instruction','R_myopic']]
        for step in milestones:
            for arm in methods:
                item=f'seed_{seed}/{arm}_at_{step}'
                ev.append(dict(kind='evaluating',id=item,output=f'evaluation/{item}',model_dir=f'jobs/seed_{seed}/{arm}/milestones/steps_{step}'))
    prior=read(ROOT.parent/'2026-09-10_sequential_long_training/manifest.json')
    files=list(p.glob('*.py'))+[p/'provenance.json',p/'rule_selection.json']
    for folder in ['source','configs','initial']:
        files.extend(f for f in (p/folder).rglob('*') if f.is_file())
    if not base:files.append(p/'preflight_results.json')
    m=dict(schema=1,purpose='supplementary_seed_'+phase,seeds=[seed],methods=methods,jobs=jobs,evaluation_jobs=ev,
        evaluation_items=[j['id'] for j in ev],milestones=milestones,steps_per_method=steps,total_training_steps=len(jobs)*steps,batch=4000,
        evaluation_seeds=plan['evaluation_seeds'],scenarios=prior['scenarios'],evaluation_episodes=len(ev)*13*20,
        evaluation_split='previously_used_development_seeds',resources=dict(cores=[0] if base else [0,1,2],
            training_cores=list(range(len(jobs))),seconds_per_update_estimate=1.9,maximum_training_workers=len(jobs)),
        input_hashes={str(f.relative_to(p)):sha(f) for f in sorted(set(files))})
    immutable(p/'manifest.json',m);write(p/'status.json',dict(state='prepared',total_training_steps=m['total_training_steps']))
    return m

def materialize_continuation():
    plan=verify();seed=plan['seed'];base=ROOT/f'base/jobs/seed_{seed}/joint'
    assert read(base/'status.json')['state']=='complete'
    entry=read(base/'checkpoints/index.json')['current'];source=base/'checkpoints'/entry['file'];assert sha(source)==entry['sha256']
    assert entry['update']==2500
    target=ROOT/f'continuation/initial/seed_{seed}/parent_checkpoint.pt'
    if target.exists():assert sha(target)==entry['sha256']
    else:
        import shutil
        tmp=target.with_suffix('.tmp');shutil.copy2(source,tmp);assert sha(tmp)==entry['sha256'];tmp.replace(target)
    for arm in plan['arms']:
        folder=ROOT/f'continuation/configs/seed_{seed}';cfg=read(folder/f'{arm}.template.json')
        cfg['continuation']['parent_sha256']=entry['sha256'];immutable(folder/f'{arm}.json',cfg)
    check=ROOT/'continuation/preflight_results.json'
    if not check.exists():
        with (ROOT/'continuation/transfer_validation.log').open('a') as log:
            subprocess.run([sys.executable,'-u',str(ROOT/'continuation/validate_transfer.py')],stdout=log,stderr=subprocess.STDOUT,check=True)
    assert read(check)['state']=='PASS'
    make_manifest('continuation')

def report():
    plan=verify();seed=plan['seed'];old=ROOT.parent/'2026-09-10_resource_selector_training'
    for f,h in read(old/'report/artifact_hashes.json').items():assert sha(old/f)==h
    prior=read(old/'report/results.json');new=read(ROOT/'continuation/report/results.json')
    assert read(ROOT/'continuation/report/audit.json')['state']=='PASS'
    rows=[]
    for s in [85,218,966,seed]:
        data=prior if s!=seed else new
        scores={a:100*data['per_item'][f'seed_{s}/{a}_at_2000000']['overall']['common_reward'] for a in plan['arms']}
        baseline=100*data['per_item']['rules/R_myopic']['overall']['common_reward']
        rows.append(dict(seed=s,**scores,R_myopic=baseline,selector_minus_myopic=scores['selector']-baseline,
                         selector_minus_joint=scores['selector']-scores['joint_continue']))
    for row in rows:assert abs(row['R_myopic']-rows[0]['R_myopic'])<1e-8
    means=lambda data:{k:sum(r[k] for r in data)/len(data) for k in ['joint_continue','selector','selector_minus_myopic','selector_minus_joint']}
    write(ROOT/'comparison.json',dict(rows=rows,original_three_seed_mean=means(rows[:3]),all_four_seed_mean=means(rows),
        interpretation='Post-hoc supplementary seed, not an independent validation set. Original seed 966 retained.'))
    lines=[f'新增种子{seed}训练及评估完成。该种子在训练前一次抽定；原966结果全部保留。','',
        '新种子完成1000万步joint基础训练，再分别进行200万步选择器训练与原方案续训。奖励、环境、算法参数与原实验保持一致。原评估环境种子重复使用，本结果属于补充开发实验。','',
        '| 训练种子 | 原方案续训 | 资源选择器 | 一步择优规则 | 选择器减一步择优 |','|---|---:|---:|---:|---:|']
    for r in rows:lines.append(f'| {r["seed"]} | {r["joint_continue"]:.4f} | {r["selector"]:.4f} | {r["R_myopic"]:.4f} | {r["selector_minus_myopic"]:+.4f} |')
    for label,data in [('原三个种子',rows[:3]),('全部四个种子',rows)]:
        a=means(data);lines.append(f'\n{label}：选择器均分{a["selector"]:.4f}；相对一步择优差值{a["selector_minus_myopic"]:+.4f}。')
    lines+=['','没有因得分低删除966，也未反复换种子直到获胜。此处完整报告追加结果；不把追加后的结果当作预先设定的三种子独立验证。']
    (ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    write(ROOT/'report_hashes.json',{f:sha(ROOT/f) for f in ['REPORT.md','comparison.json']})

def run():
    plan=verify();lock=(ROOT/'.pipeline.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    stopping=[]
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda n,f:stopping.append(n))
    try:
        for phase in ['base','continuation']:
            write(ROOT/'status.json',dict(state='preparing',stage=phase,updated_utc=stamp(),seed=plan['seed']))
            if phase=='base':make_manifest(phase)
            else:materialize_continuation()
            p=ROOT/phase;s=read(p/'status.json')
            if s.get('state')!='complete':
                current=read(p/'current_execution.json') if (p/'current_execution.json').exists() else {}
                if not alive(current):
                    cmd=[sys.executable,'-u',str(p/'supervisor.py'),'start']
                    if any((p/j['output']/'status.json').exists() for j in read(p/'manifest.json')['jobs']):cmd.append('--resume')
                    subprocess.run(cmd,check=True)
                while True:
                    s=read(p/'status.json')
                    previous=10000000 if phase=='continuation' else 0
                    phase_steps=s.get('completed_training_steps',sum(r.get('steps',0) for r in s.get('training_progress',{}).values()))
                    write(ROOT/'status.json',dict(state=s['state'],stage=phase,updated_utc=stamp(),seed=plan['seed'],
                        completed_training_steps=previous+phase_steps,total_training_steps=14000000,
                        phase_status=s,automatic_next_stage=True))
                    if s['state']=='complete':break
                    if s['state'] in ['attention_required','failed']:raise RuntimeError(f'{phase} needs attention: {s}')
                    if stopping:
                        subprocess.run([sys.executable,str(p/'supervisor.py'),'pause'],check=True)
                        write(ROOT/'status.json',dict(state='pause_requested',stage=phase,updated_utc=stamp()))
                        return 75
                    current=read(p/'current_execution.json')
                    if not alive(current):raise RuntimeError(f'{phase} supervisor exited before completion')
                    time.sleep(3)
        report();verify()
        write(ROOT/'status.json',dict(state='complete',updated_utc=stamp(),seed=plan['seed'],completed_training_steps=14000000,
            total_training_steps=14000000,evaluation_episodes=1820,report=str(ROOT/'REPORT.md'),original_seed_966_retained=True))
        return 0
    except BaseException as exc:
        write(ROOT/'status.json',dict(state='attention_required',updated_utc=stamp(),error=repr(exc)))
        raise
    finally:lock.close()

def start():
    verify()
    with (ROOT/'.launcher.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        record=read(ROOT/'execution.json') if (ROOT/'execution.json').exists() else {}
        assert not alive(record),'Pipeline already active'
        with (ROOT/'pipeline.log').open('a') as log:
            p=subprocess.Popen([sys.executable,'-u',__file__,'run'],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        write(ROOT/'execution.json',dict(**process_record(p.pid),utc=stamp()))
        print(json.dumps(dict(state='started',pid=p.pid,seed=read(ROOT/'PLAN.json')['seed'],root=str(ROOT))))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['start','run','status','pause','freeze-base']);args=parser.parse_args()
    if args.command=='start':start()
    elif args.command=='run':raise SystemExit(run())
    elif args.command=='freeze-base':make_manifest('base')
    elif args.command=='status':print(json.dumps(read(ROOT/'status.json'),ensure_ascii=False,indent=2))
    else:
        r=read(ROOT/'execution.json')
        if alive(r):os.kill(r['pid'],signal.SIGTERM)
