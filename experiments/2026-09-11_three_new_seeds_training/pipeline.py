"""Three predeclared supplementary seeds: bases -> paired continuations -> reporting."""
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
    plan=verify();p=ROOT/phase;seeds=plan['seeds'];base=phase=='base'
    if (p/'manifest.json').exists():
        m=read(p/'manifest.json')
        for f,h in m['input_hashes'].items():assert sha(p/f)==h,f
        return m
    methods=['joint'] if base else plan['arms'];steps=10000000 if base else 2000000
    milestones=[10000000] if base else [1000000,2000000]
    jobs=[dict(kind='training',id=f'seed_{seed}/{a}',seed=seed,method=a,device='cpu',core=i,
        config=f'configs/seed_{seed}/{a}.json',output=f'jobs/seed_{seed}/{a}')
        for i,(seed,a) in enumerate((s,a) for s in seeds for a in methods)]
    ev=[]
    if not base:
        ev=[dict(kind='evaluating',id=f'rules/{r}',output=f'evaluation/rules/{r}') for r in ['R_single','R_instruction','R_myopic']]
        for step in milestones:
            for seed in seeds:
                for arm in methods:
                    item=f'seed_{seed}/{arm}_at_{step}'
                    ev.append(dict(kind='evaluating',id=item,output=f'evaluation/{item}',model_dir=f'jobs/seed_{seed}/{arm}/milestones/steps_{step}'))
    prior=read(ROOT.parent/'2026-09-10_sequential_long_training/manifest.json')
    files=list(p.glob('*.py'))+[p/'provenance.json',p/'rule_selection.json']
    for folder in ['source','configs','initial']:
        files.extend(f for f in (p/folder).rglob('*') if f.is_file())
    if not base:files.append(p/'preflight_results.json')
    m=dict(schema=1,purpose='three_new_seeds_'+phase,seeds=seeds,methods=methods,jobs=jobs,evaluation_jobs=ev,
        evaluation_items=[j['id'] for j in ev],milestones=milestones,steps_per_method=steps,total_training_steps=len(jobs)*steps,batch=4000,
        evaluation_seeds=plan['evaluation_seeds'],scenarios=prior['scenarios'],evaluation_episodes=len(ev)*13*20,
        evaluation_split='previously_used_development_seeds',resources=dict(cores=[0,1,2] if base else list(range(8)),
            training_cores=list(range(len(jobs))),seconds_per_update_estimate=1.9,maximum_training_workers=3 if base else 4,
            maximum_active_workers=3 if base else 4),
        input_hashes={str(f.relative_to(p)):sha(f) for f in sorted(set(files))})
    immutable(p/'manifest.json',m);write(p/'status.json',dict(state='prepared',total_training_steps=m['total_training_steps']))
    return m

def materialize_continuation():
    plan=verify()
    for seed in plan['seeds']:
        base=ROOT/f'base/jobs/seed_{seed}/joint'
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
    plan=verify();old=ROOT.parent/'2026-09-11_additional_seed_training'
    for f,h in read(old/'report_hashes.json').items():assert sha(old/f)==h
    prior=read(old/'comparison.json');new=read(ROOT/'continuation/report/results.json')
    assert read(ROOT/'continuation/report/audit.json')['state']=='PASS'
    rows=prior['rows'].copy();assert [r['seed'] for r in rows]==[85,218,966,268966]
    for s in plan['seeds']:
        data=new
        scores={a:100*data['per_item'][f'seed_{s}/{a}_at_2000000']['overall']['common_reward'] for a in plan['arms']}
        baseline=100*data['per_item']['rules/R_myopic']['overall']['common_reward']
        rows.append(dict(seed=s,**scores,R_myopic=baseline,selector_minus_myopic=scores['selector']-baseline,
                         selector_minus_joint=scores['selector']-scores['joint_continue']))
    for row in rows:assert abs(row['R_myopic']-rows[0]['R_myopic'])<1e-8
    means=lambda data:{k:sum(r[k] for r in data)/len(data) for k in ['joint_continue','selector','selector_minus_myopic','selector_minus_joint']}
    write(ROOT/'comparison.json',dict(rows=rows,original_four_seed_mean=means(rows[:4]),new_three_seed_mean=means(rows[4:]),all_seven_seed_mean=means(rows),
        new_batch_selector_wins_myopic=sum(r['selector_minus_myopic']>0 for r in rows[4:]),
        interpretation='Three seeds drawn once before training; supplementary development experiment. All seven seed results retained.'))
    lines=[f'新增种子{plan["seeds"]}训练及评估完成。三个种子在训练前一次抽定，全部报告；原四个种子结果保留。','',
        '每个新种子完成1000万步joint基础训练，再分别进行200万步选择器训练与原方案续训。奖励、环境、算法参数与原实验保持一致。原评估环境种子重复使用，本结果属于补充开发实验。','',
        '| 训练种子 | 原方案续训 | 资源选择器 | 一步择优规则 | 选择器减一步择优 |','|---|---:|---:|---:|---:|']
    for r in rows:lines.append(f'| {r["seed"]} | {r["joint_continue"]:.4f} | {r["selector"]:.4f} | {r["R_myopic"]:.4f} | {r["selector_minus_myopic"]:+.4f} |')
    for label,data in [('原四个种子',rows[:4]),('本轮三个新种子',rows[4:]),('全部七个种子',rows)]:
        a=means(data);lines.append(f'\n{label}：选择器均分{a["selector"]:.4f}；相对一步择优差值{a["selector_minus_myopic"]:+.4f}。')
    lines+=['','这三个新种子没有根据得分进行筛选。完整保留所有七个结果；当前未开展独立环境测试，也没有新增隐藏指令训练消融。']
    (ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    write(ROOT/'report_hashes.json',{f:sha(ROOT/f) for f in ['REPORT.md','comparison.json']})

def run():
    plan=verify();lock=(ROOT/'.pipeline.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    stopping=[]
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda n,f:stopping.append(n))
    try:
        for phase in ['base','continuation']:
            write(ROOT/'status.json',dict(state='preparing',stage=phase,updated_utc=stamp(),seeds=plan['seeds']))
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
                    previous=30000000 if phase=='continuation' else 0
                    phase_steps=s.get('completed_training_steps',sum(r.get('steps',0) for r in s.get('training_progress',{}).values()))
                    write(ROOT/'status.json',dict(state=s['state'],stage=phase,updated_utc=stamp(),seeds=plan['seeds'],
                        completed_training_steps=previous+phase_steps,total_training_steps=42000000,
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
        write(ROOT/'status.json',dict(state='complete',updated_utc=stamp(),seeds=plan['seeds'],completed_training_steps=42000000,
            total_training_steps=42000000,evaluation_episodes=3900,report=str(ROOT/'REPORT.md'),original_four_seeds_retained=True))
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
        print(json.dumps(dict(state='started',pid=p.pid,seeds=read(ROOT/'PLAN.json')['seeds'],root=str(ROOT))))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['start','run','status','pause','freeze-base']);args=parser.parse_args()
    if args.command=='start':start()
    elif args.command=='run':raise SystemExit(run())
    elif args.command=='freeze-base':make_manifest('base')
    elif args.command=='status':print(json.dumps(read(ROOT/'status.json'),ensure_ascii=False,indent=2))
    else:
        r=read(ROOT/'execution.json')
        if alive(r):os.kill(r['pid'],signal.SIGTERM)
