"""Run independent aggregation twice and require byte-identical numeric results."""
from repair_support import *
import subprocess

def main():
    guard();verify_inputs()
    hashes=[]
    for attempt in (1,2):
        with (HERE/f'aggregate_run{attempt}.log').open('a') as log:
            p=subprocess.run([sys.executable,str(HERE/'aggregate.py')],stdout=log,stderr=subprocess.STDOUT,cwd=HERE)
        assert p.returncode==0,f'Aggregation {attempt} failed; inspect aggregate_run{attempt}.log'
        hashes.append({p:sha(HERE/'report'/p) for p in ['results.json','audit.json','physical_statistics.json','REPORT.md']})
    assert hashes[0]==hashes[1],hashes
    verify_inputs()
    write(HERE/'report/reproducibility.json',dict(state='PASS',runs=2,identical_outputs=hashes[0],
        analysis_manifest_sha256=sha(HERE/'analysis_manifest.json'),original_hashes_rechecked=True))
    status=read(HERE/'status.json');status.update(state='complete',reaggregation='PASS_twice_identical')
    write(HERE/'status.json',status)
    print(json.dumps(dict(state='complete',hashes=hashes[0])))

if __name__=='__main__':main()
