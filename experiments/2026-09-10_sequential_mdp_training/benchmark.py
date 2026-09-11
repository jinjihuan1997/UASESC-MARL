"""Run the actual sequential trainer on the requested device, no formal weights."""
import argparse,time
from helpers import *
from tensor_train import TensorTrainer

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',required=True);p.add_argument('--name',required=True);p.add_argument('--updates',type=int,default=3);a=p.parse_args()
    t=TensorTrainer(config(),a.device);rows=[]
    for u in range(1,a.updates+1):
        t.set_training_update(u);t.synchronize();start=time.monotonic();t.collect();t.synchronize();middle=time.monotonic();out=t.update();t.synchronize();end=time.monotonic()
        assert all(torch.isfinite(v).all() for v in out.values())
        t.buffer.after_update();rows.append(dict(update=u,collect_seconds=middle-start,update_seconds=end-middle,total_seconds=end-start))
        print(a.name,rows[-1],flush=True)
    write(HERE/'preflight'/f'benchmark_{a.name}.json',dict(device=a.device,rows=rows,mean_seconds_per_update=float(np.mean([r['total_seconds'] for r in rows[1:]]))))
