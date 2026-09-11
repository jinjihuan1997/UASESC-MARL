"""Aggregate joint packet-quality moments without changing the measured CC table."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent


def prepare():
    folder = ROOT/'inputs'
    profile = np.load(folder/'profile.npz', allow_pickle=False)
    protocol = json.loads((folder/'profile_protocol.json').read_text())
    rows = [json.loads(line) for line in (folder/'packet_statistics.jsonl').read_text().splitlines()]
    rows = [r for r in rows if r['partition']=='calibration']
    result = {key: np.zeros((20,11),np.float64) for key in ('success_psnr_mass','failure_psnr_mass','success_gain_mass')}
    min_quality = float('inf')
    for mi in range(20):
        for si in range(11):
            group = [r for r in rows if r['mode_id']==mi and r['snr_index']==si]
            counts = Counter(r['video_id'] for r in group)
            assert len(group)==96 and len(counts)==8
            w = np.array([1/len(counts)/counts[r['video_id']] for r in group])
            prob = np.array([r['p_success'] for r in group])
            good = np.array([r['q_success'] for r in group])
            bad = np.array([r['q_outage'] for r in group])
            min_quality = min(min_quality,float(good.min()))
            result['success_psnr_mass'][mi,si] = np.dot(w,prob*good)
            result['failure_psnr_mass'][mi,si] = np.dot(w,(1-prob)*bad)
            result['success_gain_mass'][mi,si] = np.dot(w,prob*np.maximum((good-21.)/12.,0))
            np.testing.assert_allclose(np.dot(w,prob),profile['deliver_prob'][mi,si],atol=1e-12,rtol=0)
    np.testing.assert_allclose(result['success_psnr_mass']+result['failure_psnr_mass'],profile['q_hat_mean'],atol=1e-12,rtol=0)
    fields=dict(**result,deliver_prob=profile['deliver_prob'],snr_grid_db=profile['snr_grid_db'],
                q_min_eval=21.,q_normalization_reference=33.,profile_sha256=hashlib.sha256((folder/'profile.npz').read_bytes()).hexdigest(),
                packet_records_sha256=hashlib.sha256((folder/'packet_statistics.jsonl').read_bytes()).hexdigest())
    with (folder/'delivery_moments.npz').open('xb') as f: np.savez_compressed(f,**fields)
    registry=dict(recommended_default=protocol['semantic_model_set'],sets={protocol['semantic_model_set']:dict(
        description='Measured H264 QP x LDPC rate; IDs are actions, not semantic checkpoints',
        modes=[dict(id=str(m['mode_id']),family='h264_ldpc',h264_qp=m['qp'],ldpc_code_rate=m['rate']) for m in protocol['modes']])})
    (folder/'mode_registry.json').write_text(json.dumps(registry,indent=2)+'\n')
    print(json.dumps(dict(modes=20,snr_points=11,min_success_quality_db=min_quality,quality_mixture_identity='PASS')))


if __name__=='__main__': prepare()
