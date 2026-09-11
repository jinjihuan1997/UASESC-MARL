"""Versioned RGB8 H.264/5G-LDPC measurements; no changes to active RL inputs."""
from pathlib import Path
from fractions import Fraction
from concurrent.futures import ThreadPoolExecutor
import argparse
import datetime
import hashlib
import json
import math
import os
import shutil
import struct
import subprocess
import time
import zlib
import numpy as np
from PIL import Image
from scipy.stats import beta

ROOT = Path(__file__).resolve().parents[2]
HEADER_FORMAT = ">4sBBHII"
HEADER_BYTES = struct.calcsize(HEADER_FORMAT)
SC_HASH = "a6f721686b28d6c28ccf811ab8b63e80ae8c135a08b9d7e1b4bbfe7ca13d91cc"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    tmp.replace(path)


def packet_header(payload, mode_id):
    return struct.pack(HEADER_FORMAT, b"CCP1", 1, mode_id, 0, len(payload), zlib.crc32(payload))


def block_geometry(payload_bytes, rate, k=1024, bits_per_symbol=2):
    n = math.ceil(Fraction(k, 1) / Fraction(rate) / bits_per_symbol) * bits_per_symbol
    source_bits = 8 * (payload_bytes + HEADER_BYTES)
    blocks = math.ceil(source_bits / k)
    return dict(source_bits=source_bits, transport_header_bits=HEADER_BYTES*8, k=k, n=n,
                actual_code_rate=k/n, blocks=blocks, padding_bits=blocks*k-source_bits,
                coded_bits=blocks*n, channel_uses=blocks*n//bits_per_symbol)


def psnr_mse(source, reconstruction):
    mse = float(np.mean(((source.astype(np.float64)-reconstruction.astype(np.float64))/255.)**2))
    return float(-10*np.log10(max(mse, 1e-12))), mse


def binomial_interval(errors, count):
    low = 0. if errors == 0 else float(beta.ppf(.025, errors, count-errors+1))
    high = 1. if errors == count else float(beta.ppf(.975, errors+1, count-errors))
    return low, high


def source_frames(clip):
    images = []
    for entry in clip["frames"]:
        if sha(entry["path"]) != entry["sha256"]:
            raise RuntimeError("Source image changed after protocol freeze")
        with Image.open(entry["path"]) as image:
            if image.size != (256,256):
                raise ValueError("Input must already be 256x256; no implicit resize")
            images.append(np.asarray(image.convert("RGB"), dtype=np.uint8))
    array = np.stack(images)
    if array.shape != (8,256,256,3):
        raise ValueError(f"Wrong source geometry {array.shape}")
    return array


def h264_encode_decode(frames, qp, ffmpeg, fps=240):
    if frames.dtype != np.uint8 or frames.shape != (8,256,256,3):
        raise ValueError("Expected eight uint8 RGB frames")
    base = [ffmpeg,"-hide_banner","-loglevel","error","-threads","1","-filter_threads","1"]
    command = base + ["-f","rawvideo","-pix_fmt","rgb24","-s","256x256","-r",str(fps),
                     "-i","pipe:0","-an","-c:v","libx264","-preset","medium","-qp",str(qp),
                     "-g","8","-keyint_min","8","-sc_threshold","0","-threads","1",
                     "-pix_fmt","yuv420p","-f","h264","pipe:1"]
    encoded = subprocess.run(command,input=frames.tobytes(),capture_output=True,check=True).stdout
    if not encoded:
        raise RuntimeError("Empty H264 stream")
    # Raw H264 timestamps must not trigger the rawvideo muxer's frame dropping.
    decoded = subprocess.run(base+["-r",str(fps),"-f","h264","-i","pipe:0","-an","-pix_fmt","rgb24",
                                   "-fps_mode","passthrough","-f","rawvideo","pipe:1"],
                             input=encoded,capture_output=True,check=True).stdout
    if len(decoded) != frames.nbytes:
        raise RuntimeError(f"Incomplete H264 decode: {len(decoded)} / {frames.nbytes} bytes; no padding or repetition")
    quality,mse = psnr_mse(frames,np.frombuffer(decoded,np.uint8).reshape(frames.shape))
    return encoded, dict(psnr_rgb_db=quality,mse_rgb=mse,h264_bytes=len(encoded),h264_bits=len(encoded)*8,
                         decoded_rgb_sha256=hashlib.sha256(decoded).hexdigest(),
                         source_rgb_sha256=hashlib.sha256(frames.tobytes()).hexdigest())


def prepare(root,out,mc_blocks,calibration_per_video,validation_per_video):
    if out.exists():
        raise FileExistsError("Use a new output directory")
    import torch
    crl = root/"CRL-SemCom-VidCI"
    index_path = crl/"data/nfs_block_rgb_256_8f/test/nfs_block_file_locations.pt"
    index = torch.load(index_path,map_location="cpu",weights_only=True)
    if sorted(index) != list(range(20)):
        raise RuntimeError("Unexpected video index; inspect partition before proceeding")
    clips = []
    for video_id in range(14):
        part = "calibration" if video_id<8 else "validation"
        count = calibration_per_video if part=="calibration" else validation_per_video
        keys = sorted(index[video_id])
        positions = sorted({min(len(keys)-1,int((i+.5)*len(keys)/count)) for i in range(count)})
        for pos in positions:
            clip_id = keys[pos]
            frames = []
            for old in index[video_id][clip_id][:8]:
                path = Path(old)
                if path.parts[0] == "..":
                    path = crl/Path(*path.parts[1:])
                elif not path.is_absolute():
                    path = crl/path
                path = path.resolve()
                if not path.is_relative_to(crl/"data"):
                    raise RuntimeError("Unexpected source path")
                frames.append(dict(path=str(path),sha256=sha(path)))
            numbers = [int(Path(f["path"]).stem) for f in frames]
            if len(numbers)!=8 or any(b-a!=1 for a,b in zip(numbers,numbers[1:])):
                raise RuntimeError("Eight consecutive frames required")
            clips.append(dict(id=f"v{video_id:02d}_c{clip_id:05d}",video_id=video_id,
                              clip_id=clip_id,partition=part,frames=frames))
    ffmpeg = str(Path(shutil.which("ffmpeg")).resolve())
    sc = root/"HARL/HARL/harl/envs/uav_escs/semantic_models/profiles/derived_coupled_profile_snr0_20.npz"
    if sha(sc)!=SC_HASH:
        raise RuntimeError("SC reference changed")
    qps,rates = [42,38,34,30,26],["1/2","2/3","3/4","5/6"]
    p = dict(schema="cc_rgb8_profile_protocol_v1",created_utc=now(),project_root=str(root),
             semantic_model_set="h264_ldpc_rgb8_20modes_v1",index_path=str(index_path),index_sha256=sha(index_path),
             clips=clips,partition_video_ids=dict(calibration=list(range(8)),validation=list(range(8,14)),reserved=list(range(14,20))),
             sampling="Fixed midpoint clip-index quantiles per video; equal weight per video",
             source_shape_thwc=[8,256,256,3],external_reference_frames=False,
             frame_selection="First eight indexed frame paths, checked consecutive, no old model loader",fps=240,
             ffmpeg=ffmpeg,ffmpeg_sha256=sha(ffmpeg),
             ffmpeg_version=subprocess.check_output([ffmpeg,"-version"],text=True).splitlines()[0],
             codec="libx264 medium, yuv420p, independent 8-frame GOP; RGB PSNR includes color conversion",qps=qps,rates=rates,
             modes=[dict(mode_id=qi*len(rates)+ri,qp=qp,rate=rate,rate_index=ri,
                         name=f"h264_qp{qp}_ldpc_{rate.replace('/', '_')}")
                    for qi,qp in enumerate(qps) for ri,rate in enumerate(rates)],
             snr_grid_db=[0.,1.,2.,3.,4.,5.,6.,8.,10.,15.,20.],snr_definition="complex symbol Es/N0",
             snr_interpolation="Linear in dB; no measurements outside [0,20]",seed=20260909,k=1024,bits_per_symbol=2,
             modulation="Gray QPSK",decoder_iterations=20,decoder="Sionna LDPC5GDecoder boxplus-phi; exact bit comparison",
             mc_blocks_per_cell=mc_blocks,mc_batch_blocks=128,channel_device="CPU",cpu_affinity=[0,2],intra_op_threads=2,
             transport_header=dict(format=HEADER_FORMAT,bytes=HEADER_BYTES,
                                   fields=["magic CCP1","version u8","mode u8","reserved u16","length u32","CRC32 u32"]),
             packet_success_model="(1-BLER)^full_LDPC_blocks; independent AWGN blocks and detected packet errors; no ARQ",
             failure_display="Fixed RGB value 128 known at receiver; no free previous/future source reference",
             quality_semantics="Expected per-clip RGB PSNR including neutral display on packet loss; average of dB values",
             profile_level="Measured H264 clips + measured Monte Carlo LDPC BLER + analytic packet reliability; not actual corrupted H264 decoding",
             aoi_integration="Not implemented here; failed packets are not successful updates; align SC/CC quality and delivery abstractions explicitly",
             mc_confidence="95% Clopper-Pearson per cell; not simultaneous confidence or video population uncertainty",
             sc_reference_path=str(sc),sc_reference_sha256=SC_HASH,source_sha256=sha(__file__),no_performance_based_mode_filtering=True)
    out.mkdir(parents=True)
    for folder in ["source","codec","bitstreams","channel"]:
        (out/folder).mkdir()
    shutil.copyfile(__file__,out/"source/rebuild_cc.py")
    write_json(out/"protocol.json",p)
    write_json(out/"status.json",dict(state="prepared",updated_utc=now(),clips=len(clips)))
    return p


def load_protocol(out):
    p=json.loads((out/"protocol.json").read_text())
    if sha(__file__)!=p["source_sha256"]:
        raise RuntimeError("Execute the frozen source for this output")
    for path_key,hash_key in [("index_path","index_sha256"),("ffmpeg","ffmpeg_sha256"),("sc_reference_path","sc_reference_sha256")]:
        if sha(p[path_key])!=p[hash_key]:
            raise RuntimeError(f"Frozen input changed: {path_key}")
    return p,sha(out/"protocol.json")


def run_codec(out,p,protocol_hash):
    def one(clip):
        frames=source_frames(clip)
        bad_q,bad_mse=psnr_mse(frames,np.full_like(frames,128))
        for qp in p["qps"]:
            path=out/"codec"/f"{clip['id']}_qp{qp}.json"
            stream_path=out/"bitstreams"/f"{clip['id']}_qp{qp}.h264"
            if path.exists():
                record=json.loads(path.read_text())
                if record["protocol_sha256"]!=protocol_hash or sha(stream_path)!=record["h264_sha256"]:
                    raise RuntimeError("Existing codec record failed integrity check")
                continue
            encoded,stats=h264_encode_decode(frames,qp,p["ffmpeg"],p["fps"])
            if stream_path.exists() and stream_path.read_bytes()!=encoded:
                raise RuntimeError("Interrupted codec output differs on deterministic rerun")
            stream_path.write_bytes(encoded)
            write_json(path,dict(clip_id=clip["id"],video_id=clip["video_id"],partition=clip["partition"],qp=qp,
                                 protocol_sha256=protocol_hash,h264_sha256=sha(stream_path),
                                 outage_psnr_rgb_db=bad_q,outage_mse_rgb=bad_mse,**stats))
        return clip["id"]
    with ThreadPoolExecutor(max_workers=2) as pool:
        for done,_ in enumerate(pool.map(one,p["clips"]),1):
            write_json(out/"status.json",dict(state="codec",updated_utc=now(),completed_clips=done,total_clips=len(p["clips"])))
            if done%12==0 or done==len(p["clips"]):
                print(json.dumps(dict(codec_clips=done,total=len(p["clips"]))),flush=True)


def run_channel(out,p,protocol_hash):
    os.environ["CUDA_VISIBLE_DEVICES"]=""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL","3")
    os.environ.setdefault("MPLCONFIGDIR","/tmp/cc_profile_matplotlib")
    import tensorflow as tf
    import sionna
    from sionna.fec.ldpc import LDPC5GEncoder,LDPC5GDecoder
    from sionna.mapping import Mapper,Demapper
    tf.config.set_visible_devices([],"GPU")
    tf.config.threading.set_intra_op_parallelism_threads(p["intra_op_threads"])
    tf.config.threading.set_inter_op_parallelism_threads(1)
    mapper,demapper=Mapper("qam",2),Demapper("app","qam",2)
    done=0
    for ri,rate in enumerate(p["rates"]):
        geo=block_geometry(0,rate,p["k"],2)
        encoder=LDPC5GEncoder(geo["k"],geo["n"])
        decoder=LDPC5GDecoder(encoder,hard_out=True,num_iter=p["decoder_iterations"])

        @tf.function
        def trial(no,seed):
            seeds=tf.random.experimental.stateless_split(seed,3)
            bits=tf.cast(tf.random.stateless_uniform([p["mc_batch_blocks"],p["k"]],seeds[0])>.5,tf.float32)
            x=mapper(encoder(bits))
            re=tf.random.stateless_normal(tf.shape(x),seeds[1])
            im=tf.random.stateless_normal(tf.shape(x),seeds[2])
            noise=tf.cast(tf.sqrt(no/2),tf.complex64)*tf.complex(re,im)
            difference=tf.not_equal(decoder(demapper([x+noise,no])),bits)
            return (tf.reduce_sum(tf.cast(tf.reduce_any(difference,axis=1),tf.int32)),
                    tf.reduce_sum(tf.cast(difference,tf.int32)),tf.reduce_sum(tf.square(tf.abs(noise))),tf.size(x))

        for si,snr in enumerate(p["snr_grid_db"]):
            dest=out/"channel"/f"rate{ri}_snr{si}.json"
            if dest.exists():
                cell=json.loads(dest.read_text())
                if cell["protocol_sha256"]!=protocol_hash:
                    raise RuntimeError("Existing channel result belongs to another protocol")
            else:
                batches=[]
                started=time.monotonic()
                for bi in range(p["mc_blocks_per_cell"]//p["mc_batch_blocks"]):
                    seed=[p["seed"]+ri,si*10000+bi]
                    errors,bit_errors,energy,symbols=trial(tf.constant(10.**(-snr/10),tf.float32),tf.constant(seed,tf.int32))
                    batches.append(dict(batch_index=bi,seed=seed,blocks=p["mc_batch_blocks"],block_errors=int(errors),
                                        bit_errors=int(bit_errors),noise_energy=float(energy),complex_symbols=int(symbols)))
                count=sum(b["blocks"] for b in batches)
                errors=sum(b["block_errors"] for b in batches)
                low,high=binomial_interval(errors,count)
                measured=sum(b["noise_energy"] for b in batches)/sum(b["complex_symbols"] for b in batches)
                cell=dict(rate_index=ri,rate=rate,snr_db=snr,snr_index=si,protocol_sha256=protocol_hash,
                          tensorflow=tf.__version__,sionna=sionna.__version__,k=geo["k"],n=geo["n"],actual_code_rate=geo["actual_code_rate"],
                          blocks=count,block_errors=errors,bler=errors/count,bler_ci_low=low,bler_ci_high=high,
                          empirical_es_n0_db=-10*math.log10(measured),elapsed_seconds=time.monotonic()-started,batches=batches)
                write_json(dest,cell)
            done+=1
            write_json(out/"status.json",dict(state="channel",updated_utc=now(),completed_cells=done,total_cells=len(p["rates"])*len(p["snr_grid_db"])))
            print(json.dumps(dict(channel_cells=done,rate=rate,snr_db=snr,errors=cell["block_errors"],blocks=cell["blocks"])),flush=True)


def read_measurements(out,p,protocol_hash):
    codec,channel={},{}
    for clip in p["clips"]:
        for qp in p["qps"]:
            r=json.loads((out/"codec"/f"{clip['id']}_qp{qp}.json").read_text())
            stream=out/"bitstreams"/f"{clip['id']}_qp{qp}.h264"
            if r["protocol_sha256"]!=protocol_hash or sha(stream)!=r["h264_sha256"] or stream.stat().st_size!=r["h264_bytes"]:
                raise RuntimeError("Codec record/hash/length mismatch")
            codec[clip["id"],qp]=r
    for ri,_ in enumerate(p["rates"]):
        for si,_ in enumerate(p["snr_grid_db"]):
            r=json.loads((out/"channel"/f"rate{ri}_snr{si}.json").read_text())
            count=sum(b["blocks"] for b in r["batches"])
            errors=sum(b["block_errors"] for b in r["batches"])
            if r["protocol_sha256"]!=protocol_hash or count!=p["mc_blocks_per_cell"] or errors!=r["block_errors"]:
                raise RuntimeError("Channel record count mismatch")
            if abs(errors/count-r["bler"])>1e-15 or abs(r["empirical_es_n0_db"]-r["snr_db"])>.1:
                raise RuntimeError("BLER/SNR sanity check failed")
            channel[ri,si]=r
    return codec,channel


def aggregate(out,p,protocol_hash):
    codec,channel=read_measurements(out,p,protocol_hash)
    rows=[]
    for clip in p["clips"]:
        for mode in p["modes"]:
            c=codec[clip["id"],mode["qp"]]
            geo=block_geometry(c["h264_bytes"],mode["rate"],p["k"],p["bits_per_symbol"])
            for si,snr in enumerate(p["snr_grid_db"]):
                cell=channel[mode["rate_index"],si]
                if geo["n"]!=cell["n"]:
                    raise RuntimeError("Packet/MC LDPC lengths differ")
                success=(1-cell["bler"])**geo["blocks"]
                low=(1-cell["bler_ci_high"])**geo["blocks"]
                high=(1-cell["bler_ci_low"])**geo["blocks"]
                q,bad=c["psnr_rgb_db"],c["outage_psnr_rgb_db"]
                rows.append(dict(clip_id=clip["id"],video_id=clip["video_id"],partition=clip["partition"],mode_id=mode["mode_id"],
                                 snr_index=si,snr_db=snr,**geo,q_success=q,q_outage=bad,p_success=success,
                                 p_success_ci_low=low,p_success_ci_high=high,q_expected=success*q+(1-success)*bad,
                                 q_second_moment=success*q*q+(1-success)*bad*bad,
                                 q_channel_ci_low=min(low*q+(1-low)*bad,high*q+(1-high)*bad),
                                 q_channel_ci_high=max(low*q+(1-low)*bad,high*q+(1-high)*bad),
                                 expected_mse=success*c["mse_rgb"]+(1-success)*c["outage_mse_rgb"]))
    keys=["q_hat_mean","q_hat_std","bar_ls_main_mean","avg_kept_real_symbols_mean","deliver_prob",
          "deliver_prob_ci_low","deliver_prob_ci_high","q_success_mean","q_outage_mean","q_channel_ci_low",
          "q_channel_ci_high","sample_count","video_count","psnr_from_expected_mse"]
    arrays={k:np.zeros((len(p["modes"]),len(p["snr_grid_db"])),np.float64) for k in keys}
    mapping=dict(q_hat_mean="q_expected",bar_ls_main_mean="channel_uses",deliver_prob="p_success",
                 deliver_prob_ci_low="p_success_ci_low",deliver_prob_ci_high="p_success_ci_high",q_success_mean="q_success",
                 q_outage_mean="q_outage",q_channel_ci_low="q_channel_ci_low",q_channel_ci_high="q_channel_ci_high")
    for mi in range(len(p["modes"])):
        for si in range(len(p["snr_grid_db"])):
            group=[r for r in rows if r["partition"]=="calibration" and r["mode_id"]==mi and r["snr_index"]==si]
            videos=sorted({r["video_id"] for r in group})
            def mean(key):
                return float(np.mean([np.mean([r[key] for r in group if r["video_id"]==v]) for v in videos]))
            for dest,key in mapping.items():
                arrays[dest][mi,si]=mean(key)
            arrays["q_hat_std"][mi,si]=math.sqrt(max(0.,mean("q_second_moment")-mean("q_expected")**2))
            arrays["avg_kept_real_symbols_mean"][mi,si]=2*mean("channel_uses")
            arrays["sample_count"][mi,si],arrays["video_count"][mi,si]=len(group),len(videos)
            arrays["psnr_from_expected_mse"][mi,si]=-10*math.log10(max(mean("expected_mse"),1e-12))
    validation=[]
    for mi in range(len(p["modes"])):
        group=[r for r in rows if r["partition"]=="validation" and r["mode_id"]==mi]
        videos=sorted({r["video_id"] for r in group})
        q_mae=np.mean([np.mean([abs(r["q_expected"]-arrays["q_hat_mean"][mi,r["snr_index"]]) for r in group if r["video_id"]==v]) for v in videos])
        mape=np.mean([np.mean([abs(r["channel_uses"]-arrays["bar_ls_main_mean"][mi,r["snr_index"]])/r["channel_uses"] for r in group if r["video_id"]==v]) for v in videos])
        validation.append(dict(mode_id=mi,video_count=len(videos),quality_expected_mae_db=float(q_mae),load_mape=float(mape)))
    return arrays,rows,validation


def assemble(out,p,protocol_hash):
    arrays,rows,validation=aggregate(out,p,protocol_hash)
    artifact=out/"profile.npz"
    fields=dict(schema_version="cc_measured_rgb8_v1",semantic_model_set=p["semantic_model_set"],dataset_name="nfs_rgb8_calibration8videos",
                quality_metric="psnr_rgb_expected",mode_ids=np.arange(len(p["modes"])),mode_names=np.array([m["name"] for m in p["modes"]]),
                h264_qp_modes=np.array([m["qp"] for m in p["modes"]]),ldpc_code_rate_modes=np.array([float(Fraction(m["rate"])) for m in p["modes"]]),
                ldpc_actual_code_rate_modes=np.array([block_geometry(0,m["rate"])["actual_code_rate"] for m in p["modes"]]),
                snr_grid_db=np.array(p["snr_grid_db"]),source_shape_thwc=np.array(p["source_shape_thwc"]),num_bits_per_symbol=p["bits_per_symbol"],
                protocol_sha256=protocol_hash,source_sha256=p["source_sha256"],quality_semantics=p["quality_semantics"],
                profile_level=p["profile_level"],failure_display=p["failure_display"],**arrays)
    if artifact.exists():
        with np.load(artifact,allow_pickle=False) as saved:
            if set(saved.files)!=set(fields) or any(not np.array_equal(saved[k],v) for k,v in fields.items()):
                raise RuntimeError("Existing profile differs from reaggregation")
    else:
        with artifact.open("xb") as f:
            np.savez_compressed(f,**fields)
    raw_path=out/"packet_statistics.jsonl"
    text="".join(json.dumps(r,separators=(",",":"),allow_nan=False)+"\n" for r in rows)
    if raw_path.exists() and raw_path.read_text()!=text:
        raise RuntimeError("Existing packet statistics differ")
    if not raw_path.exists():
        raw_path.write_text(text)
    if any(not np.all(np.isfinite(a)) for a in arrays.values()):
        raise RuntimeError("Nonfinite profile")
    assert np.all(arrays["bar_ls_main_mean"]>0)
    assert np.all(arrays["bar_ls_main_mean"]==arrays["bar_ls_main_mean"][:,[0]])
    assert np.all((arrays["deliver_prob"]>=0)&(arrays["deliver_prob"]<=1))
    assert np.all(arrays["deliver_prob_ci_low"]<=arrays["deliver_prob"])
    assert np.all(arrays["deliver_prob_ci_high"]>=arrays["deliver_prob"])
    files=[out/"protocol.json",out/"source/rebuild_cc.py",artifact,raw_path]
    files+=sorted((out/"codec").glob("*.json"))+sorted((out/"channel").glob("*.json"))
    manifest={str(f.relative_to(out)):sha(f) for f in files}
    if (out/"manifest.json").exists():
        if json.loads((out/"manifest.json").read_text())!=manifest:
            raise RuntimeError("Reaggregated manifest differs")
    else:
        write_json(out/"manifest.json",manifest)
    result=dict(state="complete",verified_utc=now(),profile_sha256=sha(artifact),modes=len(p["modes"]),snr_points=len(p["snr_grid_db"]),
                codec_measurements=len(p["clips"])*len(p["qps"]),channel_cells=len(p["rates"])*len(p["snr_grid_db"]),
                channel_mc_blocks=len(p["rates"])*len(p["snr_grid_db"])*p["mc_blocks_per_cell"],
                per_clip_mode_snr_statistics=len(rows),validation=validation,
                quality_range_db=[float(arrays["q_hat_mean"].min()),float(arrays["q_hat_mean"].max())],
                load_range_channel_uses=[float(arrays["bar_ls_main_mean"].min()),float(arrays["bar_ls_main_mean"].max())],
                sc_profile_unchanged=sha(p["sc_reference_path"])==SC_HASH,
                rl_integration="NOT_STARTED; explicit quality and delivery semantics required",
                science_limit="Rebuilt CC average profile; SC remains derived, so empirical SC-vs-CC fairness is not established by this alone")
    write_json(out/"validation.json",result)
    write_json(out/"status.json",dict(state="complete",updated_utc=now(),profile_sha256=sha(artifact)))
    print(json.dumps({k:v for k,v in result.items() if k!="validation"}),flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=["prepare","run","verify"])
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--project-root",type=Path,default=ROOT)
    parser.add_argument("--mc-blocks",type=int,default=8192)
    parser.add_argument("--calibration-per-video",type=int,default=12)
    parser.add_argument("--validation-per-video",type=int,default=8)
    args=parser.parse_args()
    out=args.output.resolve()
    if args.action=="prepare":
        if args.mc_blocks<128 or args.mc_blocks%128 or min(args.calibration_per_video,args.validation_per_video)<1:
            parser.error("Positive sample counts and MC blocks divisible by 128 required")
        p=prepare(args.project_root.resolve(),out,args.mc_blocks,args.calibration_per_video,args.validation_per_video)
        print(json.dumps(dict(state="prepared",clips=len(p["clips"]),modes=len(p["modes"]),output=str(out))))
        return
    p,protocol_hash=load_protocol(out)
    os.sched_setaffinity(0,set(p["cpu_affinity"]))
    if args.action=="run":
        run_codec(out,p,protocol_hash)
        run_channel(out,p,protocol_hash)
    assemble(out,p,protocol_hash)


if __name__=="__main__":
    main()
