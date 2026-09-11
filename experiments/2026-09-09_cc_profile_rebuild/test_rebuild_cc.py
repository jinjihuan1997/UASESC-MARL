import importlib.util
import json
import os
from pathlib import Path
import shutil
import struct
import tempfile
import unittest
import zlib

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/cc_profile_matplotlib")
import numpy as np
from PIL import Image

spec = importlib.util.spec_from_file_location("cc_builder", Path(__file__).with_name("rebuild_cc.py"))
cc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cc)


class ProfileTests(unittest.TestCase):
    def test_packet_header_and_segment_boundary(self):
        payload = b"x" * 112
        header = cc.packet_header(payload,19)
        self.assertEqual(len(header),16)
        self.assertEqual(struct.unpack(cc.HEADER_FORMAT,header),
                         (b"CCP1",1,19,0,112,zlib.crc32(payload)))
        self.assertEqual(cc.block_geometry(112,"1/2")["blocks"],1)
        self.assertEqual(cc.block_geometry(113,"1/2")["blocks"],2)
        self.assertEqual(cc.block_geometry(113,"3/4")["coded_bits"],2*1366)
        self.assertEqual(cc.block_geometry(113,"3/4")["channel_uses"],1366)
        self.assertEqual(cc.block_geometry(113,"5/6")["n"],1230)

    def test_zero_errors_does_not_imply_perfect_reliability(self):
        low,high = cc.binomial_interval(0,8192)
        self.assertEqual(low,0)
        self.assertGreater(high,0)
        self.assertLess((1-high)**100,1)
        low,high = cc.binomial_interval(8192,8192)
        self.assertLess(low,1)
        self.assertEqual(high,1)

    def test_exact_rgb8_loader_and_tamper_detection(self):
        with tempfile.TemporaryDirectory() as folder:
            frames=[]
            for i in range(8):
                p=Path(folder)/f"{i}.png"
                Image.fromarray(np.full((256,256,3),i,np.uint8)).save(p)
                frames.append(dict(path=str(p),sha256=cc.sha(p)))
            result=cc.source_frames(dict(frames=frames))
            self.assertEqual(result.shape,(8,256,256,3))
            self.assertEqual(int(result[7,0,0,0]),7)
            Path(frames[0]["path"]).write_bytes(b"changed")
            with self.assertRaises(RuntimeError):
                cc.source_frames(dict(frames=frames))

    def test_real_h264_roundtrip_deterministic_and_rgb_metric(self):
        yy,xx=np.indices((256,256))
        frames=np.stack([np.stack([(xx+i*2)%256,yy,(xx+yy)//2],axis=-1) for i in range(8)]).astype(np.uint8)
        stream,a=cc.h264_encode_decode(frames,26,shutil.which("ffmpeg"))
        repeated,b=cc.h264_encode_decode(frames,26,shutil.which("ffmpeg"))
        _,coarse=cc.h264_encode_decode(frames,42,shutil.which("ffmpeg"))
        self.assertEqual(stream,repeated)
        self.assertEqual(a,b)
        self.assertGreater(a["psnr_rgb_db"],coarse["psnr_rgb_db"])
        self.assertAlmostEqual(a["psnr_rgb_db"],-10*np.log10(a["mse_rgb"]))
        with self.assertRaises(ValueError):
            cc.h264_encode_decode(frames[:7],26,shutil.which("ffmpeg"))

    def test_per_clip_packet_probability_video_weighting_and_heldout(self):
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder)
            for name in ["codec","channel","bitstreams"]:
                (out/name).mkdir()
            clips=[dict(id="a",video_id=0,partition="calibration"),dict(id="b",video_id=0,partition="calibration"),
                   dict(id="c",video_id=1,partition="calibration"),dict(id="v",video_id=2,partition="validation")]
            for clip,size,q in zip(clips,[112,113,112,112],[40.,40.,34.,100.]):
                stream=out/"bitstreams"/f"{clip['id']}_qp42.h264"
                stream.write_bytes(b"x"*size)
                cc.write_json(out/"codec"/f"{clip['id']}_qp42.json",
                              dict(protocol_sha256="test",h264_sha256=cc.sha(stream),h264_bytes=size,
                                   psnr_rgb_db=q,outage_psnr_rgb_db=10.,mse_rgb=.001,outage_mse_rgb=.1))
            cc.write_json(out/"channel/rate0_snr0.json",dict(protocol_sha256="test",blocks=128,block_errors=64,
                          bler=.5,bler_ci_low=.4,bler_ci_high=.6,n=2048,empirical_es_n0_db=10.,snr_db=10.,
                          batches=[dict(blocks=128,block_errors=64)]))
            p=dict(clips=clips,qps=[42],rates=["1/2"],snr_grid_db=[10.],mc_blocks_per_cell=128,k=1024,bits_per_symbol=2,
                   modes=[dict(mode_id=0,qp=42,rate="1/2",rate_index=0)])
            arrays,rows,_=cc.aggregate(out,p,"test")
            self.assertAlmostEqual(arrays["q_hat_mean"][0,0],21.625)
            self.assertAlmostEqual(arrays["deliver_prob"][0,0],.4375)
            self.assertEqual(arrays["bar_ls_main_mean"][0,0],1280)
            self.assertEqual(arrays["sample_count"][0,0],3)
            self.assertEqual(len(rows),4)

    def test_noiseless_ldpc_qpsk_roundtrip_and_unit_power(self):
        import tensorflow as tf
        tf.config.set_visible_devices([],"GPU")
        tf.config.threading.set_intra_op_parallelism_threads(2)
        tf.config.threading.set_inter_op_parallelism_threads(1)
        from sionna.fec.ldpc import LDPC5GEncoder,LDPC5GDecoder
        from sionna.mapping import Mapper,Demapper
        mapper,demapper=Mapper("qam",2),Demapper("app","qam",2)
        for rate in ["1/2","2/3","3/4","5/6"]:
            geometry=cc.block_geometry(112,rate)
            encoder=LDPC5GEncoder(1024,geometry["n"])
            decoder=LDPC5GDecoder(encoder,hard_out=True,num_iter=20)
            bits=tf.cast(tf.random.stateless_uniform([8,1024],[182,2])>.5,tf.float32)
            x=mapper(encoder(bits))
            recovered=decoder(demapper([x,tf.constant(1e-4,tf.float32)]))
            self.assertTrue(np.array_equal(bits.numpy(),recovered.numpy()))
            self.assertAlmostEqual(float(tf.reduce_mean(tf.square(tf.abs(x)))),1.,places=6)


if __name__=="__main__":
    unittest.main()
