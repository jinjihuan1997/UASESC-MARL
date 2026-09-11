from pathlib import Path
import copy
import importlib.util
import json
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from common import ROOT,configuration,network_hash
from tensor_env import TensorCCEnv
from tensor_train import TensorTrainer
from cc_metrics import metrics,external_reference,array
from training_checkpoint import capture,restore,save_checkpoint,load_checkpoint
from multiseed_protocol import prepare,verify_run
from three_seed_train import ThermalControl
torch.set_num_threads(1)


def actions(env,mode=0):
    result=[torch.zeros((env.count,s.shape[0]),device=env.device) for s in env.action_space]
    for a in result[1:]: a[:,mode]=1.
    return result


class CCChecks(unittest.TestCase):
    def env(self,seed=85):
        e=TensorCCEnv(configuration()['env_args'],count=1,seed=seed,device='cpu');e.reset();return e

    def test_all_twenty_modes_and_no_side_load(self):
        e=self.env();self.assertEqual(e.p.semantic_mode_lookup_by_bucket_mu.shape,(4,20))
        self.assertEqual(e.action_space[1].shape,(20,))
        for mode in range(20):
            e.reset();e.p.B_sut_sat=3e9;e.p.B_uav_sut=1e9
            e.quality.fill_(35.);e.success_quality.fill_(35.);e.success_quality_gain.fill_(14/12);e.delivery_probability.fill_(1.)
            e.step(actions(e,mode),auto_reset=False)
            self.assertTrue(bool((e.last['mode']==mode).all()))
        torch.testing.assert_close(e.load,e.lz,rtol=0,atol=0)

    def test_failure_cost_aoi_and_single_attempt_cache(self):
        e=self.env();e.aoi.fill_(10.);e.delivery_probability.zero_()
        _,_,_,_,info,_=e.step(actions(e),auto_reset=False)
        self.assertGreater(int(info['attempted'].sum()),0)
        self.assertEqual(int(info['served'].sum()),0)
        self.assertGreater(float(info['usage'].sum()),0)
        self.assertTrue(bool((e.aoi==11).all()))
        self.assertFalse(bool(e.q[info['attempted']].any()))
        self.assertTrue(bool((e.tau[info['attempted']]==-1).all()))
        result=metrics(e,info,0);self.assertEqual(result['deliveries'],0)
        self.assertEqual(result['attempts'],result['failed_packets'])

    def test_success_and_mixed_delivery_independent_reward(self):
        for probability in (1.,.5):
            e=self.env();e.aoi.fill_(10.);e.p.B_sut_sat=3e9;e.p.B_uav_sut=1e9
            e.delivery_probability.fill_(probability)
            e.delivery_uniforms[0,0]=torch.arange(e.K)[None,:].expand(e.U,-1)/e.K
            _,_,_,_,info,_=e.step(actions(e),auto_reset=False)
            expected=info['attempted'] & (e.delivery_uniforms[0]<probability)
            torch.testing.assert_close(info['served'],expected)
            self.assertTrue(bool((e.aoi[expected]==1).all()))
            self.assertTrue(bool((e.aoi[~expected]==11).all()))
            metrics(e,info,0)

    def test_moments_interpolation_not_product_of_means(self):
        e=self.env();raw=np.load(ROOT/'inputs/profile.npz',allow_pickle=False)
        for snr in (0.,1.5,4.3,6.,7.,20.):
            e.gamma_bh.fill_(10**(snr/10));e.update_tables()
            q=np.array([np.interp(snr,raw['snr_grid_db'],row) for row in raw['q_hat_mean']])
            np.testing.assert_allclose(array(e.quality)[0,0],q,atol=1e-12,rtol=0)
            reconstructed=e.delivery_probability*e.success_quality+(1-e.delivery_probability)*e.failure_quality
            torch.testing.assert_close(reconstructed,e.quality,atol=1e-12,rtol=0)
        e.gamma_bh.fill_(100.);e.update_tables()
        self.assertGreater(float(e.success_quality_gain.max()),1.)

    def test_sc_external_pairing_and_actions_cannot_change_rng(self):
        schedule=[[0,0],[10,1],[30,2]];cfg=configuration()
        cfg['env_args'].update(instruction_mode_strategy='explicit_evaluation',explicit_instruction_schedule=schedule)
        e=TensorCCEnv(cfg['env_args'],count=1,seed=20260931,device='cpu');e.reset()
        ref,_=external_reference(cfg,20260931,schedule)
        np.testing.assert_array_equal(array(e.pos_ds)[0],ref.pos_ds)
        other=TensorCCEnv(cfg['env_args'],count=1,seed=20260931,device='cpu');other.reset()
        for slot in range(40):
            np.testing.assert_allclose(array(e.gamma_us)[0],ref.gamma_uav_sut,atol=1e-10,rtol=1e-12)
            self.assertEqual(int(e.instructions[slot,0]),next(g for t,g in reversed(schedule) if t<=slot))
            e.step(actions(e,0),auto_reset=False);other.step(actions(other,19),auto_reset=False)
            torch.testing.assert_close(e.delivery_uniforms,other.delivery_uniforms,rtol=0,atol=0)
            torch.testing.assert_close(e.noise_us,other.noise_us,rtol=0,atol=0)
            if slot<39: ref._update_channels()

    def test_exact_resume_across_episode_reset_and_optimizer_updates(self):
        c=configuration();c['env_args']['T']=40.
        c['algo_args']['train'].update(n_rollout_threads=2,episode_length=20,num_env_steps=120)
        identity={'test':'cc exact resume'}
        def update(t):
            t.collect();t.update();t.buffer.after_update()
        a=TensorTrainer(copy.deepcopy(c),'cpu');update(a)
        with tempfile.TemporaryDirectory() as tmp:
            save_checkpoint(tmp,capture(a,1,identity));state,_=load_checkpoint(tmp)
            update(a);update(a)
            expected=[network_hash(x.actor) for x in a.actors]+[network_hash(a.critic.critic)]
            b=TensorTrainer(copy.deepcopy(c),'cpu');self.assertEqual(restore(b,state,identity),1)
            update(b);update(b)
            self.assertEqual(expected,[network_hash(x.actor) for x in b.actors]+[network_hash(b.critic.critic)])
            for key in ('aoi','q','tau','delivery_uniforms'):
                torch.testing.assert_close(getattr(a.env,key),getattr(b.env,key),rtol=0,atol=0)

    def test_external_hash_matches_actual_sc_step_loop(self):
        import hashlib
        schedule=[[0,0],[300,1],[400,2]];cfg=configuration()
        ref,reference_hash=external_reference(cfg,20260931,schedule)
        actual,actual_hash=external_reference(cfg,20260931,schedule)
        for slot in range(600):
            gid=next(g for t,g in reversed(schedule) if t<=slot)
            reference_hash.update(ref.gamma_uav_sut.tobytes())
            reference_hash.update(np.asarray([ref.gamma_sut_sat,gid],dtype=np.float64).tobytes())
            actual_hash.update(actual.gamma_uav_sut.tobytes())
            actual_hash.update(np.asarray([actual.gamma_sut_sat,actual.current_instruction_id],dtype=np.float64).tobytes())
            actual.step([np.zeros(space.shape,np.float32) for space in actual.action_space])
            if slot<599:ref._update_channels()
        self.assertEqual(reference_hash.hexdigest(),actual_hash.hexdigest())

    def test_freeze_pairing_and_tamper_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'cc';m=prepare(out,smoke=True)
            self.assertEqual(len(m['jobs']),6);self.assertEqual(m['seeds'],[85,218,966])
            self.assertEqual(m['resources']['cpu_parallel_jobs'],3)
            self.assertEqual(verify_run(out),m)
            p=out/m['jobs'][0]['config'];p.write_text(p.read_text()+' ')
            with self.assertRaises(RuntimeError): verify_run(out)
            with self.assertRaises(ValueError): prepare(Path(tmp)/'bad',seeds=[1,2,3],smoke=True)

    def test_thermal_pause_and_full_cooldown(self):
        t=ThermalControl();self.assertIsNone(t.tick(86,42,0));self.assertEqual(t.tick(86,42,5),'pause')
        self.assertIsNone(t.tick(74,42,125));self.assertEqual(t.tick(74,42,185),'resume')
        self.assertEqual(t.tick(95,42,200),'pause')


if __name__=='__main__':unittest.main()
