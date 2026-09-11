"""Protect the frozen study, checkpoint controls and thermal recovery semantics."""
from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from multiseed_protocol import prepare, verify_run, read
from three_seed_train import ThermalControl, recorded_process_alive, snapshot
from formal_train import train


class LongTraining(unittest.TestCase):
    def test_complete_design_and_tamper_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/'study'
            m = prepare(out, [85,218,966])
            self.assertEqual(m['total_training_steps'], 210_000_000)
            self.assertEqual(m['total_evaluation_slots'], 3_900_000)
            self.assertEqual(sum(j['device']=='cuda:0' for j in m['jobs']),6)
            self.assertEqual(m['resources']['gpu_parallel_jobs'],2)
            self.assertEqual(verify_run(out),m)
            for job in m['jobs']:
                c=read(out/job['config'])
                self.assertEqual(c['env_args']['backhaul_availability'],.6)
                self.assertEqual(c['env_args']['aoi_reward_ref'],10)
                self.assertIsNone(c['algo_args']['train']['model_dir'])
                self.assertEqual(c['algo_args']['train']['num_env_steps'],10_000_000)
            self.assertEqual(snapshot(out)['completed_training_steps'],0)
            cfg=out/m['jobs'][0]['config']
            cfg.write_text(cfg.read_text()+' ')
            with self.assertRaises(RuntimeError): verify_run(out)

    def test_no_seed_overlap_no_budget_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            for seeds in ([1,1,2],[1,2,1001]):
                with self.assertRaises(ValueError): prepare(Path(tmp)/'run',seeds)
            with self.assertRaises(ValueError): prepare(Path(tmp)/'run',[85,218,966],steps=1_000_000)

    def test_original_actor_mask_is_preserved(self):
        from common import configuration
        from harl.envs.uav_escs.SC.uav_escs_env_sc import SCUAVEnv
        c=configuration()['env_args'];c['explicit_instruction_schedule']=[[0,2]]
        e=SCUAVEnv(c);e.seed(20260931);_,_,a=e.reset()
        self.assertTrue((a[1:,:16].sum(axis=1)<16).all())
        e.close()

    def test_both_sensors_and_full_cooldown(self):
        t=ThermalControl()
        self.assertIsNone(t.tick(86,60,0))
        self.assertEqual(t.tick(86,60,5),'pause')
        self.assertIsNone(t.tick(74,74,100))
        self.assertIsNone(t.tick(74,74,184))
        self.assertEqual(t.tick(74,74,185),'resume')
        self.assertEqual(t.tick(60,83,186),'pause')
        self.assertIsNone(t.tick(60,76,400))
        self.assertIsNone(t.tick(60,74,401))
        self.assertEqual(t.tick(60,74,461),'resume')
        self.assertEqual(t.tick(95,60,462),'pause')
        with self.assertRaises(RuntimeError):t.tick(None,60,463)

    def test_existing_checkpoints_and_pid_identity_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);(p/'status.json').write_text('{"state":"complete"}')
            with self.assertRaises(FileExistsError):train('unused',p,'cpu','unused')
            (p/'process.json').write_text(json.dumps(dict(pid=123,uid=1000,start_ticks=10)))
            with patch('three_seed_train.process_info',return_value=dict(pid=123,uid=1000,start_ticks=20,state='S')):
                self.assertFalse(recorded_process_alive(p))

if __name__=='__main__':unittest.main()
