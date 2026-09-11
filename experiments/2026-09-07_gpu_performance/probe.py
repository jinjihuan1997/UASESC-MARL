"""Isolated CPU/GPU stage timing and CPU-rollout/GPU-update experiment.

Never edits the frozen production runner, configuration, or checkpoints.
"""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
HARL = ROOT / "HARL/HARL"
sys.path[:0] = [str(HARL), str(HARL / "examples")]
import evaluate_instruction_constraints
import numpy as np
import torch
from harl.runners.on_policy_ha_runner import OnPolicyHARunner
from harl.common.valuenorm import ValueNorm


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)


def state_hash(network):
    h = hashlib.sha256()
    for key, value in sorted(network.state_dict().items()):
        h.update(key.encode())
        h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def to_cpu_copy(agent, network_name):
    result = copy.deepcopy(agent)
    result.device = torch.device("cpu")
    result.tpdv = dict(dtype=torch.float32, device=result.device)
    network = getattr(result, network_name)
    network.to(result.device)
    for module in network.modules():
        if hasattr(module, "tpdv"):
            module.tpdv = result.tpdv
    result.prep_rollout()
    return result


class ProbeRunner(OnPolicyHARunner):
    def __init__(self, config, destination, mode):
        self.destination = destination
        self.mode = mode
        self.hybrid = mode.startswith("hybrid")
        self.stages = {}
        self.records = []
        self.started = time.perf_counter()
        super().__init__(config["main_args"], config["algo_args"], config["env_args"])
        self.initial_hashes = [state_hash(a.actor) for a in self.actor]
        self.normalizer_original_keys = list(self.value_normalizer.state_dict())
        self.normalizer_cpu = None
        self.cpu_actors = None
        if self.hybrid or mode == "gpu_vector":
            # Register GPU statistics as Parameters before moving the module.
            # The original ValueNorm constructs Parameters with .to(cuda),
            # which can leave plain, unregistered tensors in this Torch version.
            previous = self.value_normalizer
            fixed = ValueNorm(1, device=torch.device("cpu")).to(self.device)
            fixed.tpdv = dict(dtype=torch.float32, device=self.device)
            with torch.no_grad():
                for name in ("running_mean", "running_mean_sq", "debiasing_term"):
                    getattr(fixed, name).copy_(getattr(previous, name))
            self.value_normalizer = fixed
        if self.hybrid:
            self.cpu_actors = [to_cpu_copy(a, "actor") for a in self.actor]
            self.cpu_critic = to_cpu_copy(self.critic, "critic")
            self.normalizer_cpu = ValueNorm(1, device=torch.device("cpu"))
            self.sync_inference_copies()
        for name in ("collect", "compute", "insert", "train", "save"):
            setattr(self, name, self.timed(name, getattr(self, name)))
        self.envs.step = self.timed("environment_and_ipc", self.envs.step)
        self.logger.per_step = self.timed("logging_per_step", self.logger.per_step)
        self.logger.episode_log = self.timed("logging_update", self.logger.episode_log)
        original_start = self.logger.episode_init
        def start(episode):
            self.stages = {}
            self.update_started = time.perf_counter()
            return original_start(episode)
        self.logger.episode_init = start
        self.init_seconds = time.perf_counter() - self.started

    def timed(self, name, fn):
        def call(*args, **kwargs):
            start = time.perf_counter()
            result = fn(*args, **kwargs)
            self.stages[name] = self.stages.get(name, 0.) + time.perf_counter() - start
            return result
        return call

    @torch.no_grad()
    def sync_inference_copies(self):
        for src, dst in zip(self.actor, self.cpu_actors):
            dst.actor.load_state_dict(src.actor.state_dict(), strict=True)
        self.cpu_critic.critic.load_state_dict(self.critic.critic.state_dict(), strict=True)
        for name in ("running_mean", "running_mean_sq", "debiasing_term"):
            getattr(self.normalizer_cpu, name).copy_(getattr(self.value_normalizer, name).detach().cpu())

    def collect(self, step):
        if not self.hybrid:
            return super().collect(step)
        actors, critic = self.actor, self.critic
        try:
            self.actor, self.critic = self.cpu_actors, self.cpu_critic
            return super().collect(step)
        finally:
            self.actor, self.critic = actors, critic

    def compute(self):
        if not self.hybrid:
            return super().compute()
        critic, normalizer = self.critic, self.value_normalizer
        try:
            self.critic, self.value_normalizer = self.cpu_critic, self.normalizer_cpu
            return super().compute()
        finally:
            self.critic, self.value_normalizer = critic, normalizer

    def train(self):
        result = super().train()
        for network in [*(a.actor for a in self.actor), self.critic.critic]:
            if not all(torch.isfinite(p).all() for p in network.parameters()):
                raise FloatingPointError("Non-finite learned parameters")
        assert np.isfinite(self.critic_buffer.returns).all()
        for metrics in [*result[0], result[1]]:
            assert all(np.isfinite(float(v)) for v in metrics.values())
        if self.hybrid:
            begin = time.perf_counter()
            self.sync_inference_copies()
            self.stages["hybrid_sync_within_train"] = time.perf_counter() - begin
        return result

    def after_update(self):
        super().after_update()
        record = {"update": len(self.records)+1, "seconds": time.perf_counter()-self.update_started,
                  "stages": self.stages.copy()}
        self.records.append(record)
        write_json(self.destination / "progress.json", {"state": "training", "mode": self.mode,
                   "completed_steps": len(self.records)*4000, "records": self.records})

    @torch.no_grad()
    def validate_hybrid(self):
        if not self.hybrid:
            return None
        errors = []
        for i, (gpu, cpu) in enumerate(zip(self.actor, self.cpu_actors)):
            assert state_hash(gpu.actor) == state_hash(cpu.actor)
            b = self.actor_buffer[i]
            args = (b.obs[0], b.rnn_states[0], b.masks[0], b.available_actions[0])
            gpu.actor.eval(); cpu.actor.eval()
            ga, gl, _ = gpu.get_actions(*args, deterministic=True)
            ca, cl, _ = cpu.get_actions(*args, deterministic=True)
            np.testing.assert_allclose(ga.cpu().numpy(), ca.numpy(), rtol=1e-4, atol=2e-5)
            np.testing.assert_allclose(gl.cpu().numpy(), cl.numpy(), rtol=1e-4, atol=2e-5)
            # Evaluate the SAME sampled action on both devices, not RNG equality.
            action, _, _ = cpu.get_actions(*args, deterministic=False)
            eval_args = (args[0], args[1], action.numpy(), args[2], args[3], b.active_masks[0])
            g_log, g_ent, _ = gpu.evaluate_actions(*eval_args)
            c_log, c_ent, _ = cpu.evaluate_actions(*eval_args)
            np.testing.assert_allclose(g_log.cpu().numpy(), c_log.numpy(), rtol=1e-4, atol=2e-5)
            np.testing.assert_allclose(g_ent.cpu().numpy(), c_ent.numpy(), rtol=1e-4, atol=2e-5)
            errors.append(float((g_log.cpu()-c_log).abs().max()))
        assert state_hash(self.critic.critic) == state_hash(self.cpu_critic.critic)
        b = self.critic_buffer
        value_args = (b.share_obs[0], b.rnn_states_critic[0], b.masks[0])
        gv, _ = self.critic.get_values(*value_args)
        cv, _ = self.cpu_critic.get_values(*value_args)
        np.testing.assert_allclose(gv.cpu().numpy(), cv.numpy(), rtol=1e-4, atol=2e-5)
        values = np.linspace(-2, 2, 21, dtype=np.float32).reshape(-1, 1)
        np.testing.assert_allclose(self.value_normalizer.denormalize(values), self.normalizer_cpu.denormalize(values), rtol=1e-5, atol=1e-5)
        return {"synchronized_parameters_exact": True, "deterministic_actions_close": True,
                "same_action_log_probability_max_abs_error": max(errors),
                "critic_values_close": True, "normalizer_denormalization_close": True,
                "note": "Device RNG streams differ; this checks fixed-input policy equivalence, not identical training trajectories."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["cpu", "gpu", "hybrid", "cpu_vector", "gpu_vector", "hybrid_vector"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--updates", type=int, default=6)
    args = parser.parse_args()
    out = Path(__file__).resolve().parent / f"{args.mode}_seed{args.seed}"
    out.mkdir(exist_ok=False)
    source = ROOT / "experiments/2026-09-07_happo_instruction_ab/configs/B_explicit_instruction_seed1.json"
    config = json.loads(source.read_text())
    config["main_args"]["exp_name"] = f"stage_probe_{args.mode}_seed{args.seed}"
    algo = config["algo_args"]
    algo["device"].update(cuda=not args.mode.startswith("cpu"), cuda_deterministic=False, torch_threads=1)
    algo["seed"].update(seed=args.seed, seed_specify=True)
    algo["train"].update(num_env_steps=args.updates*4000, log_interval=1, eval_interval=args.updates)
    algo["logger"]["log_dir"] = str(out / "training")
    write_json(out / "config.json", config)
    if not args.mode.startswith("cpu") and not torch.cuda.is_available():
        raise RuntimeError("Run this GPU probe through the approved host execution path")
    if args.mode.endswith("_vector"):
        from vectorized_simplex import install
        install()
    runner = None
    try:
        runner = ProbeRunner(config, out, args.mode)
        runner.run()
        runner.save()
        elapsed = time.perf_counter() - runner.started
        checks = runner.validate_hybrid()
        final_hashes = [state_hash(a.actor) for a in runner.actor]
        assert all(a != b for a, b in zip(runner.initial_hashes, final_hashes))
        steady = runner.records[1:]
        seconds = sum(r["seconds"] for r in steady)
        stage_totals = {}
        for row in steady:
            for name, value in row["stages"].items():
                stage_totals[name] = stage_totals.get(name, 0.) + value
        result = {"state": "complete", "mode": args.mode, "seed": args.seed,
                  "steps": args.updates*4000, "elapsed_seconds": elapsed,
                  "initialization_seconds": runner.init_seconds,
                  "steady_steps": 4000*len(steady), "steady_seconds": seconds,
                  "steady_steps_per_second": 4000*len(steady)/seconds,
                  "steady_stage_seconds": stage_totals, "updates": runner.records,
                  "original_normalizer_state_keys": runner.normalizer_original_keys,
                  "saved_normalizer_state_keys": list(runner.value_normalizer.state_dict()),
                  "all_actors_updated": True, "finite_parameters_and_metrics": True,
                  "hybrid_equivalence_checks": checks,
                  "checkpoint_hashes": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(runner.save_dir).glob("*.pt")},
                  "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        write_json(out / "result.json", result)
        write_json(out / "progress.json", {"state": "complete", "completed_steps": args.updates*4000})
        print(json.dumps({k:result[k] for k in ["mode", "seed", "elapsed_seconds", "steady_steps_per_second", "steady_stage_seconds", "saved_normalizer_state_keys", "hybrid_equivalence_checks"]}), flush=True)
    except BaseException as exc:
        write_json(out / "progress.json", {"state": "failed", "error": repr(exc)})
        raise
    finally:
        if runner is not None:
            runner.close()


if __name__ == "__main__":
    main()
