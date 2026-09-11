"""One training job; invoked by run.py in a separate foreground subprocess."""
from pathlib import Path
import argparse
import copy
import csv
import os
import time
from protocol import activate_runtime, network_hash, read, sha, stamp, verify_run, write


def train(out, method, attempt):
    manifest = verify_run(out)
    activate_runtime()
    import numpy as np
    import torch
    from harl.runners import RUNNER_REGISTRY
    from harl.runners.hybrid_inference import HybridInferenceMixin
    from harl.envs.uav_escs.SC.uav_escs_env_sc import SCUAVEnv
    config = read(out / "configs" / f"{method}.json")
    main, algo, env_args = config["main_args"], copy.deepcopy(config["algo_args"]), config["env_args"]
    if algo["device"]["cuda"] and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; explicitly select --device cpu")
    work = out / "jobs" / method / f"attempt_{attempt:02d}"
    work.mkdir(parents=True, exist_ok=False)
    algo["logger"]["log_dir"] = str(work / "training")
    status_path = work / "status.json"
    latest_path = out / "jobs" / method / "status.json"
    batch = algo["train"]["n_rollout_threads"] * algo["train"]["episode_length"]
    started = time.monotonic()
    probe = SCUAVEnv(env_args)
    active_ids = probe.active_agent_ids
    probe.close()

    class TrackedRunner(HybridInferenceMixin, RUNNER_REGISTRY[main["algo"]]):
        def __init__(self):
            super().__init__(main, algo, env_args)
            self.configure_hybrid()
            self.completed_updates = 0
            self.initial_actors = [network_hash(actor.actor) for actor in self.actor]
            self.initial_critic = network_hash(self.critic.critic)
            self.last_train_metrics = None
            self.steady_seconds = 0.0
            self.performance = {}
            self.stages = {}
            for name in ("collect", "compute", "insert", "train", "save"):
                setattr(self, name, self.timed(name, getattr(self, name)))
            self.envs.step = self.timed("environment_and_ipc", self.envs.step)
            original_start = self.logger.episode_init
            def begin_update(episode):
                self.stages = {}
                self.update_started = time.perf_counter()
                return original_start(episode)
            self.logger.episode_init = begin_update
            self.report("initialized")

        def timed(self, name, function):
            def wrapped(*args, **kwargs):
                start = time.perf_counter()
                result = function(*args, **kwargs)
                self.stages[name] = self.stages.get(name, 0.0) + time.perf_counter()-start
                return result
            return wrapped

        def after_update(self):
            super().after_update()
            seconds = time.perf_counter()-self.update_started
            if self.completed_updates > 1:
                self.steady_seconds += seconds
            self.performance = dict(last_update_seconds=seconds,
                steady_steps=max(0, self.completed_updates-1)*batch,
                steady_seconds=self.steady_seconds,
                steady_steps_per_second=((self.completed_updates-1)*batch/self.steady_seconds)
                    if self.steady_seconds else None,
                stages=self.stages.copy())
            path = work/"update_timing.csv"
            fields = ["update", "steps", "seconds", "collect", "environment_and_ipc", "insert", "compute", "train", "save"]
            with path.open("a", newline="") as fp:
                writer = csv.DictWriter(fp, fieldnames=fields)
                if self.completed_updates == 1:
                    writer.writeheader()
                writer.writerow(dict(update=self.completed_updates, steps=self.completed_updates*batch,
                                     seconds=seconds, **{k:self.stages.get(k, 0.0) for k in fields[3:]}))
            self.report("training")

        def report(self, state, **extra):
            value = dict(method=method, attempt=attempt, state=state, updated_utc=stamp(),
                         completed_steps=self.completed_updates*batch,
                         target_steps=algo["train"]["num_env_steps"],
                         elapsed_seconds=time.monotonic()-started, run_dir=str(self.run_dir),
                         model_dir=str(self.save_dir), active_agent_ids=active_ids,
                         pid=os.getpid(), execution_device=manifest["device"],
                         rollout_device="cpu" if self.hybrid else str(self.device),
                         update_device=str(self.device), performance=self.performance,
                         runner_class=RUNNER_REGISTRY[main["algo"]].__name__,
                         initial_actor_hashes=self.initial_actors,
                         initial_critic_hash=self.initial_critic,
                         last_train_metrics=self.last_train_metrics, **extra)
            write(status_path, value)
            write(latest_path, value)

        def train(self):
            result = super().train()
            for network in [*(a.actor for a in self.actor), self.critic.critic]:
                if not all(torch.isfinite(p).all() for p in network.parameters()):
                    raise FloatingPointError("Non-finite learned parameters")
            if not np.isfinite(self.critic_buffer.returns).all():
                raise FloatingPointError("Non-finite returns")
            metrics = [{k: float(v) for k, v in values.items()} for values in [*result[0], result[1]]]
            if any(not np.isfinite(value) for row in metrics for value in row.values()):
                raise FloatingPointError("Non-finite update metrics")
            self.last_train_metrics = metrics
            self.completed_updates += 1
            return result

        def save(self):
            super().save()
            step = self.completed_updates*batch
            dest = work / "checkpoints" / f"step_{step:09d}"
            if not dest.exists():
                import shutil
                dest.mkdir(parents=True)
                for path in Path(self.save_dir).glob("*.pt"):
                    shutil.copyfile(path, dest/path.name)
                write(dest / "checkpoint.json", {
                    "steps": step, "active_agent_ids": active_ids,
                    "weights_only": True,
                    "note": "No optimizer/RNG/environment state; this is not exact training resume",
                    "hashes": {p.name: sha(p) for p in dest.glob("*.pt")},
                })

    runner = None
    try:
        runner = TrackedRunner()
        runner.run()
        runner.save()
        final_actors = [network_hash(a.actor) for a in runner.actor]
        final_critic = network_hash(runner.critic.critic)
        if any(a == b for a, b in zip(runner.initial_actors, final_actors)):
            raise RuntimeError("A trainable actor did not update")
        if runner.initial_critic == final_critic:
            raise RuntimeError("Critic did not update")
        if runner.completed_updates*batch != manifest["steps_per_method"]:
            raise RuntimeError("Training budget mismatch")
        hybrid_checks = runner.validate_hybrid()
        hashes = {}
        for path in Path(runner.save_dir).glob("*.pt"):
            state = torch.load(path, map_location="cpu", weights_only=True)
            if path.name == "value_normalizer.pt" and set(state) != {
                    "running_mean", "running_mean_sq", "debiasing_term"}:
                raise RuntimeError("ValueNorm checkpoint is missing registered statistics")
            if not all(torch.isfinite(v).all() for v in state.values()):
                raise FloatingPointError(f"Non-finite checkpoint {path}")
            hashes[str(path)] = sha(path)
        runner.report("complete", final_actor_hashes=final_actors,
                      final_critic_hash=final_critic, checkpoint_hashes=hashes,
                      hybrid_equivalence_checks=hybrid_checks)
    except BaseException as exc:
        if runner is not None:
            runner.report("failed", error=repr(exc))
        else:
            value = dict(state="failed", method=method, attempt=attempt, error=repr(exc), updated_utc=stamp())
            write(status_path, value)
            write(latest_path, value)
        raise
    finally:
        if runner is not None:
            runner.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--method", required=True)
    p.add_argument("--attempt", type=int, required=True)
    args = p.parse_args()
    train(args.output.resolve(), args.method, args.attempt)
