"""Frozen comparison protocol, file integrity, and shared utilities."""
from pathlib import Path
import copy
import datetime
import hashlib
import json
import sys

VERSION = Path(__file__).resolve().parent
TABLE_SHA = "a6f721686b28d6c28ccf811ab8b63e80ae8c135a08b9d7e1b4bbfe7ca13d91cc"
METHODS = [
    ("IC_HAPPO", "happo", {}),
    ("IC_MAPPO", "mappo", {}),
    ("HAPPO_hidden_instruction", "happo", {"actor_observe_instruction": False}),
    ("MAPPO_hidden_instruction", "mappo", {"actor_observe_instruction": False}),
    ("HAPPO_fixed_mode_rule", "happo", {"fixed_mode_rule": True}),
    ("HAPPO_equal_resources", "happo", {"fixed_resources": True}),
    ("HAPPO_no_task_aux_reward", "happo", {"use_task_auxiliary_reward": False}),
]
RULES = ["R_fixed", "G_local_greedy", "Random", "RoundRobin"]
SCENARIOS = {f"fixed_{g}": [[0, g]] for g in range(3)}
SCENARIOS.update({f"switch300_{a}_to_{b}": [[0, a], [300, b]]
                  for a in range(3) for b in range(3) if a != b})
SCENARIOS.update({
    "early50_0_to_1": [[0, 0], [50, 1]],
    "late550_0_to_2": [[0, 0], [550, 2]],
    "multi_0_1_2": [[0, 0], [200, 1], [400, 2]],
    "multi_2_1_0": [[0, 2], [200, 1], [400, 0]],
})


def stamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def activate_runtime():
    sys.path.insert(0, str(VERSION / "runtime"))
    import harl.envs.uav_escs.SC.uav_escs_env_sc as module
    if not Path(module.__file__).resolve().is_relative_to(VERSION / "runtime"):
        raise RuntimeError("Imported an external HARL version; refusing mixed source")


def configurations(seed=1, steps=10_000_000, threads=10, device="cpu", log_root=None):
    algo = read(VERSION / "config/base_algo.json")
    env = read(VERSION / "config/base_env.json")
    env.update(semantic_registry_path=str(VERSION / "inputs/mode_registry.json"),
               semantic_profile_path=str(VERSION / "inputs/profile.npz"))
    algo["seed"].update(seed=seed, seed_specify=True)
    if device not in ("cpu", "cuda", "hybrid"):
        raise ValueError(device)
    algo["device"].update(cuda=(device != "cpu"), cuda_deterministic=(device != "cpu"))
    batch = threads * algo["train"]["episode_length"]
    if threads < 1 or steps < 2*batch or steps % batch:
        raise ValueError(f"steps must be a multiple of {batch}, and at least {2*batch}")
    checkpoint_updates = max(1, 500_000 // batch)
    algo["train"].update(num_env_steps=steps, n_rollout_threads=threads,
                         eval_interval=checkpoint_updates,
                         log_interval=1 if steps <= 80_000 else 5,
                         hybrid_inference=(device == "hybrid"))
    if log_root is not None:
        algo["logger"]["log_dir"] = str(log_root)
    configs = {}
    for name, algorithm, overrides in METHODS:
        config = {"main_args": {"algo": algorithm, "env": "uav_escs_sc", "exp_name": name},
                  "algo_args": copy.deepcopy(algo), "env_args": copy.deepcopy(env)}
        config["env_args"].update(overrides)
        configs[name] = config
    return configs


def verify_run(out):
    out = Path(out)
    manifest = read(out / "manifest.json")
    changed = [str(out / p) for p, expected in manifest["input_hashes"].items()
               if not (out / p).is_file() or sha(out / p) != expected]
    if changed:
        raise RuntimeError(f"Frozen inputs changed: {changed}")
    if sha(out / "frozen/inputs/profile.npz") != TABLE_SHA:
        raise RuntimeError("The final average table changed")
    return manifest


def verify_checkpoint(status):
    if status["state"] != "complete":
        raise ValueError("Incomplete checkpoint")
    for path, expected in status["checkpoint_hashes"].items():
        if sha(path) != expected:
            raise RuntimeError(f"Checkpoint changed: {path}")


def network_hash(network):
    h = hashlib.sha256()
    for name, value in sorted(network.state_dict().items()):
        h.update(name.encode())
        h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()
