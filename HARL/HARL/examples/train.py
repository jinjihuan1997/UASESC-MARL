"""Train an algorithm."""
import argparse
import json
import os
import sys
import types


def _suppress_gym_notice_banner():
    """Silence gym_notices banner before any gym import."""
    if os.environ.get("HARL_SHOW_GYM_NOTICE", "0") == "1":
        return

    pkg_name = "gym_notices"
    mod_name = "gym_notices.notices"

    if mod_name in sys.modules:
        mod = sys.modules[mod_name]
        if not hasattr(mod, "notices"):
            setattr(mod, "notices", {})
        return

    notices_mod = types.ModuleType(mod_name)
    notices_mod.notices = {}

    pkg_mod = sys.modules.get(pkg_name)
    if pkg_mod is None:
        pkg_mod = types.ModuleType(pkg_name)
        pkg_mod.__path__ = []
        sys.modules[pkg_name] = pkg_mod

    sys.modules[mod_name] = notices_mod
    setattr(pkg_mod, "notices", notices_mod)


_suppress_gym_notice_banner()

from harl.utils.configs_tools import get_defaults_yaml_args, update_args


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--algo",
        type=str,
        default="happo",
        choices=[
            "happo",
            "hatrpo",
            "haa2c",
            "haddpg",
            "hatd3",
            "hasac",
            "had3qn",
            "maddpg",
            "matd3",
            "mappo",
        ],
        help=(
            "Algorithm name. Choose from: "
            "happo, hatrpo, haa2c, haddpg, hatd3, hasac, had3qn, maddpg, matd3, mappo."
        ),
    )
    parser.add_argument(
        "--env",
        type=str,
        default="pettingzoo_mpe",
        choices=[
            "smac",
            "mamujoco",
            "pettingzoo_mpe",
            "gym",
            "football",
            "dexhands",
            "smacv2",
            "lag",
            "uav_escs_sc",
            "uav_escs_cc",
        ],
        help=(
            "Environment name. Choose from: "
            "smac, mamujoco, pettingzoo_mpe, gym, football, dexhands, smacv2, lag, "
            "uav_escs_sc, uav_escs_cc."
        ),
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default="installtest",
        help="Experiment name.",
    )
    parser.add_argument(
        "--load_config",
        type=str,
        default="",
        help="If set, load existing experiment config file instead of reading from yaml config file.",
    )

    # 解析已知参数，其余参数先保留为 unparsed_args，后面交给 update_args 去覆盖 yaml
    args, unparsed_args = parser.parse_known_args()

    def process(arg):
        """把命令行字符串尝试 eval 一下，方便传 bool / list / 数字."""
        try:
            return eval(arg)
        except Exception:
            return arg

    def parse_bool(v):
        """Robust bool parser for CLI values like true/false/1/0/on/off."""
        if isinstance(v, bool):
            return v
        if isinstance(v, (int, float)):
            return bool(v)
        s = str(v).strip().lower()
        if s in ("1", "true", "yes", "y", "on"):
            return True
        if s in ("0", "false", "no", "n", "off"):
            return False
        return bool(v)

    # 把 "--xxx v" 这样的残余参数转成 dict：{xxx: v}
    keys = [k[2:] for k in unparsed_args[0::2]]  # remove leading "--"
    values = [process(v) for v in unparsed_args[1::2]]
    unparsed_dict = {k: v for k, v in zip(keys, values)}

    # argparse.Namespace -> dict，后面传给 runner 用
    args = vars(args)

    # ============= 读取配置：已有 json 或 yaml 默认 =============
    if args["load_config"] != "":  # 从已有 config.json 加载
        with open(args["load_config"], encoding="utf-8") as file:
            all_config = json.load(file)

        args["algo"] = all_config["main_args"]["algo"]
        args["env"] = all_config["main_args"]["env"]

        # ★ 保持 exp_name
        if "exp_name" in all_config.get("main_args", {}):
            args["exp_name"] = all_config["main_args"]["exp_name"]

        algo_args = all_config["algo_args"]
        env_args = all_config["env_args"]
    else:  # 从对应 algo/env 的 yaml 默认配置读取
        algo_args, env_args = get_defaults_yaml_args(args["algo"], args["env"])

    # 合并参数：命令行显式值覆盖 yaml/config 中的默认值。
    update_args(unparsed_dict, algo_args, env_args, override=True)

    # 例外：少数关键项保留显式归一化，保证 bool/int 解析稳定。
    if "cuda" in unparsed_dict:
        if "device" not in algo_args or not isinstance(algo_args["device"], dict):
            algo_args["device"] = {}
        algo_args["device"]["cuda"] = parse_bool(unparsed_dict["cuda"])

    if "seed_specify" in unparsed_dict:
        if "seed" not in algo_args or not isinstance(algo_args["seed"], dict):
            algo_args["seed"] = {}
        algo_args["seed"]["seed_specify"] = parse_bool(unparsed_dict["seed_specify"])

    if "seed" in unparsed_dict:
        if "seed" not in algo_args or not isinstance(algo_args["seed"], dict):
            algo_args["seed"] = {}
        algo_args["seed"]["seed"] = int(unparsed_dict["seed"])
        # 显式给出 seed 时，强制启用固定种子，确保可复现实验。
        algo_args["seed"]["seed_specify"] = True

    # Keep lightweight run metadata for result organization.
    os.environ["UAVESCS_ALGO"] = str(args["algo"])
    os.environ["UAVESCS_ENV_NAME"] = str(args["env"])
    os.environ["UAVESCS_EXP_NAME"] = str(args.get("exp_name", "default_exp"))

    # dexhands 特殊处理
    if args["env"] == "dexhands":
        import isaacgym  # isaacgym 必须先于 torch 导入

    if args["env"] == "dexhands":
        # isaac gym 不支持独立 eval 环境，所以禁用 eval
        algo_args["eval"]["use_eval"] = False
        algo_args["train"]["episode_length"] = env_args["hands_episode_length"]

    # UAV-ESCS: place training results under <repo>/examples/results by default.
    # Keep explicit non-default logger paths unchanged.
    if args["env"] in ("uav_escs_sc", "uav_escs_cc"):
        if "logger" not in algo_args or not isinstance(algo_args["logger"], dict):
            algo_args["logger"] = {}
        cur_log_dir = str(algo_args["logger"].get("log_dir", "")).strip()
        if cur_log_dir in ("", "./results", "results", "./examples/results", "examples/results"):
            examples_dir = os.path.dirname(os.path.abspath(__file__))
            algo_args["logger"]["log_dir"] = os.path.join(examples_dir, "results")

    if args["env"] in ("uav_escs_sc", "uav_escs_cc"):
        env_args["env_name"] = str(env_args.get("env_name", "semantic_video_acquisition_fixed_multiobj"))

    # 启动训练
    from harl.runners import RUNNER_REGISTRY

    runner_cls = RUNNER_REGISTRY[args["algo"]]
    runner = runner_cls(args, algo_args, env_args)
    runner.run()
    runner.close()


if __name__ == "__main__":
    main()
