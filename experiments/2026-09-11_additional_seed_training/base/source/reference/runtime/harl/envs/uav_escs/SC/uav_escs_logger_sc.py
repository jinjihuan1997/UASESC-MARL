import os

import numpy as np

from harl.common.base_logger import BaseLogger


class SCUAVLogger(BaseLogger):
    """Logger for the fixed multi-objective semantic video acquisition env."""

    METRIC_KEYS = (
        "utility",
        "quality_gain",
        "avg_quality_gain",
        "load_cost",
        "A_fair",
        "avg_aoi",
        "max_aoi",
        "scheduled_count",
        "avg_lambda_usage",
        "avg_phi_budget",
        "sum_backhaul_rate",
        "instruction_id",
        "base_reward",
        "total_reward",
        "quality_violation_bucketwise",
        "aoi_violation_max",
        "recv_aoi_bonus",
        "common_evaluation_reward",
    )

    def get_task_name(self):
        return self.env_args.get("env_name", "semantic_video_acquisition_fixed_multiobj")

    @staticmethod
    def _as_float(value, default=0.0):
        try:
            return float(value)
        except Exception:
            return float(default)

    @staticmethod
    def _extract_info(info):
        if isinstance(info, dict):
            return info
        if isinstance(info, (list, tuple, np.ndarray)):
            for item in info:
                if isinstance(item, dict):
                    return item
        return {}

    @staticmethod
    def _safe_open_append(path, header):
        try:
            need_header = (not os.path.exists(path)) or (os.path.getsize(path) == 0)
            fp = open(path, "a", buffering=1)
            if need_header:
                fp.write(header.rstrip("\n") + "\n")
            return fp
        except Exception:
            return None

    @staticmethod
    def _append_csv_row(fp, values):
        if fp is None:
            return
        try:
            fp.write(",".join(map(str, values)) + "\n")
        except Exception:
            pass

    def _infer_log_dir(self):
        try:
            log_path = getattr(self.log_file, "name", "")
            if log_path:
                return os.path.dirname(log_path)
        except Exception:
            pass
        return "."

    def _reset_metric_windows(self):
        self.train_metrics = {key: [] for key in self.METRIC_KEYS}

    def _ensure_metric_io(self):
        if not hasattr(self, "train_metrics"):
            self._reset_metric_windows()
        if getattr(self, "train_metrics_fp", None) is None:
            log_dir = self._infer_log_dir()
            self.train_metrics_fp = self._safe_open_append(
                os.path.join(log_dir, "train_semantic_metrics.csv"),
                ",".join(("total_num_steps", "average_step_rewards", "average_episode_rewards", *self.METRIC_KEYS)),
            )

    def _metric_dicts_from_infos(self, infos):
        if infos is None:
            return []
        try:
            iterable = list(infos)
        except Exception:
            return []
        out = []
        for item in iterable:
            info = self._extract_info(item)
            if info:
                out.append(info)
        return out

    def _update_window(self, metrics, infos):
        for key in self.METRIC_KEYS:
            vals = [self._as_float(info.get(key, 0.0)) for info in infos]
            if vals:
                metrics[key].append(float(np.mean(vals)))

    def eval_init(self):
        super().eval_init()
        self._ensure_metric_io()

    def per_step(self, data):
        self._ensure_metric_io()
        super().per_step(data)
        infos = data[4] if len(data) > 4 else None
        info_dicts = self._metric_dicts_from_infos(infos)
        self._update_window(self.train_metrics, info_dicts)

    def eval_per_step(self, eval_data):
        super().eval_per_step(eval_data)

    def episode_log(self, actor_train_infos, critic_train_info, actor_buffer, critic_buffer):
        self._ensure_metric_io()
        avg_ep_reward = float(np.mean(self.done_episodes_rewards)) if len(self.done_episodes_rewards) > 0 else 0.0
        super().episode_log(actor_train_infos, critic_train_info, actor_buffer, critic_buffer)

        env_infos = {}
        for key, vals in self.train_metrics.items():
            if vals:
                env_infos[f"train/{key}"] = [float(np.mean(vals))]
        if env_infos:
            self.log_env(env_infos)

        row = [
            self.total_num_steps,
            float(critic_train_info.get("average_step_rewards", 0.0)),
            avg_ep_reward,
        ]
        row.extend(float(np.mean(self.train_metrics[key])) if self.train_metrics[key] else 0.0 for key in self.METRIC_KEYS)
        self._append_csv_row(self.train_metrics_fp, row)
        self._reset_metric_windows()

    def eval_log(self, eval_episode):
        chunks = [rewards for rewards in self.eval_episode_rewards if len(rewards) > 0]
        if chunks:
            self.eval_episode_rewards = np.concatenate(chunks)
            average_episode_rewards = float(np.mean(self.eval_episode_rewards))
            max_episode_rewards = float(np.max(self.eval_episode_rewards))
        else:
            average_episode_rewards = 0.0
            max_episode_rewards = 0.0

        self.log_env(
            {
                "eval_average_episode_rewards": [average_episode_rewards],
                "eval_max_episode_rewards": [max_episode_rewards],
            }
        )

    def close(self):
        for name in ("train_metrics_fp",):
            fp = getattr(self, name, None)
            if fp is None:
                continue
            try:
                fp.flush()
                fp.close()
            except Exception:
                pass
            setattr(self, name, None)
