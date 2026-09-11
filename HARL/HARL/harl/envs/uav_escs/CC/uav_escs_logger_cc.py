from harl.envs.uav_escs.SC.uav_escs_logger_sc import SCUAVLogger


class CCUAVLogger(SCUAVLogger):
    """CC logger entry, reusing SC logger implementation."""

    def get_task_name(self):
        return self.env_args.get("env_name", "uav_escs_cc")
