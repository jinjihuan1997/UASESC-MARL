"""
HARL package init.

By default we silence Gym's legacy `gym_notices` banner to avoid multi-process
stderr spam (one notice per worker process).

Set `HARL_SHOW_GYM_NOTICE=1` to restore upstream Gym notice behavior.
"""

from __future__ import annotations

import os
import sys
import types


def _suppress_gym_notice() -> None:
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
        # mark as namespace-like package so submodule import works
        pkg_mod.__path__ = []
        sys.modules[pkg_name] = pkg_mod

    sys.modules[mod_name] = notices_mod
    setattr(pkg_mod, "notices", notices_mod)


_suppress_gym_notice()
