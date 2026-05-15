# UASESE-MARL

This repository contains a slimmed monorepo for the UASESE-MARL work:

- `HARL/`: the UAV-ESCS multi-agent reinforcement learning code based on HARL, keeping the SUT/UAV environments, MAPPO/HAPPO training code, configs, and paper experiment scripts.
- `CRL-SemCom-VidCI/`: the semantic communication and video compressed imaging code, keeping source modules, tests, requirements, and experiment scripts.

Large generated artifacts are intentionally excluded from GitHub, including virtual environments, training logs, checkpoints, downloaded datasets, cached bytecode, and local IDE metadata. Recreate dependencies from each subproject's `requirements.txt` and regenerate datasets/checkpoints outside version control.
