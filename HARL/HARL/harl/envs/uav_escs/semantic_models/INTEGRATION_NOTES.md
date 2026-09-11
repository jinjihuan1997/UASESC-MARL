# SemCom Results Integration Notes

This directory connects trained `CRL-SemCom-VidCI` artifacts to the HARL
instruction-conditioned semantic video acquisition environment.

## What The Results Are

`SCI_M`

- Role: fixed SCI sampler/reconstruction backbone.
- Checkpoint content: `shutter` and `decoder` state dict keys.
- HARL use: represents the fixed DS/UAV/RCC synchronized SCI front-end. It is
  not an RL action by itself.

`SemCom_full_mu*_snr*`

- Role: full semantic communication candidate checkpoints trained at specific
  SNR and communication penalty settings.
- Checkpoint content: `transmission.encoder`, `transmission.ran`,
  `transmission.decoder`, plus frozen SCI backbone keys.
- HARL use: can define semantic modes where each mode is a complete
  SCE/RAN/SCD operating point.

`SemCom_from_MST_adaptive_smallscale_mu*_snr*`

- Role: fixed-SCI SemCom checkpoints with learned RAN behavior.
- Checkpoint content: `transmission.encoder`, `transmission.ran`,
  `transmission.decoder`, plus frozen SCI backbone keys.
- HARL use: default source of `M_sc`. Each checkpoint becomes one UAV-level
  semantic mode.

## How HARL Uses Them

The HARL environment does not run video reconstruction networks during RL
training. Instead, it loads checkpoint metadata from `semcom_model_sets.json`.
This converts each trained SemCom checkpoint into an online semantic mode with:

- `trained_snr_db`: the channel condition where that mode was trained.
- `mu_comm`: the quality/load tradeoff used during training.
- `checkpoint`: the model path for later offline evaluation or deployment.
- `component_prefixes`: where SCE/RAN/SCD weights live inside the checkpoint.

During each HARL step, this metadata drives:

- `Lambda_sem[n,k,m]`: estimated semantic backhaul load of scheduling DS `k`
  from UAV `n` using mode `m`.
- `Q_hat_rec[n,k,m]`: predicted reconstruction quality used for online
  feasibility and reward.
- `M_feas[n,k,m]`: quality-threshold action mask for instruction `g[t]`.

The actual UAV action still chooses only:

- one common mode `m` for the UAV in the slot;
- a scheduling priority over cached DS streams.

The environment then enforces cache causality, common-mode consistency, quality
feasibility, and UAV backhaul budget.

## Default Mode Set

The default config uses:

`ran_adaptive_snr10_mu_sweep_4modes`

This maps four fixed-SCI adaptive SemCom checkpoints trained at the same SNR
10 dB to four HARL modes. The mode axis is `mu_comm`:

- mode 0: `mu_comm=1e-4`
- mode 1: `mu_comm=5e-4`
- mode 2: `mu_comm=1e-3`
- mode 3: `mu_comm=2e-3`

Practical interpretation:

- Online channel SNR is not a mode; it is already part of the environment state
  through `gamma_bh`.
- Low `mu_comm` modes spend more semantic symbols and target higher reconstruction quality.
- High `mu_comm` modes are more bandwidth-frugal and represent lower-load operating points.
- The fixed training SNR is the baseline condition used to compare these quality/load tradeoffs.

## Why This Matters

These trained results are not rewards by themselves. Their role is to define the
semantic action space and the physical meaning of each mode:

- They give `M_sc` real checkpoint identities instead of synthetic mode labels.
- They determine which modes are feasible under a given RCC instruction.
- They determine the semantic load cost paid by the scheduler.
- They determine the predicted quality gain credited in the shared reward.
- They make trained HARL policies deployable: selected `mode_id` can be mapped
  back to a concrete SCE/RAN/SCD checkpoint.

## Files

- `semcom_manifest.json`: all discovered CRL-SemCom-VidCI runs.
- `semcom_manifest.csv`: compact table view.
- `semcom_model_sets.json`: curated mode-set presets used by HARL.
- `semantic_registry.py`: metadata adapter used by the environment.

## Current Limitation

The training-time `Q_hat_rec` and `Lambda_sem` are metadata-based surrogates.
For final paper/evaluation numbers, run selected checkpoints through the real
CRL-SemCom-VidCI SCE/RAN/SCD pipeline and fit or replace `phi_Q`.
