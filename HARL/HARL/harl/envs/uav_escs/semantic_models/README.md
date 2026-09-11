# CRL-SemCom-VidCI Model Inventory

Source project: `/home/king/Downloads/Projects/TON/CRL-SemCom-VidCI`

## Counts
- `mst_fixed_baseline`: 1
- `ran_adaptive_fixed_sci`: 20
- `sci_baseline`: 1
- `semcom_full_sci_sce_scd_ran`: 15

## Recommended HARL Mode Mapping
- Use fixed-SNR `mu_comm` sweeps from `family=ran_adaptive_fixed_sci` as semantic modes when modes should represent quality/load Pareto points.
- Use `family=semcom_full_sci_sce_scd_ran` as candidate SCI/SCE/SCD modes trained at different SNR and `mu_comm`.
- Use `latest_checkpoint` unless a validation table identifies a better epoch.

## Runs
### sci_baseline
| id | mu | snr_db | latest_epoch | ckpts | freeze_shutter | freeze_decoder | latest_checkpoint |
|---|---:|---:|---:|---:|---|---|---|
| SCI_M | None | None | 299 | 300 | False | False | `logs/26-04-17/26-04-17-MST/SCI_M/v_1/checkpoints/model_epoch_0299.pth` |

### semcom_full_sci_sce_scd_ran
| id | mu | snr_db | latest_epoch | ckpts | freeze_shutter | freeze_decoder | latest_checkpoint |
|---|---:|---:|---:|---:|---|---|---|
| SemCom_full_mu1e-4_snr0 | 0.0001 | 0.0 | 49 | 37 | True | True | `logs/26-04-19/26-04-19-MST/SemCom_full_mu1e-4_snr0/v_0/checkpoints/model_epoch_0049.pth` |
| SemCom_full_mu1e-4_snr5 | 0.0001 | 5.0 | 49 | 37 | True | True | `logs/26-04-19/26-04-19-MST/SemCom_full_mu1e-4_snr5/v_0/checkpoints/model_epoch_0049.pth` |
| SemCom_full_mu1e-4_snr10 | 0.0001 | 10.0 | 49 | 37 | True | True | `logs/26-04-19/26-04-19-MST/SemCom_full_mu1e-4_snr10/v_0/checkpoints/model_epoch_0049.pth` |
| SemCom_full_mu1e-4_snr15 | 0.0001 | 15.0 | 49 | 37 | True | True | `logs/26-04-19/26-04-19-MST/SemCom_full_mu1e-4_snr15/v_0/checkpoints/model_epoch_0049.pth` |
| SemCom_full_mu1e-4_snr20 | 0.0001 | 20.0 | 49 | 37 | True | True | `logs/26-04-19/26-04-19-MST/SemCom_full_mu1e-4_snr20/v_0/checkpoints/model_epoch_0049.pth` |
| SemCom_full_mu5e-4_snr0 | 0.0005 | 0.0 | 49 | 37 | True | True | `logs/26-04-20/26-04-20-MST/SemCom_full_mu5e-4_snr0/v_0/checkpoints/model_epoch_0049.pth` |
| SemCom_full_mu5e-4_snr5 | 0.0005 | 5.0 | 49 | 37 | True | True | `logs/26-04-20/26-04-20-MST/SemCom_full_mu5e-4_snr5/v_0/checkpoints/model_epoch_0049.pth` |
| SemCom_full_mu5e-4_snr10 | 0.0005 | 10.0 | 49 | 37 | True | True | `logs/26-04-20/26-04-20-MST/SemCom_full_mu5e-4_snr10/v_0/checkpoints/model_epoch_0049.pth` |
| SemCom_full_mu5e-4_snr15 | 0.0005 | 15.0 | 49 | 37 | True | True | `logs/26-04-20/26-04-20-MST/SemCom_full_mu5e-4_snr15/v_0/checkpoints/model_epoch_0049.pth` |
| SemCom_full_mu5e-4_snr20 | 0.0005 | 20.0 | 49 | 37 | True | True | `logs/26-04-20/26-04-20-MST/SemCom_full_mu5e-4_snr20/v_0/checkpoints/model_epoch_0049.pth` |
| SemCom_full_mu1e-3_snr0 | 0.001 | 0.0 | 49 | 37 | True | True | `logs/26-04-21/26-04-21-MST/SemCom_full_mu1e-3_snr0/v_0/checkpoints/model_epoch_0049.pth` |
| SemCom_full_mu1e-3_snr5 | 0.001 | 5.0 | 49 | 37 | True | True | `logs/26-04-21/26-04-21-MST/SemCom_full_mu1e-3_snr5/v_0/checkpoints/model_epoch_0049.pth` |
| SemCom_full_mu1e-3_snr10 | 0.001 | 10.0 | 49 | 37 | True | True | `logs/26-04-21/26-04-21-MST/SemCom_full_mu1e-3_snr10/v_0/checkpoints/model_epoch_0049.pth` |
| SemCom_full_mu1e-3_snr15 | 0.001 | 15.0 | 49 | 37 | True | True | `logs/26-04-21/26-04-21-MST/SemCom_full_mu1e-3_snr15/v_0/checkpoints/model_epoch_0049.pth` |
| SemCom_full_mu1e-3_snr20 | 0.001 | 20.0 | 49 | 37 | True | True | `logs/26-04-21/26-04-21-MST/SemCom_full_mu1e-3_snr20/v_0/checkpoints/model_epoch_0049.pth` |

### ran_adaptive_fixed_sci
| id | mu | snr_db | latest_epoch | ckpts | freeze_shutter | freeze_decoder | latest_checkpoint |
|---|---:|---:|---:|---:|---|---|---|
| SemCom_from_MST_adaptive_smallscale_mu1e-4_snr0 | 0.0001 | 0.0 | 49 | 37 | True | True | `logs/26-04-22/26-04-22-MST/SemCom_from_MST_adaptive_smallscale_mu1e-4_snr0/v_0/checkpoints/model_epoch_0049.pth` |
| SemCom_from_MST_adaptive_smallscale_mu1e-4_snr5 | 0.0001 | 5.0 | 49 | 37 | True | True | `logs/26-04-22/26-04-22-MST/SemCom_from_MST_adaptive_smallscale_mu1e-4_snr5/v_0/checkpoints/model_epoch_0049.pth` |
| SemCom_from_MST_adaptive_smallscale_mu1e-4_snr10 | 0.0001 | 10.0 | 49 | 37 | True | True | `logs/26-04-22/26-04-22-MST/SemCom_from_MST_adaptive_smallscale_mu1e-4_snr10/v_0/checkpoints/model_epoch_0049.pth` |
| SemCom_from_MST_adaptive_smallscale_mu1e-4_snr15 | 0.0001 | 15.0 | 49 | 37 | True | True | `logs/26-04-22/26-04-22-MST/SemCom_from_MST_adaptive_smallscale_mu1e-4_snr15/v_0/checkpoints/model_epoch_0049.pth` |
| SemCom_from_MST_adaptive_smallscale_mu1e-4_snr20 | 0.0001 | 20.0 | 49 | 37 | True | True | `logs/26-04-22/26-04-22-MST/SemCom_from_MST_adaptive_smallscale_mu1e-4_snr20/v_0/checkpoints/model_epoch_0049.pth` |
| SemCom_from_MST_adaptive_smallscale_mu5e-4_snr0 | 0.0005 | 0.0 | 49 | 37 | True | True | `logs/26-04-23/26-04-23-MST/SemCom_from_MST_adaptive_smallscale_mu5e-4_snr0/v_0/checkpoints/model_epoch_0049.pth` |
| SemCom_from_MST_adaptive_smallscale_mu5e-4_snr5 | 0.0005 | 5.0 | 49 | 37 | True | True | `logs/26-04-23/26-04-23-MST/SemCom_from_MST_adaptive_smallscale_mu5e-4_snr5/v_0/checkpoints/model_epoch_0049.pth` |
| SemCom_from_MST_adaptive_smallscale_mu5e-4_snr10 | 0.0005 | 10.0 | 49 | 37 | True | True | `logs/26-04-23/26-04-23-MST/SemCom_from_MST_adaptive_smallscale_mu5e-4_snr10/v_0/checkpoints/model_epoch_0049.pth` |
| SemCom_from_MST_adaptive_smallscale_mu5e-4_snr15 | 0.0005 | 15.0 | 49 | 37 | True | True | `logs/26-04-23/26-04-23-MST/SemCom_from_MST_adaptive_smallscale_mu5e-4_snr15/v_0/checkpoints/model_epoch_0049.pth` |
| SemCom_from_MST_adaptive_smallscale_mu5e-4_snr20 | 0.0005 | 20.0 | 49 | 37 | True | True | `logs/26-04-23/26-04-23-MST/SemCom_from_MST_adaptive_smallscale_mu5e-4_snr20/v_0/checkpoints/model_epoch_0049.pth` |
| SemCom_from_MST_adaptive_smallscale_mu1e-3_snr0 | 0.001 | 0.0 | 49 | 37 | True | True | `logs/26-04-23/26-04-23-MST/SemCom_from_MST_adaptive_smallscale_mu1e-3_snr0/v_0/checkpoints/model_epoch_0049.pth` |
| SemCom_from_MST_adaptive_smallscale_mu1e-3_snr5 | 0.001 | 5.0 | 49 | 37 | True | True | `logs/26-04-24/26-04-24-MST/SemCom_from_MST_adaptive_smallscale_mu1e-3_snr5/v_0/checkpoints/model_epoch_0049.pth` |
| SemCom_from_MST_adaptive_smallscale_mu1e-3_snr10 | 0.001 | 10.0 | 49 | 37 | True | True | `logs/26-04-24/26-04-24-MST/SemCom_from_MST_adaptive_smallscale_mu1e-3_snr10/v_0/checkpoints/model_epoch_0049.pth` |
| SemCom_from_MST_adaptive_smallscale_mu1e-3_snr15 | 0.001 | 15.0 | 49 | 37 | True | True | `logs/26-04-24/26-04-24-MST/SemCom_from_MST_adaptive_smallscale_mu1e-3_snr15/v_0/checkpoints/model_epoch_0049.pth` |
| SemCom_from_MST_adaptive_smallscale_mu1e-3_snr20 | 0.001 | 20.0 | 49 | 37 | True | True | `logs/26-04-24/26-04-24-MST/SemCom_from_MST_adaptive_smallscale_mu1e-3_snr20/v_0/checkpoints/model_epoch_0049.pth` |
| SemCom_from_MST_adaptive_smallscale_mu2e-3_snr0 | 0.002 | 0.0 | 49 | 37 | True | True | `logs/26-04-24/26-04-24-MST/SemCom_from_MST_adaptive_smallscale_mu2e-3_snr0/v_0/checkpoints/model_epoch_0049.pth` |
| SemCom_from_MST_adaptive_smallscale_mu2e-3_snr5 | 0.002 | 5.0 | 49 | 37 | True | True | `logs/26-04-24/26-04-24-MST/SemCom_from_MST_adaptive_smallscale_mu2e-3_snr5/v_0/checkpoints/model_epoch_0049.pth` |
| SemCom_from_MST_adaptive_smallscale_mu2e-3_snr10 | 0.002 | 10.0 | 49 | 37 | True | True | `logs/26-04-25/26-04-25-MST/SemCom_from_MST_adaptive_smallscale_mu2e-3_snr10/v_0/checkpoints/model_epoch_0049.pth` |
| SemCom_from_MST_adaptive_smallscale_mu2e-3_snr15 | 0.002 | 15.0 | 49 | 37 | True | True | `logs/26-04-25/26-04-25-MST/SemCom_from_MST_adaptive_smallscale_mu2e-3_snr15/v_0/checkpoints/model_epoch_0049.pth` |
| SemCom_from_MST_adaptive_smallscale_mu2e-3_snr20 | 0.002 | 20.0 | 49 | 37 | True | True | `logs/26-04-25/26-04-25-MST/SemCom_from_MST_adaptive_smallscale_mu2e-3_snr20/v_0/checkpoints/model_epoch_0049.pth` |

### mst_fixed_baseline
| id | mu | snr_db | latest_epoch | ckpts | freeze_shutter | freeze_decoder | latest_checkpoint |
|---|---:|---:|---:|---:|---|---|---|
| MST_fixed | None | None | 133 | 4 | False | False | `logs/24-03-08/24-03-08-MST/MST_fixed/v_4/checkpoints/model_epoch_0133.pth` |
## Model Set Presets

Additional preset mappings are stored in `semcom_model_sets.json`:

- `ran_adaptive_snr10_mu_sweep_4modes`: default fixed SNR 10 dB, mu sweep 1e-4/5e-4/1e-3/2e-3 for quality/load modes.
- `ran_adaptive_snr{0,5,10,15,20}_mu_sweep_4modes`: fixed-SNR alternatives.
- `ran_adaptive_snr0_15_mu_grid_16modes`: HARL default that combines fixed SNR 0/5/10/15 dB with mu sweep 1e-4/5e-4/1e-3/2e-3. The SNR 20 dB checkpoints are omitted because the current disaster backhaul operates mostly below that range.
- `ran_adaptive_mu5e-4_snr_sweep_5modes`: SNR-sweep ablation only; not the default semantic mode interpretation.
- `ran_adaptive_mu1e-3_snr_sweep_5modes`: SNR-sweep ablation only with stronger communication penalty.
- `full_semcom_mu5e-4_snr_sweep_5modes`: full SemCom family across SNR 0/5/10/15/20 dB.

Checkpoint component prefixes verified by inspection:

- SCI baseline: `shutter`, `decoder` only.
- SemCom checkpoints: `transmission.encoder`, `transmission.ran`, `transmission.decoder`, plus frozen `shutter` and `decoder`.
