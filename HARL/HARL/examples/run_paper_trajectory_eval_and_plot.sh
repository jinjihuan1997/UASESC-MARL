#!/usr/bin/env bash
set -euo pipefail

# Run slot-level trajectory evaluation and generate paper figures for:
# 1) Effectiveness of Cooperative Learning Policies
# 2) Instruction Response Analysis
#
# This script does not train models and does not modify environment/reward code.
# It writes all new evaluation outputs under examples/results/paper_trajectory_eval
# by default, then runs the trajectory-based plotting script.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

ENV_NAME="${ENV_NAME:-uav_escs_sc}"
EPISODES="${EPISODES:-3}"
EPISODE_LENGTH="${EPISODE_LENGTH:-600}"
TRAIN_SEEDS="${TRAIN_SEEDS:-1,2,3}"
UNSEEN_SEEDS="${UNSEEN_SEEDS:-101,102,103}"

RESULTS_ROOT="${RESULTS_ROOT:-examples/results}"
EVAL_ROOT="${EVAL_ROOT:-${RESULTS_ROOT}/paper_trajectory_eval}"
PLOT_RESULTS_ROOT="${PLOT_RESULTS_ROOT:-${EVAL_ROOT}}"
PLOT_OUTPUT_DIR="${PLOT_OUTPUT_DIR:-${RESULTS_ROOT}/plot_results/semantic_video_paper_figures}"

RUN_BASELINES="${RUN_BASELINES:-1}"
RUN_HAPPO="${RUN_HAPPO:-1}"
RUN_MAPPO="${RUN_MAPPO:-1}"
RUN_PLOT="${RUN_PLOT:-1}"
RUN_SWITCH300_SUMMARY="${RUN_SWITCH300_SUMMARY:-0}"

HAPPO_RUN_DIR="${HAPPO_RUN_DIR:-examples/results/uav_escs_sc/semantic_video_acquisition_fixed_multiobj/happo/ichappo_random_switch_once_clean_load/seed-00001-2026-05-07-23-30-37}"
MAPPO_RUN_DIR="${MAPPO_RUN_DIR:-examples/results/uav_escs_sc/semantic_video_acquisition_fixed_multiobj/mappo/icmappo_random_switch_once_clean_load/seed-00001-2026-05-07-23-32-42}"

HAPPO_CONFIG="${HAPPO_CONFIG:-${HAPPO_RUN_DIR}/config.json}"
HAPPO_MODEL_DIR="${HAPPO_MODEL_DIR:-${HAPPO_RUN_DIR}/models}"
MAPPO_CONFIG="${MAPPO_CONFIG:-${MAPPO_RUN_DIR}/config.json}"
MAPPO_MODEL_DIR="${MAPPO_MODEL_DIR:-${MAPPO_RUN_DIR}/models}"

SMOOTH_WINDOW="${SMOOTH_WINDOW:-5}"
SWITCH_STEP="${SWITCH_STEP:-200}"
SWITCH_GRID_STEPS="${SWITCH_GRID_STEPS:-250,300,350}"
INCLUDE_SWITCH_GRID="${INCLUDE_SWITCH_GRID:-False}"
PLOT_STD="${PLOT_STD:-False}"
SAVE_CSV="${SAVE_CSV:-True}"

COMMON_ENV_OVERRIDES=(
  --instruction_mode_strategy random_switch_once
  --instruction_switch_min_step 200
  --instruction_switch_max_step 400
  --instruction_before_candidates "[0]"
  --instruction_after_candidates "[1,2]"
  --allow_same_instruction_after_switch False
)

run_baselines_for_seed() {
  local seed="$1"
  local group="$2"
  local scenario="random_switch_once_middle"
  local out_dir="${EVAL_ROOT}/01_effectiveness_cooperative_learning_policies/baselines/${scenario}/${group}_seed_${seed}"

  echo "[baseline] scenario=${scenario} seed=${seed} group=${group} output=${out_dir}"
  python examples/evaluate_uav_algorithm_baselines.py \
    --env "${ENV_NAME}" \
    --algo happo \
    --baselines random,round_robin,utility_greedy \
    --episodes "${EPISODES}" \
    --episode_length "${EPISODE_LENGTH}" \
    --seed "${seed}" \
    --scenario_name "${scenario}" \
    --output_dir "${out_dir}" \
    --dump_episode_trace True \
    --trace_max_episodes "${EPISODES}" \
    --trace_output_dir "${out_dir}/traces" \
    "${COMMON_ENV_OVERRIDES[@]}"
}

run_ic_happo_for_seed() {
  local seed="$1"
  local group="$2"
  local out_dir="${EVAL_ROOT}/02_instruction_response_analysis/ic_happo/${group}_seed_${seed}"

  if [[ ! -d "${HAPPO_MODEL_DIR}" || ! -f "${HAPPO_CONFIG}" ]]; then
    echo "WARNING: skip IC-HAPPO seed=${seed}; missing HAPPO_CONFIG=${HAPPO_CONFIG} or HAPPO_MODEL_DIR=${HAPPO_MODEL_DIR}"
    return 0
  fi

  echo "[IC-HAPPO] seed=${seed} group=${group} output=${out_dir}"
  python examples/evaluate_instruction_constraints.py \
    --env "${ENV_NAME}" \
    --algo happo \
    --load_config "${HAPPO_CONFIG}" \
    --model_dir "${HAPPO_MODEL_DIR}" \
    --episodes "${EPISODES}" \
    --episode_length "${EPISODE_LENGTH}" \
    --seed "${seed}" \
    --output_dir "${out_dir}" \
    --deterministic True \
    --include_switch_grid "${INCLUDE_SWITCH_GRID}" \
    --switch_grid_steps "${SWITCH_GRID_STEPS}" \
    "${COMMON_ENV_OVERRIDES[@]}"
}

run_ic_mappo_for_seed() {
  local seed="$1"
  local group="$2"
  local out_dir="${EVAL_ROOT}/02_instruction_response_analysis/ic_mappo/${group}_seed_${seed}"

  if [[ ! -d "${MAPPO_MODEL_DIR}" || ! -f "${MAPPO_CONFIG}" ]]; then
    echo "WARNING: skip IC-MAPPO seed=${seed}; missing MAPPO_CONFIG=${MAPPO_CONFIG} or MAPPO_MODEL_DIR=${MAPPO_MODEL_DIR}"
    return 0
  fi

  echo "[IC-MAPPO] seed=${seed} group=${group} output=${out_dir}"
  python examples/evaluate_instruction_constraints.py \
    --env "${ENV_NAME}" \
    --algo mappo \
    --load_config "${MAPPO_CONFIG}" \
    --model_dir "${MAPPO_MODEL_DIR}" \
    --episodes "${EPISODES}" \
    --episode_length "${EPISODE_LENGTH}" \
    --seed "${seed}" \
    --output_dir "${out_dir}" \
    --deterministic True \
    --include_switch_grid "${INCLUDE_SWITCH_GRID}" \
    --switch_grid_steps "${SWITCH_GRID_STEPS}" \
    "${COMMON_ENV_OVERRIDES[@]}"
}

run_seed_group() {
  local seeds_csv="$1"
  local group="$2"
  local IFS=","
  read -ra seeds <<< "${seeds_csv}"
  for seed in "${seeds[@]}"; do
    seed="$(echo "${seed}" | xargs)"
    [[ -z "${seed}" ]] && continue
    if [[ "${RUN_BASELINES}" == "1" ]]; then
      run_baselines_for_seed "${seed}" "${group}"
    fi
    if [[ "${RUN_HAPPO}" == "1" ]]; then
      run_ic_happo_for_seed "${seed}" "${group}"
    fi
    if [[ "${RUN_MAPPO}" == "1" ]]; then
      run_ic_mappo_for_seed "${seed}" "${group}"
    fi
  done
}

echo "Project directory: ${PROJECT_DIR}"
echo "Evaluation root: ${EVAL_ROOT}"
echo "Plot input root: ${PLOT_RESULTS_ROOT}"
echo "Plot output dir: ${PLOT_OUTPUT_DIR}"
echo "Instruction strategy: random_switch_once, switch range: [200, 400]"
echo "Switch grid steps: ${SWITCH_GRID_STEPS} (include=${INCLUDE_SWITCH_GRID})"

run_seed_group "${TRAIN_SEEDS}" "train"
run_seed_group "${UNSEEN_SEEDS}" "unseen"

if [[ "${RUN_PLOT}" == "1" ]]; then
  echo "[plot] generating paper figures"
  python examples/plot_semantic_video_paper_figures.py \
    --results_root "${PLOT_RESULTS_ROOT}" \
    --output_dir "${PLOT_OUTPUT_DIR}" \
    --smooth_window "${SMOOTH_WINDOW}" \
    --switch_step "${SWITCH_STEP}" \
    --plot_std "${PLOT_STD}" \
    --save_csv "${SAVE_CSV}" \
    --train_seeds "${TRAIN_SEEDS}" \
    --unseen_seeds "${UNSEEN_SEEDS}"
fi

if [[ "${RUN_SWITCH300_SUMMARY}" == "1" ]]; then
  echo "[summary] generating focused switch-300 reward/quality/AoI comparison"
  python examples/summarize_switch300_policy_comparison.py \
    --results_root "${EVAL_ROOT}" \
    --output_dir "${PLOT_OUTPUT_DIR}/00_switch300_policy_comparison" \
    --train_seeds "${TRAIN_SEEDS}" \
    --unseen_seeds "${UNSEEN_SEEDS}"
fi

echo "Done."
echo "Evaluation outputs: ${EVAL_ROOT}"
echo "Figure outputs: ${PLOT_OUTPUT_DIR}"
