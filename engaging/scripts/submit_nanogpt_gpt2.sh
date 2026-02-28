#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

PARTITION="${PARTITION:-mit_normal_gpu}"
GPU_TYPE="${GPU_TYPE:-h100}"
GPUS_PER_NODE="${GPUS_PER_NODE:-1}"
CPUS_PER_TASK="${CPUS_PER_TASK:-16}"
TIME="${TIME:-02:00:00}"
GRAD_ACC_STEPS="${GRAD_ACC_STEPS:-40}"
WANDB_LOG="${WANDB_LOG:-False}"

CMD=(
  sbatch
  -p "${PARTITION}"
  --gres="gpu:${GPU_TYPE}:${GPUS_PER_NODE}"
  --cpus-per-task="${CPUS_PER_TASK}"
  --time="${TIME}"
)

echo "[submit nanogpt] partition=${PARTITION} gpu=${GPU_TYPE}x${GPUS_PER_NODE} cpus=${CPUS_PER_TASK} time=${TIME} grad_acc=${GRAD_ACC_STEPS}"
GPU_TYPE="${GPU_TYPE}" GPUS_PER_NODE="${GPUS_PER_NODE}" GRAD_ACC_STEPS="${GRAD_ACC_STEPS}" WANDB_LOG="${WANDB_LOG}" \
  "${CMD[@]}" "${REPO_ROOT}/slurm/nanogpt/train_gpt2.sh"
