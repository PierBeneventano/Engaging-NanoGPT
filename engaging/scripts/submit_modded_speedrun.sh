#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

PARTITION="${PARTITION:-mit_normal_gpu}"
GPU_TYPE="${GPU_TYPE:-h100}"
NUM_GPUS="${NUM_GPUS:-1}"
CPUS_PER_TASK="${CPUS_PER_TASK:-16}"
TIME="${TIME:-00:30:00}"

case "${NUM_GPUS}" in
  1|2|4|8)
    ;;
  *)
    echo "NUM_GPUS must be one of: 1, 2, 4, 8 (current: ${NUM_GPUS})" >&2
    exit 1
    ;;
esac

CMD=(
  sbatch
  -p "${PARTITION}"
  --gres="gpu:${GPU_TYPE}:${NUM_GPUS}"
  --cpus-per-task="${CPUS_PER_TASK}"
  --time="${TIME}"
)

echo "[submit modded] partition=${PARTITION} gpu=${GPU_TYPE}x${NUM_GPUS} cpus=${CPUS_PER_TASK} time=${TIME}"
GPU_TYPE="${GPU_TYPE}" NUM_GPUS="${NUM_GPUS}" \
  "${CMD[@]}" "${REPO_ROOT}/slurm/modded/train_speedrun.sh"
