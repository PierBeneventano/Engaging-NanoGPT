#!/usr/bin/env bash
set -euo pipefail

PARTITION="${PARTITION:-mit_normal_gpu}"
GPU_TYPE="${GPU_TYPE:-h100}"
GPUS="${GPUS:-1}"
CPUS="${CPUS:-8}"
MEM="${MEM:-32G}"
TIME="${TIME:-01:00:00}"

CMD=(
  salloc
  -p "${PARTITION}"
  -t "${TIME}"
  -c "${CPUS}"
  --mem="${MEM}"
  --gres="gpu:${GPU_TYPE}:${GPUS}"
)

echo "[interactive] ${CMD[*]}"
"${CMD[@]}"
