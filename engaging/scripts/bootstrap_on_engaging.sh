#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

export CONDA_SH="${CONDA_SH:-${HOME}/miniforge3/etc/profile.d/conda.sh}"
export ENV_PATH="${ENV_PATH:-${HOME}/conda_envs/nanogpt_env}"

echo "[bootstrap] repo: ${REPO_ROOT}"
echo "[bootstrap] CONDA_SH=${CONDA_SH}"
echo "[bootstrap] ENV_PATH=${ENV_PATH}"

echo "[bootstrap] Running setup_env.sh"
bash setup_env.sh

echo "[bootstrap] Complete. Activate env with: source activate_env.sh"
