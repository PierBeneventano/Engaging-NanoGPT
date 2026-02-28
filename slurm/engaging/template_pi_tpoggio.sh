#!/usr/bin/env bash
#SBATCH --job-name=poggio-job
#SBATCH --partition=pi_tpoggio
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

CONDA_SH="${CONDA_SH:-${HOME}/miniforge3/etc/profile.d/conda.sh}"
ENV_PATH="${ENV_PATH:-${HOME}/conda_envs/nanogpt_env}"

mkdir -p logs

if command -v module >/dev/null 2>&1; then
  module load cuda/12.4.0 cudnn/9.8.0.87-cuda12
fi

source "${CONDA_SH}"
conda activate "${ENV_PATH}"

echo "Running on $(hostname) in partition ${SLURM_JOB_PARTITION:-unknown}"
nvidia-smi || true
python train.py
