#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"

echo "[1/4] Creating virtualenv at ${VENV_DIR}"
python3 -m venv "${VENV_DIR}"

echo "[2/4] Upgrading pip/setuptools/wheel"
source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip setuptools wheel

echo "[3/4] Installing CPU-only PyTorch stack"
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

echo "[4/4] Installing scientific + plotting dependencies"
python -m pip install -r "${ROOT_DIR}/requirements.txt"

python - <<'PY'
import torch
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CPU threads:", torch.get_num_threads())
PY

echo "Setup complete."
echo "Activate with: source ${VENV_DIR}/bin/activate"
