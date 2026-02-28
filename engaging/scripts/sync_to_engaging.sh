#!/usr/bin/env bash
set -euo pipefail

ENGAGING_HOST="${ENGAGING_HOST:-orcd-login.mit.edu}"
ENGAGING_USER="${ENGAGING_USER:-}"
REMOTE_ROOT="${REMOTE_ROOT:-}"

if [[ -z "${ENGAGING_USER}" ]]; then
  echo "Set ENGAGING_USER first, e.g. ENGAGING_USER=<kerberos_username> bash engaging/scripts/sync_to_engaging.sh" >&2
  exit 1
fi

if [[ -z "${REMOTE_ROOT}" ]]; then
  REMOTE_ROOT="/home/${ENGAGING_USER}/Engaging-NanoGPT"
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

mkdir -p "${REPO_ROOT}/logs"

echo "[sync] ${REPO_ROOT} -> ${ENGAGING_USER}@${ENGAGING_HOST}:${REMOTE_ROOT}"

rsync -avP \
  --exclude '.git/' \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  --exclude '.DS_Store' \
  --exclude 'out/' \
  "${REPO_ROOT}/" "${ENGAGING_USER}@${ENGAGING_HOST}:${REMOTE_ROOT}/"

echo "[sync] Done"
