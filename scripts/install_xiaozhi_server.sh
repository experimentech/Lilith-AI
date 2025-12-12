#!/usr/bin/env bash
set -euo pipefail

# Optional, isolated env for the Xiaozhi server surface.
# Flags:
#   TORCH_CPU_ONLY=1   install CPU-only torch to avoid CUDA megawheels (default: 1)
#   TMPDIR=/path       override temp dir if /tmp is small

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${ROOT}/.venv-xiaozhi"
TORCH_CPU_ONLY="${TORCH_CPU_ONLY:-1}"

# Ensure a roomy temp dir to avoid /tmp exhaustion during wheel builds/downloads.
TMPDIR="${TMPDIR:-${ROOT}/tmp/pip}"
mkdir -p "${TMPDIR}"
export TMPDIR

if [[ ! -d "${VENV}" ]]; then
  python3 -m venv "${VENV}"
fi

source "${VENV}/bin/activate"
python -m pip install --upgrade pip

# Core project deps plus Xiaozhi-facing server bits.
if [[ "${TORCH_CPU_ONLY}" == "1" ]]; then
  python -m pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu torch
else
  python -m pip install --no-cache-dir torch
fi

python -m pip install --no-cache-dir -r "${ROOT}/requirements.txt"
python -m pip install --no-cache-dir fastapi "uvicorn[standard]" websockets paho-mqtt cryptography pytest-asyncio httpx

echo "Xiaozhi server venv ready. Activate with: source ${VENV}/bin/activate"
