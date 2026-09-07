#!/usr/bin/env bash
# Idempotent install for the SimMeR prototype in a Cloud Agent environment.
# Safe to run repeatedly: it only installs what is missing.
set -euo pipefail

cd "$(dirname "$0")/.."

# --- system dependency: python venv support (missing from some base images) ---
if ! python3 -c "import ensurepip" >/dev/null 2>&1; then
  echo "[install] installing python3-venv..."
  sudo apt-get update -qq
  sudo apt-get install -y -qq python3-venv >/dev/null
fi

# --- project virtual environment ---
if [ ! -d ".venv" ]; then
  echo "[install] creating virtual environment..."
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
. .venv/bin/activate

python -m pip install --upgrade pip -q

# --- CPU-only PyTorch from the dedicated index, then the rest ---
echo "[install] installing PyTorch (CPU) ..."
python -m pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.4.1" -q

echo "[install] installing scientific stack ..."
python -m pip install "numpy>=1.26,<3" "pandas>=2.1" "scikit-learn>=1.3" "scipy>=1.11" "matplotlib>=3.7" "pytest>=7" -q

# --- install the package itself (editable) ---
python -m pip install -e . -q

echo "[install] verifying import ..."
python -c "import simmer, torch; print('simmer', simmer.__version__, '| torch', torch.__version__)"
echo "[install] done."
