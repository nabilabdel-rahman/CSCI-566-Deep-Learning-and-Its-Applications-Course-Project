#!/usr/bin/env bash
# One-time setup on a rented Linux GPU machine (Lambda, RunPod, Vast.ai, ...).
# Run from the repo root:  bash scripts/setup_gpu.sh
# Then activate the environment in every new shell:  source .venv/bin/activate
# Needs ~40 GB free disk (archive + extracted dataset). Set DATA_ROOT to put data elsewhere.
#
# On Lambda, PyTorch + CUDA come preinstalled (Lambda Stack). The venv is created with
# --system-site-packages so that install is reused instead of downloading a second copy,
# and so pip never touches system packages (avoids "externally-managed-environment").
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-data}"

if [ ! -d .venv ]; then
  python3 -m venv --system-site-packages .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

bash scripts/download_camelyon17.sh "$DATA_ROOT"
mkdir -p logs

python - <<'EOF'
import torch, wilds
print("torch", torch.__version__, "| wilds", wilds.__version__)
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available - check that this machine has an NVIDIA GPU and a CUDA build of PyTorch.")
print("GPU:", torch.cuda.get_device_name(0))
EOF
echo "Setup complete. Dataset at $DATA_ROOT/camelyon17_v1.0"
echo "Next: source .venv/bin/activate && tmux new -s train"
