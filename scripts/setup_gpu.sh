#!/usr/bin/env bash
# One-time setup on a rented Linux GPU machine (RunPod, Lambda, Vast.ai, ...).
# Run from the repo root:  bash scripts/setup_gpu.sh
# Needs ~40 GB free disk (archive + extracted dataset). Set DATA_ROOT to put data elsewhere.
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-data}"
PYTHON="${PYTHON:-python3}"

# Reuses a preinstalled CUDA PyTorch if the image has one; otherwise pip installs it.
$PYTHON -m pip install --upgrade pip
$PYTHON -m pip install -r requirements.txt

bash scripts/download_camelyon17.sh "$DATA_ROOT"
mkdir -p logs

$PYTHON - <<'EOF'
import torch, wilds
print("torch", torch.__version__, "| wilds", wilds.__version__)
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available - check that this machine has an NVIDIA GPU and a CUDA build of PyTorch.")
print("GPU:", torch.cuda.get_device_name(0))
EOF
echo "Setup complete. Dataset at $DATA_ROOT/camelyon17_v1.0"
