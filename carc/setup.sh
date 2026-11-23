#!/usr/bin/env bash
# One-time setup on CARC Discovery. Run from the repo root on a login node:
#   bash carc/setup.sh
set -euo pipefail

ENV_NAME=csci566
DATA_ROOT="/scratch1/$USER/data"

module purge
module load conda
eval "$(conda shell.bash hook)"

if ! conda env list | grep -q "^$ENV_NAME "; then
  conda create -y -n "$ENV_NAME" python=3.11
fi
conda activate "$ENV_NAME"
pip install --upgrade pip
pip install -r requirements.txt

bash scripts/download_camelyon17.sh "$DATA_ROOT"
mkdir -p logs

python -c "import torch, wilds; print('torch', torch.__version__, '| wilds', wilds.__version__)"
echo "Setup complete. Dataset at $DATA_ROOT/camelyon17_v1.0"
