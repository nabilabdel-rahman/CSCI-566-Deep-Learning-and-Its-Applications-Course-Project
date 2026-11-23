#!/bin/bash
# Usage (from the repo root):
#   sbatch --array=0-2 carc/train.slurm configs/phase1_baseline.yaml [extra.overrides=...]
# The array index is used as the random seed.

#SBATCH --account=REPLACE_WITH_PROJECT_ID
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --job-name=camelyon
#SBATCH --output=logs/%x_%A_%a.out

set -euo pipefail

CONFIG="$1"
shift
SEED="${SLURM_ARRAY_TASK_ID:-0}"

module purge
eval "$(conda shell.bash hook)"
conda activate csci566

export TQDM_MININTERVAL=60

echo "Job $SLURM_JOB_ID | node $(hostname) | seed $SEED | config $CONFIG"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python train.py --config "$CONFIG" --seed "$SEED" --override \
  data.root_dir="/scratch1/$USER/data" \
  data.download=false \
  data.num_workers=8 \
  "$@"
