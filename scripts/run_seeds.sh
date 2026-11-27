#!/usr/bin/env bash
# Trains one config over several seeds, then runs the extra evaluations (last epoch, AdaBN)
# for each finished seed. Each seed logs to logs/<config>_seed<N>.log.
# Run from the repo root, ideally inside tmux so it survives SSH disconnects:
#   bash scripts/run_seeds.sh configs/phase1_baseline.yaml [extra.overrides=...]
#
# Environment variables:
#   SEEDS="0 1 2"    seeds to run
#   MAX_PARALLEL=1   how many seeds train at once on the GPU (3 works well for DenseNet-121)
#   NUM_WORKERS=8    data-loading workers per run (MAX_PARALLEL x NUM_WORKERS should be <= CPU cores)
#   DATA_ROOT=data   where scripts/setup_gpu.sh put the dataset
#   EVAL_VARIANTS=1  set to 0 to skip the extra evaluations
# Re-running the same command resumes unfinished seeds from their last epoch.
set -euo pipefail

CONFIG="$1"
shift
SEEDS="${SEEDS:-0 1 2}"
MAX_PARALLEL="${MAX_PARALLEL:-1}"
NUM_WORKERS="${NUM_WORKERS:-8}"
DATA_ROOT="${DATA_ROOT:-data}"
EVAL_VARIANTS="${EVAL_VARIANTS:-1}"
PYTHON="${PYTHON:-python3}"
NAME="$(basename "$CONFIG" .yaml)"
export DATA_ROOT NUM_WORKERS PYTHON

EXP_DIR="$($PYTHON -c '
import sys
from src.utils import load_config
cfg = load_config(sys.argv[1], sys.argv[2:])
print(cfg["output_dir"] + "/" + cfg["experiment_name"])
' "$CONFIG" "$@")"

export TQDM_MININTERVAL=30
mkdir -p logs

run_seed() {
  local seed="$1"
  shift
  local log="logs/${NAME}_seed${seed}.log"
  echo "Starting seed $seed -> $log"
  if ! $PYTHON -u train.py --config "$CONFIG" --seed "$seed" --override \
    data.root_dir="$DATA_ROOT" \
    data.download=false \
    data.num_workers="$NUM_WORKERS" \
    "$@" >> "$log" 2>&1; then
    echo "FAILED seed $seed (training) - see $log"
    return
  fi
  if [ "$EVAL_VARIANTS" = "1" ] && ! bash scripts/eval_variants.sh "$EXP_DIR/seed$seed" >> "$log" 2>&1; then
    echo "FAILED seed $seed (extra evaluations) - see $log"
    return
  fi
  echo "Finished seed $seed"
}

for seed in $SEEDS; do
  while [ "$(jobs -rp | wc -l)" -ge "$MAX_PARALLEL" ]; do
    wait -n 2>/dev/null || wait  # bash < 4.3 (macOS) lacks wait -n
  done
  run_seed "$seed" "$@" &
done
wait
echo "All seeds done: $EXP_DIR"
