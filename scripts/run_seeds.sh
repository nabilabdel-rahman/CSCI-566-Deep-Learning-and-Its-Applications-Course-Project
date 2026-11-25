#!/usr/bin/env bash
# Runs one config over several seeds and logs each run to logs/.
# Run from the repo root, ideally inside tmux so it survives SSH disconnects:
#   bash scripts/run_seeds.sh configs/phase1_baseline.yaml [extra.overrides=...]
#
# Environment variables:
#   SEEDS="0 1 2"    seeds to run
#   PARALLEL=0       1 = run all seeds at once on the same GPU (faster for small models)
#   NUM_WORKERS=8    data-loading workers per run
#   DATA_ROOT=data   where scripts/setup_gpu.sh put the dataset
# Re-running the same command resumes unfinished seeds from their last epoch.
set -euo pipefail

CONFIG="$1"
shift
SEEDS="${SEEDS:-0 1 2}"
PARALLEL="${PARALLEL:-0}"
NUM_WORKERS="${NUM_WORKERS:-8}"
DATA_ROOT="${DATA_ROOT:-data}"
PYTHON="${PYTHON:-python3}"
NAME="$(basename "$CONFIG" .yaml)"

export TQDM_MININTERVAL=30
mkdir -p logs

run_seed() {
  local seed="$1"
  shift
  local log="logs/${NAME}_seed${seed}.log"
  echo "Starting seed $seed -> $log"
  if $PYTHON -u train.py --config "$CONFIG" --seed "$seed" --override \
    data.root_dir="$DATA_ROOT" \
    data.download=false \
    data.num_workers="$NUM_WORKERS" \
    "$@" >> "$log" 2>&1; then
    echo "Finished seed $seed"
  else
    echo "FAILED seed $seed - see $log"
  fi
}

for seed in $SEEDS; do
  if [ "$PARALLEL" = "1" ]; then
    run_seed "$seed" "$@" &
  else
    run_seed "$seed" "$@"
  fi
done
wait
echo "All seeds done. Results: outputs/*/seed*/metrics.json"
