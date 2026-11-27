#!/usr/bin/env bash
# Extra evaluations for finished training runs: last-epoch checkpoint and test-time BatchNorm
# adaptation (AdaBN) on both checkpoints. Writes eval_*.json files read by scripts/summarize.py.
#
# Usage (from the repo root): bash scripts/eval_variants.sh outputs/<experiment>[/seed<N>] ...
# Environment variables: DATA_ROOT=data, NUM_WORKERS=8, PYTHON=python3
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-data}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PYTHON="${PYTHON:-python3}"

for target in "$@"; do
  if [ -f "$target/best.pt" ]; then
    seed_dirs=("$target")
  else
    seed_dirs=("$target"/seed*)
  fi
  for dir in "${seed_dirs[@]}"; do
    for args in "last.pt" "best.pt --adapt-bn" "last.pt --adapt-bn"; do
      read -r ckpt flag <<< "$args"
      echo "== $dir/$ckpt ${flag:-}"
      $PYTHON -u evaluate.py --checkpoint "$dir/$ckpt" ${flag:-} \
        --data-root "$DATA_ROOT" --num-workers "$NUM_WORKERS"
    done
  done
done
