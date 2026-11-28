#!/usr/bin/env bash
# Extra evaluations for finished training runs, on both best.pt and last.pt:
#   eval_last.json                  last-epoch checkpoint, no adaptation
#   eval_{best,last}_adabn.json     BatchNorm re-estimated on the target hospital's unlabeled images
#   eval_{best,last}_srcbn.json     BatchNorm re-estimated on clean training images (no target data)
# Evaluations whose output file already exists are skipped, as are missing checkpoints.
#
# Usage (from the repo root): bash scripts/eval_variants.sh outputs/<experiment>[/seed<N>] ...
# Environment variables: DATA_ROOT=data, NUM_WORKERS=8, PYTHON=python3
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-data}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PYTHON="${PYTHON:-python3}"

VARIANTS=(
  "last.pt||eval_last.json"
  "best.pt|--adapt-bn|eval_best_adabn.json"
  "last.pt|--adapt-bn|eval_last_adabn.json"
  "best.pt|--adapt-bn-source|eval_best_srcbn.json"
  "last.pt|--adapt-bn-source|eval_last_srcbn.json"
)

for target in "$@"; do
  if [ -f "$target/best.pt" ]; then
    seed_dirs=("$target")
  else
    seed_dirs=("$target"/seed*)
  fi
  for dir in "${seed_dirs[@]}"; do
    for variant in "${VARIANTS[@]}"; do
      IFS="|" read -r ckpt flag out <<< "$variant"
      if [ -f "$dir/$out" ]; then
        echo "== skip $dir/$out (exists)"
        continue
      fi
      if [ ! -f "$dir/$ckpt" ]; then
        echo "== skip $dir/$out ($ckpt missing)"
        continue
      fi
      echo "== $dir/$ckpt ${flag}"
      $PYTHON -u evaluate.py --checkpoint "$dir/$ckpt" ${flag} \
        --data-root "$DATA_ROOT" --num-workers "$NUM_WORKERS"
    done
  done
done
