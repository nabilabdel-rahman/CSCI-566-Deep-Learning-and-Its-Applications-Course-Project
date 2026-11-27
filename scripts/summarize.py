"""Aggregate results across seeds into results/summary.md and results/summary.csv.

Each experiment lives in <outputs>/<experiment>/seed<N>/. For each seed this reads:
  - preds_<split>.npz      -> "best"        (checkpoint selected on the OOD validation hospital)
  - eval_last.json         -> "last"        (final epoch)
  - eval_best_adabn.json   -> "best+adabn"  (test-time BatchNorm adaptation)
  - eval_last_adabn.json   -> "last+adabn"

Usage:
    python scripts/summarize.py [--outputs outputs] [--results results]
"""
import argparse
import csv
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.metrics import compute_metrics  # noqa: E402

SPLITS = ("id_val", "val", "test")
EVAL_FILES = {"last": "eval_last.json", "best+adabn": "eval_best_adabn.json", "last+adabn": "eval_last_adabn.json"}
COLUMNS = [
    ("test", "acc"), ("test", "sensitivity"), ("test", "specificity"), ("test", "auroc"), ("test", "ece"),
    ("val", "acc"), ("id_val", "acc"),
]


def metrics_from_preds(seed_dir):
    out = {}
    for split in SPLITS:
        path = os.path.join(seed_dir, f"preds_{split}.npz")
        if not os.path.exists(path):
            return None
        d = np.load(path)
        out[split] = compute_metrics(d["probs"], d["y"], d["hospital"])
    return out


def collect(outputs_dir):
    """Returns {(experiment, variant): {seed: {split: metrics}}}."""
    results = defaultdict(dict)
    for seed_dir in sorted(glob.glob(os.path.join(outputs_dir, "**", "seed*"), recursive=True)):
        if not os.path.isdir(seed_dir):
            continue
        experiment = os.path.basename(os.path.dirname(seed_dir))
        seed = int(os.path.basename(seed_dir).removeprefix("seed"))
        best = metrics_from_preds(seed_dir)
        if best:
            results[(experiment, "best")][seed] = best
        for variant, fname in EVAL_FILES.items():
            path = os.path.join(seed_dir, fname)
            if os.path.exists(path):
                with open(path) as f:
                    data = json.load(f)
                if all(s in data for s in SPLITS):
                    results[(experiment, variant)][seed] = data
    return results


def summarize(results):
    rows = []
    for (experiment, variant), seeds in sorted(results.items()):
        row = {"experiment": experiment, "variant": variant, "seeds": ",".join(map(str, sorted(seeds)))}
        for split, metric in COLUMNS:
            values = np.array([seeds[s][split][metric] for s in sorted(seeds)], dtype=float)
            row[f"{split}_{metric}_mean"] = values.mean()
            row[f"{split}_{metric}_std"] = values.std(ddof=1) if len(values) > 1 else float("nan")
        rows.append(row)
    return rows


def fmt(mean, std, metric):
    scale = 1 if metric in ("auroc", "ece") else 100
    digits = 3 if scale == 1 else 1
    s = f"{mean * scale:.{digits}f}"
    return s if np.isnan(std) else f"{s} ± {std * scale:.{digits}f}"


def to_markdown(rows):
    header = ["Experiment", "Variant", "Seeds"] + [f"{split} {metric}" for split, metric in COLUMNS]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in rows:
        cells = [r["experiment"], r["variant"], r["seeds"]]
        cells += [fmt(r[f"{s}_{m}_mean"], r[f"{s}_{m}_std"], m) for s, m in COLUMNS]
        lines.append("| " + " | ".join(cells) + " |")
    note = ("\nAccuracy, sensitivity and specificity in %; AUROC and ECE as fractions. "
            "Mean ± sample std across seeds. test = hospital 2, val = hospital 1, "
            "id_val = held-out patches from training hospitals 0, 3, 4.\n")
    return "\n".join(lines) + "\n" + note


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs", default="outputs")
    parser.add_argument("--results", default="results")
    args = parser.parse_args()

    rows = summarize(collect(args.outputs))
    if not rows:
        raise SystemExit(f"No results found under {args.outputs}/")
    os.makedirs(args.results, exist_ok=True)
    md = to_markdown(rows)
    with open(os.path.join(args.results, "summary.md"), "w") as f:
        f.write(md)
    with open(os.path.join(args.results, "summary.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(md)


if __name__ == "__main__":
    main()
