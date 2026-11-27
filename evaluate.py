"""Evaluate a saved checkpoint (best.pt or last.pt) on one or more Camelyon17-WILDS splits.

Results are written next to the checkpoint as eval_<checkpoint>[_adabn].json, which
scripts/summarize.py picks up.

Examples:
    python evaluate.py --checkpoint outputs/phase1_erm_densenet121/seed0/last.pt
    python evaluate.py --checkpoint outputs/phase1_erm_densenet121/seed0/best.pt --adapt-bn
"""
import argparse
import json
import os

import torch
import yaml

from src.adapt import adapt_batchnorm
from src.data import build_loaders, hospital_column, load_dataset
from src.engine import evaluate
from src.models import build_model
from src.utils import get_device

EVAL_SPLITS = ("id_val", "val", "test")


def load_checkpoint_config(ckpt, ckpt_path):
    if "cfg" in ckpt:
        return ckpt["cfg"]
    with open(os.path.join(os.path.dirname(ckpt_path), "config.yaml")) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--splits", nargs="+", default=list(EVAL_SPLITS), choices=EVAL_SPLITS)
    parser.add_argument("--adapt-bn", action="store_true",
                        help="re-estimate BatchNorm statistics on each split's unlabeled images before evaluating")
    parser.add_argument("--adapt-samples", type=int, default=20000)
    parser.add_argument("--data-root", default=None, help="override data.root_dir from the checkpoint config")
    parser.add_argument("--num-workers", type=int, default=None)
    args = parser.parse_args()

    device = get_device()
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = load_checkpoint_config(ckpt, args.checkpoint)
    cfg["data"]["download"] = False
    if args.data_root:
        cfg["data"]["root_dir"] = args.data_root
    if args.num_workers is not None:
        cfg["data"]["num_workers"] = args.num_workers
    amp = cfg["train"].get("amp", False) and device.type == "cuda"

    model = build_model(cfg["model"]["arch"], pretrained=False).to(device)
    model.load_state_dict(ckpt["model"])

    dataset = load_dataset(cfg)
    hosp_col = hospital_column(dataset)
    loaders = build_loaders(cfg, dataset, splits=args.splits)

    results = {"checkpoint": os.path.basename(args.checkpoint), "epoch": ckpt.get("epoch"), "adapt_bn": args.adapt_bn}
    for split in args.splits:
        eval_model = model
        if args.adapt_bn:
            eval_model = adapt_batchnorm(
                model, loaders[split].dataset, device,
                max_samples=args.adapt_samples, num_workers=cfg["data"]["num_workers"], seed=cfg["seed"],
            )
        results[split], _ = evaluate(eval_model, loaders[split], device, hosp_col, amp, desc=split)
        m = results[split]
        print(f"{split:7s} acc {m['acc']:.4f} | sens {m['sensitivity']:.4f} | spec {m['specificity']:.4f} "
              f"| auroc {m['auroc']:.4f} | ece {m['ece']:.4f}", flush=True)

    stem = os.path.splitext(os.path.basename(args.checkpoint))[0]
    out_path = os.path.join(os.path.dirname(args.checkpoint), f"eval_{stem}{'_adabn' if args.adapt_bn else ''}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
