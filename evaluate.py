"""Evaluate a saved checkpoint (best.pt or last.pt) on one or more Camelyon17-WILDS splits.

Results are written next to the checkpoint as eval_<checkpoint>[_adabn|_srcbn].json, which
scripts/summarize.py picks up.

  --adapt-bn         re-estimate BatchNorm statistics on each evaluated split's own unlabeled images
                     (test-time adaptation: uses images from the target hospital)
  --adapt-bn-source  re-estimate BatchNorm statistics once on clean (unaugmented) training images
                     (domain generalization: never sees the target hospital)

Examples:
    python evaluate.py --checkpoint outputs/phase1_erm_densenet121/seed0/last.pt
    python evaluate.py --checkpoint outputs/phase1_erm_densenet121/seed0/best.pt --adapt-bn
    python evaluate.py --checkpoint outputs/phase1_erm_densenet121/seed0/best.pt --adapt-bn-source
"""
import argparse
import json
import os

import torch
import yaml

from src.adapt import adapt_batchnorm
from src.data import build_loaders, clean_subset, hospital_column, load_dataset
from src.engine import evaluate
from src.models import build_model
from src.utils import get_device, set_seed

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
    adapt = parser.add_mutually_exclusive_group()
    adapt.add_argument("--adapt-bn", action="store_true",
                       help="re-estimate BatchNorm statistics on each split's unlabeled images before evaluating")
    adapt.add_argument("--adapt-bn-source", action="store_true",
                       help="re-estimate BatchNorm statistics on clean training images before evaluating")
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
    set_seed(cfg["seed"])

    model = build_model(cfg["model"]["arch"], pretrained=False).to(device)
    model.load_state_dict(ckpt["model"])

    dataset = load_dataset(cfg)
    hosp_col = hospital_column(dataset)
    loaders = build_loaders(cfg, dataset, splits=args.splits)

    def adapt(adapt_dataset):
        return adapt_batchnorm(
            model, adapt_dataset, device,
            max_samples=args.adapt_samples, num_workers=cfg["data"]["num_workers"], seed=cfg["seed"],
        )

    if args.adapt_bn_source:
        model = adapt(clean_subset(cfg, dataset, "train"))

    results = {
        "checkpoint": os.path.basename(args.checkpoint),
        "epoch": ckpt.get("epoch"),
        "adapt_bn": args.adapt_bn,
        "adapt_bn_source": args.adapt_bn_source,
    }
    for split in args.splits:
        eval_model = adapt(loaders[split].dataset) if args.adapt_bn else model
        results[split], _ = evaluate(eval_model, loaders[split], device, hosp_col, amp, desc=split)
        m = results[split]
        print(f"{split:7s} acc {m['acc']:.4f} | sens {m['sensitivity']:.4f} | spec {m['specificity']:.4f} "
              f"| auroc {m['auroc']:.4f} | ece {m['ece']:.4f}", flush=True)

    stem = os.path.splitext(os.path.basename(args.checkpoint))[0]
    suffix = "_adabn" if args.adapt_bn else "_srcbn" if args.adapt_bn_source else ""
    out_path = os.path.join(os.path.dirname(args.checkpoint), f"eval_{stem}{suffix}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
