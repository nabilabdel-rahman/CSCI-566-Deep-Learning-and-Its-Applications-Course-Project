"""Evaluate a saved checkpoint on one or more Camelyon17-WILDS splits.

Example:
    python evaluate.py --checkpoint outputs/phase1_erm_densenet121/seed0/best.pt --splits val test
"""
import argparse
import json

import torch

from src.data import SPLITS, build_loaders, hospital_column, load_dataset
from src.engine import evaluate
from src.models import build_model
from src.utils import get_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--splits", nargs="+", default=["test"], choices=SPLITS)
    args = parser.parse_args()

    device = get_device()
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = ckpt["cfg"]

    model = build_model(cfg["model"]["arch"], pretrained=False).to(device)
    model.load_state_dict(ckpt["model"])

    dataset = load_dataset(cfg)
    loaders = build_loaders(cfg, dataset, splits=args.splits)
    results = {}
    for split in args.splits:
        results[split], _ = evaluate(model, loaders[split], device, hospital_column(dataset), desc=split)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
