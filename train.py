"""Train a tumor classifier on Camelyon17-WILDS.

Example:
    python train.py --config configs/phase1_baseline.yaml --seed 0
    python train.py --config configs/phase1_baseline.yaml --override data.frac=0.02 train.epochs=1
"""
import argparse
import json
import os
import time

import torch
import yaml

from src.algorithms import build_algorithm
from src.data import SPLITS, build_loaders, hospital_column, load_dataset
from src.engine import evaluate, save_predictions, train_one_epoch
from src.models import build_model
from src.utils import get_device, load_config, set_seed


def build_optimizer(cfg, model):
    t = cfg["train"]
    if t["optimizer"] == "sgd":
        return torch.optim.SGD(model.parameters(), lr=t["lr"], momentum=t["momentum"], weight_decay=t["weight_decay"])
    if t["optimizer"] == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    raise ValueError(f"Unknown optimizer: {t['optimizer']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--override", nargs="*", default=[], help="dotted overrides, e.g. train.lr=0.01")
    args = parser.parse_args()

    cfg = load_config(args.config, args.override)
    if args.seed is not None:
        cfg["seed"] = args.seed
    set_seed(cfg["seed"])
    device = get_device()
    torch.backends.cudnn.benchmark = True

    out_dir = os.path.join(cfg["output_dir"], cfg["experiment_name"], f"seed{cfg['seed']}")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print(f"Device: {device} | Output: {out_dir}")

    dataset = load_dataset(cfg)
    hosp_col = hospital_column(dataset)
    loaders = build_loaders(cfg, dataset)
    print({split: len(loader.dataset) for split, loader in loaders.items()})

    model = build_model(cfg["model"]["arch"], cfg["model"]["pretrained"]).to(device)
    algorithm = build_algorithm(cfg)
    optimizer = build_optimizer(cfg, model)
    amp = cfg["train"]["amp"] and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=amp)

    select_on = cfg["eval"]["select_on"]
    best_score, history, start_epoch = -1.0, [], 1
    ckpt_path = os.path.join(out_dir, "best.pt")
    last_path = os.path.join(out_dir, "last.pt")

    if os.path.exists(last_path):
        state = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        best_score, history = state["best_score"], state["history"]
        start_epoch = state["epoch"] + 1
        print(f"Resuming from epoch {start_epoch} ({last_path})")

    for epoch in range(start_epoch, cfg["train"]["epochs"] + 1):
        start = time.time()
        train_loss = train_one_epoch(
            model, algorithm, loaders["train"], optimizer, device, scaler, amp, desc=f"epoch {epoch}"
        )
        record = {"epoch": epoch, "train_loss": train_loss}
        for split in ("id_val", "val"):
            metrics, _ = evaluate(model, loaders[split], device, hosp_col, amp, desc=split)
            record[split] = metrics
        record["minutes"] = (time.time() - start) / 60
        history.append(record)

        score = record[select_on]["acc"]
        is_best = score > best_score
        if is_best:
            best_score = score
            torch.save({"model": model.state_dict(), "epoch": epoch, "cfg": cfg}, ckpt_path)
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch,
                "best_score": best_score,
                "history": history,
            },
            last_path,
        )
        print(
            f"epoch {epoch} | loss {train_loss:.4f} | id_val acc {record['id_val']['acc']:.4f} "
            f"| val (OOD) acc {record['val']['acc']:.4f} | {record['minutes']:.1f} min"
            + (" | *best*" if is_best else ""),
            flush=True,
        )
        with open(os.path.join(out_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    final = {"best_epoch": ckpt["epoch"], "selected_on": select_on}
    for split in SPLITS:
        metrics, preds = evaluate(model, loaders[split], device, hosp_col, amp, desc=f"final {split}")
        final[split] = metrics
        save_predictions(os.path.join(out_dir, f"preds_{split}.npz"), *preds)
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(final, f, indent=2)

    print(f"\nBest epoch {final['best_epoch']} (selected on {select_on})")
    for split in SPLITS:
        m = final[split]
        print(f"  {split:7s} acc {m['acc']:.4f} | f1 {m['f1_macro']:.4f} | auroc {m['auroc']:.4f} | ece {m['ece']:.4f}")


if __name__ == "__main__":
    main()
