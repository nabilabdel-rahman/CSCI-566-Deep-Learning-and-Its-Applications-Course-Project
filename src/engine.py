import numpy as np
import torch
from tqdm import tqdm

from .metrics import compute_metrics


def autocast(device, enabled):
    return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=enabled)


def train_one_epoch(model, algorithm, loader, optimizer, device, scaler, amp=False, desc="train"):
    model.train()
    total, n = 0.0, 0
    bar = tqdm(loader, desc=desc, leave=False)
    for x, y, metadata in bar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with autocast(device, amp):
            loss = algorithm.loss(model, x, y, metadata.to(device))
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total += loss.item() * len(y)
        n += len(y)
        bar.set_postfix(loss=f"{total / n:.4f}")
    return total / n


@torch.no_grad()
def predict(model, loader, device, hospital_col, amp=False, desc="eval"):
    model.eval()
    probs, ys, hospitals = [], [], []
    for x, y, metadata in tqdm(loader, desc=desc, leave=False):
        with autocast(device, amp):
            logits = model(x.to(device, non_blocking=True))
        probs.append(torch.softmax(logits.float(), dim=1).cpu())
        ys.append(y)
        hospitals.append(metadata[:, hospital_col])
    return (
        torch.cat(probs).numpy(),
        torch.cat(ys).numpy(),
        torch.cat(hospitals).numpy(),
    )


def evaluate(model, loader, device, hospital_col, amp=False, desc="eval"):
    probs, y, hospitals = predict(model, loader, device, hospital_col, amp, desc)
    return compute_metrics(probs, y, hospitals), (probs, y, hospitals)


def save_predictions(path, probs, y, hospitals):
    np.savez_compressed(path, probs=probs, y=y, hospital=hospitals)
