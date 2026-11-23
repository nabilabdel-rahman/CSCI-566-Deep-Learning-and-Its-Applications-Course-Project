import numpy as np
import torch
from tqdm import tqdm

from .metrics import compute_metrics


def train_one_epoch(model, algorithm, loader, optimizer, device, desc="train"):
    model.train()
    total, n = 0.0, 0
    bar = tqdm(loader, desc=desc, leave=False)
    for x, y, metadata in bar:
        x, y = x.to(device), y.to(device)
        loss = algorithm.loss(model, x, y, metadata.to(device))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(y)
        n += len(y)
        bar.set_postfix(loss=f"{total / n:.4f}")
    return total / n


@torch.no_grad()
def predict(model, loader, device, hospital_col, desc="eval"):
    model.eval()
    probs, ys, hospitals = [], [], []
    for x, y, metadata in tqdm(loader, desc=desc, leave=False):
        probs.append(torch.softmax(model(x.to(device)), dim=1).cpu())
        ys.append(y)
        hospitals.append(metadata[:, hospital_col])
    return (
        torch.cat(probs).numpy(),
        torch.cat(ys).numpy(),
        torch.cat(hospitals).numpy(),
    )


def evaluate(model, loader, device, hospital_col, desc="eval"):
    probs, y, hospitals = predict(model, loader, device, hospital_col, desc)
    return compute_metrics(probs, y, hospitals), (probs, y, hospitals)


def save_predictions(path, probs, y, hospitals):
    np.savez_compressed(path, probs=probs, y=y, hospital=hospitals)
