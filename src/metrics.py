import numpy as np
from sklearn.metrics import f1_score, roc_auc_score


def expected_calibration_error(probs, y, n_bins=15):
    conf = probs.max(axis=1)
    correct = probs.argmax(axis=1) == y
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


def compute_metrics(probs, y, hospitals):
    """probs: (N, 2) softmax outputs; y: (N,) labels; hospitals: (N,) hospital ids."""
    preds = probs.argmax(axis=1)
    metrics = {
        "n": int(len(y)),
        "acc": float((preds == y).mean()),
        "sensitivity": float((preds[y == 1] == 1).mean()) if (y == 1).any() else float("nan"),
        "specificity": float((preds[y == 0] == 0).mean()) if (y == 0).any() else float("nan"),
        "pred_tumor_frac": float(preds.mean()),
        "f1_macro": float(f1_score(y, preds, average="macro")),
        "auroc": float(roc_auc_score(y, probs[:, 1])) if len(np.unique(y)) > 1 else float("nan"),
        "ece": expected_calibration_error(probs, y),
    }
    for h in np.unique(hospitals):
        mask = hospitals == h
        metrics[f"acc_hospital_{int(h)}"] = float((preds[mask] == y[mask]).mean())
    return metrics
