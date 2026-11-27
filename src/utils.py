import os
import random

import numpy as np
import torch
import yaml


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _deep_merge(base, update):
    merged = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _read_config(path):
    """Reads a YAML config, resolving an optional ``base:`` key (path relative to this file)."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    base = cfg.pop("base", None)
    if base is None:
        return cfg
    base_cfg = _read_config(os.path.join(os.path.dirname(path), base))
    for key in _flatten(cfg):
        node = base_cfg
        for part in key.split("."):
            if not isinstance(node, dict) or part not in node:
                raise KeyError(f"{path}: key '{key}' is not in base config {base}")
            node = node[part]
    return _deep_merge(base_cfg, cfg)


def _flatten(cfg, prefix=""):
    for key, value in cfg.items():
        if isinstance(value, dict):
            yield from _flatten(value, f"{prefix}{key}.")
        else:
            yield f"{prefix}{key}"


def load_config(path, overrides=()):
    """Loads a YAML config and applies dotted overrides like ``train.lr=0.01``."""
    cfg = _read_config(path)
    for item in overrides:
        key, value = item.split("=", 1)
        node = cfg
        *parents, leaf = key.split(".")
        for p in parents:
            node = node[p]
        if leaf not in node:
            raise KeyError(f"Unknown config key: {key}")
        node[leaf] = yaml.safe_load(value)
    return cfg
