import torch.nn.functional as F


class ERM:
    """Empirical risk minimization: plain cross-entropy on pooled training hospitals."""

    def __init__(self, cfg):
        self.cfg = cfg

    def loss(self, model, x, y, metadata):
        return F.cross_entropy(model(x), y)


ALGORITHMS = {
    "erm": ERM,
}


def build_algorithm(cfg):
    name = cfg["train"]["algorithm"]
    if name not in ALGORITHMS:
        raise ValueError(f"Unknown algorithm: {name}. Choose from {list(ALGORITHMS)}")
    return ALGORITHMS[name](cfg)
