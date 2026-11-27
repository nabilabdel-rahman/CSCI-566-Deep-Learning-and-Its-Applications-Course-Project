import copy

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm


@torch.no_grad()
def adapt_batchnorm(model, dataset, device, batch_size=256, max_samples=20000, num_workers=4, seed=0):
    """Test-time BatchNorm adaptation (AdaBN; Li et al., 2016).

    Returns a copy of ``model`` whose BatchNorm running statistics are re-estimated on unlabeled
    images from ``dataset`` (the target hospital). No weights are trained and labels are not used.
    """
    adapted = copy.deepcopy(model)
    bn_layers = [m for m in adapted.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    if not bn_layers:
        raise ValueError("Model has no BatchNorm layers to adapt")

    if len(dataset) > max_samples:
        g = torch.Generator().manual_seed(seed)
        dataset = Subset(dataset, torch.randperm(len(dataset), generator=g)[:max_samples].tolist())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    adapted.eval()
    for bn in bn_layers:
        bn.reset_running_stats()
        bn.momentum = None  # cumulative average over all adaptation batches
        bn.train()
    for x, _, _ in tqdm(loader, desc="adapt bn", leave=False):
        adapted(x.to(device))
    adapted.eval()
    return adapted
