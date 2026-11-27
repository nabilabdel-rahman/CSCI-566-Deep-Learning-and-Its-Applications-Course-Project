import torch

# Normalized optical-density vectors for hematoxylin, eosin and DAB (Ruifrok & Johnston, 2001).
_RGB_FROM_HED = torch.tensor(
    [
        [0.65, 0.70, 0.29],
        [0.07, 0.99, 0.11],
        [0.27, 0.57, 0.78],
    ]
)
_HED_FROM_RGB = torch.linalg.inv(_RGB_FROM_HED)
_EPS = 1e-6


class StainColorJitter:
    """H&E stain color jitter (Tellez et al., 2018), as used by Gao et al. (2023) on Camelyon17.

    Operates on a (3, H, W) RGB tensor in [0, 1]. Each stain channel is scaled by
    alpha ~ U(1 - sigma, 1 + sigma) and shifted by beta ~ U(-sigma, sigma) in optical-density space.
    """

    def __init__(self, sigma=0.1, p=0.5):
        self.sigma = sigma
        self.p = p

    def __call__(self, x):
        if torch.rand(()) >= self.p:
            return x
        od = -torch.log(x.clamp_min(0) + _EPS)
        stains = torch.einsum("chw,cd->dhw", od, _HED_FROM_RGB)
        alpha = 1 + (torch.rand(3) * 2 - 1) * self.sigma
        beta = (torch.rand(3) * 2 - 1) * self.sigma
        stains = stains * alpha[:, None, None] + beta[:, None, None]
        rgb = torch.exp(-torch.einsum("dhw,dc->chw", stains, _RGB_FROM_HED)) - _EPS
        return rgb.clamp(0, 1)

    def __repr__(self):
        return f"{type(self).__name__}(sigma={self.sigma}, p={self.p})"
