import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T
from wilds import get_dataset

from .augment import StainColorJitter

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

SPLITS = ("train", "id_val", "val", "test")
HOSPITAL_FIELD = "hospital"


AUGMENTS = ("flip", "stain")


def build_transform(train_cfg, train):
    """train_cfg["augment"] is a list drawn from AUGMENTS (the legacy string "none" means [])."""
    augment = train_cfg["augment"]
    augment = [] if augment in (None, "none") else [augment] if isinstance(augment, str) else augment
    unknown = set(augment) - set(AUGMENTS)
    if unknown:
        raise ValueError(f"Unknown augment(s): {sorted(unknown)}. Choose from {AUGMENTS}")

    steps = []
    if train and "flip" in augment:
        steps += [T.RandomHorizontalFlip(), T.RandomVerticalFlip()]
    steps.append(T.ToTensor())
    if train and "stain" in augment:
        steps.append(
            StainColorJitter(
                sigma=train_cfg["stain_sigma"],
                p=train_cfg["stain_p"],
                intensity_scale=train_cfg.get("stain_intensity_scale", 1.0),
            )
        )
    steps.append(T.Normalize(IMAGENET_MEAN, IMAGENET_STD))
    return T.Compose(steps)


def load_dataset(cfg):
    return get_dataset(
        dataset="camelyon17",
        root_dir=cfg["data"]["root_dir"],
        download=cfg["data"]["download"],
    )


def build_loaders(cfg, dataset, splits=SPLITS):
    """Returns {split: DataLoader}. Each batch is (x, y, metadata)."""
    frac = cfg["data"]["frac"]
    num_workers = cfg["data"]["num_workers"]
    loaders = {}
    for split in splits:
        is_train = split == "train"
        subset = dataset.get_subset(
            split,
            frac=frac,
            transform=build_transform(cfg["train"], train=is_train),
        )
        loaders[split] = DataLoader(
            subset,
            batch_size=cfg["train"]["batch_size"] if is_train else cfg["eval"]["batch_size"],
            shuffle=is_train,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0,
            drop_last=is_train,
        )
    return loaders


def clean_subset(cfg, dataset, split):
    """A split with the evaluation transform (no augmentation), e.g. clean training images."""
    return dataset.get_subset(split, frac=cfg["data"]["frac"], transform=build_transform(cfg["train"], train=False))


def hospital_column(dataset):
    return dataset.metadata_fields.index(HOSPITAL_FIELD)
