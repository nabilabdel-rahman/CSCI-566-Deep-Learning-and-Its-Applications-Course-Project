import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T
from wilds import get_dataset

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

SPLITS = ("train", "id_val", "val", "test")
HOSPITAL_FIELD = "hospital"


def build_transform(augment, train):
    steps = []
    if train and augment == "flip":
        steps += [T.RandomHorizontalFlip(), T.RandomVerticalFlip()]
    elif train and augment != "none":
        raise ValueError(f"Unknown augment: {augment}")
    steps += [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
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
            transform=build_transform(cfg["train"]["augment"], train=is_train),
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


def hospital_column(dataset):
    return dataset.metadata_fields.index(HOSPITAL_FIELD)
