import torch.nn as nn
from torchvision import models

_BUILDERS = {
    "densenet121": (models.densenet121, models.DenseNet121_Weights.IMAGENET1K_V1),
    "resnet18": (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1),
    "resnet50": (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V2),
}


def build_model(arch, pretrained, num_classes=2):
    if arch not in _BUILDERS:
        raise ValueError(f"Unknown arch: {arch}. Choose from {list(_BUILDERS)}")
    builder, weights = _BUILDERS[arch]
    model = builder(weights=weights if pretrained else None)

    if arch.startswith("densenet"):
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    else:
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
