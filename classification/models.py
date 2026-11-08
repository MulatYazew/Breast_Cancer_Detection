"""Small ImageNet classifiers. EfficientNet-B0 is the primary model.

Every network here is tens of megabytes, not gigabytes. ConvNeXt, ResNet50,
and any language model are intentionally absent from this factory.
"""

from __future__ import annotations

from torch import nn
from torchvision.models import (
    DenseNet121_Weights,
    EfficientNet_B0_Weights,
    MobileNet_V3_Small_Weights,
    ResNet18_Weights,
    densenet121,
    efficientnet_b0,
    mobilenet_v3_small,
    resnet18,
)

PRIMARY_BACKBONE = "efficientnet_b0"
COMPARISON_BACKBONES = ("densenet121", "mobilenet_v3_small", "resnet18")
ALLOWED_BACKBONES = (PRIMARY_BACKBONE,) + COMPARISON_BACKBONES

_HEADS = {
    "efficientnet_b0": ("classifier",),
    "densenet121": ("classifier",),
    "mobilenet_v3_small": ("classifier",),
    "resnet18": ("fc",),
}


def build_image_classifier(backbone: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    """Replace the ImageNet head. ``pretrained=False`` skips the weight download."""
    if backbone not in ALLOWED_BACKBONES:
        raise ValueError(f"{backbone} is not in the hardware-approved set {ALLOWED_BACKBONES}")
    if num_classes < 2:
        raise ValueError("num_classes must match the labels that exist, at least 2")

    if backbone == "efficientnet_b0":
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = efficientnet_b0(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif backbone == "densenet121":
        weights = DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        model = densenet121(weights=weights)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif backbone == "mobilenet_v3_small":
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        model = mobilenet_v3_small(weights=weights)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    else:
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def head_prefixes(backbone: str) -> tuple[str, ...]:
    return _HEADS[backbone]


def freeze_backbone(model: nn.Module, backbone: str, frozen: bool) -> None:
    prefixes = head_prefixes(backbone)
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name.startswith(prefixes) or not frozen


def parameter_counts(model: nn.Module) -> dict:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {"total": int(total), "trainable": int(trainable)}
