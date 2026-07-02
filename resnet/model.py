"""Stage 4 classification model: ImageNet-pretrained ResNet50 transfer learning."""

from __future__ import annotations

from torch import nn
from torchvision.models import ResNet50_Weights, resnet50


def build_resnet50(pretrained: bool = True, num_classes: int = 2, freeze_backbone: bool = False) -> nn.Module:
    """Build ResNet50 with its final FC layer replaced for benign/malignant classification.

    Args:
        pretrained: Load ImageNet weights (recommended — BUSI is too small to
            train ResNet50 from scratch).
        num_classes: 2 (benign, malignant).
        freeze_backbone: If True, freeze every layer except the new FC head
            (linear-probe warmup phase before fine-tuning; see
            ``resnet.train.train_resnet``'s ``freeze_backbone_epochs``).
    """
    weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model = resnet50(weights=weights)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    """Toggle every parameter except ``fc`` between frozen/trainable (fine-tuning switch)."""
    for name, param in model.named_parameters():
        if not name.startswith("fc."):
            param.requires_grad = trainable
