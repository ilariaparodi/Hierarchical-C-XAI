import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights


class ResNet50Baseline(nn.Module):

    def __init__(self, num_classes, pretrained=True, freeze_backbone=False):
        super().__init__()

        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        self.backbone = models.resnet50(weights=weights)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)


def build_resnet50_baseline(num_classes, pretrained=True, freeze_backbone=False):
    return ResNet50Baseline(
        num_classes=num_classes,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
    )