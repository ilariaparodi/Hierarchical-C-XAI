import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights


class ResNetHierarchical(nn.Module):

    def __init__(self, num_classes, num_fine=110, num_coarse=11, pretrained=True, freeze_backbone=False):
        super().__init__()
        # load weights
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        self.backbone = models.resnet50(weights=weights)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
                
        # Remove the original fully connected layer to get raw features (dim: 2048)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        # fully-connected layer for fine-grained classification (dim: 110)
        self.fine_head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Linear(512, num_fine)
        )  

        # fully-connected layer for coarse-grained classification (dim: 11)
        self.coarse_head = nn.Sequential(
            nn.Linear(num_fine, 64),
            nn.ReLU(),
            nn.Linear(64, num_coarse)
        )

        # fully-connected layer for final classification (dim: 4)
        self.classifier_head = nn.Sequential(
            nn.Linear(num_coarse, 16),
            nn.ReLU(),
            nn.Linear(16, num_classes)
        ) 

    def forward(self, x):
        features = self.backbone(x)
        fine_logits = self.fine_head(features)
        coarse_logits = self.coarse_head(fine_logits)
        class_logits = self.classifier_head(coarse_logits)
        return {
            "fine": fine_logits,
            "coarse": coarse_logits,
            "class": class_logits
        }

def build_resnet50_hierarchical(num_classes, num_fine=110, num_coarse=11, pretrained=True, freeze_backbone=False):
    return ResNetHierarchical(
        num_classes=num_classes,
        num_fine=num_fine,
        num_coarse=num_coarse,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
    )
