# src/models/multitask_resnet.py
import torch
import torch.nn as nn
from torchvision import models

class MultiTaskResNet(nn.Module):
    def __init__(self, pretrained=True, num_gender_classes=2):
        super().__init__()
        base = models.resnet18(pretrained=pretrained)
        in_features = base.fc.in_features
        base.fc = nn.Identity()
        self.backbone = base

        self.age_head = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)  # regression
        )

        self.gender_head = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_gender_classes)  # logits for classes
        )

    def forward(self, x):
        feats = self.backbone(x)
        age_out = self.age_head(feats).squeeze(1)
        gender_out = self.gender_head(feats)
        return age_out, gender_out
