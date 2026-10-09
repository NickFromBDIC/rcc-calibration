from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import torch
from torch import nn
from monai.networks.nets import resnet18


class FusionResNet18(nn.Module):

    def __init__(self, radiomics_dim: int):
        super().__init__()
        self.backbone = resnet18(spatial_dims=3, n_input_channels=1, num_classes=2)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.radiomics_norm = nn.Identity()
        self.classifier = nn.Linear(in_features + radiomics_dim, 2)

    def forward(self, x_img, x_rad):
        feat_img = self.backbone(x_img)
        x_rad = self.radiomics_norm(x_rad)
        x = torch.cat([feat_img, x_rad], dim=1)
        logits = self.classifier(x)
        return logits
