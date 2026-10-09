from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import torch
from torch import nn
from monai.networks.nets import resnet18


class GatedFusionResNet18(nn.Module):

    def __init__(self, radiomics_dim: int, hidden_dim: int=256):
        super().__init__()
        self.backbone = resnet18(spatial_dims=3, n_input_channels=1, num_classes=2)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.radiomics_dim = radiomics_dim
        self.img_feat_dim = in_features
        self.rad_proj = nn.Linear(radiomics_dim, in_features)
        self.gate_mlp = nn.Sequential(nn.Linear(in_features, hidden_dim), nn.ReLU(inplace=True), nn.Linear(hidden_dim, in_features), nn.Sigmoid())
        self.classifier = nn.Linear(in_features, 2)

    def forward(self, x_img, x_rad):
        feat_img = self.backbone(x_img)
        rad_proj = self.rad_proj(x_rad)
        gate = self.gate_mlp(rad_proj)
        fused = gate * feat_img + (1.0 - gate) * rad_proj
        logits = self.classifier(fused)
        return logits

    def forward_with_gates(self, x_img, x_rad):
        feat_img = self.backbone(x_img)
        rad_proj = self.rad_proj(x_rad)
        gate = self.gate_mlp(rad_proj)
        fused = gate * feat_img + (1.0 - gate) * rad_proj
        logits = self.classifier(fused)
        return (logits, gate, feat_img, rad_proj)
