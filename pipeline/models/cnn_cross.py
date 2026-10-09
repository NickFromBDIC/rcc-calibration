from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import torch
from torch import nn
from monai.networks.nets import resnet18


class CrossAttFusionResNet18(nn.Module):

    def __init__(self, radiomics_dim: int, d_model: int=256, num_heads: int=4):
        super().__init__()
        self.backbone = resnet18(spatial_dims=3, n_input_channels=1, num_classes=2)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.d_model = d_model
        self.radiomics_dim = radiomics_dim
        self.img_proj = nn.Linear(in_features, d_model)
        self.rad_value_proj = nn.Linear(1, d_model)
        self.rad_index_emb = nn.Embedding(radiomics_dim, d_model)
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)
        self.attn_norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(in_features + d_model, 2)

    def forward(self, x_img, x_rad):
        feat_img = self.backbone(x_img)
        q = self.img_proj(feat_img)
        q = q.unsqueeze(1)
        B, D = x_rad.shape
        assert D == self.radiomics_dim, f'Expected radiomics_dim={self.radiomics_dim}, got {D}'
        values = x_rad.view(B, D, 1)
        value_emb = self.rad_value_proj(values)
        idx = torch.arange(D, device=x_rad.device)
        idx = idx.unsqueeze(0).expand(B, D)
        index_emb = self.rad_index_emb(idx)
        rad_tokens = value_emb + index_emb
        attn_out, _ = self.cross_attn(q, rad_tokens, rad_tokens)
        attn_out = attn_out.squeeze(1)
        attn_out = self.attn_norm(attn_out)
        fused = torch.cat([feat_img, attn_out], dim=1)
        logits = self.classifier(fused)
        return logits
