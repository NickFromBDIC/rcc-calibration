from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import torch
from torch import nn
from monai.networks.nets import resnet18
from .medvae_backbone import MVAE


def extract_latent_tensor(output):
    if torch.is_tensor(output):
        return output
    if isinstance(output, dict):
        for key in ['latent', 'z', 'embedding', 'embeddings', 'features', 'x']:
            val = output.get(key, None)
            if torch.is_tensor(val):
                return val
        for val in output.values():
            if hasattr(val, 'mean') and torch.is_tensor(val.mean):
                return val.mean
    if isinstance(output, (list, tuple)):
        for item in output:
            if torch.is_tensor(item):
                return item
            if hasattr(item, 'mean') and torch.is_tensor(item.mean):
                return item.mean
    if hasattr(output, 'mean') and torch.is_tensor(output.mean):
        return output.mean
    raise RuntimeError('Unable to extract a latent tensor from MedVAE output. Please inspect the exact return format of your installed medvae version.')

def freeze_all_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = False

def unfreeze_last_fraction(module: nn.Module, frac: float=0.2):
    params = [p for p in module.parameters()]
    if len(params) == 0:
        return
    frac = float(max(0.0, min(1.0, frac)))
    n_unfreeze = max(1, int(round(len(params) * frac)))
    for p in params[-n_unfreeze:]:
        p.requires_grad = True

class MedVAE3DClassifier(nn.Module):

    def __init__(self, medvae_model_name: str='medvae_4_1_3d', medvae_modality: str='ct', proj_dim: int=512, dropout: float=0.2, finetune_mode: str='frozen', partial_ft_frac: float=0.2, force_three_channel: bool=False):
        super().__init__()
        self.force_three_channel = force_three_channel
        self.volume_encoder = MVAE(model_name=medvae_model_name, modality=medvae_modality)
        if finetune_mode == 'frozen':
            freeze_all_params(self.volume_encoder)
        elif finetune_mode == 'partial':
            freeze_all_params(self.volume_encoder)
            unfreeze_last_fraction(self.volume_encoder, frac=partial_ft_frac)
        elif finetune_mode == 'full':
            for p in self.volume_encoder.parameters():
                p.requires_grad = True
        else:
            raise ValueError(f'Unsupported finetune_mode: {finetune_mode}')
        self.feature_proj = nn.LazyLinear(proj_dim)
        self.classifier = nn.Sequential(nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(proj_dim, 2))

    def encode_image(self, x_img: torch.Tensor):
        vol = x_img
        if vol.ndim != 5:
            raise ValueError(f'Expected volume to be 5D [B,C,Z,Y,X], got {tuple(vol.shape)}')
        if self.force_three_channel and vol.shape[1] == 1:
            vol = vol.repeat(1, 3, 1, 1, 1)
        B = vol.shape[0]
        medvae_out = self.volume_encoder(vol)
        latent = extract_latent_tensor(medvae_out)
        if not torch.is_tensor(latent):
            raise RuntimeError(f'Expected latent tensor, got {type(latent)}')
        latent = latent.contiguous().reshape(B, -1)
        feat_img = self.feature_proj(latent)
        return feat_img

    def forward(self, x_img):
        feat_img = self.encode_image(x_img)
        logits = self.classifier(feat_img)
        return logits
