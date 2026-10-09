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

class SliceAttentionPool(nn.Module):

    def __init__(self, dim: int=512, attn_dim: int=128):
        super().__init__()
        self.fc1 = nn.Linear(dim, attn_dim)
        self.fc2 = nn.Linear(attn_dim, 1)

    def forward(self, x):
        a = torch.tanh(self.fc1(x))
        a = self.fc2(a).squeeze(-1)
        a = torch.softmax(a, dim=1)
        pooled = torch.sum(x * a.unsqueeze(-1), dim=1)
        return (pooled, a)

class SliceMeanPool(nn.Module):

    def forward(self, x):
        pooled = x.mean(dim=1)
        attn = None
        return (pooled, attn)

class SliceMaxPool(nn.Module):

    def forward(self, x):
        pooled, _ = x.max(dim=1)
        attn = None
        return (pooled, attn)

class MedVAE2DClassifier(nn.Module):

    def __init__(self, medvae_model_name: str='medvae_4_3_2d', medvae_modality: str='ct', top_k: int=5, proj_dim: int=512, attn_dim: int=128, finetune_mode: str='partial', partial_ft_frac: float=0.2, pooling: str='attention', dropout: float=0.2):
        super().__init__()
        self.proj_dim = proj_dim
        self.top_k = top_k
        self.pooling = pooling
        self.slice_encoder = MVAE(model_name=medvae_model_name, modality=medvae_modality)
        if finetune_mode == 'frozen':
            freeze_all_params(self.slice_encoder)
        elif finetune_mode == 'partial':
            freeze_all_params(self.slice_encoder)
            unfreeze_last_fraction(self.slice_encoder, frac=partial_ft_frac)
        elif finetune_mode == 'full':
            for p in self.slice_encoder.parameters():
                p.requires_grad = True
        else:
            raise ValueError(f'Unsupported finetune_mode: {finetune_mode}')
        self.slice_proj = nn.LazyLinear(proj_dim)
        if pooling == 'attention':
            self.slice_pool = SliceAttentionPool(dim=proj_dim, attn_dim=attn_dim)
        elif pooling == 'mean':
            self.slice_pool = SliceMeanPool()
        elif pooling == 'max':
            self.slice_pool = SliceMaxPool()
        else:
            raise ValueError(f'Unsupported pooling: {pooling}')
        self.classifier = nn.Sequential(nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(proj_dim, 2))

    @staticmethod
    def _topk_indices_from_mask(mask_3d: torch.Tensor, k: int) -> torch.Tensor:
        B, _, Z, _, _ = mask_3d.shape
        areas = mask_3d[:, 0].sum(dim=(-1, -2))
        all_idx = []
        device = mask_3d.device
        for b in range(B):
            a = areas[b]
            pos = torch.nonzero(a > 0, as_tuple=False).flatten()
            if len(pos) == 0:
                center = torch.tensor([Z // 2], device=device, dtype=torch.long)
                idx = center.repeat(k)
            else:
                order = torch.argsort(a[pos], descending=True)
                chosen = pos[order[:min(k, len(pos))]]
                chosen, _ = torch.sort(chosen)
                if len(chosen) < k:
                    pad = chosen[-1].repeat(k - len(chosen))
                    idx = torch.cat([chosen, pad], dim=0)
                else:
                    idx = chosen
            all_idx.append(idx)
        return torch.stack(all_idx, dim=0)

    def _select_topk_slices(self, x_img: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
        idx = self._topk_indices_from_mask(x_mask, self.top_k)
        selected = []
        for b in range(x_img.shape[0]):
            z_idx = idx[b]
            slices_b = x_img[b, :, z_idx, :, :].permute(1, 0, 2, 3).contiguous()
            selected.append(slices_b)
        return torch.stack(selected, dim=0)

    def encode_image(self, x_img: torch.Tensor, x_mask: torch.Tensor, return_slice_feat: bool=False):
        slices = self._select_topk_slices(x_img, x_mask)
        B, K, C, H, W = slices.shape
        slices = slices.view(B * K, C, H, W)
        if slices.ndim != 4:
            raise ValueError(f'Expected slices to be 4D [N,C,H,W], got {slices.shape}')
        if slices.shape[1] == 1:
            slices = slices.repeat(1, 3, 1, 1)
        elif slices.shape[1] != 3:
            raise ValueError(f'MedVAE expects 3 channels, but got {slices.shape[1]}')
        medvae_out = self.slice_encoder(slices)
        latent = extract_latent_tensor(medvae_out)
        latent = torch.flatten(latent, start_dim=1)
        slice_feat = self.slice_proj(latent)
        slice_feat = slice_feat.view(B, K, self.proj_dim)
        feat_img, attn = self.slice_pool(slice_feat)
        if return_slice_feat:
            return (feat_img, attn, slice_feat)
        return (feat_img, attn)

    def forward(self, x_img, x_mask):
        feat_img, _ = self.encode_image(x_img, x_mask)
        logits = self.classifier(feat_img)
        return logits
