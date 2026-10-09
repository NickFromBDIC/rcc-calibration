from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import torch
from torch import nn
from monai.networks.nets import resnet18
from .medvae_backbone import MVAE


class MedVAE2DMeanPoolLegacyGatedFusion(nn.Module):

    def __init__(self, radiomics_dim: int, medvae_model_name: str, medvae_modality: str, top_k: int, proj_dim: int, gate_hidden_dim: int) -> None:
        super().__init__()
        self.radiomics_dim = radiomics_dim
        self.proj_dim = proj_dim
        self.top_k = top_k
        self.slice_encoder = MVAE(model_name=medvae_model_name, modality=medvae_modality)
        freeze_all_params(self.slice_encoder)
        self.slice_proj = nn.LazyLinear(proj_dim)
        self.rad_proj = nn.Linear(radiomics_dim, proj_dim)
        self.gate_mlp = nn.Sequential(nn.Linear(proj_dim, gate_hidden_dim), nn.ReLU(inplace=True), nn.Linear(gate_hidden_dim, proj_dim), nn.Sigmoid())
        self.classifier = nn.Linear(proj_dim, 2)

    @staticmethod
    def _topk_indices_from_mask(mask_3d: torch.Tensor, k: int) -> torch.Tensor:
        areas = mask_3d[:, 0].sum(dim=(-1, -2))
        indices: List[torch.Tensor] = []
        for batch_index in range(mask_3d.shape[0]):
            area = areas[batch_index]
            positive = torch.nonzero(area > 0, as_tuple=False).flatten()
            if len(positive) == 0:
                chosen = torch.tensor([mask_3d.shape[2] // 2], device=mask_3d.device, dtype=torch.long).repeat(k)
            else:
                order = torch.argsort(area[positive], descending=True)
                chosen = positive[order[:min(k, len(positive))]]
                chosen, _ = torch.sort(chosen)
                if len(chosen) < k:
                    chosen = torch.cat([chosen, chosen[-1].repeat(k - len(chosen))], dim=0)
            indices.append(chosen)
        return torch.stack(indices, dim=0)

    def encode_image(self, image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        indices = self._topk_indices_from_mask(mask, self.top_k)
        selected = []
        for batch_index in range(image.shape[0]):
            slices = image[batch_index, :, indices[batch_index], :, :]
            selected.append(slices.permute(1, 0, 2, 3).contiguous())
        slices = torch.stack(selected, dim=0)
        batch_size, slice_count, channels, height, width = slices.shape
        slices = slices.view(batch_size * slice_count, channels, height, width)
        if slices.shape[1] == 1:
            slices = slices.repeat(1, 3, 1, 1)
        elif slices.shape[1] != 3:
            raise ValueError(f'Expected 1 or 3 slice channels, got {slices.shape[1]}')
        latent = extract_latent_tensor(self.slice_encoder(slices))
        slice_features = self.slice_proj(torch.flatten(latent, start_dim=1))
        slice_features = slice_features.view(batch_size, slice_count, self.proj_dim)
        return slice_features.mean(dim=1)

    def forward_with_gates(self, image: torch.Tensor, mask: torch.Tensor, radiomics: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        image_features = self.encode_image(image, mask)
        radiomics_features = self.rad_proj(radiomics)
        gate = self.gate_mlp(radiomics_features)
        fused = gate * image_features + (1.0 - gate) * radiomics_features
        return (self.classifier(fused), gate)

    def forward(self, image: torch.Tensor, mask: torch.Tensor, radiomics: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward_with_gates(image, mask, radiomics)
        return logits

from .medvae2d_image import extract_latent_tensor, freeze_all_params
