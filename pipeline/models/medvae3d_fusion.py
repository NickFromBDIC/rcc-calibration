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

def unfreeze_by_stages(module: nn.Module, stages):
    stage_to_prefixes = {'conv_in': ['model.encoder.conv_in.'], 'down0': ['model.encoder.down.0.'], 'down1': ['model.encoder.down.1.'], 'down2': ['model.encoder.down.2.'], 'mid': ['model.encoder.mid.'], 'norm_out': ['model.encoder.norm_out.'], 'conv_out': ['model.encoder.conv_out.'], 'quant_conv': ['model.quant_conv.']}
    unknown = [s for s in stages if s not in stage_to_prefixes]
    if len(unknown) > 0:
        raise ValueError(f'Unknown stages: {unknown}. Valid stages: {list(stage_to_prefixes.keys())}')
    for name, p in module.named_parameters():
        for s in stages:
            prefixes = stage_to_prefixes[s]
            if any((name.startswith(pref) for pref in prefixes)):
                p.requires_grad = True
                break

class MedVAE3DFusion(nn.Module):

    def __init__(self, radiomics_dim: int, medvae_model_name: str='medvae_4_1_3d', medvae_modality: str='ct', proj_dim: int=512, gate_hidden_dim: int=256, finetune_mode: str='frozen', partial_ft_frac: float=0.2, unfreeze_stages: Optional[List[str]]=None, fusion_strategy: str='gatedattention', cross_num_rad_tokens: int=4, force_three_channel: bool=False):
        super().__init__()
        self.radiomics_dim = radiomics_dim
        self.proj_dim = proj_dim
        self.fusion_strategy = fusion_strategy
        self.cross_num_rad_tokens = cross_num_rad_tokens
        self.force_three_channel = force_three_channel
        self.unfreeze_stages = unfreeze_stages or ['mid', 'norm_out', 'conv_out', 'quant_conv']
        self.volume_encoder = MVAE(model_name=medvae_model_name, modality=medvae_modality)
        if finetune_mode == 'frozen':
            freeze_all_params(self.volume_encoder)
        elif finetune_mode == 'partial':
            freeze_all_params(self.volume_encoder)
            unfreeze_by_stages(self.volume_encoder, self.unfreeze_stages)
        elif finetune_mode == 'full':
            for p in self.volume_encoder.parameters():
                p.requires_grad = True
        else:
            raise ValueError(f'Unsupported finetune_mode: {finetune_mode}')
        self.vol_proj = nn.LazyLinear(proj_dim)
        self.rad_proj = nn.Linear(radiomics_dim, proj_dim)
        self.concat_classifier = nn.Linear(proj_dim * 2, 2)
        self.gate_mlp = nn.Sequential(nn.Linear(proj_dim, gate_hidden_dim), nn.ReLU(inplace=True), nn.Linear(gate_hidden_dim, proj_dim), nn.Sigmoid())
        self.gated_classifier = nn.Linear(proj_dim, 2)
        self.cross_d_model = 256
        self.img_query_proj = nn.Linear(proj_dim, self.cross_d_model)
        self.rad_token_proj = nn.Linear(proj_dim, self.cross_num_rad_tokens * self.cross_d_model)
        self.cross_attn = nn.MultiheadAttention(embed_dim=self.cross_d_model, num_heads=4, batch_first=True)
        self.cross_norm1 = nn.LayerNorm(self.cross_d_model)
        self.cross_ffn = nn.Sequential(nn.Linear(self.cross_d_model, self.cross_d_model), nn.ReLU(inplace=True), nn.Linear(self.cross_d_model, self.cross_d_model))
        self.cross_norm2 = nn.LayerNorm(self.cross_d_model)
        self.cross_classifier = nn.Linear(proj_dim + self.cross_d_model, 2)
        self.classifier = self.gated_classifier

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
        latent = torch.nan_to_num(latent, nan=0.0, posinf=10000.0, neginf=-10000.0)
        latent = torch.clamp(latent, min=-10000.0, max=10000.0)
        feat_img = self.vol_proj(latent)
        return feat_img

    def forward_concat(self, feat_img, x_rad):
        rad_proj = self.rad_proj(x_rad)
        fused = torch.cat([feat_img, rad_proj], dim=1)
        logits = self.concat_classifier(fused)
        return logits

    def forward_gatedattention(self, feat_img, x_rad):
        rad_proj = self.rad_proj(x_rad)
        gate = self.gate_mlp(rad_proj)
        fused = gate * feat_img + (1.0 - gate) * rad_proj
        logits = self.gated_classifier(fused)
        return (logits, gate, rad_proj)

    def forward_crossattention(self, feat_img, x_rad):
        B, D = x_rad.shape
        q = self.img_query_proj(feat_img).unsqueeze(1)
        rad_proj = self.rad_proj(x_rad)
        rad_tokens = self.rad_token_proj(rad_proj)
        rad_tokens = rad_tokens.view(B, self.cross_num_rad_tokens, self.cross_d_model)
        attn_out, _ = self.cross_attn(q, rad_tokens, rad_tokens)
        x = self.cross_norm1(q + attn_out)
        x_ffn = self.cross_ffn(x)
        x = self.cross_norm2(x + x_ffn)
        cross_summary = x.squeeze(1)
        fused = torch.cat([feat_img, cross_summary], dim=1)
        logits = self.cross_classifier(fused)
        return logits

    def forward(self, x_img, x_rad):
        feat_img = self.encode_image(x_img)
        if self.fusion_strategy == 'concat':
            return self.forward_concat(feat_img, x_rad)
        elif self.fusion_strategy == 'crossattention':
            return self.forward_crossattention(feat_img, x_rad)
        elif self.fusion_strategy == 'gatedattention':
            logits, _, _ = self.forward_gatedattention(feat_img, x_rad)
            return logits
        else:
            raise ValueError(f'Unsupported fusion_strategy: {self.fusion_strategy}')

    def forward_from_feat(self, feat_img, x_rad):
        if self.fusion_strategy == 'concat':
            return self.forward_concat(feat_img, x_rad)
        elif self.fusion_strategy == 'crossattention':
            return self.forward_crossattention(feat_img, x_rad)
        elif self.fusion_strategy == 'gatedattention':
            logits, _, _ = self.forward_gatedattention(feat_img, x_rad)
            return logits
        else:
            raise ValueError(f'Unsupported fusion_strategy: {self.fusion_strategy}')

    def forward_with_gates(self, x_img, x_rad):
        feat_img = self.encode_image(x_img)
        if self.fusion_strategy == 'gatedattention':
            logits, gate, rad_proj = self.forward_gatedattention(feat_img, x_rad)
            return (logits, gate, feat_img, rad_proj)
        elif self.fusion_strategy == 'concat':
            logits = self.forward_concat(feat_img, x_rad)
            return (logits, None, feat_img, None)
        elif self.fusion_strategy == 'crossattention':
            logits = self.forward_crossattention(feat_img, x_rad)
            return (logits, None, feat_img, None)
        else:
            raise ValueError(f'Unsupported fusion_strategy: {self.fusion_strategy}')
