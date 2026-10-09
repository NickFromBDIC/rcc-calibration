from pathlib import Path

import torch
from monai.networks.nets import resnet18

from .models.cnn_concat import FusionResNet18
from .models.cnn_cross import CrossAttFusionResNet18
from .models.cnn_gated import GatedFusionResNet18


def build_model(config):
    family, fusion = config['family'], config['fusion']
    if family in ('resnet18', 'medicalnet'):
        if fusion == 'image_only':
            return resnet18(spatial_dims=3, n_input_channels=1, num_classes=2)
        classes = {'concat': FusionResNet18, 'cross_attention': CrossAttFusionResNet18, 'gated': GatedFusionResNet18}
        return classes[fusion](radiomics_dim=73)
    common = dict(proj_dim=config['proj_dim'], finetune_mode=config['finetune_mode'], partial_ft_frac=config['partial_ft_frac'])
    strategy = {'concat': 'concat', 'cross_attention': 'crossattention', 'gated': 'gatedattention'}.get(fusion)
    if family == 'medvae2d':
        common.update(top_k=config['top_k'], pooling='mean')
        if fusion == 'image_only':
            from .models.medvae2d_image import MedVAE2DClassifier
            return MedVAE2DClassifier(**common)
        if fusion == 'gated':
            from .models.medvae2d_gated import MedVAE2DMeanPoolLegacyGatedFusion
            from .models.medvae2d_image import unfreeze_last_fraction
            model = MedVAE2DMeanPoolLegacyGatedFusion(73, 'medvae_4_3_2d', 'ct', config['top_k'], config['proj_dim'], config['gate_hidden_dim'])
            if config['finetune_mode'] == 'partial':
                unfreeze_last_fraction(model.slice_encoder, config['partial_ft_frac'])
            return model
        if fusion == 'concat':
            from .models.medvae2d_concat import MedVAE2DSliceGatedFusion
        else:
            from .models.medvae2d_cross import MedVAE2DSliceGatedFusion
            common['cross_num_rad_tokens'] = config['cross_num_rad_tokens']
        return MedVAE2DSliceGatedFusion(radiomics_dim=73, fusion_strategy=strategy, **common)
    if family == 'medvae3d':
        if fusion == 'image_only':
            from .models.medvae3d_image import MedVAE3DClassifier
            return MedVAE3DClassifier(**common)
        from .models.medvae3d_fusion import MedVAE3DFusion
        return MedVAE3DFusion(radiomics_dim=73, fusion_strategy=strategy,
            cross_num_rad_tokens=config['cross_num_rad_tokens'], unfreeze_stages=config['unfreeze_stages'], **common)
    raise ValueError(f'Unknown family: {family}')


def initialize_pretrained(model, config, directory):
    if config['family'].startswith('medvae'):
        from .models.medvae_backbone import initialize_medvae
        initialize_medvae(getattr(model, 'slice_encoder', None) or model.volume_encoder, directory)
    elif config['family'] == 'medicalnet':
        path = Path(directory) / 'resnet_18_23dataset.pth'
        state = torch.load(path, map_location='cpu', weights_only=True)
        state = state.get('state_dict', state)
        backbone = getattr(model, 'backbone', model)
        target = backbone.state_dict()
        mapped = {}
        for key, value in state.items():
            key = key.removeprefix('module.').removeprefix('resnet.')
            if not key.startswith('fc.') and key in target and target[key].shape == value.shape:
                mapped[key] = value
        if 'conv1.weight' not in mapped:
            raise ValueError(f'{path} does not contain the expected MedicalNet backbone')
        backbone.load_state_dict(mapped, strict=False)


def forward(model, batch, config, device):
    image = batch['image'].to(device)
    args = [image]
    if config['family'] == 'medvae2d':
        args.append(batch['mask'].to(device))
    if config['fusion'] != 'image_only':
        args.append(batch['radiomics'].to(device, dtype=torch.float32))
    return model(*args)
