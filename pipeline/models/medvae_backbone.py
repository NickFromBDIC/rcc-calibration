from pathlib import Path

import torch
from medvae import MVAE as UpstreamMVAE
from medvae.models import AutoencoderKL_2D, AutoencoderKL_3D
from medvae.utils.factory import build_transform
from medvae.utils.lora import inject_trainable_lora_extended
from omegaconf import OmegaConf

from ..config import CONFIGS


class MVAE(UpstreamMVAE):
    def __init__(self, model_name, modality, gpu_dim=160):
        torch.nn.Module.__init__(self)
        self.model_name, self.modality, self.gpu_dim = model_name, modality, gpu_dim
        self.encoded_latent = self.decoded_latent = None
        # A study checkpoint includes the backbone; inference needs no network download.
        if model_name == 'medvae_4_3_2d':
            config = OmegaConf.load(CONFIGS / 'medvae_4x3.yaml').model.params
            self.model = AutoencoderKL_2D(ddconfig=config.ddconfig, embed_dim=config.embed_dim)
            inject_trainable_lora_extended(self.model, {'ResnetBlock', 'AttnBlock'}, r=4)
        elif model_name == 'medvae_4_1_3d':
            config = OmegaConf.load(CONFIGS / 'medvae_4x1.yaml')
            self.model = AutoencoderKL_3D(ddconfig=config.ddconfig, embed_dim=config.embed_dim)
        else:
            raise ValueError(f'Unsupported study backbone: {model_name}')
        self.transform = build_transform(model_name, modality)


def initialize_medvae(encoder, pretrained_dir):
    filename = 'vae_4x_3c_2D.ckpt' if '2d' in encoder.model_name else 'vae_4x_1c_3D.ckpt'
    path = Path(pretrained_dir) / filename
    if not path.is_file():
        raise FileNotFoundError(f'{path}: run download_pretrained.py first')
    encoder.model.init_from_ckpt(str(path), state_dict=True)
