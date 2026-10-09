import argparse
from pathlib import Path

from pipeline.config import read_config
from prepare_data import download


def main():
    parser = argparse.ArgumentParser(description='Download backbone initialization weights for training.')
    parser.add_argument('--output', type=Path, default=Path('work/pretrained'))
    args = parser.parse_args()
    revision = read_config('sources.json')['medvae_revision']
    for name in ['vae_4x_3c_2D.ckpt', 'vae_4x_1c_3D.ckpt']:
        download(f'https://huggingface.co/stanfordmimi/MedVAE/resolve/{revision}/model_weights/{name}', args.output / name)
    revision = '758fe285bc8ab565eb4f9f965810f1d1a3f79491'
    name = 'resnet_18_23dataset.pth'
    download(f'https://huggingface.co/TencentMedicalNet/MedicalNet-Resnet18/resolve/{revision}/{name}', args.output / name)


if __name__ == '__main__':
    main()
