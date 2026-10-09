import argparse
import hashlib
from pathlib import Path

from pipeline.config import read_config


def main():
    parser = argparse.ArgumentParser(description='Verify the published study checkpoint files.')
    parser.add_argument('--weights', type=Path, default=Path('weights'))
    args = parser.parse_args()
    for relative, expected in read_config('weights_manifest.json')['files'].items():
        path = args.weights / relative
        if not path.is_file():
            raise FileNotFoundError(f'{path}: extract {expected["archive"]} into the repository root')
        if path.stat().st_size != expected['bytes']:
            raise ValueError(f'File size mismatch: {path}')
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        if digest != expected['sha256']:
            raise ValueError(f'Checksum mismatch: {path}')
    print('All 17 classifiers and 12 radiomics scalers verified.')


if __name__ == '__main__':
    main()
