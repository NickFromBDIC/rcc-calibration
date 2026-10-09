from pathlib import Path

import numpy as np
import pandas as pd
from monai.transforms import (Compose, LoadImaged, EnsureChannelFirstd,
    NormalizeIntensityd, SpatialPadd, CenterSpatialCropd, RandFlipd,
    RandRotate90d, RandAffined, EnsureTyped)

from .config import features, read_config


def load_cohort(cohort, radiomics):
    labels = pd.read_csv(cohort)
    table = pd.read_csv(radiomics)
    for data in [labels, table]:
        if data.case_id.duplicated().any():
            raise ValueError('Duplicate case IDs are not allowed.')
    expected = {case: split for split, ids in read_config('splits.json').items() for case in ids}
    if set(labels.case_id) != set(expected):
        raise ValueError('Cohort must contain the complete fixed study split.')
    if any(expected[row.case_id] != row.split for row in labels.itertuples()):
        raise ValueError('Split assignments differ from configs/splits.json.')
    if not labels.y_true.isin([0, 1]).all():
        raise ValueError('Labels must be 0 or 1.')
    merged = labels.merge(table[['case_id'] + features()], on='case_id', how='left', validate='one_to_one')
    return merged.set_index('case_id').sort_index()


def records(table, root, scaler=None):
    rows = []
    names = features()
    for case, row in table.iterrows():
        vector = row[names].to_numpy(dtype=np.float32)
        if not np.isfinite(vector).all():
            raise ValueError(f'Missing or non-finite radiomics features: {case}')
        if scaler is not None:
            if list(scaler['feature_names']) != names:
                raise ValueError('Scaler feature order does not match the study feature list.')
            vector = (vector - scaler['mean']) / scaler['std']
        image, mask = Path(root) / case / 'image.nii.gz', Path(root) / case / 'mask.nii.gz'
        if not image.is_file() or not mask.is_file():
            raise FileNotFoundError(f'Run preprocess.py for {case} first.')
        rows.append({'case_id': case, 'label': int(row.y_true), 'image': str(image), 'mask': str(mask), 'radiomics': vector})
    return rows


def transforms(config, training=False):
    keys = ['image', 'mask'] if config['family'] == 'medvae2d' else ['image']
    size = (config['patch_size'],) * 3
    steps = [LoadImaged(keys), EnsureChannelFirstd(keys),
        NormalizeIntensityd('image', nonzero=True, channel_wise=False),
        SpatialPadd(keys, spatial_size=size), CenterSpatialCropd(keys, roi_size=size)]
    if training:
        steps += [RandFlipd(keys, prob=.5, spatial_axis=0), RandFlipd(keys, prob=.5, spatial_axis=1),
            RandRotate90d(keys, prob=.2, max_k=3),
            RandAffined(keys, prob=.2, rotate_range=(.1, .1, .1), scale_range=(.1, .1, .1),
                        mode=('bilinear', 'nearest') if len(keys) == 2 else 'bilinear')]
    return Compose(steps + [EnsureTyped(keys)])


def fit_scaler(training):
    values = np.ascontiguousarray(training[features()].to_numpy(dtype=np.float32))
    if not np.isfinite(values).all():
        raise ValueError('Training features contain missing or non-finite values.')
    mean, std = values.mean(0), values.std(0)
    std[std < 1e-6] = 1.
    return {'mean': mean, 'std': std, 'feature_names': np.asarray(features())}
