import unittest
import tempfile
from pathlib import Path

import torch
import numpy as np
import pandas as pd

from pipeline.config import features, read_config
from pipeline.data import fit_scaler, load_cohort
from preprocess import bounding_box
from select_features import icc31


class PipelineTests(unittest.TestCase):
    def test_splits_are_disjoint_and_complete(self):
        splits = read_config('splits.json')
        ids = sum(splits.values(), [])
        self.assertEqual({k: len(v) for k, v in splits.items()}, {'train': 237, 'val': 99, 'test': 60})
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(len(features()), 73)

    def test_scaler_uses_supplied_training_rows_and_handles_constant_features(self):
        names = features()
        training = pd.DataFrame(np.tile([2., 4.], (73, 1)).T, columns=names)
        training[names[0]] = 7.
        scaler = fit_scaler(training)
        self.assertEqual(scaler['mean'][0], 7.)
        self.assertEqual(scaler['std'][0], 1.)
        self.assertEqual(scaler['mean'][1], 3.)
        self.assertEqual(scaler['std'][1], 1.)

    def test_icc_consistency_ignores_constant_rater_offset(self):
        subject = np.arange(10, dtype=float)
        scores = np.stack([subject, subject + 4, subject - 3], axis=1)
        self.assertAlmostEqual(icc31(scores), 1.)

    def test_crop_margin_is_clipped_at_volume_boundary(self):
        mask = np.zeros((8, 9, 10), dtype=np.uint8)
        mask[0:2, 3:5, 8:10] = 1
        slices, start = bounding_box(mask, 2)
        self.assertEqual(start, [0, 1, 6])
        self.assertEqual(mask[slices].shape, (4, 6, 4))
        with self.assertRaises(ValueError):
            bounding_box(np.zeros_like(mask), 2)

    def test_duplicate_ids_are_rejected_before_merging(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            pd.DataFrame({'case_id': ['x', 'x'], 'split': ['train', 'train'], 'y_true': [0, 1]}).to_csv(path / 'cohort.csv', index=False)
            pd.DataFrame({'case_id': ['x']}).to_csv(path / 'radiomics.csv', index=False)
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                load_cohort(path / 'cohort.csv', path / 'radiomics.csv')


if __name__ == '__main__':
    unittest.main()
