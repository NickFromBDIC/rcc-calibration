import argparse
import csv
import json
import logging
from pathlib import Path

import numpy as np
from radiomics import featureextractor

from pipeline.config import case_ids, features


def extractors():
    shape = featureextractor.RadiomicsFeatureExtractor({
        'imageType': {'Original': {}}, 'featureClass': {'shape': []},
        'setting': {'normalize': False}})
    texture = featureextractor.RadiomicsFeatureExtractor({
        'imageType': {'Original': {}, 'LoG': {'sigma': [1., 2., 3.]}},
        'featureClass': {key: [] for key in ['firstorder', 'glcm', 'glrlm', 'glszm', 'gldm', 'ngtdm']},
        'setting': {'binWidth': 25, 'normalize': False, 'interpolator': 'sitkBSpline'}})
    return shape, texture


def extract_pair(image, mask, engines):
    values = {}
    for engine in engines:
        for key, value in engine.execute(str(image), str(mask)).items():
            if key.startswith(('original_', 'log-sigma-', 'LoG-sigma-')):
                values[key] = float(np.asarray(value))
    return values


def main():
    parser = argparse.ArgumentParser(description='Extract study shape, Original and LoG radiomics features.')
    parser.add_argument('--preprocessed', type=Path, default=Path('work/preprocessed'))
    parser.add_argument('--output', type=Path, default=Path('work/radiomics.csv'))
    parser.add_argument('--include-feature-selection', action='store_true')
    parser.add_argument('--all-features', action='store_true')
    parser.add_argument('--annotator', type=int, choices=[1, 2, 3])
    parser.add_argument('--case-id')
    args = parser.parse_args()
    logging.getLogger('radiomics').setLevel(logging.ERROR)
    ids = case_ids(args.include_feature_selection)
    if args.case_id:
        if args.case_id not in ids:
            parser.error('--case-id is outside the selected cohort')
        ids = [args.case_id]
    engines = extractors()
    rows = []
    for index, case in enumerate(ids, 1):
        root = args.preprocessed / case
        if args.annotator:
            base = root / 'annotations' / f'annotation-{args.annotator}'
            mask, image = base.with_suffix('.nii.gz'), base.with_name(base.name + '_image.nii.gz')
            if not mask.is_file():
                continue
        else:
            paths = json.loads((root / 'radiomics_paths.json').read_text())
            image, mask = root / paths['image'], root / paths['mask']
        print(f'[{index}/{len(ids)}] {case}', flush=True)
        row = extract_pair(image, mask, engines)
        names = list(row) if args.all_features else features()
        # Small ROIs can lack LoG features; retain NaNs for the explicit selection step.
        rows.append({'case_id': case, **{name: row.get(name, np.nan) for name in names}})
    if not rows:
        raise ValueError('No annotated cases were extracted.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    names = list(dict.fromkeys(key for row in rows for key in row))
    with args.output.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == '__main__':
    main()
