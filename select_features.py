import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from pipeline.config import features, read_config


def feature_columns(table):
    return [c for c in table.columns if c.startswith(('original_', 'log-sigma-', 'LoG-sigma-'))]


def read_table(path):
    table = pd.read_csv(path).rename(columns={'Case': 'case_id'})
    if table.case_id.duplicated().any():
        raise ValueError(f'Duplicate cases in {path}')
    return table.set_index('case_id').sort_index()


def icc31(matrix):
    n, k = matrix.shape
    if n < 2 or k < 2:
        return np.nan
    row, col, mean = matrix.mean(1, keepdims=True), matrix.mean(0, keepdims=True), matrix.mean()
    msr = k * ((row - mean) ** 2).sum() / (n - 1)
    mse = ((matrix - row - col + mean) ** 2).sum() / ((n - 1) * (k - 1))
    denominator = msr + (k - 1) * mse
    return (msr - mse) / denominator if denominator else np.nan


def icc_features(raters):
    common = sorted(set.intersection(*(set(t.index) for t in raters)))
    names = sorted(set.intersection(*(set(feature_columns(t)) for t in raters)))
    records = []
    for name in names:
        matrix = np.stack([t.loc[common, name].to_numpy(float) for t in raters], axis=1)
        matrix = matrix[np.isfinite(matrix).all(1)]
        if len(matrix) < 2 or (matrix.var(0) == 0).all():
            continue
        records.append({'feature': name, 'icc': icc31(matrix), 'n': len(matrix)})
    table = pd.DataFrame(records).sort_values(['icc', 'n'], ascending=[False, False])
    return table.loc[table.icc >= .75, 'feature'].tolist()


def reduce_features(table, names):
    values = table[names].replace([np.inf, -np.inf], np.nan).dropna().astype(float)
    mad = (values - values.median()).abs().median().sort_values()
    retained = mad[mad >= np.percentile(mad, 20)].index.tolist()
    corr = values[retained].corr().abs()
    order = corr.mean(axis=1).sort_values(ascending=False).index
    kept = []
    for name in order:
        if not any(corr.loc[name, prior] > .95 for prior in kept):
            kept.append(name)
    distance = 1 - values[kept].corr().abs()
    np.fill_diagonal(distance.values, 0)
    groups = fcluster(linkage(squareform(distance.values, checks=False), method='average'), t=.2, criterion='distance')
    clusters = pd.DataFrame({'feature': kept, 'cluster': groups}).sort_values(['cluster', 'feature'])
    return [max(group.feature.tolist(), key=lambda name: mad[name]) for _, group in clusters.groupby('cluster')]


def main():
    parser = argparse.ArgumentParser(description='Rebuild ICC, MAD, Pearson and clustering feature selection.')
    parser.add_argument('--consensus', type=Path, required=True)
    parser.add_argument('--annotators', type=Path, nargs=3, required=True)
    parser.add_argument('--output', type=Path, default=Path('work/selected_features.txt'))
    parser.add_argument('--check-study-list', action='store_true')
    args = parser.parse_args()
    cases = read_config('feature_selection_cases.json')
    # Match the original unsupervised selection population, before classification exclusions.
    consensus = read_table(args.consensus).loc[cases['filter_cases']]
    raters = [read_table(p) for p in args.annotators]
    raters = [t.loc[t.index.intersection(cases['icc_cases'])] for t in raters]
    selected = reduce_features(consensus, icc_features(raters))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text('\n'.join(selected) + '\n')
    if args.check_study_list and selected != features():
        raise ValueError('Recomputed feature names/order differ from configs/radiomics_features.txt.')
    print(f'Saved {len(selected)} features to {args.output}')


if __name__ == '__main__':
    main()
