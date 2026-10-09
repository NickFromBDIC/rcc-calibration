import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / 'configs'


def read_config(name):
    return json.loads((CONFIGS / name).read_text(encoding='utf-8'))


def features():
    return (CONFIGS / 'radiomics_features.txt').read_text().splitlines()


def case_ids(include_selection=False):
    groups = read_config('splits.json')
    ids = set().union(*map(set, groups.values()))
    if include_selection:
        ids.update(set().union(*map(set, read_config('feature_selection_cases.json').values())))
    return sorted(ids)
