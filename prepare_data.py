import argparse
import json
import time
import urllib.request
from pathlib import Path

from pipeline.config import case_ids, read_config


def download(url, destination):
    destination = Path(destination)
    if destination.is_file() and destination.stat().st_size:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + '.partial')
    for attempt in range(3):
        try:
            request = urllib.request.Request(url, headers={'User-Agent': 'rcc-calibration'})
            with urllib.request.urlopen(request, timeout=120) as response, partial.open('wb') as stream:
                while chunk := response.read(1024 * 1024):
                    stream.write(chunk)
            partial.replace(destination)
            return
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


def download_dataset(output, selected):
    sources = read_config('sources.json')
    revision = sources['kits23_revision']
    base = f'https://raw.githubusercontent.com/neheller/kits23/{revision}/'
    download(base + 'dataset/kits23.json', output / 'kits23.json')
    index = output / 'annotation_index.json'
    download(f'https://api.github.com/repos/neheller/kits23/git/trees/{revision}?recursive=1', index)
    tree = json.loads(index.read_text())
    if tree.get('truncated'):
        raise RuntimeError('Upstream annotation listing is incomplete.')
    wanted = set(selected)
    paths = [entry['path'] for entry in tree['tree'] if entry['type'] == 'blob']
    for number, case in enumerate(selected, 1):
        annotation_paths = [p for p in paths if p.startswith(f'dataset/{case}/instances/tumor_instance-1_annotation-') and p.endswith('.nii.gz')]
        if not annotation_paths:
            raise FileNotFoundError(f'No public tumour annotations for {case}')
        print(f'[{number}/{len(wanted)}] {case}', flush=True)
        for path in annotation_paths:
            download(base + path, output / Path(path).relative_to('dataset'))
        image_url = ('https://huggingface.co/datasets/neheller/KiTS-Challenge-Imaging/resolve/'
                     + sources['imaging_revision'] + f'/images/{case}.nii.gz')
        download(image_url, output / case / 'imaging.nii.gz')


def prepare_labels(metadata_path, output):
    import pandas as pd
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    records = {row['case_id']: row for row in metadata}
    allowed = {'clear_cell_rcc', 'papillary_rcc', 'chromophobe_rcc',
               'transitional_cell_carcinoma', 'other', 'clear_cell_papillary',
               'rcc_unclassified', 'multilocular_cystic_rcc'}
    rows = []
    for split, ids in read_config('splits.json').items():
        for case in ids:
            subtype = records[case]['tumor_histologic_subtype']
            if subtype not in allowed:
                raise ValueError(f'Unexpected histology for locked case {case}: {subtype}')
            rows.append({'case_id': case, 'y_true': int(subtype == 'clear_cell_rcc'), 'split': split})
    table = pd.DataFrame(rows).sort_values('case_id')
    if table.case_id.duplicated().any() or len(table) != 396 or table.y_true.sum() != 280:
        raise ValueError('Metadata does not match the study cohort.')
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False)
    print(f'Saved {len(table)} labels and fixed split assignments to {output}')


def main():
    parser = argparse.ArgumentParser(description='Download public KiTS23 inputs or generate study labels.')
    sub = parser.add_subparsers(dest='action', required=True)
    p = sub.add_parser('download')
    p.add_argument('--output', type=Path, default=Path('work/raw'))
    p.add_argument('--include-feature-selection', action='store_true')
    p.add_argument('--case-id', help='Download one study case for a smoke test.')
    p = sub.add_parser('labels')
    p.add_argument('--metadata', type=Path, default=Path('work/raw/kits23.json'))
    p.add_argument('--output', type=Path, default=Path('work/cohort.csv'))
    args = parser.parse_args()
    if args.action == 'download':
        ids = case_ids(args.include_feature_selection)
        if args.case_id:
            if args.case_id not in ids:
                parser.error('--case-id must belong to the selected cohort')
            ids = [args.case_id]
        download_dataset(args.output, ids)
    else:
        prepare_labels(args.metadata, args.output)


if __name__ == '__main__':
    main()
