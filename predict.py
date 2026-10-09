import argparse
import gc
from pathlib import Path

import torch
import joblib
import numpy as np
import pandas as pd
from monai.data import DataLoader, Dataset
from monai.utils import set_determinism

from pipeline.config import features, read_config
from pipeline.data import load_cohort, records, transforms
from pipeline.model_factory import build_model, forward


def predict_neural(name, config, table, preprocessed, weights, device, batch_size):
    model_dir = Path(weights) / name
    scaler = np.load(model_dir / 'radiomics_scaler.npz', allow_pickle=False) if config['fusion'] != 'image_only' else None
    dataset = Dataset(records(table, preprocessed, scaler), transforms(config))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    set_determinism(seed=config['seed'])
    model = build_model(config)
    state = torch.load(model_dir / 'best_model.pt', map_location='cpu', weights_only=True, mmap=True)
    model.load_state_dict(state, strict=True)
    del state
    model.to(device).eval()
    values = {}
    with torch.inference_mode():
        for batch in loader:
            with torch.autocast(device_type=device.type, enabled=device.type == 'cuda' and config['inference_amp']):
                logits = forward(model, batch, config, device)
            probabilities = torch.softmax(logits, dim=1)[:, 1].float().cpu().numpy()
            values.update(zip(batch['case_id'], map(float, probabilities)))
    del model, loader, dataset
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return pd.Series(values, name=name)


def main():
    configs = read_config('models.json')
    parser = argparse.ArgumentParser(description='Generate native ccRCC probabilities for calibration.')
    parser.add_argument('--cohort', type=Path, default=Path('work/cohort.csv'))
    parser.add_argument('--radiomics', type=Path, default=Path('work/radiomics.csv'))
    parser.add_argument('--preprocessed', type=Path, default=Path('work/preprocessed'))
    parser.add_argument('--weights', type=Path, default=Path('weights'))
    parser.add_argument('--split', choices=['train', 'val', 'test'], default='test')
    parser.add_argument('--models', nargs='+', default=['all'], choices=['all', 'radiomics_svc'] + list(configs))
    parser.add_argument('--case-id', help='Run one case from the chosen split.')
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--output', type=Path, default=Path('work/predictions.csv'))
    args = parser.parse_args()
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device)
    table = load_cohort(args.cohort, args.radiomics)
    table = table[table.split == args.split]
    if args.case_id:
        if args.case_id not in table.index:
            parser.error('--case-id is outside the chosen split')
        table = table.loc[[args.case_id]]
    if not np.isfinite(table[features()].to_numpy(float)).all():
        raise ValueError('Selected cases lack finite radiomics features.')
    output = table[['y_true']].copy()
    names = ['radiomics_svc'] + list(configs) if 'all' in args.models else args.models
    for name in names:
        print(f'Predicting {name}: {len(table)} cases', flush=True)
        if name == 'radiomics_svc':
            model = joblib.load(args.weights / name / 'model.joblib')
            output[name] = model.predict_proba(table[features()].to_numpy(float))[:, list(model.classes_).index(1)]
        else:
            output[name] = predict_neural(name, configs[name], table, args.preprocessed, args.weights, device, args.batch_size)
    if output.isna().any().any():
        raise ValueError('Prediction alignment failed.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.reset_index().to_csv(args.output, index=False)
    print(f'Saved {args.output}')


if __name__ == '__main__':
    main()
