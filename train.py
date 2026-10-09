import argparse
import gc
import json
from pathlib import Path

import torch
import joblib
import numpy as np
from monai.data import DataLoader, Dataset
from monai.utils import set_determinism
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from pipeline.config import features, read_config
from pipeline.data import fit_scaler, load_cohort, records, transforms
from pipeline.model_factory import build_model, forward, initialize_pretrained


def ordered_splits(table, seed=42):
    ids = table.index.to_numpy()
    train_ids, holdout = train_test_split(ids, test_size=.4, random_state=seed, stratify=table.y_true)
    val_ids, test_ids = train_test_split(holdout, test_size=.15 / .4, random_state=seed, stratify=table.loc[holdout].y_true)
    groups = {'train': train_ids, 'val': val_ids, 'test': test_ids}
    for split, values in groups.items():
        if set(values) != set(table.index[table.split == split]):
            raise ValueError('Reconstructed training order disagrees with the fixed split.')
    return {split: table.loc[sorted(values)] for split, values in groups.items()}


def train_svc(training, output, jobs):
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('selector', SelectFromModel(LogisticRegression(solver='liblinear', l1_ratio=1., class_weight='balanced', max_iter=5000, random_state=42), threshold=1e-8)),
        ('svc', SVC(kernel='rbf', probability=True, random_state=42)),
    ])
    grid = {'selector__estimator__C': [.1, .3, 1., 3., 10.],
            'svc__C': [.1, 1., 10., 100.], 'svc__gamma': ['scale', .001, .01, .1, 1.],
            'svc__class_weight': [None, 'balanced']}
    # All fitting, including scaling and L1 selection, stays inside training folds.
    search = GridSearchCV(pipeline, grid, scoring='roc_auc',
        cv=StratifiedKFold(5, shuffle=True, random_state=42), n_jobs=jobs, refit=True, error_score=np.nan)
    search.fit(training[features()].to_numpy(float), training.y_true.to_numpy())
    if not np.isfinite(search.best_score_):
        raise RuntimeError('All SVC tuning candidates failed.')
    joblib.dump(search.best_estimator_, output / 'model.joblib')


def train_epoch(model, loader, optimizer, criterion, device, config, scaler):
    model.train()
    losses = []
    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=device.type == 'cuda'):
            logits = forward(model, batch, config, device)
            loss = criterion(logits, batch['label'].to(device))
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite training loss')
        scaler.scale(loss).backward()
        if config['gradient_clip'] is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'])
        scaler.step(optimizer)
        scaler.update()
        losses.append(loss.item())
    return float(np.mean(losses))


def validation_auc(model, loader, device, config):
    model.eval()
    labels, probabilities = [], []
    with torch.inference_mode():
        for batch in loader:
            with torch.autocast(device_type=device.type, enabled=device.type == 'cuda'):
                logits = forward(model, batch, config, device)
            prob = torch.softmax(logits, dim=1)[:, 1].float().cpu().numpy()
            labels.extend(batch['label'].tolist())
            probabilities.extend(prob.tolist())
    return float(roc_auc_score(labels, probabilities))


def train_neural(config, groups, preprocessed, pretrained, output, device, workers):
    set_determinism(seed=config['seed'])
    standardizer = fit_scaler(groups['train'])
    if config['fusion'] != 'image_only':
        np.savez(output / 'radiomics_scaler.npz', **standardizer)
    train_data = Dataset(records(groups['train'], preprocessed, standardizer), transforms(config, training=True))
    val_data = Dataset(records(groups['val'], preprocessed, standardizer), transforms(config))
    train_loader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True, num_workers=workers)
    val_loader = DataLoader(val_data, batch_size=config['batch_size'], shuffle=False, num_workers=workers)
    model = build_model(config)
    initialize_pretrained(model, config, pretrained)
    model.to(device)
    # Materialize lazy projections before constructing the optimizer.
    model.eval()
    with torch.no_grad(), torch.autocast(device_type=device.type, enabled=device.type == 'cuda'):
        forward(model, next(iter(val_loader)), config, device)
    encoder, heads = [], []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            (encoder if name.startswith(('slice_encoder.', 'volume_encoder.')) else heads).append(parameter)
    optimizer = torch.optim.Adam([
        {'params': encoder, 'lr': config['encoder_learning_rate']},
        {'params': heads, 'lr': config['learning_rate']}], weight_decay=config['weight_decay'])
    counts = np.bincount(groups['train'].y_true.to_numpy(), minlength=2)
    class_weights = counts.sum() / (2 * counts.astype(np.float32))
    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(class_weights, device=device))
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=.5, patience=5)
    best, stale = -np.inf, 0
    (output / 'training_config.json').write_text(json.dumps(config, indent=2) + '\n')
    for epoch in range(config['epochs']):
        loss = train_epoch(model, train_loader, optimizer, criterion, device, config, scaler)
        auc = validation_auc(model, val_loader, device, config)
        scheduler.step(auc)
        print(f'Epoch {epoch+1}: loss={loss:.5f}, validation AUC={auc:.5f}', flush=True)
        if auc > best:
            best, stale = auc, 0
            torch.save(model.state_dict(), output / 'best_model.pt')
        else:
            stale += 1
        if stale >= config['patience']:
            break
    del model, train_loader, val_loader
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()


def main():
    configs = read_config('models.json')
    parser = argparse.ArgumentParser(description='Train classifiers on the fixed study partitions.')
    parser.add_argument('--model', choices=['all', 'radiomics_svc'] + list(configs), required=True)
    parser.add_argument('--cohort', type=Path, default=Path('work/cohort.csv'))
    parser.add_argument('--radiomics', type=Path, default=Path('work/radiomics.csv'))
    parser.add_argument('--preprocessed', type=Path, default=Path('work/preprocessed'))
    parser.add_argument('--pretrained', type=Path, default=Path('work/pretrained'))
    parser.add_argument('--output', type=Path, default=Path('work/trained_weights'))
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--jobs', type=int, default=1, help='SVC grid-search jobs.')
    parser.add_argument('--epochs', type=int, help='Optional training-length override.')
    args = parser.parse_args()
    if args.epochs is not None and args.epochs < 1:
        parser.error('--epochs must be positive')
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device)
    table = load_cohort(args.cohort, args.radiomics)
    groups = ordered_splits(table)
    names = ['radiomics_svc'] + list(configs) if args.model == 'all' else [args.model]
    for name in names:
        destination = args.output / name
        if destination.exists() and any(destination.iterdir()):
            raise FileExistsError(f'Choose an empty model output directory: {destination}')
        destination.mkdir(parents=True, exist_ok=True)
        print(f'Training {name}', flush=True)
        if name == 'radiomics_svc':
            train_svc(groups['train'], destination, args.jobs)
        else:
            config = dict(configs[name])
            if args.epochs is not None:
                config['epochs'] = args.epochs
            train_neural(config, groups, args.preprocessed, args.pretrained, destination, device, args.workers)


if __name__ == '__main__':
    main()
