# RCC Probability Calibration

Code accompanying **Beyond Discrimination: Evaluating Probability Calibration in Radiomics–Image Fusion for Renal Cell Carcinoma Classification**.

This repository provides data preparation, radiomics extraction and selection, model training, inference, and probability calibration analysis. Please refer to the paper for experimental results and discussion. One prediction record is included as a lightweight demo.

## Calibration demo

Use Python 3.11 and run commands from the repository root:

```bash
python -m pip install -r requirements.txt
python evaluate_calibration.py --input data/demo_predictions.csv --demo --output results/demo
```

The single record demonstrates loading, ensemble averaging and per-sample scoring. Cohort analysis requires both classes.

## From KiTS23 to model probabilities

### 1. Install the pipeline

Use Python 3.11 for the models and a separate Python 3.9 environment for PyRadiomics. Install its compiled extensions from the official radiomics Conda channel:

```bash
conda create -n rcc python=3.11 -y
conda activate rcc
python -m pip install -r requirements-full.txt
conda create -n rcc-radiomics -c radiomics -c conda-forge python=3.9 pyradiomics=3.1.0a2 numpy=1.26.4 -y
conda run -n rcc-radiomics python -m pip install -r requirements-radiomics.txt
```

For GPU execution, install a matching PyTorch 2.11 / torchvision 0.26 build using the [PyTorch installation instructions](https://pytorch.org/get-started/locally/). Preprocessing, radiomics and calibration run on CPU; a CUDA GPU is recommended for the neural networks.

### 2. Download and prepare data

```bash
python prepare_data.py download --output work/raw
python prepare_data.py labels --metadata work/raw/kits23.json --output work/cohort.csv
python preprocess.py --raw work/raw --output work/preprocessed
conda run -n rcc-radiomics python extract_radiomics.py --preprocessed work/preprocessed --output work/radiomics.csv
```

The downloader uses the official [KiTS23](https://github.com/neheller/kits23) sources, pinned in `configs/sources.json`. It downloads only the study cases and resumes completed files. Data storage is substantially larger than this code repository.

`configs/splits.json` fixes the 237 training, 99 validation and 60 test patients. Labels come from public histology metadata: `1` is ccRCC and `0` is the study's non-ccRCC group. Multifocal and other excluded cases are absent from this fixed cohort.

Preprocessing retains the original study array convention: RAS reorientation and 1 mm resampling, HU clipping to `[-150, 200]`, annotation-specific tumour crops, STAPLE consensus at 0.5 for multiple annotations, and a 12-voxel context margin for the image models. Smaller connected components are retained.

Radiomics uses Original shape features and Original/LoG first-order and texture features with sigma 1, 2 and 3 mm and a bin width of 25. Default extraction follows the fixed 73-feature list. Fusion-model scaling is fitted on training patients and saved with each checkpoint.

### 3. Load study weights or train models

The study checkpoints are provided separately from the code. Extract the following archives into the repository root; each contains a `weights/` directory. Check the repository's [Releases](https://github.com/NickFromBDIC/rcc-calibration/releases) for checkpoint assets.

```text
weights-resnet18.zip    (also contains the radiomics SVC)
weights-medicalnet.zip
weights-medvae2d.zip
weights-medvae3d-1.zip
weights-medvae3d-2.zip
```

```bash
python verify_weights.py --weights weights
```

To train your own checkpoints instead:

```bash
python download_pretrained.py --output work/pretrained
python train.py --model all --pretrained work/pretrained --output work/trained_weights
```

Use `--model resnet18_gated`, for example, to train one model. Training reads the fixed partitions and `configs/models.json`; validation AUC controls checkpoint selection, scheduling and early stopping. SVC tuning uses five-fold cross-validation within the training partition. Model-specific optimizer and fine-tuning settings retain the source implementation. Training is stochastic; checkpoint inference is the direct route to evaluate the supplied study models.

### 4. Predict and analyse calibration

```bash
python predict.py --weights weights --split test --output work/predictions.csv
python evaluate_calibration.py --input work/predictions.csv --output results/cohort --n-bins 5 --n-bootstrap 1000 --seed 42 --plots
```

For newly trained models, use `--weights work/trained_weights`. `predict.py` aligns all 17 native ccRCC probabilities by patient ID. The calibration script constructs each family's equal-weight average of its three fusion models and calculates metrics, bootstrap intervals and reliability diagrams. These outputs are generated locally.

For a one-case pipeline check, add `--case-id case_00007` to the download, preprocessing, extraction and prediction commands, then pass `--demo` to the calibration command. Still generate the full cohort label file; prediction selects only that one test patient.

## Rebuild radiomics feature selection

The optional selection workflow uses the original annotation and filtering populations in `configs/feature_selection_cases.json`, which precede the final classification exclusions:

```bash
python prepare_data.py download --include-feature-selection
python preprocess.py --include-feature-selection
conda run -n rcc-radiomics python extract_radiomics.py --include-feature-selection --all-features --output work/consensus_features.csv
conda run -n rcc-radiomics python extract_radiomics.py --include-feature-selection --all-features --annotator 1 --output work/ann1.csv
conda run -n rcc-radiomics python extract_radiomics.py --include-feature-selection --all-features --annotator 2 --output work/ann2.csv
conda run -n rcc-radiomics python extract_radiomics.py --include-feature-selection --all-features --annotator 3 --output work/ann3.csv
python select_features.py --consensus work/consensus_features.csv --annotators work/ann1.csv work/ann2.csv work/ann3.csv --check-study-list
```

Selection applies ICC(3,1) ≥ 0.75, a 20th-percentile MAD cutoff, greedy removal at |Pearson r| > 0.95, and average-linkage clustering at distance 0.2. The largest-MAD feature represents each cluster. The resulting names and order are checked against the study list.

## Code map

| Location | Purpose |
| --- | --- |
| `prepare_data.py`, `preprocess.py` | Public data, labels, CT and mask preparation |
| `extract_radiomics.py`, `select_features.py` | Feature extraction and selection |
| `pipeline/models/`, `configs/models.json` | Model definitions and training settings |
| `train.py`, `predict.py` | Training and native-probability inference |
| `evaluate_calibration.py`, `rcc_calibration.py` | Calibration evaluation and bootstrap |
| `prediction_io.py`, `plot_reliability.py` | Ensembles and reliability plots |

Run checks with `python -m unittest discover -s tests` in the model environment.

## License

Code: [MIT](LICENSE). Data and upstream components retain their respective terms; see [third-party notices](THIRD_PARTY_NOTICES.md).
