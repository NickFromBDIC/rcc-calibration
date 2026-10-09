# RCC Probability Calibration

Code accompanying **Beyond Discrimination: Evaluating Probability Calibration in Radiomics–Image Fusion for Renal Cell Carcinoma Classification**.

The scripts evaluate saved model probabilities and construct probability-average ensembles for ResNet-18, MedicalNet, 2D MedVAE, and 3D MedVAE. Please refer to the paper for experimental results and discussion.

## Installation

Use Python 3.11. A CPU is sufficient. Run the commands from the repository root.

```bash
python -m pip install -r requirements.txt
```

## Demo

```bash
python evaluate_calibration.py --input data/demo_predictions.csv --demo --output results/demo
```

The demo contains one study prediction record, identified as `demo_001`, with a class label and 17 model probabilities. It demonstrates data loading, ensemble averaging, and per-sample scoring. The output is saved in `results/demo/`.

Use a cohort containing both classes for the full analysis below.

## Cohort analysis

Prepare a CSV using the same columns as the demo, with one row per patient:

| Column | Description |
| --- | --- |
| `case_id` | Unique patient identifier |
| `y_true` | `1` for clear-cell RCC; `0` for non-clear-cell RCC |
| `radiomics_svc` | Radiomics-only SVC probability |
| `resnet18_*`, `medicalnet_*`, `medvae2d_*`, `medvae3d_*` | Image-only and fusion-model probabilities |

For each image-model family, include all four suffixes: `image_only`, `concat`, `cross_attention`, and `gated`. Each probability is the model's native P(ccRCC) in `[0, 1]`. Align all predictions by patient ID.

```bash
python evaluate_calibration.py --input private_data/predictions.csv --output results/cohort --n-bins 5 --n-bootstrap 1000 --seed 42 --plots
```

The analysis calculates AUC, AP, ECE, Brier score, NLL, joint calibration intercept and slope, bootstrap intervals, and paired ensemble-versus-image-only differences. Each ensemble is the equal-weight average of the three fusion probabilities.

Metric tables, analysis summaries, and reliability plots are written to `results/cohort/`. The defaults use five equal-frequency bins and 1,000 patient bootstrap samples; reliability plots show Wilson 95% intervals. Choose a new or empty output directory for each run.

## Files

| File | Purpose |
| --- | --- |
| `evaluate_calibration.py` | Demo and cohort-analysis entry point |
| `prediction_io.py` | Prediction loading and ensemble construction |
| `rcc_calibration.py` | Metrics and bootstrap calculations |
| `plot_reliability.py` | Reliability diagrams |
| `data/demo_predictions.csv` | Single-record example |

## License

The code is released under the [MIT License](LICENSE). The study uses [KiTS23](https://github.com/neheller/kits23); see its [data terms and attribution](https://github.com/neheller/kits23#license-and-attribution).
