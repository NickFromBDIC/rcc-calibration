"""Run the paper's calibration evaluation from saved native probabilities."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from prediction_io import (
    BOOTSTRAP_SEED_OFFSETS,
    FAMILIES,
    VARIANTS,
    add_probability_ensembles,
    load_predictions,
    model_order,
)
from rcc_calibration import (
    EPS,
    PRIMARY_METRICS,
    binary_calibration_metrics,
    binary_nll,
    bootstrap_metric_intervals,
    calibration_bins,
    paired_bootstrap_differences,
)


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(json_safe(value), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def run_demo(frame: pd.DataFrame, output: Path):
    if len(frame) != 1:
        raise ValueError("--demo requires exactly one patient.")
    y = frame["y_true"].to_numpy()
    predictions = {}
    for model in model_order():
        p = frame[model].to_numpy()
        predictions[model] = {
            "probability_ccrcc": float(p[0]),
            "squared_error": float((np.clip(p[0], EPS, 1 - EPS) - y[0]) ** 2),
            "negative_log_likelihood": binary_nll(y, p),
        }
    write_json(output / "demo_result.json", {
        "mode": "single_patient_demo",
        "case_id": str(frame.iloc[0]["case_id"]),
        "y_true": int(y[0]),
        "note": "One patient demonstrates file handling, ensemble averaging and per-patient scoring. "
                "Cohort calibration, discrimination and confidence intervals are not estimated.",
        "predictions": predictions,
    })


def evaluate_cohort(frame, output, n_bins, n_bootstrap, seed):
    y = frame["y_true"].to_numpy()
    results, metric_rows, reliability_rows = {}, [], []
    for model in model_order():
        probabilities = frame[model].to_numpy()
        point = binary_calibration_metrics(y, probabilities, n_bins)
        model_seed = seed + BOOTSTRAP_SEED_OFFSETS.get(model, 0)
        bootstrap = bootstrap_metric_intervals(y, probabilities, n_bins, n_bootstrap, model_seed)
        results[model] = {"point": point, "bootstrap": bootstrap}
        row = {"model": model, "bootstrap_seed": model_seed, **point}
        for name in PRIMARY_METRICS:
            interval = bootstrap["metrics"][name]
            low, high = interval["ci_95"] or (None, None)
            row.update({
                f"{name}_ci_lower": low,
                f"{name}_ci_upper": high,
                f"{name}_ci_status": interval["ci_status"],
                f"{name}_nonestimable_replicates": interval["nonestimable_replicates"],
            })
        metric_rows.append(row)
        reliability_rows.extend(
            {"model": model, **row} for row in calibration_bins(y, probabilities, n_bins)
        )
        print(f"Evaluated {model}", flush=True)

    paired, paired_rows, fusion_rows = {}, [], []
    for family in FAMILIES:
        baseline = results[f"{family}_image_only"]["point"]
        for variant in VARIANTS[1:]:
            fusion = results[f"{family}_{variant}"]["point"]
            fusion_rows.append({
                "family": family,
                "fusion": variant,
                **{f"delta_{metric}": fusion[metric] - baseline[metric] for metric in PRIMARY_METRICS},
            })
        paired[family] = paired_bootstrap_differences(
            y, frame[f"{family}_image_only"], frame[f"{family}_ensemble"],
            n_bins, n_bootstrap, seed,
        )
        for metric, values in paired[family].items():
            low, high = values["ci_95"] or (None, None)
            paired_rows.append({
                "family": family,
                "metric": metric,
                "image_only": values["image_only"],
                "ensemble": values["ensemble"],
                "ensemble_minus_image_only": values["ensemble_minus_image_only"],
                "ci_lower": low,
                "ci_upper": high,
                "ci_status": values["ci_status"],
            })

    pd.DataFrame(metric_rows).to_csv(output / "metrics.csv", index=False)
    pd.DataFrame(reliability_rows).to_csv(output / "reliability_bins.csv", index=False)
    pd.DataFrame(fusion_rows).to_csv(output / "fusion_vs_image_only.csv", index=False)
    pd.DataFrame(paired_rows).to_csv(output / "paired_ensemble_differences.csv", index=False)
    write_json(output / "calibration_summary.json", {
        "models": results,
        "paired_ensemble_minus_image_only": paired,
    })


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="CSV with one aligned row per patient.")
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--demo", action="store_true", help="Run the bundled one-patient example.")
    parser.add_argument("--n-bins", type=int, default=5)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42, help="Base seed; original model-specific offsets are retained.")
    parser.add_argument("--plots", action="store_true", help="Save cohort reliability diagrams.")
    args = parser.parse_args()
    if args.n_bins < 2 or args.n_bootstrap < 1 or args.seed < 0:
        parser.error("Require n-bins >= 2, n-bootstrap >= 1 and seed >= 0.")
    if args.demo and args.plots:
        parser.error("Reliability diagrams require cohort data; omit --plots in demo mode.")
    return args


def main():
    args = parse_args()
    frame = add_probability_ensembles(load_predictions(args.input))
    if args.demo:
        if len(frame) != 1:
            raise ValueError("--demo requires exactly one patient.")
    elif len(frame) < args.n_bins or frame["y_true"].nunique() < 2:
        raise ValueError(
            "Cohort evaluation requires both outcome classes and at least n-bins patients. "
            "Use --demo for the bundled single-patient example."
        )
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Output directory is not empty; choose a new --output directory.")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.demo:
        run_demo(frame, args.output)
    else:
        evaluate_cohort(frame, args.output, args.n_bins, args.n_bootstrap, args.seed)
    frame.to_csv(args.output / "predictions_with_ensembles.csv", index=False)
    if args.plots:
        from plot_reliability import plot_reliability
        plot_reliability(args.output / "reliability_bins.csv", args.output / "figures")
    write_json(args.output / "run_manifest.json", {
        "mode": "single_patient_demo" if args.demo else "cohort_evaluation",
        "input_file": args.input.name,
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "n_patients": len(frame),
        "n_ccrcc": int(frame["y_true"].sum()),
        "probability_type": "model_native",
        "ensemble_weights": {variant: 1 / 3 for variant in VARIANTS[1:]},
        "n_bins": None if args.demo else args.n_bins,
        "binning": "equal_frequency_stable_sort",
        "n_bootstrap": 0 if args.demo else args.n_bootstrap,
        "bootstrap_method": "ordinary_patient",
        "seed": args.seed,
        "model_bootstrap_seeds": {
            model: args.seed + BOOTSTRAP_SEED_OFFSETS.get(model, 0) for model in model_order()
        } if not args.demo else {},
        "paired_bootstrap_seed": args.seed if not args.demo else None,
        "probability_clipping": EPS,
        "reliability_intervals": "Wilson 95%",
        "figure_size_inches": {"family": [10.5, 6.8], "svc": [3.8, 3.6]} if args.plots else None,
        "figure_formats": ["png", "pdf"] if args.plots else [],
        "figure_dpi": 300 if args.plots else None,
        "python": platform.python_version(),
        "packages": {name: version(name) for name in ("numpy", "pandas", "matplotlib")},
        "source_sha256": {
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ("rcc_calibration.py", "prediction_io.py", "evaluate_calibration.py", "plot_reliability.py")
        },
    })
    print(f"Saved results to {args.output.resolve()}")


if __name__ == "__main__":
    main()
