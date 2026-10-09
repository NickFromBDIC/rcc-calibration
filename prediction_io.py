"""Load aligned patient predictions and construct the four paper ensembles."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd

FAMILIES = {
    "resnet18": "ResNet-18",
    "medicalnet": "MedicalNet",
    "medvae2d": "2D MedVAE",
    "medvae3d": "3D MedVAE",
}
VARIANTS = ("image_only", "concat", "cross_attention", "gated")
PREDICTION_COLUMNS = (
    "radiomics_svc",
    *(f"{family}_{variant}" for family in FAMILIES for variant in VARIANTS),
)
BOOTSTRAP_SEED_OFFSETS = {
    "radiomics_svc": 900,
    "medicalnet_image_only": 100,
    "resnet18_gated": 200,
    "medicalnet_gated": 300,
    "medvae2d_image_only": 400,
    "medvae3d_image_only": 500,
    "medvae2d_gated": 600,
    "medvae3d_gated": 700,
    "medvae2d_concat": 800,
    "medvae2d_cross_attention": 900,
    "medvae3d_concat": 1000,
    "medvae3d_cross_attention": 1100,
}


def load_predictions(path: str | Path) -> pd.DataFrame:
    """Read one row per patient, with all 17 native probability columns."""
    path = Path(path)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        header = next(csv.reader(handle), [])
    required = {"case_id", "y_true", *PREDICTION_COLUMNS}
    if len(header) != len(set(header)):
        raise ValueError("Duplicate column names are not allowed.")
    if set(header) != required:
        raise ValueError(
            f"CSV columns differ from the paper schema. "
            f"Missing: {sorted(required - set(header))}; extra: {sorted(set(header) - required)}"
        )
    frame = pd.read_csv(path, dtype={"case_id": str})
    if frame.empty:
        raise ValueError("The prediction CSV is empty.")
    if frame.isna().any().any():
        raise ValueError("Missing values are not allowed.")
    frame["case_id"] = frame["case_id"].str.strip()
    if frame["case_id"].eq("").any() or frame["case_id"].duplicated().any():
        raise ValueError("Each patient must have a unique, nonempty case_id.")
    if not frame["y_true"].isin([0, 1]).all():
        raise ValueError("y_true must contain only 0 and 1.")
    frame["y_true"] = frame["y_true"].astype(int)
    probabilities = frame[list(PREDICTION_COLUMNS)].apply(pd.to_numeric, errors="raise")
    values = probabilities.to_numpy(float)
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("All probabilities must be finite and in [0, 1].")
    frame[list(PREDICTION_COLUMNS)] = probabilities
    return frame[["case_id", "y_true", *PREDICTION_COLUMNS]].sort_values(
        "case_id", kind="mergesort"
    ).reset_index(drop=True)


def add_probability_ensembles(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for family in FAMILIES:
        components = [f"{family}_{variant}" for variant in VARIANTS[1:]]
        result[f"{family}_ensemble"] = result[components].mean(axis=1, skipna=False)
    return result


def model_order() -> list[str]:
    return [
        "radiomics_svc",
        *(f"{family}_{variant}" for family in FAMILIES for variant in (*VARIANTS, "ensemble")),
    ]
