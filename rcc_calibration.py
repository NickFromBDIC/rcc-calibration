"""Probability-calibration metrics for binary ccRCC classification."""

from __future__ import annotations

import math
from typing import Dict, Sequence, Tuple

import numpy as np

EPS = 1e-7
PRIMARY_METRICS = (
    "roc_auc", "average_precision", "ece_equal_frequency",
    "brier_score", "negative_log_likelihood",
    "calibration_slope", "calibration_intercept",
)
ERROR_METRICS = ("ece_equal_frequency", "brier_score", "negative_log_likelihood")


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    out = np.empty_like(values)
    positive = values >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    out[~positive] = exp_values / (1.0 + exp_values)
    return out


def _logit(probabilities: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(probabilities, dtype=float), EPS, 1.0 - EPS)
    return np.log(p) - np.log1p(-p)


def wilson_interval(successes: int, total: int, z_value: float = 1.959964) -> Tuple[float, float]:
    if total <= 0:
        return float("nan"), float("nan")
    proportion = float(successes) / float(total)
    denominator = 1.0 + z_value**2 / total
    centre = (proportion + z_value**2 / (2.0 * total)) / denominator
    half_width = (
        z_value
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z_value**2 / (4.0 * total**2)
        )
        / denominator
    )
    return max(0.0, centre - half_width), min(1.0, centre + half_width)


def _logistic_log_likelihood(design: np.ndarray, labels: np.ndarray, beta: np.ndarray) -> float:
    fitted = np.clip(_sigmoid(design @ beta), EPS, 1.0 - EPS)
    return float(np.sum(labels * np.log(fitted) + (1 - labels) * np.log1p(-fitted)))


def _fit_logistic(
    design: np.ndarray,
    labels: np.ndarray,
    ridge: float = 1e-7,
    max_iter: int = 200,
) -> Tuple[np.ndarray, float, bool]:
    x = np.asarray(design, dtype=float)
    y = np.asarray(labels, dtype=float).reshape(-1)
    if x.ndim != 2 or x.shape[0] != y.size:
        raise ValueError("invalid logistic calibration design")
    if np.unique(y).size < 2:
        return np.full(x.shape[1], np.nan), float("nan"), False

    beta = np.zeros(x.shape[1], dtype=float)
    penalty = np.eye(x.shape[1], dtype=float) * ridge
    penalty[0, 0] = 0.0
    converged = False
    for _ in range(max_iter):
        fitted = np.clip(_sigmoid(x @ beta), EPS, 1.0 - EPS)
        gradient = x.T @ (fitted - y) + penalty @ beta
        weights = np.clip(fitted * (1.0 - fitted), 1e-8, None)
        hessian = (x.T * weights) @ x + penalty
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hessian, gradient, rcond=None)[0]

        current_loss = -_logistic_log_likelihood(x, y, beta) + 0.5 * float(beta @ penalty @ beta)
        scale = 1.0
        accepted = False
        for _ in range(30):
            candidate = np.clip(beta - scale * step, -30.0, 30.0)
            candidate_loss = -_logistic_log_likelihood(x, y, candidate) + 0.5 * float(
                candidate @ penalty @ candidate
            )
            if candidate_loss <= current_loss + 1e-12:
                accepted = True
                break
            scale *= 0.5
        if not accepted:
            break
        if np.max(np.abs(candidate - beta)) < 1e-8:
            beta = candidate
            converged = True
            break
        beta = candidate
    return beta, _logistic_log_likelihood(x, y, beta), converged


def _calibration_regression(y: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    linear_predictor = _logit(p)
    design = np.column_stack([np.ones(y.size), linear_predictor])
    beta, log_likelihood, converged = _fit_logistic(design, y)
    return {
        "calibration_intercept": float(beta[0]),
        "calibration_slope": float(beta[1]),
        "calibration_regression_log_likelihood": float(log_likelihood),
        "calibration_regression_converged": bool(converged),
    }


def _roc_auc(labels: np.ndarray, probabilities: np.ndarray) -> float:
    y = np.asarray(labels, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    n_positive = int(np.sum(y == 1))
    n_negative = int(np.sum(y == 0))
    if n_positive == 0 or n_negative == 0:
        return float("nan")
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(p.size, dtype=float)
    start = 0
    while start < p.size:
        stop = start + 1
        while stop < p.size and p[order[stop]] == p[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + 1 + stop)
        start = stop
    positive_rank_sum = float(np.sum(ranks[y == 1]))
    return float(
        (positive_rank_sum - n_positive * (n_positive + 1) / 2)
        / (n_positive * n_negative)
    )


def _average_precision(labels: np.ndarray, probabilities: np.ndarray) -> float:
    y = np.asarray(labels, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    n_positive = int(np.sum(y == 1))
    if n_positive == 0:
        return float("nan")
    thresholds = np.unique(p)[::-1]
    previous_recall = 0.0
    score = 0.0
    for threshold in thresholds:
        selected = p >= threshold
        true_positive = int(np.sum((y == 1) & selected))
        false_positive = int(np.sum((y == 0) & selected))
        recall = true_positive / n_positive
        precision = true_positive / max(1, true_positive + false_positive)
        score += (recall - previous_recall) * precision
        previous_recall = recall
    return float(score)


def _as_binary_arrays(labels, probabilities):
    y = np.asarray(labels, dtype=float).reshape(-1)
    p = np.asarray(probabilities, dtype=float).reshape(-1)
    if y.shape != p.shape or y.size == 0:
        raise ValueError("Labels and probabilities must have the same nonzero length.")
    if not np.all(np.isin(y, [0, 1])):
        raise ValueError("Labels must be 0 (non-ccRCC) or 1 (ccRCC).")
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Probabilities must be finite and in [0, 1].")
    # Retain the original analysis's numerical clipping for calibration metrics.
    return y.astype(int), np.clip(p, EPS, 1.0 - EPS)


def binary_nll(labels, probabilities) -> float:
    y, p = _as_binary_arrays(labels, probabilities)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log1p(-p)))


def calibration_bins(labels, probabilities, n_bins: int = 5) -> list[dict]:
    y, p = _as_binary_arrays(labels, probabilities)
    if not isinstance(n_bins, (int, np.integer)) or n_bins < 1:
        raise ValueError("n_bins must be a positive integer.")
    # Stable sorting and array_split preserve the source implementation's tie handling.
    order = np.argsort(p, kind="mergesort")
    groups = np.array_split(order, min(n_bins, y.size))
    rows = []
    for index, group in enumerate(groups, start=1):
        predicted = float(np.mean(p[group]))
        observed = float(np.mean(y[group]))
        lower, upper = wilson_interval(int(y[group].sum()), int(group.size))
        rows.append({
            "bin": index,
            "n": int(group.size),
            "probability_min": float(np.min(p[group])),
            "probability_max": float(np.max(p[group])),
            "mean_probability": predicted,
            "observed_fraction": observed,
            "calibration_gap": observed - predicted,
            "absolute_gap": abs(observed - predicted),
            "observed_ci_lower": lower,
            "observed_ci_upper": upper,
        })
    return rows


def error_metrics(labels, probabilities, n_bins: int = 5) -> dict:
    y, p = _as_binary_arrays(labels, probabilities)
    bins = calibration_bins(y, p, n_bins)
    return {
        "ece_equal_frequency": float(sum(row["n"] * row["absolute_gap"] for row in bins) / y.size),
        "brier_score": float(np.mean((p - y) ** 2)),
        "negative_log_likelihood": binary_nll(y, p),
    }


def binary_calibration_metrics(labels, probabilities, n_bins: int = 5) -> dict:
    """Evaluate native probabilities; regression parameters are diagnostic only."""
    y, p = _as_binary_arrays(labels, probabilities)
    raw = np.asarray(probabilities, dtype=float).reshape(-1)
    regression = _calibration_regression(y, p)
    identifiable = np.unique(y).size == 2 and np.ptp(_logit(p)) > 0
    interior = all(abs(regression[name]) < 30 for name in ("calibration_slope", "calibration_intercept"))
    estimable = identifiable and interior and regression["calibration_regression_converged"]
    return {
        "n": int(y.size),
        "n_non_ccrcc": int(np.sum(y == 0)),
        "n_ccrcc": int(np.sum(y == 1)),
        "roc_auc": _roc_auc(y, raw),
        "average_precision": _average_precision(y, raw),
        **error_metrics(y, p, n_bins),
        "calibration_slope": regression["calibration_slope"] if estimable else float("nan"),
        "calibration_intercept": regression["calibration_intercept"] if estimable else float("nan"),
        "calibration_regression_estimable": bool(estimable),
    }


def _interval_summary(samples: np.ndarray) -> dict:
    finite = samples[np.isfinite(samples)]
    conditional = np.percentile(finite, [2.5, 97.5]).tolist() if finite.size else None
    complete = finite.size == samples.size and finite.size > 0
    return {
        "ci_95": conditional if complete else None,
        "conditional_ci_95_given_estimable": conditional if not complete else None,
        "estimable_replicates": int(finite.size),
        "nonestimable_replicates": int(samples.size - finite.size),
        "ci_status": "ordinary_percentile" if complete else "not_unconditionally_estimable",
    }


def bootstrap_metric_intervals(labels, probabilities, n_bins=5, n_bootstrap=1000, seed=42) -> dict:
    """Ordinary patient bootstrap with explicit reporting of nonestimable fits."""
    y, _ = _as_binary_arrays(labels, probabilities)
    p = np.asarray(probabilities, dtype=float).reshape(-1)
    if not isinstance(n_bootstrap, (int, np.integer)) or n_bootstrap < 1:
        raise ValueError("n_bootstrap must be a positive integer.")
    samples = {name: np.full(n_bootstrap, np.nan) for name in PRIMARY_METRICS}
    rng = np.random.default_rng(seed)
    single_class = 0
    regression_failures = 0
    for iteration in range(n_bootstrap):
        indices = rng.choice(np.arange(y.size), size=y.size, replace=True)
        sample_y = y[indices]
        metrics = binary_calibration_metrics(sample_y, p[indices], n_bins)
        single_class += int(np.unique(sample_y).size < 2)
        regression_failures += int(not metrics["calibration_regression_estimable"])
        for name in PRIMARY_METRICS:
            samples[name][iteration] = metrics[name]
    return {
        "method": "ordinary_patient",
        "n_bootstrap": n_bootstrap,
        "seed": seed,
        "single_class_replicates": single_class,
        "regression_nonestimable_replicates": regression_failures,
        "metrics": {name: _interval_summary(values) for name, values in samples.items()},
    }


def paired_bootstrap_differences(labels, image_probabilities, ensemble_probabilities,
                                 n_bins=5, n_bootstrap=1000, seed=42) -> dict:
    """Return ensemble-minus-image-only differences for ECE, Brier score and NLL."""
    y, image = _as_binary_arrays(labels, image_probabilities)
    _, ensemble = _as_binary_arrays(labels, ensemble_probabilities)
    if not isinstance(n_bootstrap, (int, np.integer)) or n_bootstrap < 1:
        raise ValueError("n_bootstrap must be a positive integer.")
    image_point = error_metrics(y, image, n_bins)
    ensemble_point = error_metrics(y, ensemble, n_bins)
    samples = {name: np.empty(n_bootstrap) for name in ERROR_METRICS}
    rng = np.random.default_rng(seed)
    for iteration in range(n_bootstrap):
        # Both predictors use the same resampled patients; bins are rebuilt each time.
        indices = rng.choice(np.arange(y.size), size=y.size, replace=True)
        first = error_metrics(y[indices], image[indices], n_bins)
        second = error_metrics(y[indices], ensemble[indices], n_bins)
        for name in ERROR_METRICS:
            samples[name][iteration] = second[name] - first[name]
    return {
        name: {
            "image_only": image_point[name],
            "ensemble": ensemble_point[name],
            "ensemble_minus_image_only": ensemble_point[name] - image_point[name],
            **_interval_summary(samples[name]),
        }
        for name in ERROR_METRICS
    }
