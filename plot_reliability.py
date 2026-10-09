"""Plot equal-frequency reliability bins and Wilson 95% confidence intervals."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from prediction_io import FAMILIES


def plot_reliability(bins_path: Path, output: Path):
    bins = pd.read_csv(bins_path)
    output.mkdir(parents=True, exist_ok=True)
    variants = (
        ("image_only", "Image only"),
        ("concat", "Concatenation"),
        ("cross_attention", "Cross-attention"),
        ("gated", "Gated fusion"),
        ("ensemble", "Probability ensemble"),
    )
    groups = [
        (family, title, [(f"{family}_{key}", label) for key, label in variants])
        for family, title in FAMILIES.items()
    ]
    groups.append(("radiomics_svc", "Radiomics", [("radiomics_svc", "SVC")]))
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42}):
        for family, title, models in groups:
            if len(models) == 1:
                fig, ax = plt.subplots(figsize=(3.8, 3.6), constrained_layout=True)
                axes = [ax]
            else:
                fig, grid = plt.subplots(2, 3, figsize=(10.5, 6.8), constrained_layout=True)
                axes = grid.ravel()
            for ax, (model, label) in zip(axes, models):
                rows = bins.loc[bins["model"] == model]
                x = rows["mean_probability"].to_numpy()
                y = rows["observed_fraction"].to_numpy()
                errors = np.maximum(0, np.vstack([
                    y - rows["observed_ci_lower"].to_numpy(),
                    rows["observed_ci_upper"].to_numpy() - y,
                ]))
                ax.plot([0, 1], [0, 1], "--", color="#666666", linewidth=1)
                ax.errorbar(x, y, yerr=errors, fmt="o", color="#0072B2", capsize=3, markersize=5)
                ax.set(xlim=(-0.02, 1.02), ylim=(-0.02, 1.02),
                       xlabel="Mean predicted P(ccRCC)", ylabel="Observed ccRCC fraction",
                       title=f"{label} (n={int(rows['n'].sum())})", aspect="equal")
                ax.grid(alpha=0.2)
            for ax in axes[len(models):]:
                ax.set_visible(False)
            count = len(bins.loc[bins["model"] == models[0][0]])
            fig.suptitle(f"{title}\n{count} equal-frequency bins | Wilson 95% CI", fontsize=10)
            for extension in ("png", "pdf"):
                fig.savefig(output / f"{family}_reliability.{extension}", dpi=300, facecolor="white")
            plt.close(fig)
