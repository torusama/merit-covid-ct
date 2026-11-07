"""Plot the reproducible dataset summary created by audit_zenodo20.py."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis" / "zenodo20"


def main() -> None:
    with (OUT / "case_statistics.csv").open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    names = [row["id"].replace("coronacases_", "C").replace("radiopaedia_", "R") for row in rows]
    positive = np.array([int(row["positive_slices"]) for row in rows])
    empty = np.array([int(row["empty_slices"]) for row in rows])
    z_spacing = np.array([float(row["spacing_z_mm"]) for row in rows])

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), constrained_layout=True)
    x = np.arange(len(rows))
    axes[0].bar(x, positive, color="#e34850", label="Infection present")
    axes[0].bar(x, empty, bottom=positive, color="#a7b0bf", label="No infection")
    axes[0].set_ylabel("Number of axial slices")
    axes[0].legend(loc="upper right")
    axes[0].set_title("COVID-19 CT Seg: lesion presence by volume")
    axes[0].set_xticks(x, names, rotation=60, ha="right", fontsize=8)
    axes[0].axvline(9.5, color="#333", linestyle="--", linewidth=1)

    ratio = [100 * float(row["infection_fraction_lung"]) for row in rows]
    axes[1].scatter(x, ratio, c=z_spacing, cmap="viridis", s=70, edgecolors="black", linewidths=0.3)
    axes[1].set_xticks(x, names, rotation=60, ha="right", fontsize=8)
    axes[1].set_ylabel("Infected voxels / lung voxels (%)")
    axes[1].set_yscale("log")
    axes[1].grid(axis="y", alpha=0.2)
    axes[1].axvline(9.5, color="#333", linestyle="--", linewidth=1)
    mappable = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin=z_spacing.min(), vmax=z_spacing.max()))
    fig.colorbar(mappable, ax=axes[1], label="Axial spacing (mm)")
    fig.savefig(OUT / "dataset_overview.png", dpi=160)
    plt.close(fig)

    examples = ["coronacases_001", "radiopaedia_4_85506_1", "radiopaedia_40_86625_0"]
    area = {item["id"]: item["areas_by_slice"] for item in summary["per_case_slice_areas"]}
    spacing = {row["id"]: float(row["spacing_z_mm"]) for row in rows}
    fig, ax = plt.subplots(figsize=(11, 4), constrained_layout=True)
    for name in examples:
        values = np.array(area[name])
        ax.plot(np.arange(len(values)) * spacing[name], values, label=f"{name} ({spacing[name]:g} mm)")
    ax.set_xlabel("Axial distance from first slice (mm)")
    ax.set_ylabel("Infection area (pixels)")
    ax.set_title("Ground-truth lesion trajectories across slices")
    ax.legend(fontsize=8)
    fig.savefig(OUT / "lesion_trajectories.png", dpi=160)
    plt.close(fig)
    print(OUT / "dataset_overview.png")
    print(OUT / "lesion_trajectories.png")


if __name__ == "__main__":
    main()
