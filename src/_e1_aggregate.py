"""Generate the aggregate component-decomposition figure from stored sweep results."""

import glob
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "paper_results"
FIGURES = Path(os.environ.get("CUTPURSE_FIGURES", HERE.parent / "figures"))
OUTPUT = FIGURES / "decomposition.png"
DATASETS = ["crop", "adult", "breast", "nursery", "mushroom"]
MODELS = ["lr", "nb", "knn", "dt", "rdf"]
LABELS = ["LR", "NB", "KNN", "DT", "RF"]


def main():
    cells = {}
    for filename in glob.glob(str(RESULTS / "lime_divsweep_n1_k3_ds*.json")):
        for row in json.load(open(filename)):
            boundary = 100 * (row["ours"]["0"]["fid_mean"][-1] - row["base"]["fid_mean"][-1])
            diverse = 100 * (row["ours"]["15"]["fid_mean"][-1] - row["ours"]["0"]["fid_mean"][-1])
            cells[(row["dataset"], row["model"])] = (boundary, diverse)

    boundary_means, diverse_means = [], []
    for model in MODELS:
        values = [cells[(dataset, model)] for dataset in DATASETS if (dataset, model) in cells]
        if not values:
            raise RuntimeError(f"missing decomposition results for {model}")
        boundary_means.append(np.mean([value[0] for value in values]))
        diverse_means.append(np.mean([value[1] for value in values]))

    FIGURES.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(MODELS))
    width = 0.38
    fig, ax = plt.subplots(figsize=(6.4, 3.3))
    ax.bar(x - width / 2, boundary_means, width, color="tab:orange", label="boundary (Phase 2)")
    ax.bar(x + width / 2, diverse_means, width, color="tab:green", label="diverse (Phase 1)")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x, LABELS)
    ax.set_ylabel(r"$\Delta$ fidelity vs Autolycus (pp)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT, dpi=150)
    print("wrote", OUTPUT)


if __name__ == "__main__":
    main()
