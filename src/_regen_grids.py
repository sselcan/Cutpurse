"""Regenerate the E4 fidelity-vs-budget grids with a FIXED, shared y-axis so subplots
are visually comparable (per-subplot autoscaling was misleading). Writes the 4 figures the
paper includes, directly into paper/figures/. Reads the existing sweep JSONs in paper_results/.

    python _regen_grids.py
"""
import json, glob, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

_HERE = Path(__file__).resolve().parent
FIGP = Path(os.environ.get('CUTPURSE_FIGURES', _HERE.parent / 'figures'))
RES = _HERE / 'paper_results'
FIGP.mkdir(parents=True, exist_ok=True)
YLIM = (0.4, 1.0)   # uniform across every subplot/grid
DATASET_ORDER = ['crop', 'adult', 'breast', 'nursery', 'mushroom']
MODELS_ALL = ['LR', 'NB', 'KNN', 'DT', 'RF']   # display order
# LIME sweeps use lowercase model keys ('rdf'); SHAP sweeps use uppercase ('RF'). Normalise both.
NORM = {'lr': 'LR', 'nb': 'NB', 'knn': 'KNN', 'dt': 'DT', 'rdf': 'RF',
        'LR': 'LR', 'NB': 'NB', 'KNN': 'KNN', 'DT': 'DT', 'RF': 'RF'}


def _se(all_by_q):
    return np.array([np.std(v) / np.sqrt(len(v)) if len(v) else 0.0 for v in all_by_q])


def load(pat):
    d = {}
    for f in glob.glob(str(RES / pat)):
        for r in json.load(open(f)):
            d[(NORM.get(r['model'], r['model'].upper()), r['dataset'])] = r
    return d


def grid(cells, ours_get, out):
    fig, axes = plt.subplots(len(MODELS_ALL), len(DATASET_ORDER),
                             figsize=(2.7 * len(DATASET_ORDER), 2.3 * len(MODELS_ALL)), squeeze=False)
    legend_ax = None
    for i, m in enumerate(MODELS_ALL):
        for j, ds in enumerate(DATASET_ORDER):
            ax = axes[i][j]; r = cells.get((m, ds))
            if not r:
                ax.axis('off'); continue
            if legend_ax is None:
                legend_ax = ax
            xb = np.array(r['q_cap'], float)
            bm = np.array(r['base']['fid_mean']); bse = _se(r['base']['fid_all'])
            o = ours_get(r); om = np.array(o['fid_mean']); ose = _se(o['fid_all'])
            ax.plot(xb, bm, '--', color='tab:gray', marker='s', ms=3, label='Autolycus')
            ax.fill_between(xb, bm - bse, bm + bse, color='tab:gray', alpha=0.2)
            ax.plot(xb, om, '-', color='tab:blue', marker='o', ms=3, label='ours')
            ax.fill_between(xb, om - ose, om + ose, color='tab:blue', alpha=0.2)
            ax.set_ylim(*YLIM)
            ax.set_title(f"{ds} / {m}", fontsize=9); ax.grid(alpha=0.25)
            if i == len(MODELS_ALL) - 1:
                ax.set_xlabel('query budget')
            if j == 0:
                ax.set_ylabel('fidelity')
    (legend_ax or axes[0][0]).legend(fontsize=8, loc='lower right')
    plt.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print('wrote', out)


if __name__ == '__main__':
    lime_ours = lambda r: r['ours']['15']   # DIV_CAP_FIXED = 15
    shap_main = lambda r: r['main']
    grid(load('lime_divsweep_n1_k3_ds*.json'), lime_ours, FIGP / 'lime_grid.png')
    grid(load('shap_sweep_n5_k3_ds*.json'),    shap_main, FIGP / 'shap_grid.png')
    grid(load('lime_divsweep_n3_k1_ds*.json'), lime_ours, FIGP / 'lime_grid_n3_k1.png')
    grid(load('lime_divsweep_n3_k3_ds*.json'), lime_ours, FIGP / 'lime_grid_n3_k3.png')
