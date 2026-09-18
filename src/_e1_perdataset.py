"""E1 companion: per-dataset decomposition of the LIME fidelity gain over Autolycus into its two
components (boundary search vs diverse generation), broken out by dataset (one panel each) x model.
  boundary_pp = 100*(ours[cap=0] - base)      # Phase-2 boundary, no diverse
  diverse_pp  = 100*(ours[cap=15] - ours[cap=0])  # diverse generation on top
Top budget, mean over the S=10 seed sets. Writes paper/figures/decomposition_per_dataset.png.

    python _e1_perdataset.py
"""
import json, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

RES = Path('/Users/sesame/FaithfulDefense/src/paper_results')
OUT = Path('/Users/sesame/FaithfulDefense/paper/figures/decomposition_per_dataset.png')
DATASET_ORDER = ['crop', 'adult', 'breast', 'nursery', 'mushroom']
MODELS = ['lr', 'nb', 'knn', 'dt', 'rdf']
DISP = {'lr': 'LR', 'nb': 'NB', 'knn': 'KNN', 'dt': 'DT', 'rdf': 'RF'}
C_BND, C_DIV = 'tab:orange', 'tab:green'

cells = {}
for f in glob.glob(str(RES / 'lime_divsweep_n1_k3_ds*.json')):
    for r in json.load(open(f)):
        b = r['base']['fid_mean'][-1]
        o0 = r['ours']['0']['fid_mean'][-1]
        o15 = r['ours']['15']['fid_mean'][-1]
        cells[(r['dataset'], r['model'])] = (100 * (o0 - b), 100 * (o15 - o0))

# shared y-limits across panels
vals = [v for bd in cells.values() for v in bd]
ylo, yhi = min(vals) - 1, max(vals) + 1

fig, axes = plt.subplots(1, len(DATASET_ORDER), figsize=(3.0 * len(DATASET_ORDER), 3.3),
                         squeeze=False, sharey=True)
x = np.arange(len(MODELS)); w = 0.38
for j, ds in enumerate(DATASET_ORDER):
    ax = axes[0][j]
    bnd = [cells.get((ds, m), (0, 0))[0] for m in MODELS]
    div = [cells.get((ds, m), (0, 0))[1] for m in MODELS]
    ax.bar(x - w / 2, bnd, w, color=C_BND, label='boundary (Phase 2)')
    ax.bar(x + w / 2, div, w, color=C_DIV, label='diverse (Phase 1)')
    ax.axhline(0, color='k', lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels([DISP[m] for m in MODELS], fontsize=8)
    ax.set_title(ds, fontsize=10); ax.grid(axis='y', alpha=0.25)
    ax.set_ylim(ylo, yhi)
axes[0][0].set_ylabel(r'$\Delta$ fidelity vs Autolycus (pp)')
axes[0][0].legend(fontsize=7, loc='best')
plt.suptitle('E1 decomposition per dataset: boundary search vs diverse generation (top budget)', fontsize=11)
plt.tight_layout()
fig.savefig(OUT, dpi=150)
print('wrote', OUT)
