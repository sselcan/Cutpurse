"""Plot the three-curve defense budget sweep from stored JSON results."""
import os, json, glob, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RES = os.path.join(os.path.dirname(__file__), 'paper_results')
YLIM = (0.3, 1.0)
DATASET_ORDER = ['crop', 'pendigits', 'breast', 'adult', 'nursery', 'mushroom']
# The attack snaps queries to a bin edge, so it operates only where a continuous axis exists.
# Split the grid on that condition rather than mixing the two regimes in one panel set.
CONTINUOUS = ['crop', 'pendigits', 'breast', 'adult']
CATEGORICAL = ['nursery', 'mushroom']
MODELS = ['LR', 'NB', 'KNN', 'DT', 'RF']
STYLE = {'autolycus': ('--', 'tab:gray', 's', 'Autolycus'),
         'defended':  (':',  'tab:red',  '^', 'Autolycus, bins withheld'),
         'odysseus':  ('-',  'tab:blue', 'o', r'\textsc{Cutpurse}')}
PLAIN = {'odysseus': 'Cutpurse'}


def load():
    cells = {}
    for f in glob.glob(os.path.join(RES, 'defense_sweep_ds*.json')):
        for r in json.load(open(f)):
            cells[(r['dataset'], r['model'].upper().replace('RDF', 'RF'))] = r
    return cells


def main(out, only=None, transpose=False):
    cells = load()
    order = only if only else DATASET_ORDER
    ds_present = [d for d in order if any(k[0] == d for k in cells)]
    md_present = [m for m in MODELS if any(k[1] == m for k in cells)]
    if not ds_present:
        print('no defense_sweep_ds*.json yet'); return
    # With few datasets the default (models down, datasets across) is tall and narrow, which
    # overflows a full-width float. Transposing puts models across and datasets down instead.
    nrow, ncol = (len(ds_present), len(md_present)) if transpose else (len(md_present), len(ds_present))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.9 * ncol, 1.95 * nrow), squeeze=False)
    for r_i, m in enumerate(md_present):
        for c_i, ds in enumerate(ds_present):
            ax = axes[c_i][r_i] if transpose else axes[r_i][c_i]
            r = cells.get((ds, m))
            if not r:
                ax.axis('off'); continue
            x = np.array(r['q_list'], float)
            n = r['hms']
            for arm, (ls, col, mk, lab) in STYLE.items():
                mu = np.array(r[f'{arm}_mean'], float)
                se = np.array(r[f'{arm}_std'], float) / max(np.sqrt(n), 1)
                ax.plot(x, mu, ls, color=col, marker=mk, ms=3, lw=1.4,
                        label=PLAIN.get(arm, lab))
                ax.fill_between(x, mu - se, mu + se, color=col, alpha=0.15)
            ax.set_ylim(*YLIM)
            ax.grid(alpha=0.25, lw=0.5)
            top, left, bottom = ((c_i == 0, r_i == 0, c_i == len(ds_present) - 1) if transpose
                                 else (r_i == 0, c_i == 0, r_i == len(md_present) - 1))
            if top:
                ax.set_title(m if transpose else ds, fontsize=10)
            if left:
                ax.set_ylabel((f'{ds}\nfidelity') if transpose else (f'{m}\nfidelity'), fontsize=9)
            if bottom:
                ax.set_xlabel('query budget $Q$', fontsize=9)
            ax.tick_params(labelsize=7)
    h, l = axes[0][0].get_legend_handles_labels()
    if not h:
        for row in axes:
            for ax in row:
                h, l = ax.get_legend_handles_labels()
                if h: break
            if h: break
    fig.legend(h, l, loc='upper center', ncol=3, fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, 1.005))
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('wrote', out)

    print(f"\n{'dataset':10s}{'mdl':4s}{'Q':>26s}")
    for ds in ds_present:
        for m in md_present:
            r = cells.get((ds, m))
            if not r: continue
            print(f"{ds:10s}{m:4s} Q={r['q_list']}")
            print(f"{'':14s}defense gap (aut-def) {r['defense_gap']}")
            print(f"{'':14s}odysseus - defended   {r['ody_vs_defended']}")
            print(f"{'':14s}odysseus - autolycus  {r['ody_vs_autolycus']}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(os.path.dirname(__file__),
                                                  '..', 'paper', 'figures', 'defense_sweep.png'))
    ap.add_argument('--transpose', action='store_true',
                    help='models across, datasets down (use when few datasets)')
    ap.add_argument('--only', default=None,
                    help="comma-separated datasets, or 'continuous' / 'categorical'")
    a = ap.parse_args()
    sel = {'continuous': CONTINUOUS, 'categorical': CATEGORICAL}.get(
        a.only, a.only.split(',') if a.only else None)
    main(a.out, only=sel, transpose=a.transpose)
