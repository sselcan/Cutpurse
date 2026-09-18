"""Recompute the E2/E3 significance markers from the stored per-split arrays, reporting tied pairs.

Why: the paired Wilcoxon signed-rank test operates on the 10 per-split differences. Where the two
arms produce identical fidelity on a split the difference is exactly zero, and scipy's default
(zero_method='wilcox') DISCARDS those pairs -- so a cell with 8 ties is tested on 2 observations and
scipy falls back to a normal approximation it simultaneously warns is invalid at that size. Those
p-values are not interpretable and should not carry a significance marker.

Ties are not noise: they mean the ablation changed nothing on that split (typically a near-ceiling or
degenerate target). We therefore report the tie count alongside each cell and suppress the marker
when fewer than MIN_EFFECTIVE differences remain.

No experiments are re-run; everything is recomputed from paper_results/*.json.

    python _recompute_stats.py
"""
import json, glob, os, warnings
import numpy as np
from scipy.stats import wilcoxon

warnings.filterwarnings('ignore')
RES = os.path.join(os.path.dirname(__file__), 'paper_results')
MIN_EFFECTIVE = 6          # need at least this many non-tied pairs for a reportable test
DS_ORDER = ['crop', 'pendigits', 'breast', 'adult', 'nursery', 'mushroom']
MODELS = ['LR', 'NB', 'KNN', 'DT', 'RF']


def paired(a, b):
    """Return (delta_pp, p, n_ties, n_effective). p is None when the test is not reportable."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    ok = ~(np.isnan(a) | np.isnan(b)); a, b = a[ok], b[ok]
    d = a - b
    ties = int(np.sum(d == 0)); eff = int(np.sum(d != 0))
    delta = float(a.mean() - b.mean()) * 100
    if eff < MIN_EFFECTIVE:
        return delta, None, ties, eff
    return delta, float(wilcoxon(a, b).pvalue), ties, eff


def mark(p):
    if p is None:
        return ' n/t'          # not testable: too many tied splits
    return '**  ' if p < .01 else ('*   ' if p < .05 else '    ')


def main():
    cells = {}
    for f in glob.glob(os.path.join(RES, 'autolycus_ablation_ms_ds*.json')):
        for r in json.load(open(f)):
            if 'default_all' in r:
                cells[(r['dataset'], r['model'].upper().replace('RDF', 'RF'))] = r

    for chan, arm in [('ATTRIBUTION (default - randfeat)', 'randfeat_all'),
                      ('THRESHOLD  (default - nothresh)', 'nothresh_all')]:
        print(f"\n=== {chan} ===")
        print(f"{'dataset':10s} {'mdl':4s} {'delta':>8s} {'sig':>5s} {'ties':>6s}")
        vals, sp, sn, nt = [], 0, 0, 0
        for ds in DS_ORDER:
            for m in MODELS:
                r = cells.get((ds, m))
                if not r or arm not in r:
                    continue
                delta, p, ties, eff = paired(r['default_all'], r[arm])
                vals.append(delta)
                if p is not None and p < .05:
                    sp += delta > 0; sn += delta < 0
                if p is None:
                    nt += 1
                print(f"{ds:10s} {m:4s} {delta:+8.1f} {mark(p):>5s} {ties:4d}/10")
        print(f"  -> n={len(vals)}  mean={np.mean(vals):+.2f}pp  median={np.median(vals):+.2f}"
              f"  sig+={sp}  sig-={sn}  not-testable={nt}")


if __name__ == '__main__':
    main()
