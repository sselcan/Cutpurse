"""Emit the LaTeX ladder table (tab:ladder) and its summary line from ladder_ds*.json.

    python _ladder_table.py            # LaTeX body to stdout
    python _ladder_table.py --plain    # readable version for eyeballing

Significance is the paired Wilcoxon stored by _ladder.py. A contrast whose splits are mostly tied
carries no marker: with fewer than MIN_EFFECTIVE non-tied pairs the test is not interpretable
(scipy discards zero differences, then falls back to a normal approximation it warns is invalid).
"""
import json, glob, os, argparse
import numpy as np

RES = os.path.join(os.path.dirname(__file__), 'paper_results')
DS_ORDER = ['crop', 'pendigits', 'breast', 'adult', 'nursery', 'mushroom']
MODELS = ['LR', 'NB', 'KNN', 'DT', 'RF']
ARMS = ['blind', 'autolycus', 'selfgrid', 'ours']
CONTRASTS = [('c4_ours_vs_autolycus', 'ours$-$\\textsc{aut}'),
             ('c5_ours_vs_selfgrid', 'ours$-$\\textsc{sg}'),
             ('selfgrid_vs_autolycus', '\\textsc{sg}$-$\\textsc{aut}')]
MIN_EFFECTIVE = 6


def star(p, ties, nsplit):
    if p is None or np.isnan(p) or (nsplit - ties) < MIN_EFFECTIVE:
        return ''
    return '^{**}' if p < .01 else ('^{*}' if p < .05 else '')


def load():
    cells = {}
    for f in glob.glob(os.path.join(RES, 'ladder_ds*.json')):
        for r in json.load(open(f)):
            cells[(r['dataset'], r['model'].upper())] = r
    return cells


def main(plain=False):
    cells = load()
    rows, agg = [], {c: [] for c, _ in CONTRASTS}
    sig = {c: [0, 0] for c, _ in CONTRASTS}
    for ds in DS_ORDER:
        for m in MODELS:
            r = cells.get((ds, m))
            if not r:
                continue
            f = [r[f'{a}_mean'] * 100 for a in ARMS]
            cs = []
            for c, _ in CONTRASTS:
                d = r[c] * 100
                s = star(r.get(f'{c}_p'), r.get(f'{c}_ties', 0), r['nsplit'])
                cs.append((d, s))
                if not np.isnan(d):
                    agg[c].append(d)
                    if s:
                        sig[c][0 if d > 0 else 1] += 1
            rows.append((ds, m, f, cs, r['n_eval']))

    if plain:
        print(f"{'dataset':10s}{'mdl':4s}{'n_ev':>6s}" + ''.join(f'{a:>11s}' for a in ARMS)
              + ''.join(f'{c:>14s}' for c in ['ours-aut', 'ours-sg', 'sg-aut']))
        for ds, m, f, cs, ne in rows:
            print(f'{ds:10s}{m:4s}{ne:6d}' + ''.join(f'{x:11.1f}' for x in f)
                  + ''.join(f"{d:+11.1f}{s.replace('^',''):>3s}" for d, s in cs))
        print()
        for c, lab in CONTRASTS:
            v = np.array(agg[c])
            print(f"{lab:26s} mean {v.mean():+6.2f}pp  median {np.median(v):+6.2f}  "
                  f"sig+ {sig[c][0]:2d}  sig- {sig[c][1]:2d}  n={len(v)}")
        return

    for ds, m, f, cs, _ in rows:
        cells_tex = ' & '.join(f'${x:.1f}$' for x in f)
        con_tex = ' & '.join(f'${d:+.1f}{s}$' for d, s in cs)
        print(f'{ds:10s} & {m:4s} & {cells_tex} & {con_tex} \\\\')
    print('\\midrule')
    means = [np.mean([r[3][i][0] for r in rows]) for i in range(len(CONTRASTS))]
    arm_means = [np.mean([r[2][i] for r in rows]) for i in range(len(ARMS))]
    print('\\multicolumn{2}{l}{\\emph{mean}} & '
          + ' & '.join(f'${x:.1f}$' for x in arm_means) + ' & '
          + ' & '.join(f'${x:+.1f}$' for x in means) + ' \\\\')
    print('% ' + '; '.join(
        f"{lab}: mean {np.mean(agg[c]):+.2f}pp, sig+ {sig[c][0]}, sig- {sig[c][1]}, n={len(agg[c])}"
        for c, lab in CONTRASTS))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--plain', action='store_true')
    main(**vars(ap.parse_args()))
