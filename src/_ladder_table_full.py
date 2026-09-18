"""Emit tab:ladder, the five-arm adversary table, from BOTH result sources.

`_ladder_table.py` only reads paper_results/ladder_ds*.json and so cannot produce the `defended`
column or the ours-def contrast, which come from the defense sweep. It also silently prefers a
stale Q=500 pendigits ladder over the Q=1000 one the paper reports. This script fixes both:

  * ladder cells are keyed (dataset, model) and the LARGEST q wins, so paper_results_q1000/
    supersedes the Q=500 pendigits file rather than losing to glob order
  * `defended` and its per-split array come from defense_sweep at its last budget, and that budget
    is asserted equal to the ladder's q, so the two sources cannot silently disagree

    python _ladder_table_full.py --plain     # eyeball
    python _ladder_table_full.py             # LaTeX rows
"""
import json, glob, os, argparse
import numpy as np
from scipy.stats import wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
LADDER_DIRS = ['paper_results', 'paper_results_q1000']
SWEEP_DIR = 'paper_results'
DS_ORDER = ['crop', 'pendigits', 'breast', 'adult', 'nursery', 'mushroom']
MODELS = ['LR', 'NB', 'KNN', 'DT', 'RF']
MIN_EFFECTIVE = 6


def star(p, ties, nsplit):
    if p is None or (isinstance(p, float) and np.isnan(p)) or (nsplit - ties) < MIN_EFFECTIVE:
        return ''
    return '^{**}' if p < .01 else ('^{*}' if p < .05 else '')


def paired(a, b):
    """ours - defended: mean difference, Wilcoxon p, and tie count."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    ok = ~np.isnan(d)
    ties = int(np.sum(np.isclose(d[ok], 0)))
    try:
        p = wilcoxon(a[ok], b[ok]).pvalue
    except Exception:
        p = float('nan')
    return float(np.nanmean(d)), p, ties


def load():
    lad = {}
    for d in LADDER_DIRS:
        for f in glob.glob(os.path.join(HERE, d, 'ladder_ds*.json')):
            for r in json.load(open(f)):
                k = (r['dataset'], r['model'].upper().replace('RDF', 'RF'))
                if k not in lad or r['q'] > lad[k]['q']:
                    lad[k] = r
    swp = {}
    for f in glob.glob(os.path.join(HERE, SWEEP_DIR, 'defense_sweep_ds*.json')):
        for r in json.load(open(f)):
            swp[(r['dataset'], r['model'].upper().replace('RDF', 'RF'))] = r
    return lad, swp


def main(plain=False):
    lad, swp = load()
    rows, agg = [], {'sg_aut': [], 'ours_aut': [], 'ours_def': []}
    sig = {k: [0, 0] for k in agg}
    for ds in DS_ORDER:
        for m in MODELS:
            r, s = lad.get((ds, m)), swp.get((ds, m))
            if not r or not s:
                if r or s:
                    print(f'% WARNING {ds}/{m}: ladder={bool(r)} sweep={bool(s)}')
                continue
            assert s['q_list'][-1] == r['q'], f"{ds}/{m}: sweep Q={s['q_list'][-1]} != ladder q={r['q']}"
            defended = s['defended_mean'][-1] * 100
            d3, p3, t3 = paired(r['ours_all'], s['defended_all'][-1])
            cs = [(r['selfgrid_vs_autolycus'] * 100,
                   star(r.get('selfgrid_vs_autolycus_p'), r.get('selfgrid_vs_autolycus_ties', 0), r['nsplit']),
                   'sg_aut'),
                  (r['c4_ours_vs_autolycus'] * 100,
                   star(r.get('c4_ours_vs_autolycus_p'), r.get('c4_ours_vs_autolycus_ties', 0), r['nsplit']),
                   'ours_aut'),
                  (d3 * 100, star(p3, t3, r['nsplit']), 'ours_def')]
            fid = [r['blind_mean'] * 100, defended, r['autolycus_mean'] * 100,
                   r['selfgrid_mean'] * 100, r['ours_mean'] * 100]
            for v, st, key in cs:
                agg[key].append(v)
                if st:
                    sig[key][0 if v > 0 else 1] += 1
            rows.append((ds, m, fid, cs, r['n_eval'], r['q']))

    if plain:
        print(f"{'dataset':10s}{'mdl':4s}{'Q':>6s}{'n_ev':>6s}"
              + ''.join(f'{a:>10s}' for a in ['blind', 'defended', 'autolyc', 'selfgrid', 'ours'])
              + ''.join(f'{c:>13s}' for c in ['sg-aut', 'ours-aut', 'ours-def']))
        for ds, m, f, cs, ne, q in rows:
            print(f'{ds:10s}{m:4s}{q:6d}{ne:6d}' + ''.join(f'{x:10.1f}' for x in f)
                  + ''.join(f"{v:+10.1f}{st.replace('^',''):>3s}" for v, st, _ in cs))
        print()
        for k, lab in [('sg_aut', 'sg-aut'), ('ours_aut', 'ours-aut'), ('ours_def', 'ours-def')]:
            v = np.array(agg[k])
            print(f'{lab:10s} mean {v.mean():+6.2f}pp  median {np.median(v):+6.2f}  '
                  f'sig+ {sig[k][0]:2d}  sig- {sig[k][1]:2d}  n={len(v)}')
        # the split the paper reports
        CONT = {'crop', 'pendigits', 'breast', 'adult'}
        for k, lab in [('ours_aut', 'ours-aut'), ('ours_def', 'ours-def')]:
            c = [v for (ds, m, f, cs, ne, q), v in zip(rows, agg[k]) if ds in CONT]
            g = [v for (ds, m, f, cs, ne, q), v in zip(rows, agg[k]) if ds not in CONT]
            print(f'{lab:10s} continuous {np.mean(c):+6.2f} (n={len(c)})   categorical {np.mean(g):+6.2f} (n={len(g)})')
        return

    prev = None
    for ds, m, f, cs, _, _ in rows:
        if prev is not None and ds != prev:
            print('\\addlinespace')
        prev = ds
        print(f'{ds:10s} & {m:4s} & ' + ' & '.join(f'${x:.1f}$' for x in f) + ' & '
              + ' & '.join(f'${v:+.1f}{st}$' for v, st, _ in cs) + ' \\\\')
    print('\\midrule')
    arm_means = [np.mean([r[2][i] for r in rows]) for i in range(5)]
    con_means = [np.mean(agg[k]) for k in ['sg_aut', 'ours_aut', 'ours_def']]
    print('\\multicolumn{2}{l}{\\emph{mean}} & ' + ' & '.join(f'${x:.1f}$' for x in arm_means)
          + ' & ' + ' & '.join(f'${x:+.1f}$' for x in con_means) + ' \\\\')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--plain', action='store_true')
    main(**vars(ap.parse_args()))
