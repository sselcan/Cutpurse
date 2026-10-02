"""Assemble step-size-matched Cutpurse and Autolycus contrasts.

The script reads the step-size and ladder outputs, checks their expected pairing,
and prints the reported aggregate contrasts.

Column names: `ours` and `eps1` are Cutpurse at the per-feature and unit step; `aut` and `autE` are
Autolycus at the unit and per-feature step; `def` is defended; `sg` is self-grid; `best` is the
higher-scoring of the two Autolycus configurations in that cell; `step` is ours minus eps1.
"""
import os, glob, json, argparse
import numpy as np
from scipy.stats import wilcoxon

SRC = os.path.dirname(os.path.abspath(__file__))
CONTINUOUS = ['crop', 'pendigits', 'breast', 'adult']      # the datasets behind the +16.3/+7.0 means
MODELS = ['LR', 'NB', 'KNN', 'DT', 'RF']


def _norm(m):
    return {'rdf': 'RF', 'dt': 'DT', 'lr': 'LR', 'nb': 'NB', 'knn': 'KNN'}.get(str(m).lower(), str(m).upper())


def load_stored():
    """Load maximum-budget ladder arrays indexed by dataset and model."""
    out = {}
    for f in sorted(glob.glob(os.path.join(SRC, 'paper_results*', 'ladder_ds*.json'))):
        for r in json.load(open(f)):
            k = (r['dataset'], _norm(r['model'])); q = r.get('q', 0)
            d = out.setdefault(k, {})
            if q >= d.get('_q_ladder', -1):
                d['_q_ladder'] = q
                for a in ('selfgrid', 'autolycus', 'ours', 'blind'):
                    if f'{a}_all' in r:
                        d[a] = list(r[f'{a}_all'])
    for f in sorted(glob.glob(os.path.join(SRC, 'paper_results*', 'defense_sweep_ds*.json'))):
        for r in json.load(open(f)):
            k = (r['dataset'], _norm(r['model'])); q = (r.get('q_list') or [0])[-1]
            d = out.setdefault(k, {})
            if q >= d.get('_q_sweep', -1):
                d['_q_sweep'] = q
                for a, key in (('defended', 'defended_all'), ('autolycus_sweep', 'autolycus_all')):
                    if key in r:
                        v = r[key]
                        d[a] = list(v[-1] if v and isinstance(v[0], list) else v)
    return out


def _best(a, b):
    """Return the higher-scoring baseline value for each paired observation."""
    if a is None:
        return b
    if b is None:
        return a
    return a if np.nanmean(np.asarray(a, float)) >= np.nanmean(np.asarray(b, float)) else b


def _wp(u, v):
    u, v = np.asarray(u, float), np.asarray(v, float)
    ok = ~(np.isnan(u) | np.isnan(v))
    if ok.sum() < 2:
        return float('nan'), 1.0
    u, v = u[ok], v[ok]
    return (float(u.mean() - v.mean()),
            float(wilcoxon(u, v).pvalue) if not np.allclose(u, v) else 1.0)


def _cell_bootstrap_interval(values, rng, n_resamples=100_000):
    """Return a percentile bootstrap interval for the mean across evaluated cells."""
    values = np.asarray(values, float)
    draws = rng.choice(values, size=(n_resamples, len(values)), replace=True).mean(axis=1)
    return tuple(np.quantile(draws, [0.025, 0.975]))


def main(out_dir):
    stored = load_stored()
    files = sorted(glob.glob(os.path.join(SRC, out_dir, 'ladder_eps_ds*.json')))
    if not files:
        raise SystemExit(f'no results in {out_dir} yet (written only after the final split).')
    rows = []
    for f in files:
        rows += json.load(open(f))

    # ---- pairing check: our autolycus arm must reproduce the stored one exactly ----
    print('PAIRING CHECK: re-run `autolycus` vs stored (both base LIME, eps=1)\n')
    ok_all = True
    for r in rows:
        k = (r['dataset'], _norm(r['model']))
        ref = stored.get(k, {}).get('autolycus') or stored.get(k, {}).get('autolycus_sweep')
        if ref is None:
            print(f"  {k[0]:10s} {k[1]:4s}  no stored autolycus to compare"); ok_all = False; continue
        same = np.allclose(np.asarray(r['autolycus_all'], float), np.asarray(ref, float), atol=5e-4)
        ok_all &= same
        if not same:
            print(f"  {k[0]:10s} {k[1]:4s}  MISMATCH")
            print(f"       ours  {r['autolycus_all']}")
            print(f"       store {ref}")
    print(f"  -> {'ALL MATCH, pairing verified' if ok_all else 'MISMATCH, do NOT pair across files'}\n")
    if not ok_all:
        raise SystemExit('pairing check failed; re-run defended/selfgrid in-script instead.')

    print('CONTRASTS  (pp; * p<.05, ** p<.01)\n')
    hdr = (f"{'dataset':10s} {'mdl':4s} {'epsx':>6s} | {'ours-def':>10s} {'eps1-def':>10s} | "
           f"{'ours-aut':>10s} {'ours-autE':>10s} {'ours-best':>10s} | {'autE-aut':>10s} | "
           f"{'ours-sg':>10s} | {'step':>9s}")
    print(hdr); print('-' * len(hdr))
    st = lambda p: '**' if p < .01 else ('*' if p < .05 else '')
    acc = {c: [] for c in ('ours-def', 'eps1-def', 'ours-aut', 'ours-autE', 'ours-best',
                           'eps1-aut', 'autE-aut', 'ours-sg', 'eps1-sg', 'step')}
    for r in sorted(rows, key=lambda r: (CONTINUOUS.index(r['dataset']) if r['dataset'] in CONTINUOUS else 9,
                                         MODELS.index(_norm(r['model'])) if _norm(r['model']) in MODELS else 9)):
        k = (r['dataset'], _norm(r['model'])); s = stored.get(k, {})
        O, O1, A = r['ours_all'], r['ours_eps1_all'], r['autolycus_all']
        AE = r.get('autolycus_eps_all')
        cells = {}
        for name, u, v in (('ours-def', O, s.get('defended')), ('eps1-def', O1, s.get('defended')),
                           ('ours-aut', O, A), ('ours-autE', O, AE), ('eps1-aut', O1, A),
                           ('autE-aut', AE, A),
                           ('ours-sg', O, s.get('selfgrid')), ('eps1-sg', O1, s.get('selfgrid')),
                           ('ours-best', O, _best(A, AE)),
                           ('step', O, O1)):
            if u is None:
                cells[name] = ('   n/a', None); continue
            if v is None:
                cells[name] = ('   n/a', None); continue
            d, p = _wp(u, v)
            cells[name] = (f"{100*d:+.1f}{st(p)}", d)
            if r['dataset'] in CONTINUOUS:
                acc[name].append(d)
        print(f"{r['dataset']:10s} {_norm(r['model']):4s} {r.get('eps_median', float('nan')):6.2f} | "
              + " ".join(f"{cells[c][0]:>10s}" for c in ('ours-def', 'eps1-def')) + " | "
              + " ".join(f"{cells[c][0]:>10s}" for c in ('ours-aut', 'ours-autE', 'ours-best')) + " | "
              + f"{cells['autE-aut'][0]:>10s}" + " | "
              + f"{cells['ours-sg'][0]:>10s}" + " | "
              + f"{cells['step'][0]:>9s}")
    print('-' * len(hdr))
    print(f"mean over continuous-feature cells (n per contrast in brackets):")
    for c in ('ours-def', 'eps1-def', 'ours-aut', 'ours-autE', 'ours-best',
              'eps1-aut', 'autE-aut', 'ours-sg', 'eps1-sg', 'step'):
        v = acc[c]
        if v:
            print(f"   {c:10s} {100*np.mean(v):+7.2f}pp  [{len(v)} cells]")

    # Descriptive uncertainty for the appendix table. Each draw resamples the 20 reported
    # dataset--model cells, not the individual target instantiations within a cell.
    print("\n95% percentile bootstrap intervals over continuous-feature cells (100,000 resamples):")
    ci_rng = np.random.default_rng(0)
    for c in ('ours-aut', 'ours-autE', 'eps1-aut', 'autE-aut', 'ours-best'):
        if acc[c]:
            lo, hi = _cell_bootstrap_interval(acc[c], ci_rng)
            print(f"   {c:10s} [{100*lo:+.2f}, {100*hi:+.2f}] pp")
    print("\npaper reports: ours-def +16.3, ours-aut +7.0, ours-sg +5.9 (continuous only)")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='paper_results_eps2')
    a = ap.parse_args()
    main(a.out)
