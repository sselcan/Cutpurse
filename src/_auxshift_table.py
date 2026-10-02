"""Assemble the grid-fitting distribution-shift results.

The table contrasts uniform subsampling, class-support restriction, and the
service grid using the JSON shards produced by ``_lime_auxshift.py``.
"""
import os, glob, json, argparse
import numpy as np
from scipy.stats import wilcoxon

SRC = os.path.dirname(os.path.abspath(__file__))
ORDER = ['crop', 'pendigits']
MODELS = ['LR', 'NB', 'KNN', 'DT', 'RF']
POOLS = ['aux_full', 'unif_mid', 'hold_mid', 'unif_low', 'hold_low']


def _st(p):
    return '**' if p < .01 else ('*' if p < .05 else '')


def _wp(u, v):
    u, v = np.asarray(u, float), np.asarray(v, float)
    ok = ~(np.isnan(u) | np.isnan(v))
    if ok.sum() < 2:
        return float('nan'), 1.0
    u, v = u[ok], v[ok]
    return float(u.mean() - v.mean()), (float(wilcoxon(u, v).pvalue) if not np.allclose(u, v) else 1.0)


def main(out_dir):
    rows = []
    for f in sorted(glob.glob(os.path.join(SRC, out_dir, 'lime_auxshift_ds*_*.json'))):
        rows += json.load(open(f))
    if not rows:
        raise SystemExit(f'no shards in {out_dir} yet (each is written after its final split).')

    print('PAIRING CHECK: aux_full must reproduce the stored RQ3 `aux` arm\n')
    ok_all = True
    for r in rows:
        a = np.asarray(r['aux_full_all'], float)
        b = np.asarray(r['stored_aux_all'], float)
        same = len(a) == len(b) and np.allclose(a, b, atol=5e-4)
        ok_all &= same
        print(f"  {r['dataset']:10s} {r['model']:4s}  {'MATCH' if same else 'MISMATCH'}")
        if not same:
            print(f'       rerun  {list(a)}\n       stored {list(b)}')
    print(f"  -> {'pairing verified' if ok_all else 'NOT paired'}\n")
    if not ok_all:
        raise SystemExit('pairing check failed; nothing below would be comparable.')

    disp = {}
    for f in glob.glob(os.path.join(SRC, out_dir, 'grid_disp_ds*.json')):
        d = json.load(open(f))
        disp[d['dataset']] = {k: float(np.mean(v)) for k, v in d['disp'].items()}

    print('Bin-edge gain over no-bin-edge traversal (pp). tgt = target grid.\n')
    hdr = (f"{'dataset':10s} {'mdl':4s} {'tgt':>7s} {'full':>7s} | {'unifM':>7s} {'holdM':>7s} "
           f"{'shiftM':>9s} | {'unifL':>7s} {'holdL':>7s} {'shiftL':>9s} | {'tgt-holdL':>10s}")
    print(hdr); print('-' * len(hdr))
    acc = {k: [] for k in ('shift_mid', 'shift_low', 'adv_hold_mid', 'adv_hold_low',
                           'gain_hold_mid', 'gain_hold_low', 'gain_aux_full')}
    for r in sorted(rows, key=lambda r: (ORDER.index(r['dataset']) if r['dataset'] in ORDER else 9,
                                         MODELS.index(r['model']) if r['model'] in MODELS else 9)):
        N = np.asarray(r['nothresh_all'], float)
        T = np.asarray(r['tgt_all'], float)
        g = {p: _wp(np.asarray(r[f'{p}_all'], float), N) for p in POOLS}
        sm, smp = _wp(np.asarray(r['hold_mid_all'], float), np.asarray(r['unif_mid_all'], float))
        sl, slp = _wp(np.asarray(r['hold_low_all'], float), np.asarray(r['unif_low_all'], float))
        am, amp = _wp(T, np.asarray(r['hold_mid_all'], float))
        al, alp = _wp(T, np.asarray(r['hold_low_all'], float))
        print(f"{r['dataset']:10s} {r['model']:4s} {100*r['thr_tgt']:+7.1f} "
              f"{100*g['aux_full'][0]:+7.1f} | {100*g['unif_mid'][0]:+7.1f} "
              f"{100*g['hold_mid'][0]:+7.1f} {f'{100*sm:+.1f}{_st(smp)}':>9s} | "
              f"{100*g['unif_low'][0]:+7.1f} {100*g['hold_low'][0]:+7.1f} "
              f"{f'{100*sl:+.1f}{_st(slp)}':>9s} | {f'{100*al:+.1f}{_st(alp)}':>10s}")
        for k, v in (('shift_mid', sm), ('shift_low', sl), ('adv_hold_mid', am),
                     ('adv_hold_low', al), ('gain_hold_mid', g['hold_mid'][0]),
                     ('gain_hold_low', g['hold_low'][0]), ('gain_aux_full', g['aux_full'][0])):
            acc[k].append(v)
    print('-' * len(hdr))
    print(f'mean over {len(rows)} cells:')
    for k in ('gain_aux_full', 'gain_hold_mid', 'gain_hold_low',
              'shift_mid', 'shift_low', 'adv_hold_mid', 'adv_hold_low'):
        print(f'   {k:14s} {100*np.mean(acc[k]):+7.2f}pp')

    if disp:
        print('\nGrid displacement from the target grid, mean |dedge|/sigma (zero queries):')
        print(f"   {'dataset':10s} " + " ".join(f'{p:>10s}' for p in
                                               ['full', 'unif_mid', 'hold_mid', 'unif_low', 'hold_low']))
        for ds in ORDER:
            if ds in disp:
                print(f'   {ds:10s} ' + " ".join(f"{disp[ds][p]:10.4f}" for p in
                                                 ['full', 'unif_mid', 'hold_mid', 'unif_low', 'hold_low']))
    print('\nshiftX = holdX - unifX, the shift at matched pool size. Negative means the '
          'subpopulation grid is worse than a same-size i.i.d. one.')
    print('tgt-holdL positive means the target grid retains an advantage the shifted attacker '
          'cannot reproduce.')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='paper_results_shift')
    a = ap.parse_args()
    main(a.out)
