"""Assemble the 2x2 seed-source by grid-source shift decomposition.

It combines the grid-only and end-to-end restriction outputs into aggregate
seed, grid, and interaction contrasts.
"""
import os, glob, json, argparse
import numpy as np
from scipy.stats import wilcoxon

SRC = os.path.dirname(os.path.abspath(__file__))
ORDER = ['crop', 'pendigits']
MODELS = ['LR', 'NB', 'KNN', 'DT', 'RF']


def _st(p):
    return '**' if p < .01 else ('*' if p < .05 else '')


def _wp(u, v):
    u, v = np.asarray(u, float), np.asarray(v, float)
    ok = ~(np.isnan(u) | np.isnan(v))
    if ok.sum() < 2:
        return float('nan'), 1.0
    u, v = u[ok], v[ok]
    return float(u.mean() - v.mean()), (float(wilcoxon(u, v).pvalue) if not np.allclose(u, v) else 1.0)


def load(pattern, out_dir):
    out = {}
    for f in sorted(glob.glob(os.path.join(SRC, out_dir, pattern))):
        for r in json.load(open(f)):
            out[(r['dataset'], r['model'])] = r
    return out


def main(out_dir):
    g = load('lime_auxshift_ds*.json', out_dir)          # grid-only run
    e = load('lime_auxshift_e2e_ds*.json', out_dir)      # end-to-end run
    if not e:
        raise SystemExit(f'no e2e shards in {out_dir} yet (each written after its final split).')
    cells = [k for k in e if k in g]
    missing = sorted(set(e) - set(g))
    if missing:
        print(f'  note: no grid-only row for {missing}, excluded from the 2x2\n')

    print('CROSS-RUN PAIRING CHECK: aux_full is the same arm in both drivers\n')
    ok_all = True
    for k in sorted(cells):
        a = np.asarray(g[k]['aux_full_all'], float)
        b = np.asarray(e[k]['aux_full_all'], float)
        same = len(a) == len(b) and np.allclose(a, b, atol=5e-4)
        ok_all &= same
        print(f"  {k[0]:10s} {k[1]:4s}  {'MATCH' if same else 'MISMATCH'}")
        if not same:
            print(f'       grid-run {list(a)}\n       e2e-run  {list(b)}')
    print(f"  -> {'pairing verified' if ok_all else 'NOT paired'}\n")
    if not ok_all:
        raise SystemExit('cross-run pairing failed; the 2x2 would mix incomparable runs.')

    order = sorted(cells, key=lambda k: (ORDER.index(k[0]) if k[0] in ORDER else 9,
                                         MODELS.index(k[1]) if k[1] in MODELS else 9))
    for tag, lab in (('mid', 'MID'), ('low', 'LOW')):
        print(f'=== severity {lab}  (raw fidelity, pp; * p<.05, ** p<.01) ===\n')
        hdr = (f"{'dataset':10s} {'mdl':4s} {'seeds':>6s} | {'full':>6s} {'hold':>6s} {'seed':>6s} "
               f"{'e2e':>6s} {'ns':>6s} | {'d_grid':>8s} {'d_seed':>8s} {'d_e2e':>8s} "
               f"{'interact':>9s} | {'binedge':>9s}")
        print(hdr); print('-' * len(hdr))
        acc = {k: [] for k in ('d_grid', 'd_seed', 'd_e2e', 'interact', 'binedge')}
        for k in order:
            F = np.asarray(g[k]['aux_full_all'], float)
            H = np.asarray(g[k][f'hold_{tag}_all'], float)
            S = np.asarray(e[k][f'seed_{tag}_all'], float)
            E = np.asarray(e[k][f'e2e_{tag}_all'], float)
            NS = np.asarray(e[k][f'ns_{tag}_all'], float)
            dg, pg = _wp(H, F); ds, ps = _wp(S, F); de, pe = _wp(E, F)
            it, pit = _wp(E - S, H - F)          # paired interaction contrast
            be, pbe = _wp(E, NS)
            print(f"{k[0]:10s} {k[1]:4s} {e[k]['n_seeds'][f'e2e_{tag}']:6.0f} | "
                  f"{100*F.mean():6.1f} {100*H.mean():6.1f} {100*S.mean():6.1f} "
                  f"{100*E.mean():6.1f} {100*NS.mean():6.1f} | "
                  f"{f'{100*dg:+.1f}{_st(pg)}':>8s} {f'{100*ds:+.1f}{_st(ps)}':>8s} "
                  f"{f'{100*de:+.1f}{_st(pe)}':>8s} {f'{100*it:+.1f}{_st(pit)}':>9s} | "
                  f"{f'{100*be:+.1f}{_st(pbe)}':>9s}")
            for kk, v in (('d_grid', dg), ('d_seed', ds), ('d_e2e', de),
                          ('interact', it), ('binedge', be)):
                acc[kk].append(v)
        print('-' * len(hdr))
        print(f'mean over {len(order)} cells: ' + "  ".join(
            f'{kk} {100*np.mean(acc[kk]):+.2f}' for kk in
            ('d_grid', 'd_seed', 'd_e2e', 'interact', 'binedge')) + 'pp\n')

    print('d_grid: grid shifted, seeds intact. d_seed: seeds restricted, grid intact.')
    print('d_e2e: both, the realistic subpopulation attacker.')
    print('interact < 0 means the two penalties compound rather than add.')
    print('binedge = e2e - ns, the bin-edge gain when BOTH arms are seed-restricted. This is the '
          'only column that answers whether the channel still works for a shifted attacker; the '
          'gain columns in the grid-only table are measured against a full-seed no-bin-edge arm.')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='paper_results_shift')
    a = ap.parse_args()
    main(a.out)
