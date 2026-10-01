"""Does the auxiliary grid still match the target grid once the aux pool is NOT an i.i.d. split?

RQ3's auxiliary pool is an i.i.d. split of the same dataset, so its quartiles coincide with the
target's essentially by construction. This script quantifies that, and quantifies how far the grid
moves once the attacker's pool is a SUBPOPULATION instead: we hold out whole classes from the pool
used to fit the discretizer.

No target queries are made. LIME's QuartileDiscretizer places bin edges at the 25/50/75th
percentiles of its background data (verified against `disc.bins()` below), so grid displacement is a
property of the pool alone and costs nothing to measure.

Pools, all fitted on feature columns only:
  full        the whole aux partition            (the published RQ3 `aux` arm)
  unif_mid    uniform subsample, size-matched to hold_mid
  unif_low    uniform subsample, size-matched to hold_low
  hold_mid    class holdout, keeps KEEP[ds][0] of the classes
  hold_low    class holdout, keeps KEEP[ds][1] of the classes

The uniform pools exist because holding out classes also removes rows, and pool SIZE alone already
moves the grid (the aux-size sweep shows this). Displacement attributable to the shift is
hold - unif at matched size, not hold - full.

Displacement is reported as mean |edge_pool - edge_target| / sigma_feature, sigma from X_train, so it
is comparable across features and datasets.

    python _aux_shift_grid.py
    python _aux_shift_grid.py --ds 1 --nsplit 10
"""
import argparse, warnings, json, os
import numpy as np
from attack_utils import load_dataset

DATASETS = {1: 'crop', 9: 'pendigits'}
KEEP = {1: (11, 6), 9: (7, 4)}      # classes kept at the mid / low severity (of 17 and 10)
POOLS = ['full', 'unif_mid', 'unif_low', 'hold_mid', 'hold_low']


def cuts(pool):
    """LIME's QuartileDiscretizer edges: the 25/50/75th percentiles, per column. (3, n_features)."""
    return np.percentile(np.asarray(pool, float), [25, 50, 75], axis=0)


def _verify_against_lime(pool):
    """The percentile shortcut must equal what LIME actually fits. Checked once, not assumed."""
    import lime.lime_tabular
    ex = lime.lime_tabular.LimeTabularExplainer(np.asarray(pool, float), discretize_continuous=True)
    got = ex.discretizer.bins(np.asarray(pool, float), None)
    ref = cuts(pool)
    for f, edges in zip(ex.discretizer.to_discretize, got):
        assert np.allclose(np.asarray(edges, float), ref[:, f]), f'feature {f}: {edges} vs {ref[:, f]}'
    return len(got)


def build_pools(Xs, ys, nc, keep_mid, keep_low, rng):
    """(name -> row index array). Holdout classes are redrawn per split by the caller's rng."""
    ys = np.asarray(ys)
    idx_all = np.arange(len(Xs))
    out = {'full': idx_all}
    for tag, k in (('mid', keep_mid), ('low', keep_low)):
        keep_cls = rng.choice(np.unique(ys), size=k, replace=False)
        hold = idx_all[np.isin(ys, keep_cls)]
        out[f'hold_{tag}'] = hold
        # size-matched uniform control drawn from the FULL pool
        out[f'unif_{tag}'] = rng.choice(idx_all, size=len(hold), replace=False)
    return out


def run(ds, nsplit):
    warnings.filterwarnings('ignore')
    disp = {p: [] for p in POOLS}          # mean |dcut|/sigma per split
    dmax = {p: [] for p in POOLS}
    ncol = {p: [] for p in POOLS}          # features whose 3 edges collapse to fewer under np.unique
    sizes = {p: [] for p in POOLS}
    verified = False
    for s in range(nsplit):
        a1, a2 = load_dataset(ds, seed=s)
        X_train, _, _, _, _, X_test_s, _, y_test_s = a1
        nc = a2[2]
        Xs = X_test_s.values.astype(float)
        sigma = X_train.values.astype(float).std(axis=0)
        sigma[sigma == 0] = 1.0
        tgt = cuts(X_train.values.astype(float))
        if not verified:
            n = _verify_against_lime(Xs)
            print(f'  verified percentile==LIME bins on {n} features', flush=True)
            verified = True
        rng = np.random.default_rng(7000 + s)
        pools = build_pools(Xs, y_test_s, nc, *KEEP[ds], rng)
        for p in POOLS:
            sub = Xs[pools[p]]
            c = cuts(sub)
            d = np.abs(c - tgt) / sigma
            disp[p].append(float(d.mean()))
            dmax[p].append(float(d.max()))
            ncol[p].append(int(sum(len(np.unique(c[:, f])) < 3 for f in range(c.shape[1]))))
            sizes[p].append(len(sub))
    name = DATASETS[ds]
    print(f'\n{name}: grid displacement from the TARGET grid, mean |dedge|/sigma over '
          f'{nsplit} splits\n')
    print(f"{'pool':10s} {'rows':>6s} {'mean':>8s} {'max':>8s} {'collapsed':>10s}")
    print('-' * 46)
    for p in POOLS:
        print(f'{p:10s} {np.mean(sizes[p]):6.0f} {np.mean(disp[p]):8.4f} '
              f'{np.mean(dmax[p]):8.4f} {np.mean(ncol[p]):10.1f}')
    print('-' * 46)
    for tag in ('mid', 'low'):
        h, u = np.mean(disp[f'hold_{tag}']), np.mean(disp[f'unif_{tag}'])
        print(f'  shift-attributable at {tag:3s}: hold - unif = {h - u:+.4f} sigma '
              f'(hold {h:.4f}, unif {u:.4f})')
    os.makedirs('paper_results_shift', exist_ok=True)
    with open(f'paper_results_shift/grid_disp_ds{ds}.json', 'w') as f:
        json.dump({'dataset': name, 'nsplit': nsplit, 'keep': list(KEEP[ds]),
                   'disp': disp, 'dmax': dmax, 'collapsed': ncol, 'sizes': sizes}, f, indent=2)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--ds', type=int, default=None)
    ap.add_argument('--nsplit', type=int, default=10)
    a = ap.parse_args()
    for ds in ([a.ds] if a.ds else [1, 9]):
        run(ds, a.nsplit)
