"""Run the end-to-end class-support restriction experiment.

The restricted attacker uses the same limited input pool for its seeds and its
grid, enabling a comparison with the grid-only restriction experiment.
"""
import os, json, glob, time, argparse, warnings, traceback
import numpy as np
import random
import lime, lime.lime_tabular
from scipy.stats import wilcoxon
from attack_utils import (load_dataset, load_model, mega_sample_generation, _fit_one_surrogate,
                          traverse_explanations_LIME)

SRC = os.path.dirname(os.path.abspath(__file__))
DATASETS = {1: 'crop', 9: 'pendigits'}
MODELS = {0: 'DT', 1: 'LR', 2: 'NB', 3: 'KNN', 4: 'RF'}
REPS = {'dt': 20, 'rdf': 10, 'nb': 1, 'lr': 1, 'knn': 1}
Q_BY_DS = {1: 1000, 9: 1000}
KEEP = {1: (11, 6), 9: (7, 4)}
SIZE, NFE, HMS, NSPLIT = 1, 3, 1, 10
ARMS = ['aux_full', 'ns_mid', 'ns_low', 'seed_mid', 'seed_low', 'e2e_mid', 'e2e_low']
# arm -> (seed pool, grid pool, use_threshold); 'full' = the whole auxiliary partition.
#
# The ns_* arms are the no-bin-edge control AT THE RESTRICTED SEED SET. Without them every gain is
# measured against the stored `nothresh`, which used the full seed set, so the seed penalty would be
# folded into a number labelled as a bin-edge gain. e2e_X - ns_X is the honest bin-edge gain for an
# attacker confined to a subpopulation.
#
# ns_X takes its explainer from the SAME holdout pool as e2e_X rather than from the target, so
# feature selection is held constant across the pair and only the bin edge varies. This differs from
# the published `nothresh`, which selects features using the target's explainer; RQ1 found attribution
# ranking inert, so the two conventions should agree, but the deviation is deliberate and stated.
ARM_CFG = {'aux_full': ('full', 'full', True),
           'ns_mid': ('hold_mid', 'hold_mid', False), 'ns_low': ('hold_low', 'hold_low', False),
           'seed_mid': ('hold_mid', 'full', True), 'seed_low': ('hold_low', 'full', True),
           'e2e_mid': ('hold_mid', 'hold_mid', True), 'e2e_low': ('hold_low', 'hold_low', True)}


def _fid(mn, V, P, tm, Xt, nc):
    """Fit a surrogate and return its fidelity against the target model."""
    if mn == 'nb':
        V = np.clip(np.asarray(V, float), 0, None)
    P = np.asarray(P); yt = tm.predict(Xt)
    if len(np.unique(P)) < 2:
        return float(np.mean(yt == (P[0] if len(P) else 0)))
    fs = []
    for k in range(REPS.get(mn, 1)):
        try:
            sm = _fit_one_surrogate(mn, np.asarray(V, float), P, nc, seed=k)
            fs.append(float(np.mean(yt == sm.predict(Xt))))
        except Exception:
            fs.append(float('nan'))
    return float(np.nanmean(fs)) if fs else float('nan')


def seed_from_pool(Xs, ys, nc, n_per_class):
    """Sample seeds from the available classes in a restricted auxiliary pool."""
    ys = np.asarray(ys)
    out = []
    for c in range(nc):
        idx = np.where(ys == c)[0]
        if len(idx) < n_per_class:
            continue                                  # attacker has never seen this class
        for j in random.sample(idx.tolist(), n_per_class):
            out.append(np.asarray(Xs[j], float))
    return out


def build_pools(n_rows, ys, keep_mid, keep_low, rng):
    """Construct class-restricted pools for one split."""
    ys = np.asarray(ys)
    idx_all = np.arange(n_rows)
    out = {'full': idx_all}
    for tag, k in (('mid', keep_mid), ('low', keep_low)):
        keep_cls = rng.choice(np.unique(ys), size=k, replace=False)
        hold = idx_all[np.isin(ys, keep_cls)]
        out[f'hold_{tag}'] = hold
        out[f'unif_{tag}'] = rng.choice(idx_all, size=len(hold), replace=False)
    return out


def load_stored(ds, model, topQ):
    cands = []
    for f in sorted(glob.glob(os.path.join(SRC, 'paper_results*', f'lime_auxdisc_shadow_ds{ds}.json'))):
        for r in json.load(open(f)):
            if r['model'] == model and r.get('q') == topQ and 'aux_all' in r:
                cands.append((f, r))
    if not cands:
        raise SystemExit(f'no stored RQ3 row for ds{ds} {model} at q={topQ}.')
    f, r = cands[-1]
    print(f'  reusing nothresh/tgt from {os.path.relpath(f, SRC)} '
          f'(thr_tgt {r["thr_tgt"]}, thr_aux {r["thr_aux"]})', flush=True)
    return r


def run_ds(ds, models, out_dir='paper_results_shift'):
    os.makedirs(os.path.join(SRC, out_dir), exist_ok=True)
    warnings.filterwarnings('ignore')
    topQ = Q_BY_DS[ds]
    fid = {m: {a: [] for a in ARMS} for m in models}
    nseed = {a: [] for a in ARMS}
    n_eval = None; verified = False; t0 = time.time()
    for s in range(NSPLIT):
        a1, a2 = load_dataset(ds, seed=s)
        X_train, _, y_train, _, X_test_t, X_test_s, _, y_test_s = a1
        nc = a2[2]; Xt = np.asarray(X_test_t.values, float); n_eval = len(Xt)
        lb = int((topQ // nc) * 0.5 + 1); ub = int((topQ // nc) * (nc + 0.5) + 1)
        Xs = X_test_s.values
        ys = np.asarray(y_test_s)
        pools = build_pools(len(Xs), ys, *KEEP[ds], np.random.default_rng(7000 + s))

        # Every seed draw starts from the SAME rng state, so the arms differ only in their pool.
        random.seed(s); np.random.seed(s)
        baseline = mega_sample_generation(Xs, y_test_s, nc, [SIZE], HMS)[0][0]
        if not verified:
            random.seed(s); np.random.seed(s)
            mine = seed_from_pool(Xs, ys, nc, SIZE)
            assert len(mine) == len(baseline) and np.allclose(np.asarray(mine, float),
                                                              np.asarray(baseline, float)), \
                'seed_from_pool does not reproduce mega_sample_generation on the full pool'
            print(f'  verified seed_from_pool == mega_sample_generation ({len(mine)} seeds)', flush=True)
            verified = True
        seeds = {'full': baseline}
        for tag in ('mid', 'low'):
            random.seed(s); np.random.seed(s)
            idx = pools[f'hold_{tag}']
            seeds[f'hold_{tag}'] = seed_from_pool(Xs[idx], ys[idx], nc, SIZE)

        grids = {}
        for tag in set(g for _, g, _ in ARM_CFG.values()):
            grids[tag] = lime.lime_tabular.LimeTabularExplainer(Xs[pools[tag]],
                                                                discretize_continuous=True)
        for a in ARMS:
            nseed[a].append(len(seeds[ARM_CFG[a][0]]))
        for m in models:
            try:
                tm, mn = load_model(m, X_train, y_train)
            except Exception:
                traceback.print_exc(); continue
            for arm in ARMS:
                sp, gp, ut = ARM_CFG[arm]
                random.seed(1000 + s); np.random.seed(1000 + s)   # paired, matches every other driver
                try:
                    V, P, _ = traverse_explanations_LIME(seeds[sp], grids[gp], tm, lb, ub, topQ,
                                                        NFE, a2, feature_select='explanation',
                                                        use_threshold=ut)
                    fid[m][arm].append(_fid(mn, V, P, tm, Xt, nc))
                except Exception:
                    traceback.print_exc(); fid[m][arm].append(float('nan'))
        print(f'  split {s} done ({round(time.time()-t0,1)}s)', flush=True)

    def wp(X, Y):
        X, Y = np.asarray(X, float), np.asarray(Y, float)
        n = min(len(X), len(Y))
        if len(X) != len(Y):
            print(f'  WARNING: pairing lengths differ ({len(X)} vs {len(Y)})', flush=True)
        X, Y = X[:n], Y[:n]
        ok = ~(np.isnan(X) | np.isnan(Y))
        d = round(float(np.nanmean(X) - np.nanmean(Y)), 4)
        if ok.sum() < 2:
            return d, float('nan')
        X, Y = X[ok], Y[ok]
        return d, (float(wilcoxon(X, Y).pvalue) if not np.allclose(X, Y) else 1.0)

    for m in models:
        mn = MODELS.get(m, str(m))
        st = load_stored(ds, mn, topQ)
        N = np.asarray(st['nothresh_all'], float)
        T = np.asarray(st['tgt_all'], float)
        row = {'dataset': DATASETS[ds], 'model': mn, 'q': topQ, 'nsplit': NSPLIT, 'n_eval': n_eval,
               'keep': list(KEEP[ds]), 'n_seeds': {a: float(np.mean(v)) for a, v in nseed.items()},
               'stored_aux_all': [round(x, 4) for x in st['aux_all']],
               'nothresh_all': [round(x, 4) for x in N], 'tgt_all': [round(x, 4) for x in T],
               'thr_tgt': st['thr_tgt']}
        for arm in ARMS:
            A = np.asarray(fid[m][arm], float)
            row[f'{arm}_all'] = [round(x, 4) for x in A]
            row[f'gain_{arm}'], row[f'gain_{arm}_p'] = wp(A, N)
        F = np.asarray(fid[m]['aux_full'], float)
        for tag in ('mid', 'low'):
            E = np.asarray(fid[m][f'e2e_{tag}'], float)
            row[f'seedpen_{tag}'], row[f'seedpen_{tag}_p'] = wp(np.asarray(fid[m][f'seed_{tag}'], float), F)
            row[f'e2epen_{tag}'], row[f'e2epen_{tag}_p'] = wp(E, F)
            # the honest bin-edge gain under the end-to-end threat model: both arms restricted
            row[f'binedge_{tag}'], row[f'binedge_{tag}_p'] = wp(E, np.asarray(fid[m][f'ns_{tag}'], float))
        path = os.path.join(SRC, out_dir, f'lime_auxshift_e2e_ds{ds}_{mn}.json')
        with open(path, 'w') as f:
            json.dump([row], f, indent=2)
        print(f"ROW {row['dataset']} {mn} tgt {row['thr_tgt']:+.4f} | "
              + " ".join(f"{a} {row[f'gain_{a}']:+.4f}" for a in ARMS)
              + f" | binedge_mid {row['binedge_mid']:+.4f} binedge_low {row['binedge_low']:+.4f}"
              + f" | seeds {row['n_seeds']}", flush=True)
    return True


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--ds', type=int, required=True)
    ap.add_argument('--models', type=str, required=True)
    ap.add_argument('--out', type=str, default='paper_results_shift')
    a = ap.parse_args()
    run_ds(a.ds, [int(x) for x in a.models.split(',') if x != ''], out_dir=a.out)
