"""Compare service and attacker grids under class-support restriction.

Both arms use the same restricted seed pool, so their contrast isolates the
effect of the grid source.
"""
import os, json, glob, time, argparse, warnings, traceback
import numpy as np
import random
import lime, lime.lime_tabular
from scipy.stats import wilcoxon
from attack_utils import (load_dataset, load_model, mega_sample_generation, _fit_one_surrogate,
                          traverse_explanations_LIME)
from _lime_auxshift_e2e import seed_from_pool, build_pools, KEEP, Q_BY_DS, DATASETS, MODELS, REPS

SRC = os.path.dirname(os.path.abspath(__file__))
SIZE, NFE, HMS, NSPLIT = 1, 3, 1, 10
ARMS = ['aux_full', 'tgt_mid', 'tgt_low']
# arm -> (seed pool, grid source); 'target' = a discretizer on X_train, as the published `tgt` arm
ARM_CFG = {'aux_full': ('full', 'full'), 'tgt_mid': ('hold_mid', 'target'),
           'tgt_low': ('hold_low', 'target')}


def _fid(mn, V, P, tm, Xt, nc):
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


def run_ds(ds, models, out_dir='paper_results_shift'):
    os.makedirs(os.path.join(SRC, out_dir), exist_ok=True)
    warnings.filterwarnings('ignore')
    topQ = Q_BY_DS[ds]
    fid = {m: {a: [] for a in ARMS} for m in models}
    nseed = {a: [] for a in ARMS}
    n_eval = None; t0 = time.time()
    for s in range(NSPLIT):
        a1, a2 = load_dataset(ds, seed=s)
        X_train, _, y_train, _, X_test_t, X_test_s, _, y_test_s = a1
        nc = a2[2]; Xt = np.asarray(X_test_t.values, float); n_eval = len(Xt)
        lb = int((topQ // nc) * 0.5 + 1); ub = int((topQ // nc) * (nc + 0.5) + 1)
        Xs = X_test_s.values
        ys = np.asarray(y_test_s)
        pools = build_pools(len(Xs), ys, *KEEP[ds], np.random.default_rng(7000 + s))

        random.seed(s); np.random.seed(s)
        seeds = {'full': mega_sample_generation(Xs, y_test_s, nc, [SIZE], HMS)[0][0]}
        for tag in ('mid', 'low'):
            random.seed(s); np.random.seed(s)
            idx = pools[f'hold_{tag}']
            seeds[f'hold_{tag}'] = seed_from_pool(Xs[idx], ys[idx], nc, SIZE)

        grids = {'full': lime.lime_tabular.LimeTabularExplainer(Xs[pools['full']],
                                                               discretize_continuous=True),
                 'target': lime.lime_tabular.LimeTabularExplainer(X_train.values,
                                                                  discretize_continuous=True)}
        for a in ARMS:
            nseed[a].append(len(seeds[ARM_CFG[a][0]]))
        for m in models:
            try:
                tm, mn = load_model(m, X_train, y_train)
            except Exception:
                traceback.print_exc(); continue
            for arm in ARMS:
                sp, gp = ARM_CFG[arm]
                random.seed(1000 + s); np.random.seed(1000 + s)   # paired, matches every other driver
                try:
                    V, P, _ = traverse_explanations_LIME(seeds[sp], grids[gp], tm, lb, ub, topQ,
                                                        NFE, a2, feature_select='explanation',
                                                        use_threshold=True)
                    fid[m][arm].append(_fid(mn, V, P, tm, Xt, nc))
                except Exception:
                    traceback.print_exc(); fid[m][arm].append(float('nan'))
        print(f'  split {s} done ({round(time.time()-t0,1)}s)', flush=True)

    def wp(X, Y):
        X, Y = np.asarray(X, float), np.asarray(Y, float)
        ok = ~(np.isnan(X) | np.isnan(Y))
        d = round(float(np.nanmean(X) - np.nanmean(Y)), 4)
        if ok.sum() < 2:
            return d, float('nan')
        X, Y = X[ok], Y[ok]
        return d, (float(wilcoxon(X, Y).pvalue) if not np.allclose(X, Y) else 1.0)

    for m in models:
        mn = MODELS.get(m, str(m))
        row = {'dataset': DATASETS[ds], 'model': mn, 'q': topQ, 'nsplit': NSPLIT, 'n_eval': n_eval,
               'keep': list(KEEP[ds]), 'n_seeds': {a: float(np.mean(v)) for a, v in nseed.items()}}
        for arm in ARMS:
            row[f'{arm}_all'] = [round(x, 4) for x in fid[m][arm]]
            row[f'{arm}_mean'] = round(float(np.nanmean(fid[m][arm])), 4)
        path = os.path.join(SRC, out_dir, f'lime_auxshift_tgt_ds{ds}_{mn}.json')
        with open(path, 'w') as f:
            json.dump([row], f, indent=2)
        print(f"ROW {row['dataset']} {mn} | "
              + " ".join(f"{a} {row[f'{a}_mean']:.4f}" for a in ARMS)
              + f" | seeds {row['n_seeds']}", flush=True)
    return True


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--ds', type=int, required=True)
    ap.add_argument('--models', type=str, required=True)
    ap.add_argument('--out', type=str, default='paper_results_shift')
    a = ap.parse_args()
    run_ds(a.ds, [int(x) for x in a.models.split(',') if x != ''], out_dir=a.out)
