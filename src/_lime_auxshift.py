"""Run the grid-fitting distribution-shift sensitivity experiment.

Only the attacker-side pool used to fit the grid changes. Uniform subsamples
control for pool size, while class-support restriction induces the shift.
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
Q_BY_DS = {1: 1000, 9: 1000}                 # pendigits Table V is the Q=1000 run, not the ds9 default
KEEP = {1: (11, 6), 9: (7, 4)}               # classes kept at mid / low severity (of 17 and 10)
SIZE, NFE, HMS, NSPLIT = 1, 3, 1, 10         # n=1 = LIME native, matches _lime_auxdisc.py
ARMS = ['aux_full', 'unif_mid', 'unif_low', 'hold_mid', 'hold_low']


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


def load_stored(ds, model, topQ):
    """Load stored RQ3 reference results for one dataset--model cell."""
    cands = []
    for f in sorted(glob.glob(os.path.join(SRC, 'paper_results*', f'lime_auxdisc_shadow_ds{ds}.json'))):
        for r in json.load(open(f)):
            if r['model'] == model and r.get('q') == topQ and 'aux_all' in r:
                cands.append((f, r))
    if not cands:
        raise SystemExit(f'no stored RQ3 row for ds{ds} {model} at q={topQ}; cannot reuse nothresh/tgt.')
    f, r = cands[-1]
    print(f'  reusing nothresh/tgt from {os.path.relpath(f, SRC)} '
          f'(thr_tgt {r["thr_tgt"]}, thr_aux {r["thr_aux"]})', flush=True)
    return r


def build_pools(n_rows, ys, keep_mid, keep_low, rng):
    """Return row indices for full, uniform, and class-restricted pools."""
    ys = np.asarray(ys)
    idx_all = np.arange(n_rows)
    out = {'aux_full': idx_all}
    for tag, k in (('mid', keep_mid), ('low', keep_low)):
        keep_cls = rng.choice(np.unique(ys), size=k, replace=False)
        hold = idx_all[np.isin(ys, keep_cls)]
        out[f'hold_{tag}'] = hold
        out[f'unif_{tag}'] = rng.choice(idx_all, size=len(hold), replace=False)
    return out


def run_ds(ds, models, out_dir='paper_results_shift'):
    os.makedirs(os.path.join(SRC, out_dir), exist_ok=True)
    warnings.filterwarnings('ignore')
    topQ = Q_BY_DS[ds]
    fid = {m: {a: [] for a in ARMS} for m in models}
    pool_rows = {a: [] for a in ARMS}
    n_eval = None; t0 = time.time()
    for s in range(NSPLIT):
        a1, a2 = load_dataset(ds, seed=s)
        X_train, _, y_train, _, X_test_t, X_test_s, _, y_test_s = a1
        nc = a2[2]; Xt = np.asarray(X_test_t.values, float); n_eval = len(Xt)
        lb = int((topQ // nc) * 0.5 + 1); ub = int((topQ // nc) * (nc + 0.5) + 1)
        random.seed(s); np.random.seed(s)
        mega = mega_sample_generation(X_test_s.to_numpy(), y_test_s, nc, [SIZE], HMS)
        Xs = X_test_s.values
        # independent generator: does NOT consume the global stream the attack is paired on
        pools = build_pools(len(Xs), y_test_s, *KEEP[ds], np.random.default_rng(7000 + s))
        expl = {}
        for arm in ARMS:
            sub = Xs[pools[arm]]
            pool_rows[arm].append(int(len(sub)))
            expl[arm] = lime.lime_tabular.LimeTabularExplainer(sub, discretize_continuous=True)
        for m in models:
            try:
                tm, mn = load_model(m, X_train, y_train)
            except Exception:
                traceback.print_exc(); continue
            for arm in ARMS:
                random.seed(1000 + s); np.random.seed(1000 + s)   # paired, matches _lime_auxdisc.py
                try:
                    V, P, _ = traverse_explanations_LIME(mega[0][0], expl[arm], tm, lb, ub, topQ,
                                                        NFE, a2, feature_select='explanation',
                                                        use_threshold=True)
                    fid[m][arm].append(_fid(mn, V, P, tm, Xt, nc))
                except Exception:
                    traceback.print_exc(); fid[m][arm].append(float('nan'))
        print(f'  split {s} done ({round(time.time()-t0,1)}s)', flush=True)

    def wp(X, Y):
        X, Y = np.asarray(X, float), np.asarray(Y, float)
        if len(X) != len(Y):        # only a smoke run at reduced NSPLIT should ever land here
            n = min(len(X), len(Y))
            print(f'  WARNING: pairing lengths differ ({len(X)} vs {len(Y)}), '
                  f'comparing first {n}', flush=True)
            X, Y = X[:n], Y[:n]
        ok = ~(np.isnan(X) | np.isnan(Y))
        d = round(float(np.nanmean(X) - np.nanmean(Y)), 4)
        if ok.sum() < 2:                      # smoke runs at NSPLIT=1 have no paired test
            return d, float('nan')
        X, Y = X[ok], Y[ok]
        return d, (float(wilcoxon(X, Y).pvalue) if not np.allclose(X, Y) else 1.0)
    for m in models:
        mn = MODELS.get(m, str(m))
        st = load_stored(ds, mn, topQ)
        N = np.asarray(st['nothresh_all'], float)
        T = np.asarray(st['tgt_all'], float)
        row = {'dataset': DATASETS[ds], 'model': mn, 'q': topQ, 'nsplit': NSPLIT, 'n_eval': n_eval,
               'keep': list(KEEP[ds]), 'pool_rows': {a: float(np.mean(v)) for a, v in pool_rows.items()},
               'stored_aux_all': [round(x, 4) for x in st['aux_all']],
               'nothresh_all': [round(x, 4) for x in N], 'tgt_all': [round(x, 4) for x in T],
               'thr_tgt': st['thr_tgt']}
        for arm in ARMS:
            A = np.asarray(fid[m][arm], float)
            row[f'{arm}_all'] = [round(x, 4) for x in A]
            row[f'gain_{arm}'], row[f'gain_{arm}_p'] = wp(A, N)     # bin-edge gain over no-bin-edge
            row[f'adv_{arm}'], row[f'adv_{arm}_p'] = wp(T, A)       # target advantage tgt - aux
        # shift attributable to the shift rather than the row loss
        for tag in ('mid', 'low'):
            row[f'shift_{tag}'], row[f'shift_{tag}_p'] = wp(np.asarray(fid[m][f'hold_{tag}'], float),
                                                            np.asarray(fid[m][f'unif_{tag}'], float))
        path = os.path.join(SRC, out_dir, f'lime_auxshift_ds{ds}_{mn}.json')
        with open(path, 'w') as f:
            json.dump([row], f, indent=2)
        print(f"ROW {row['dataset']} {mn} tgt {row['thr_tgt']:+.4f} | "
              + " ".join(f"{a} {row[f'gain_{a}']:+.4f}" for a in ARMS)
              + f" | shift_mid {row['shift_mid']:+.4f} shift_low {row['shift_low']:+.4f}", flush=True)
    return True


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--ds', type=int, required=True)
    ap.add_argument('--models', type=str, required=True)
    ap.add_argument('--out', type=str, default='paper_results_shift')
    a = ap.parse_args()
    run_ds(a.ds, [int(x) for x in a.models.split(',') if x != ''], out_dir=a.out)
