"""LIME aux-discretizer with ONLINE UPDATE. The attacker starts with only its seed set (~10-17
samples) but refits the LIME discretizer on its growing query set every ONLINE_EVERY queries, so the
quartiles sharpen as the attack proceeds. Tests whether the residual leak seen with the tiny static
seed set (esp. pendigits) closes once the attacker uses its own accumulating queries.

Three paired arms, base Autolycus LIME, n=1, 10 splits:
  nothresh   = tgt disc, use_threshold=False
  tgt        = tgt disc, use_threshold=True                        (real service)
  aux_online = seed-set disc, use_threshold=True, refit online      (self-computed + updated)
leak = tgt - aux_online. If ~0, the discretization is fully self-computable online.

    python _lime_auxdisc_online.py --ds 9 --out paper_results
"""
import os, json, time, argparse, warnings, traceback
import numpy as np
import random
import lime, lime.lime_tabular
from scipy.stats import wilcoxon
from attack_utils import (load_dataset, load_model, mega_sample_generation, _fit_one_surrogate,
                          traverse_explanations_LIME)

DATASETS = {1: 'crop', 9: 'pendigits'}
MODELS = {0: 'DT', 1: 'LR', 2: 'NB', 3: 'KNN', 4: 'RF'}
DEFAULT_MODELS = [1, 2, 3, 0, 4]
REPS = {'dt': 20, 'rdf': 10, 'nb': 1, 'lr': 1, 'knn': 1}
Q_BY_DS = {1: [1000], 9: [500]}
SIZE, NFE, HMS, NSPLIT, ONLINE_EVERY = 1, 3, 1, 10, 50


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


def run_ds(ds, models=DEFAULT_MODELS, out_dir='paper_results'):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f'lime_auxdisc_online_ds{ds}.json')
    warnings.filterwarnings('ignore')
    topQ = Q_BY_DS[ds][-1]
    ARMS = ['nothresh', 'tgt', 'aux_online']
    fid = {m: {a: [] for a in ARMS} for m in models}
    n_eval = None; t0 = time.time()
    for s in range(NSPLIT):
        a1, a2 = load_dataset(ds, seed=s)
        X_train, X_test, y_train, y_test, X_test_t, X_test_s, y_test_t, y_test_s = a1
        nc = a2[2]; Xt = np.asarray(X_test_t.values, float); n_eval = len(Xt)
        lb = int((topQ // nc) * 0.5 + 1); ub = int((topQ // nc) * (nc + 0.5) + 1)
        random.seed(s); np.random.seed(s)
        mega = mega_sample_generation(X_test_s.to_numpy(), y_test_s, nc, [SIZE], HMS)
        seed_set = np.asarray(mega[0][0], float)
        expl_tgt = lime.lime_tabular.LimeTabularExplainer(X_train.values, discretize_continuous=True)
        expl_seed = lime.lime_tabular.LimeTabularExplainer(seed_set, discretize_continuous=True)
        # arm -> (explainer, use_threshold, online_every)
        cfg = {'nothresh': (expl_tgt, False, None), 'tgt': (expl_tgt, True, None),
               'aux_online': (expl_seed, True, ONLINE_EVERY)}
        for m in models:
            mn = MODELS.get(m, str(m))
            tm, mn = load_model(m, X_train, y_train)
            for arm in ARMS:
                expl, ut, oe = cfg[arm]
                random.seed(2000 + s); np.random.seed(2000 + s)
                V, P, _ = traverse_explanations_LIME(mega[0][0], expl, tm, lb, ub, topQ, NFE, a2,
                                                     feature_select='explanation', use_threshold=ut,
                                                     online_disc_every=oe)
                fid[m][arm].append(_fid(mn, V, P, tm, Xt, nc))
        print(f'  split {s} done ({round(time.time()-t0,1)}s)', flush=True)
    rows = []
    for m in models:
        mn = MODELS.get(m, str(m))
        N = np.array(fid[m]['nothresh'], float); T = np.array(fid[m]['tgt'], float); A = np.array(fid[m]['aux_online'], float)
        wp = lambda X, Y: (round(float(np.mean(X) - np.mean(Y)), 4),
                           float(wilcoxon(X, Y).pvalue) if not np.allclose(X, Y) else 1.0)
        row = {'dataset': DATASETS[ds], 'model': mn, 'q': topQ, 'nsplit': NSPLIT, 'n_eval': n_eval,
               'online_every': ONLINE_EVERY,
               'nothresh_mean': round(float(np.mean(N)), 4), 'tgt_mean': round(float(np.mean(T)), 4),
               'aux_mean': round(float(np.mean(A)), 4),
               'thr_tgt': wp(T, N)[0], 'thr_tgt_p': wp(T, N)[1],
               'thr_aux': wp(A, N)[0], 'thr_aux_p': wp(A, N)[1],
               'leak': wp(T, A)[0], 'leak_p': wp(T, A)[1],
               'nothresh_all': [round(x, 4) for x in N], 'tgt_all': [round(x, 4) for x in T],
               'aux_all': [round(x, 4) for x in A]}
        rows.append(row)
        print('ROW', row['dataset'], row['model'], 'thr_tgt', row['thr_tgt'], 'thr_aux', row['thr_aux'], 'leak', row['leak'], flush=True)
        with open(path, 'w') as f:
            json.dump(rows, f, indent=2)
    return rows


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--ds', type=int, required=True)
    ap.add_argument('--models', type=str, default=','.join(map(str, DEFAULT_MODELS)))
    ap.add_argument('--out', type=str, default='paper_results')
    a = ap.parse_args()
    mods = [int(x) for x in a.models.split(',') if x != '']
    run_ds(a.ds, models=mods, out_dir=a.out)
