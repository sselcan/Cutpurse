"""Run the RQ3 target-grid versus attacker-grid comparison.

The paired base-traversal configurations use no bin edge, the service grid, or
an attacker-fitted grid. Per-split results are written as JSON.
"""
import os, json, time, argparse, warnings, traceback
import numpy as np
import random
import lime, lime.lime_tabular
from scipy.stats import wilcoxon
from attack_utils import (load_dataset, load_model, mega_sample_generation, _fit_one_surrogate,
                          traverse_explanations_LIME)

DATASETS = {1: 'crop', 3: 'breast', 4: 'nursery', 5: 'mushroom', 9: 'pendigits', 10: 'letter', 11: 'waveform', 12: 'segment'}
MODELS = {0: 'DT', 1: 'LR', 2: 'NB', 3: 'KNN', 4: 'RF'}
DEFAULT_MODELS = [1, 2, 3, 0, 4]
REPS = {'dt': 20, 'rdf': 10, 'nb': 1, 'lr': 1, 'knn': 1}
Q_BY_DS = {1: [1000], 3: [100], 4: [1000], 5: [1000], 9: [500], 10: [500], 11: [1000], 12: [500]}
SIZE, NFE, HMS, NSPLIT = 1, 3, 1, 10   # n=1 = LIME native


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


def run_ds(ds, models=DEFAULT_MODELS, out_dir='paper_results', aux='shadow'):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f'lime_auxdisc_{aux}_ds{ds}.json')  # aux: 'shadow'=X_test_s pool, 'seed'=seed set
    warnings.filterwarnings('ignore')
    topQ = Q_BY_DS[ds][-1]
    ARMS = ['nothresh', 'tgt', 'aux']
    fid = {m: {a: [] for a in ARMS} for m in models}
    n_eval = None; t0 = time.time()
    for s in range(NSPLIT):
        a1, a2 = load_dataset(ds, seed=s)
        X_train, X_test, y_train, y_test, X_test_t, X_test_s, y_test_t, y_test_s = a1
        nc = a2[2]; Xt = np.asarray(X_test_t.values, float); n_eval = len(Xt)
        lb = int((topQ // nc) * 0.5 + 1); ub = int((topQ // nc) * (nc + 0.5) + 1)
        random.seed(s); np.random.seed(s)
        mega = mega_sample_generation(X_test_s.to_numpy(), y_test_s, nc, [SIZE], HMS)
        seed_set = np.asarray(mega[0][0], float)   # the attacker's actual seeds: SIZE per class, feature-only
        aux_data = seed_set if aux == 'seed' else X_test_s.values   # 'shadow' = full aux pool
        expl_tgt = lime.lime_tabular.LimeTabularExplainer(X_train.values, discretize_continuous=True)
        expl_aux = lime.lime_tabular.LimeTabularExplainer(aux_data, discretize_continuous=True)
        # (arm -> explainer, use_threshold)
        arm_cfg = {'nothresh': (expl_tgt, False), 'tgt': (expl_tgt, True), 'aux': (expl_aux, True)}
        for m in models:
            mn = MODELS.get(m, str(m))
            tm, mn = load_model(m, X_train, y_train)
            for arm in ARMS:
                expl, ut = arm_cfg[arm]
                random.seed(1000 + s); np.random.seed(1000 + s)   # harmonised with _autolycus_ablation_ms.py so that
                #   thr_tgt here == thr_delta there (same arms, same seeds) -- a cross-check between tables   # paired across arms
                V, P, _ = traverse_explanations_LIME(mega[0][0], expl, tm, lb, ub, topQ, NFE, a2,
                                                     feature_select='explanation', use_threshold=ut)
                fid[m][arm].append(_fid(mn, V, P, tm, Xt, nc))
        print(f'  split {s} done ({round(time.time()-t0,1)}s)', flush=True)
    rows = []
    for m in models:
        mn = MODELS.get(m, str(m))
        N = np.array(fid[m]['nothresh'], float); T = np.array(fid[m]['tgt'], float); A = np.array(fid[m]['aux'], float)
        wp = lambda X, Y: (round(float(np.mean(X) - np.mean(Y)), 4),
                           float(wilcoxon(X, Y).pvalue) if not np.allclose(X, Y) else 1.0)
        row = {'dataset': DATASETS[ds], 'model': mn, 'q': topQ, 'nsplit': NSPLIT, 'n_eval': n_eval,
               'nothresh_mean': round(float(np.mean(N)), 4), 'tgt_mean': round(float(np.mean(T)), 4),
               'aux_mean': round(float(np.mean(A)), 4),
               'thr_tgt': wp(T, N)[0], 'thr_tgt_p': wp(T, N)[1],   # threshold gain w/ target disc
               'thr_aux': wp(A, N)[0], 'thr_aux_p': wp(A, N)[1],   # threshold gain w/ aux disc
               'leak': wp(T, A)[0], 'leak_p': wp(T, A)[1],          # tgt - aux (privileged advantage)
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
    ap.add_argument('--q', type=int, default=None,
                    help='override the per-dataset query budget')
    ap.add_argument('--aux', type=str, default='shadow', choices=['shadow', 'seed'])
    a = ap.parse_args()
    if a.q:
        Q_BY_DS[a.ds] = [a.q]
    mods = [int(x) for x in a.models.split(',') if x != '']
    run_ds(a.ds, models=mods, out_dir=a.out, aux=a.aux)
