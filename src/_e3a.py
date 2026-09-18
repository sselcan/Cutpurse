"""E3a / H2: is the LIME bin-edge threshold load-bearing?
Within LIME3 (cap=15, main setting n=1,k=3) toggle ONLY whether the attack uses the bin edge:
  thr_on  = snap the selected feature to its bin edge (current method)
  thr_off = ignore the edge, step from the current value (what SHAP is forced to do)
Everything else identical => gap (on - off) = the threshold channel. Swept over budget to catch a
low-budget efficiency effect that a large budget's bisection might wash out. Paired, mean-agg,
Wilcoxon at top budget.

    python _e3a.py --ds 2 --out paper_results
"""
import os, json, time, argparse, warnings, traceback
import numpy as np
import random
import lime, lime.lime_tabular
from scipy.stats import wilcoxon
from attack_utils import (load_dataset, load_model, mega_sample_generation, _fit_one_surrogate,
                          traverse_explanations_LIME3)

DATASETS = {0: 'iris', 1: 'crop', 2: 'adult', 3: 'breast', 4: 'nursery', 5: 'mushroom'}
MODELS   = {0: 'DT', 1: 'LR', 2: 'NB', 3: 'KNN', 4: 'RF'}
DEFAULT_MODELS = [1, 2, 3, 0, 4]
REPS = {'dt': 20, 'rdf': 10, 'nb': 1, 'lr': 1, 'knn': 1}
Q_BY_DS = {0: [25, 50, 100], 1: [100, 250, 500, 1000], 2: [100, 250, 500, 1000],
           3: [25, 50, 100], 4: [100, 250, 500, 1000], 5: [100, 250, 500, 1000]}
SIZE, NFE, CAP, DIV_FRAC, HMS, SEED = 1, 3, 15, 0.3, 10, 0


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


def _lime3(seed_set, expl, tm, mn, lb, ub, Q, a2, X_train, y_train, use_threshold):
    return traverse_explanations_LIME3(seed_set, expl, tm, lb, ub, Q, NFE, a2, mn, X_train, y_train,
                                       use_diverse=True, diverse_method='lime_threshold',
                                       div_frac=DIV_FRAC, div_cap=CAP, use_threshold=use_threshold)


def run_ds(ds, models=DEFAULT_MODELS, out_dir='paper_results'):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f'e3a_ds{ds}.json')
    warnings.filterwarnings('ignore')
    q_list = Q_BY_DS[ds]
    a1, a2 = load_dataset(ds)
    X_train, X_test, y_train, y_test, X_test_t, X_test_s, y_test_t, y_test_s = a1
    nc = a2[2]; Xt = np.asarray(X_test_t.values, float)
    rows = []
    for m in models:
        mn = MODELS.get(m, str(m)); t0 = time.time()
        try:
            random.seed(SEED); np.random.seed(SEED)
            tm, mn = load_model(m, X_train, y_train)
            expl = lime.lime_tabular.LimeTabularExplainer(X_train.values, discretize_continuous=True)
            mega = mega_sample_generation(X_test_s.to_numpy(), y_test_s, nc, [SIZE], HMS)
            on = {Q: [] for Q in q_list}; off = {Q: [] for Q in q_list}
            for Q in q_list:
                lb = int((Q // nc) * 0.5 + 1); ub = int((Q // nc) * (nc + 0.5) + 1)
                for i in range(HMS):
                    random.seed(SEED + 1 + i); np.random.seed(SEED + 1 + i)
                    V, P, _ = _lime3(mega[i][0], expl, tm, mn, lb, ub, Q, a2, X_train, y_train, True)
                    on[Q].append(_fid(mn, V, P, tm, Xt, nc))
                    random.seed(SEED + 1 + i); np.random.seed(SEED + 1 + i)
                    V, P, _ = _lime3(mega[i][0], expl, tm, mn, lb, ub, Q, a2, X_train, y_train, False)
                    off[Q].append(_fid(mn, V, P, tm, Xt, nc))

            def pack(d):
                return {'fid_mean': [round(float(np.mean(d[Q])), 4) for Q in q_list],
                        'fid_all':  [[round(x, 4) for x in d[Q]] for Q in q_list]}
            row = {'dataset': DATASETS[ds], 'model': mn, 'q_cap': q_list,
                   'thr_on': pack(on), 'thr_off': pack(off), 'secs': round(time.time() - t0, 1)}
            topQ = q_list[-1]
            A = np.array(on[topQ], float); B = np.array(off[topQ], float)
            row['delta_top'] = round(float(A.mean() - B.mean()), 4)
            row['wilcoxon_p_top'] = (float(wilcoxon(A, B).pvalue) if not np.allclose(A, B) else 1.0)
        except Exception as e:
            traceback.print_exc()
            row = {'dataset': DATASETS[ds], 'model': mn, 'error': repr(e), 'secs': round(time.time() - t0, 1)}
        rows.append(row)
        print('ROW', row['dataset'], row['model'], 'd_top', row.get('delta_top'), 'p', row.get('wilcoxon_p_top'), flush=True)
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
