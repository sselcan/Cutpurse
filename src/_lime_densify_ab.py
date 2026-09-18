"""A/B: LIME3 (ours) at fixed diverse budget cap=15, threshold densification OFF vs ON.
Three arms on the same seeded sets, at each dataset's top budget:
  base    = Autolycus base-LIME
  nodense = LIME3(div_cap=15, densify=0)   -- current method
  dense   = LIME3(div_cap=15, densify=3)   -- + threshold densification at the bin edge
Reports mean fidelity + paired Wilcoxon (dense vs nodense, dense vs base). Writes JSON per dataset.

    python _lime_densify_ab.py --ds 2 --out paper_results
"""
import os, json, time, argparse, warnings, traceback
import numpy as np
import random
import lime, lime.lime_tabular
from scipy.stats import wilcoxon
from attack_utils import (load_dataset, load_model, mega_sample_generation, _fit_one_surrogate,
                          traverse_explanations_LIME, traverse_explanations_LIME3)

DATASETS = {0: 'iris', 1: 'crop', 2: 'adult', 3: 'breast', 4: 'nursery', 5: 'mushroom'}
MODELS   = {0: 'DT', 1: 'LR', 2: 'NB', 3: 'KNN', 4: 'RF'}
DEFAULT_MODELS = [1, 2, 3, 0, 4]
REPS = {'dt': 20, 'rdf': 10, 'nb': 1, 'lr': 1, 'knn': 1}
Q_BY_DS = {0: 100, 1: 1000, 2: 1000, 3: 100, 4: 1000, 5: 1000}
CAP, DIV_FRAC, NFE, SIZE, HMS, SEED, DENSIFY = 15, 0.3, 3, 1, 10, 0, 3


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


def _lime3(seed_set, expl, tm, mn, lb, ub, Q, a2, X_train, y_train, densify):
    return traverse_explanations_LIME3(seed_set, expl, tm, lb, ub, Q, NFE, a2, mn, X_train, y_train,
                                       use_diverse=True, diverse_method='lime_threshold',
                                       div_frac=DIV_FRAC, div_cap=CAP, densify=densify)


def run_ds(ds, models=DEFAULT_MODELS, out_dir='paper_results'):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f'densify_ab_ds{ds}.json')
    warnings.filterwarnings('ignore')
    Q = Q_BY_DS[ds]
    a1, a2 = load_dataset(ds)
    X_train, X_test, y_train, y_test, X_test_t, X_test_s, y_test_t, y_test_s = a1
    nc = a2[2]; Xt = np.asarray(X_test_t.values, float)
    lb = int((Q // nc) * 0.5 + 1); ub = int((Q // nc) * (nc + 0.5) + 1)
    rows = []
    for m in models:
        mn = MODELS.get(m, str(m)); t0 = time.time()
        try:
            random.seed(SEED); np.random.seed(SEED)
            tm, mn = load_model(m, X_train, y_train)
            expl = lime.lime_tabular.LimeTabularExplainer(X_train.values, discretize_continuous=True)
            mega = mega_sample_generation(X_test_s.to_numpy(), y_test_s, nc, [SIZE], HMS)
            base_f, nod_f, den_f = [], [], []
            for i in range(HMS):
                random.seed(SEED + 1 + i); np.random.seed(SEED + 1 + i)
                V, P, _ = traverse_explanations_LIME(mega[i][0], expl, tm, lb, ub, Q, NFE, a2)
                base_f.append(_fid(mn, V, P, tm, Xt, nc))
                random.seed(SEED + 1 + i); np.random.seed(SEED + 1 + i)
                V, P, _ = _lime3(mega[i][0], expl, tm, mn, lb, ub, Q, a2, X_train, y_train, 0)
                nod_f.append(_fid(mn, V, P, tm, Xt, nc))
                random.seed(SEED + 1 + i); np.random.seed(SEED + 1 + i)
                V, P, _ = _lime3(mega[i][0], expl, tm, mn, lb, ub, Q, a2, X_train, y_train, DENSIFY)
                den_f.append(_fid(mn, V, P, tm, Xt, nc))
            base_f, nod_f, den_f = map(lambda x: np.array(x, float), (base_f, nod_f, den_f))

            def _wp(a, b):
                return (float(wilcoxon(a, b).pvalue) if not np.allclose(a, b) else 1.0)
            row = {'dataset': DATASETS[ds], 'model': mn, 'Q': Q,
                   'base': round(float(base_f.mean()), 4), 'nodense': round(float(nod_f.mean()), 4),
                   'dense': round(float(den_f.mean()), 4),
                   'd_dense_nod': round(float(den_f.mean() - nod_f.mean()), 4),
                   'p_dense_nod': round(_wp(den_f, nod_f), 4),
                   'd_dense_base': round(float(den_f.mean() - base_f.mean()), 4),
                   'p_dense_base': round(_wp(den_f, base_f), 4), 'secs': round(time.time() - t0, 1)}
        except Exception as e:
            traceback.print_exc()
            row = {'dataset': DATASETS[ds], 'model': mn, 'error': repr(e), 'secs': round(time.time() - t0, 1)}
        rows.append(row)
        print('ROW', json.dumps(row), flush=True)
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
