"""E2 / H1: does the explanation's feature RANKING matter, or is random-k just as good?
Compares our attack with explanation-guided vs random feature selection, everything else identical.
H1 (attribution is inert) => fidelity ~ equal (small, non-significant delta).

  lime: LIME3(cap15, feature_select 'explanation' vs 'random')   n=1, k=3
        -- random still snaps to the random feature's bin edge, so the THRESHOLD channel is kept;
           only the feature CHOICE (attribution) changes.
  shap: SHAP3(use_shap_gen/traverse True vs False)               n=5, k=3

    python _e2_h1.py --explainer lime --ds 2 --out paper_results
"""
import os, json, time, argparse, warnings, traceback
import numpy as np
import random
import lime, lime.lime_tabular
from scipy.stats import wilcoxon
from attack_utils import (load_dataset, load_model, mega_sample_generation, _fit_one_surrogate,
                          load_explainer, traverse_explanations_LIME3, traverse_explanations_SHAP3)

DATASETS = {0: 'iris', 1: 'crop', 2: 'adult', 3: 'breast', 4: 'nursery', 5: 'mushroom'}
MODELS   = {0: 'DT', 1: 'LR', 2: 'NB', 3: 'KNN', 4: 'RF'}
DEFAULT_MODELS = [1, 2, 3, 0, 4]
REPS = {'dt': 20, 'rdf': 10, 'nb': 1, 'lr': 1, 'knn': 1}
Q_BY_DS = {0: 100, 1: 1000, 2: 1000, 3: 100, 4: 1000, 5: 1000}
HMS, SEED, NFE = 10, 0, 3


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


def _run(kind, which, seed_set, expl, tm, mn, lb, ub, Q, a2, X_train, y_train):
    """which = 'explanation' or 'random'."""
    if kind == 'lime':
        return traverse_explanations_LIME3(seed_set, expl, tm, lb, ub, Q, NFE, a2, mn, X_train, y_train,
                                           use_diverse=True, diverse_method='lime_threshold',
                                           div_frac=0.3, div_cap=15, feature_select=which)
    on = (which == 'explanation')
    return traverse_explanations_SHAP3(seed_set, expl, tm, lb, ub, Q, NFE, a2, mn, X_train, y_train,
                                       use_shap_gen=on, use_shap_traverse=on)


def run_ds(kind, ds, models=DEFAULT_MODELS, out_dir='paper_results'):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f'e2_{kind}_ds{ds}.json')
    warnings.filterwarnings('ignore')
    Q = Q_BY_DS[ds]; SIZE = 1 if kind == 'lime' else 5
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
            expl = (lime.lime_tabular.LimeTabularExplainer(X_train.values, discretize_continuous=True)
                    if kind == 'lime' else load_explainer(1, tm, mn, X_train))
            mega = mega_sample_generation(X_test_s.to_numpy(), y_test_s, nc, [SIZE], HMS)
            exp_f, rnd_f = [], []
            for i in range(HMS):
                random.seed(SEED + 1 + i); np.random.seed(SEED + 1 + i)
                V, P, _ = _run(kind, 'explanation', mega[i][0], expl, tm, mn, lb, ub, Q, a2, X_train, y_train)
                exp_f.append(_fid(mn, V, P, tm, Xt, nc))
                random.seed(SEED + 1 + i); np.random.seed(SEED + 1 + i)
                V, P, _ = _run(kind, 'random', mega[i][0], expl, tm, mn, lb, ub, Q, a2, X_train, y_train)
                rnd_f.append(_fid(mn, V, P, tm, Xt, nc))
            exp_f, rnd_f = np.array(exp_f, float), np.array(rnd_f, float)

            def wp(a, b):
                return (float(wilcoxon(a, b).pvalue) if not np.allclose(a, b) else 1.0)
            row = {'explainer': kind, 'dataset': DATASETS[ds], 'model': mn, 'Q': Q,
                   'expl': round(float(exp_f.mean()), 4), 'rand': round(float(rnd_f.mean()), 4),
                   'd_rand_expl': round(float(rnd_f.mean() - exp_f.mean()), 4),
                   'p': round(wp(rnd_f, exp_f), 4),
                   'expl_all': [round(x, 4) for x in exp_f.tolist()],
                   'rand_all': [round(x, 4) for x in rnd_f.tolist()], 'secs': round(time.time() - t0, 1)}
        except Exception as e:
            traceback.print_exc()
            row = {'explainer': kind, 'dataset': DATASETS[ds], 'model': mn, 'error': repr(e),
                   'secs': round(time.time() - t0, 1)}
        rows.append(row)
        print('ROW', json.dumps({k: row[k] for k in row if not k.endswith('_all')}), flush=True)
        with open(path, 'w') as f:
            json.dump(rows, f, indent=2)
    return rows


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--explainer', required=True, choices=['lime', 'shap'])
    ap.add_argument('--ds', type=int, required=True)
    ap.add_argument('--models', type=str, default=','.join(map(str, DEFAULT_MODELS)))
    ap.add_argument('--out', type=str, default='paper_results')
    a = ap.parse_args()
    mods = [int(x) for x in a.models.split(',') if x != '']
    run_ds(a.explainer, a.ds, models=mods, out_dir=a.out)
