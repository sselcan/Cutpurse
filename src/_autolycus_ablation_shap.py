"""E2 on the BASE Autolycus SHAP attack. SHAP has no bin edge, so it's just 1x2:
  shap    = explanation_type='vanilla'  (SHAP top-k feature selection)
  random  = explanation_type='random'   (random features)  == the no-explanation baseline (4)
attribution effect = shap - random. If ~0/negative, SHAP-guided == random traversal (SHAP inert).
Base SHAP setting: n=5, k=3. Swept over budget. Paired, mean-agg, Wilcoxon at top budget.

    python _autolycus_ablation_shap.py --ds 2 --out paper_results
"""
import os, json, time, argparse, warnings, traceback
import numpy as np
import random
from scipy.stats import wilcoxon
from attack_utils import (load_dataset, load_model, mega_sample_generation, _fit_one_surrogate,
                          load_explainer, traverse_explanations_SHAP)

DATASETS = {0: 'iris', 1: 'crop', 2: 'adult', 3: 'breast', 4: 'nursery', 5: 'mushroom',
            7: 'digits', 8: 'wine', 9: 'pendigits', 10: 'letter'}
MODELS   = {0: 'DT', 1: 'LR', 2: 'NB', 3: 'KNN', 4: 'RF'}
DEFAULT_MODELS = [1, 2, 3, 0, 4]
REPS = {'dt': 20, 'rdf': 10, 'nb': 1, 'lr': 1, 'knn': 1}
Q_BY_DS = {0: [100], 1: [1000], 2: [1000], 3: [100], 4: [1000], 5: [1000],
           7: [500], 8: [100], 9: [500], 10: [500]}  # match LIME ablation budgets
SIZE, NFE, HMS, SEED = 5, 3, 10, 0
ARMS = {'shap': 'vanilla', 'random': 'random'}


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


def run_ds(ds, models=DEFAULT_MODELS, out_dir='paper_results', seed=SEED):
    os.makedirs(out_dir, exist_ok=True)
    tag = '' if seed == 0 else f'_s{seed}'
    path = os.path.join(out_dir, f'autolycus_ablation_shap{tag}_ds{ds}.json')
    warnings.filterwarnings('ignore')
    q_list = Q_BY_DS[ds]
    a1, a2 = load_dataset(ds)
    X_train, X_test, y_train, y_test, X_test_t, X_test_s, y_test_t, y_test_s = a1
    nc = a2[2]; Xt = np.asarray(X_test_t.values, float)
    rows = []
    for m in models:
        mn = MODELS.get(m, str(m)); t0 = time.time()
        try:
            random.seed(seed); np.random.seed(seed)
            tm, mn = load_model(m, X_train, y_train)
            expl = load_explainer(1, tm, mn, X_train)
            mega = mega_sample_generation(X_test_s.to_numpy(), y_test_s, nc, [SIZE], HMS)
            fid = {a: {Q: [] for Q in q_list} for a in ARMS}
            for Q in q_list:
                lb = int((Q // nc) * 0.5 + 1); ub = int((Q // nc) * (nc + 0.5) + 1)
                for a, etype in ARMS.items():
                    for i in range(HMS):
                        random.seed(seed + 1 + i); np.random.seed(seed + 1 + i)
                        V, P, _ = traverse_explanations_SHAP(mega[i][0], expl, tm, lb, ub, Q, NFE, a2,
                                                             mn, X_train, y_train, explanation_type=etype,
                                                             num_exp=NFE)   # clean random-k (fix: was default 5)
                        fid[a][Q].append(_fid(mn, V, P, tm, Xt, nc))

            def pack(a):
                return {'fid_mean': [round(float(np.mean(fid[a][Q])), 4) for Q in q_list],
                        'fid_all':  [[round(x, 4) for x in fid[a][Q]] for Q in q_list]}
            row = {'dataset': DATASETS[ds], 'model': mn, 'q_cap': q_list,
                   **{a: pack(a) for a in ARMS}, 'secs': round(time.time() - t0, 1)}
            topQ = q_list[-1]
            S = np.array(fid['shap'][topQ], float); R = np.array(fid['random'][topQ], float)
            row['attr_delta_top'] = round(float(S.mean() - R.mean()), 4)   # shap - random
            row['attr_p_top'] = (float(wilcoxon(S, R).pvalue) if not np.allclose(S, R) else 1.0)
        except Exception as e:
            traceback.print_exc()
            row = {'dataset': DATASETS[ds], 'model': mn, 'error': repr(e), 'secs': round(time.time() - t0, 1)}
        rows.append(row)
        print('ROW', row['dataset'], row['model'], 'attr_d', row.get('attr_delta_top'), 'p', row.get('attr_p_top'), flush=True)
        with open(path, 'w') as f:
            json.dump(rows, f, indent=2)
    return rows


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--ds', type=int, required=True)
    ap.add_argument('--models', type=str, default=','.join(map(str, DEFAULT_MODELS)))
    ap.add_argument('--out', type=str, default='paper_results')
    ap.add_argument('--seed', type=int, default=SEED)
    a = ap.parse_args()
    mods = [int(x) for x in a.models.split(',') if x != '']
    run_ds(a.ds, models=mods, out_dir=a.out, seed=a.seed)
