"""Diagnostic (multi-split protocol): is LIME's threshold gain a self-computable data-quantile grid,
and does the attacker need the TARGET's quantiles?

Three paired arms on the BASE Autolycus SHAP traversal (no diverse, no boundary search):
  shap         = vanilla SHAP top-k, step +/-eps from current value          (base)
  shap_auxgrid = snap to quartiles of the ATTACKER's SEED SET (small, noisy)  (base + self grid)
  shap_tgtgrid = snap to quartiles of the TARGET's training set (X_train)     (base + leaked grid)

Protocol: instead of 1 fixed split x HMS=10 seed sets, we run NSPLIT=10 independent train/test
SPLITS (load_dataset seed=0..9), HMS=1 seed set each, target retrained per split. Arms are paired
within a split; Wilcoxon over the 10 splits. This measures split/model robustness (not just seed
noise), and -- because the aux grid comes from the tiny seed set -- folds in the aux-size test:
if aux ~ tgt even here, the grid is trivially self-computable; if aux < tgt, the target's leaked
quantiles genuinely matter.

    python _shap_auxgrid.py --ds 1 --out paper_results
"""
import os, json, time, argparse, warnings, traceback
import numpy as np
import random
from scipy.stats import wilcoxon
from attack_utils import (load_dataset, load_model, mega_sample_generation, _fit_one_surrogate,
                          load_explainer, traverse_explanations_SHAP)

DATASETS = {1: 'crop', 2: 'adult', 3: 'breast', 8: 'wine', 9: 'pendigits', 10: 'letter'}  # tabular only (digits=pixels dropped)
MODELS = {0: 'DT', 1: 'LR', 2: 'NB', 3: 'KNN', 4: 'RF'}
DEFAULT_MODELS = [1, 2, 3, 0, 4]
REPS = {'dt': 20, 'rdf': 10, 'nb': 1, 'lr': 1, 'knn': 1}
Q_BY_DS = {1: [1000], 2: [1000], 3: [100], 8: [100], 9: [500], 10: [500]}
NFE, HMS, NSPLIT = 3, 1, 10   # SIZE (=n, seeds/class) set via --size; n=1 is the coverage-starved regime


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


def run_ds(ds, models=DEFAULT_MODELS, out_dir='paper_results', size=5):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f'shap_auxgrid_n{size}_ds{ds}.json')
    warnings.filterwarnings('ignore')
    topQ = Q_BY_DS[ds][-1]
    ARMS = ['shap', 'shap_auxgrid', 'shap_tgtgrid']
    fid = {m: {a: [] for a in ARMS} for m in models}   # fid[model][arm] = one value per split
    n_eval = None
    t0 = time.time()
    for s in range(NSPLIT):
        a1, a2 = load_dataset(ds, seed=s)                 # independent train/test split
        X_train, X_test, y_train, y_test, X_test_t, X_test_s, y_test_t, y_test_s = a1
        nc = a2[2]; nf = a2[3]; Xt = np.asarray(X_test_t.values, float)
        n_eval = len(Xt)                                  # eval-set size (stability); report it
        Xtr = np.asarray(X_train.values, float)
        lb = int((topQ // nc) * 0.5 + 1); ub = int((topQ // nc) * (nc + 0.5) + 1)
        random.seed(s); np.random.seed(s)
        mega = mega_sample_generation(X_test_s.to_numpy(), y_test_s, nc, [size], HMS)
        seed_set = np.asarray(mega[0][0], float)          # attacker's actual seeds (feature-only)
        aux_grid = [np.percentile(seed_set[:, f], [25, 50, 75]) for f in range(nf)]  # tiny, noisy
        tgt_grid = [np.percentile(Xtr[:, f], [25, 50, 75]) for f in range(nf)]        # target's data
        for m in models:
            mn = MODELS.get(m, str(m))
            tm, mn = load_model(m, X_train, y_train)
            expl = load_explainer(1, tm, mn, X_train)
            for arm, grid in [('shap', None), ('shap_auxgrid', aux_grid), ('shap_tgtgrid', tgt_grid)]:
                random.seed(1000 + s); np.random.seed(1000 + s)   # same RNG for all arms -> paired
                V, P, _ = traverse_explanations_SHAP(mega[0][0], expl, tm, lb, ub, topQ, NFE, a2, mn,
                                                     X_train, y_train, explanation_type='vanilla',
                                                     num_exp=NFE, quantile_grid=grid)
                fid[m][arm].append(_fid(mn, V, P, tm, Xt, nc))
        print(f'  split {s} done ({round(time.time()-t0,1)}s)', flush=True)
    rows = []
    for m in models:
        mn = MODELS.get(m, str(m))
        S = np.array(fid[m]['shap'], float); G = np.array(fid[m]['shap_auxgrid'], float)
        T = np.array(fid[m]['shap_tgtgrid'], float)
        wp = lambda A: (round(float(np.mean(A) - np.mean(S)), 4),
                        float(wilcoxon(A, S).pvalue) if not np.allclose(A, S) else 1.0)
        row = {'dataset': DATASETS[ds], 'model': mn, 'q': topQ, 'nsplit': NSPLIT,
               'size': size, 'n_eval': n_eval,
               'shap_mean': round(float(np.mean(S)), 4),
               'auxgrid_mean': round(float(np.mean(G)), 4), 'tgtgrid_mean': round(float(np.mean(T)), 4),
               'auxgrid_delta': wp(G)[0], 'auxgrid_p': wp(G)[1],
               'tgtgrid_delta': wp(T)[0], 'tgtgrid_p': wp(T)[1],
               'shap_all': [round(x, 4) for x in S], 'auxgrid_all': [round(x, 4) for x in G],
               'tgtgrid_all': [round(x, 4) for x in T]}
        rows.append(row)
        print('ROW', row['dataset'], row['model'], 'aux', row['auxgrid_delta'], 'tgt', row['tgtgrid_delta'], flush=True)
        with open(path, 'w') as f:
            json.dump(rows, f, indent=2)
    return rows


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--ds', type=int, required=True)
    ap.add_argument('--models', type=str, default=','.join(map(str, DEFAULT_MODELS)))
    ap.add_argument('--out', type=str, default='paper_results')
    ap.add_argument('--size', type=int, default=5)   # n = seeds/class; use 1 for the coverage-starved regime
    a = ap.parse_args()
    mods = [int(x) for x in a.models.split(',') if x != '']
    run_ds(a.ds, models=mods, out_dir=a.out, size=a.size)
