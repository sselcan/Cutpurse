"""E2 on the BASE Autolycus SHAP attack -- MULTI-SPLIT protocol (companion to
_autolycus_ablation_ms.py, so the SHAP and LIME attribution tables share a protocol).

SHAP exposes no bin edge, so this is a 1x2 rather than a 2x2:
  shap    = explanation_type='vanilla'  (SHAP top-k feature selection)
  random  = explanation_type='random'   (random-k features) == no-explanation baseline
attribution effect = shap - random.

Protocol: NSPLIT=10 independent train/test splits (load_dataset seed=0..9), HMS=1 seed set each,
target retrained and SHAP explainer refit per split. Arms paired within a split (same RNG);
Wilcoxon over the 10 splits. Base SHAP setting: n=5, k=3.

    python _autolycus_ablation_shap_ms.py --ds 1 --out paper_results
"""
import os, json, time, argparse, warnings, traceback
import numpy as np
import random
from scipy.stats import wilcoxon
from attack_utils import (load_dataset, load_model, mega_sample_generation, _fit_one_surrogate,
                          load_explainer, traverse_explanations_SHAP)

DATASETS = {0: 'iris', 1: 'crop', 2: 'adult', 3: 'breast', 4: 'nursery', 5: 'mushroom',
            8: 'wine', 9: 'pendigits'}
MODELS = {0: 'DT', 1: 'LR', 2: 'NB', 3: 'KNN', 4: 'RF'}
DEFAULT_MODELS = [1, 2, 3, 0, 4]
REPS = {'dt': 20, 'rdf': 10, 'nb': 1, 'lr': 1, 'knn': 1}
Q_BY_DS = {0: 100, 1: 1000, 2: 1000, 3: 100, 4: 1000, 5: 1000, 8: 100, 9: 500}
SIZE, NFE, HMS, NSPLIT = 5, 3, 1, 10      # SHAP setting: n=5, k=3
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


def run_ds(ds, models=DEFAULT_MODELS, out_dir='paper_results'):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f'autolycus_ablation_shap_ms_ds{ds}.json')
    warnings.filterwarnings('ignore')
    topQ = Q_BY_DS[ds]
    fid = {m: {a: [] for a in ARMS} for m in models}
    n_eval = 0
    t0 = time.time()
    for s in range(NSPLIT):
        a1, a2 = load_dataset(ds, seed=s)
        X_train, X_test, y_train, y_test, X_test_t, X_test_s, y_test_t, y_test_s = a1
        nc = a2[2]; Xt = np.asarray(X_test_t.values, float); n_eval = len(Xt)
        lb = int((topQ // nc) * 0.5 + 1); ub = int((topQ // nc) * (nc + 0.5) + 1)
        random.seed(s); np.random.seed(s)
        mega = mega_sample_generation(X_test_s.to_numpy(), y_test_s, nc, [SIZE], HMS)
        for m in models:
            mn = MODELS.get(m, str(m))
            try:
                tm, mn = load_model(m, X_train, y_train)          # target retrained per split
                expl = load_explainer(1, tm, mn, X_train)          # SHAP explainer refit per split
                for a, etype in ARMS.items():
                    random.seed(1000 + s); np.random.seed(1000 + s)   # paired across arms
                    V, P, _ = traverse_explanations_SHAP(mega[0][0], expl, tm, lb, ub, topQ, NFE, a2,
                                                         mn, X_train, y_train, explanation_type=etype,
                                                         num_exp=NFE)
                    fid[m][a].append(_fid(mn, V, P, tm, Xt, nc))
            except Exception:
                traceback.print_exc()
                for a in ARMS:
                    if len(fid[m][a]) < s + 1:
                        fid[m][a].append(float('nan'))
        print(f'  split {s} done ({round(time.time() - t0, 1)}s)', flush=True)

    rows = []
    for m in models:
        mn = MODELS.get(m, str(m))
        S = np.array(fid[m]['shap'], float); R = np.array(fid[m]['random'], float)
        ok = ~(np.isnan(S) | np.isnan(R))
        if ok.sum() >= 2:
            d, b = S[ok], R[ok]
            delta = round(float(d.mean() - b.mean()), 4)
            p = float(wilcoxon(d, b).pvalue) if not np.allclose(d, b) else 1.0
        else:
            delta, p = float('nan'), 1.0
        row = {'dataset': DATASETS[ds], 'model': mn, 'q': topQ, 'nsplit': NSPLIT,
               'protocol': 'multisplit', 'n_eval': n_eval,
               'shap_mean': round(float(np.nanmean(S)), 4), 'random_mean': round(float(np.nanmean(R)), 4),
               'shap_all': [round(x, 4) for x in fid[m]['shap']],
               'random_all': [round(x, 4) for x in fid[m]['random']],
               'attr_delta_top': delta, 'attr_p_top': p}
        rows.append(row)
        print('ROW', row['dataset'], row['model'], 'attr', delta, f'(p={p:.3f})', flush=True)
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
    a = ap.parse_args()
    if a.q:
        Q_BY_DS[a.ds] = a.q
    mods = [int(x) for x in a.models.split(',') if x != '']
    run_ds(a.ds, models=mods, out_dir=a.out)
