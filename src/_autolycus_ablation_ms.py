"""Run paired multi-split LIME ablations for RQ1 and RQ2.

The four configurations vary attribution-based feature selection and bin-edge
use in the base Autolycus traversal. Results are written as per-split JSON.
"""
import os, json, time, argparse, warnings, traceback
import numpy as np
import random
import lime, lime.lime_tabular
from scipy.stats import wilcoxon
from attack_utils import (load_dataset, load_model, mega_sample_generation, _fit_one_surrogate,
                          traverse_explanations_LIME)

DATASETS = {0: 'iris', 1: 'crop', 2: 'adult', 3: 'breast', 4: 'nursery', 5: 'mushroom',
            7: 'digits', 8: 'wine', 9: 'pendigits'}
MODELS = {0: 'DT', 1: 'LR', 2: 'NB', 3: 'KNN', 4: 'RF'}
DEFAULT_MODELS = [1, 2, 3, 0]          # RF omitted by default: LIMExRF traversal is very slow
REPS = {'dt': 20, 'rdf': 10, 'nb': 1, 'lr': 1, 'knn': 1}
Q_BY_DS = {0: 100, 1: 1000, 2: 1000, 3: 100, 4: 1000, 5: 1000, 7: 500, 8: 100, 9: 500}
SIZE, NFE, HMS, NSPLIT = 1, 3, 1, 10   # LIME setting (n=1, k=3); 1 seed set per split
ARMS = {'default':   ('explanation', True),
        'randfeat':  ('random', True),
        'nothresh':  ('explanation', False),
        'randnobin': ('random', False)}


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
    path = os.path.join(out_dir, f'autolycus_ablation_ms_ds{ds}.json')
    warnings.filterwarnings('ignore')
    topQ = Q_BY_DS[ds]
    fid = {m: {a: [] for a in ARMS} for m in models}   # fid[model][arm] = one value per split
    t0 = time.time()
    for s in range(NSPLIT):
        a1, a2 = load_dataset(ds, seed=s)              # independent train/test split
        X_train, X_test, y_train, y_test, X_test_t, X_test_s, y_test_t, y_test_s = a1
        nc = a2[2]; Xt = np.asarray(X_test_t.values, float)
        lb = int((topQ // nc) * 0.5 + 1); ub = int((topQ // nc) * (nc + 0.5) + 1)
        random.seed(s); np.random.seed(s)
        mega = mega_sample_generation(X_test_s.to_numpy(), y_test_s, nc, [SIZE], HMS)
        for m in models:
            mn = MODELS.get(m, str(m))
            try:
                tm, mn = load_model(m, X_train, y_train)          # target retrained per split
                expl = lime.lime_tabular.LimeTabularExplainer(X_train.values, discretize_continuous=True)
                for a, (fs_, ut) in ARMS.items():
                    random.seed(1000 + s); np.random.seed(1000 + s)   # same RNG per arm -> paired
                    V, P, _ = traverse_explanations_LIME(mega[0][0], expl, tm, lb, ub, topQ, NFE, a2,
                                                         feature_select=fs_, use_threshold=ut)
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
        D = np.array(fid[m]['default'], float)

        def wp(other):
            B = np.array(fid[m][other], float)
            ok = ~(np.isnan(D) | np.isnan(B))
            if ok.sum() < 2:
                return (float('nan'), 1.0)
            d, b = D[ok], B[ok]
            return (round(float(d.mean() - b.mean()), 4),
                    float(wilcoxon(d, b).pvalue) if not np.allclose(d, b) else 1.0)

        row = {'dataset': DATASETS[ds], 'model': mn, 'q': topQ, 'nsplit': NSPLIT, 'protocol': 'multisplit',
               **{f'{a}_mean': round(float(np.nanmean(fid[m][a])), 4) for a in ARMS},
               **{f'{a}_all': [round(x, 4) for x in fid[m][a]] for a in ARMS}}
        row['attr_delta_top'], row['attr_p_top'] = wp('randfeat')
        row['thr_delta_top'], row['thr_p_top'] = wp('nothresh')
        row['total_delta_top'], row['total_p_top'] = wp('randnobin')
        rows.append(row)
        print('ROW', row['dataset'], row['model'], 'attr', row['attr_delta_top'],
              'thr', row['thr_delta_top'], f"(p={row['thr_p_top']:.3f})", flush=True)
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
