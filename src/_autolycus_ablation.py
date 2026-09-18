"""E2 + E3a on the BASE Autolycus LIME attack (the clean tests -- no boundary-search masking).
Autolycus LIME already snaps to bin edges and selects top-k features, so both channels live here.
Three arms on the same seeded sets, at the LIME main setting (n=1, k=3), swept over budget:
  default   = feature_select='explanation', use_threshold=True   (real Autolycus)
  randfeat  = feature_select='random',      use_threshold=True   (E2: random features)
  nothresh  = feature_select='explanation', use_threshold=False  (E3a: ignore the bin edge)
=>  attribution effect = default - randfeat ;  threshold effect = default - nothresh.

    python _autolycus_ablation.py --ds 2 --out paper_results
"""
import os, json, time, argparse, warnings, traceback
import numpy as np
import random
import lime, lime.lime_tabular
from scipy.stats import wilcoxon
from attack_utils import (load_dataset, load_model, mega_sample_generation, _fit_one_surrogate,
                          traverse_explanations_LIME)

DATASETS = {0: 'iris', 1: 'crop', 2: 'adult', 3: 'breast', 4: 'nursery', 5: 'mushroom',
            7: 'digits', 8: 'wine', 9: 'pendigits', 10: 'letter'}  # 7-10: extra continuous high-headroom (generalization)
MODELS   = {0: 'DT', 1: 'LR', 2: 'NB', 3: 'KNN', 4: 'RF'}
DEFAULT_MODELS = [1, 2, 3, 0, 4]
REPS = {'dt': 20, 'rdf': 10, 'nb': 1, 'lr': 1, 'knn': 1}
Q_BY_DS = {0: [100], 1: [1000], 2: [1000], 3: [100], 4: [1000], 5: [1000],
           7: [500], 8: [100], 9: [500], 10: [500]}  # 7/9 lighter budget (LIME cost); still huge headroom
SIZE, NFE, HMS, SEED = 1, 3, 10, 0
# the 2x2: {LIME-feature, random-feature} x {use bin edge, no bin}
ARMS = {'default': ('explanation', True),    # (1) LIME feature + bin  (real Autolycus)
        'randfeat': ('random', True),         # (2) random feature + bin
        'nothresh': ('explanation', False),   # (3) LIME feature, no bin
        'randnobin': ('random', False)}       # (4) random feature, no bin = NO explanation baseline


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
    path = os.path.join(out_dir, f'autolycus_ablation_ds{ds}.json')
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
            fid = {a: {Q: [] for Q in q_list} for a in ARMS}
            for Q in q_list:
                lb = int((Q // nc) * 0.5 + 1); ub = int((Q // nc) * (nc + 0.5) + 1)
                for a, (fs, ut) in ARMS.items():
                    for i in range(HMS):
                        random.seed(SEED + 1 + i); np.random.seed(SEED + 1 + i)
                        V, P, _ = traverse_explanations_LIME(mega[i][0], expl, tm, lb, ub, Q, NFE, a2,
                                                             feature_select=fs, use_threshold=ut)
                        fid[a][Q].append(_fid(mn, V, P, tm, Xt, nc))

            def pack(a):
                return {'fid_mean': [round(float(np.mean(fid[a][Q])), 4) for Q in q_list],
                        'fid_all':  [[round(x, 4) for x in fid[a][Q]] for Q in q_list]}
            row = {'dataset': DATASETS[ds], 'model': mn, 'q_cap': q_list,
                   **{a: pack(a) for a in ARMS}, 'secs': round(time.time() - t0, 1)}
            topQ = q_list[-1]
            df = np.array(fid['default'][topQ], float)

            def wp(other):
                b = np.array(fid[other][topQ], float)
                return (round(float(df.mean() - b.mean()), 4),
                        float(wilcoxon(df, b).pvalue) if not np.allclose(df, b) else 1.0)
            row['attr_delta_top'], row['attr_p_top'] = wp('randfeat')     # attribution: (1)-(2)
            row['thr_delta_top'], row['thr_p_top'] = wp('nothresh')       # threshold:   (1)-(3)
            row['total_delta_top'], row['total_p_top'] = wp('randnobin')  # total expl.: (1)-(4) baseline
        except Exception as e:
            traceback.print_exc()
            row = {'dataset': DATASETS[ds], 'model': mn, 'error': repr(e), 'secs': round(time.time() - t0, 1)}
        rows.append(row)
        print('ROW', row['dataset'], row['model'], 'attr_d', row.get('attr_delta_top'),
              'thr_d', row.get('thr_delta_top'), flush=True)
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
