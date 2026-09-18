"""The closing arm: does an adversary with NO explanation content match the explanation-guided attack?

E2 showed LIME's feature ranking gives no systematic advantage over random; _lime_auxdisc.py showed
the discretizer need not come from the target. This composes both into a single direct measurement.

Three paired arms on the BASE Autolycus LIME traversal (n=1, LIME's native setting):
  full      = LIME on X_train, feature_select='explanation', threshold ON
              (the real service: target's own feature ranking AND target's discretizer)
  randaux   = LIME on the attacker's data, feature_select='random',      threshold ON
              (NO explanation content: random features + a self-computed quantile grid)
  randnobin = feature_select='random', threshold OFF
              (the no-explanation floor: random features, local +/-eps steps only)

Deltas:
  gap        = full - randaux     -> if ~0, the explanation contributes nothing the adversary lacks
  grid_value = randaux - randnobin-> what the self-computed grid alone buys

--aux shadow : discretizer fit on the full 10% auxiliary pool (X_test_s); the "comprehensive D_A"
               the Autolycus threat model permits.
--aux seed   : discretizer fit on the n-per-class seed set; the minimal D_A the paper evaluates.

Multi-split protocol: NSPLIT=10 independent splits, HMS=1, target retrained per split, arms paired
within a split, Wilcoxon over splits.

    python _lime_noexpl.py --ds 1 --aux shadow --out paper_results
"""
import os, json, time, argparse, warnings
import numpy as np
import random
import lime, lime.lime_tabular
from scipy.stats import wilcoxon
from attack_utils import (load_dataset, load_model, mega_sample_generation, _fit_one_surrogate,
                          traverse_explanations_LIME)

DATASETS = {1: 'crop', 3: 'breast', 4: 'nursery', 5: 'mushroom', 9: 'pendigits', 10: 'letter'}
MODELS = {0: 'DT', 1: 'LR', 2: 'NB', 3: 'KNN', 4: 'RF'}
DEFAULT_MODELS = [1, 2, 3, 0, 4]
REPS = {'dt': 20, 'rdf': 10, 'nb': 1, 'lr': 1, 'knn': 1}
Q_BY_DS = {1: 1000, 3: 100, 4: 1000, 5: 1000, 9: 500, 10: 500}
SIZE, NFE, HMS, NSPLIT = 1, 3, 1, 10
ARMS = ['full', 'randaux', 'randnobin']


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
    path = os.path.join(out_dir, f'lime_noexpl_{aux}_ds{ds}.json')
    warnings.filterwarnings('ignore')
    topQ = Q_BY_DS[ds]
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
        aux_data = seed_set if aux == 'seed' else X_test_s.values
        expl_tgt = lime.lime_tabular.LimeTabularExplainer(X_train.values, discretize_continuous=True)
        expl_aux = lime.lime_tabular.LimeTabularExplainer(aux_data, discretize_continuous=True)
        # arm -> (explainer, feature_select, use_threshold)
        cfg = {'full':      (expl_tgt, 'explanation', True),
               'randaux':   (expl_aux, 'random',      True),
               'randnobin': (expl_tgt, 'random',      False)}
        for m in models:
            mn = MODELS.get(m, str(m))
            tm, mn = load_model(m, X_train, y_train)
            for arm in ARMS:
                expl, fs_, ut = cfg[arm]
                random.seed(3000 + s); np.random.seed(3000 + s)   # paired across arms
                V, P, _ = traverse_explanations_LIME(mega[0][0], expl, tm, lb, ub, topQ, NFE, a2,
                                                     feature_select=fs_, use_threshold=ut)
                fid[m][arm].append(_fid(mn, V, P, tm, Xt, nc))
        print(f'  split {s} done ({round(time.time()-t0,1)}s)', flush=True)

    rows = []
    for m in models:
        mn = MODELS.get(m, str(m))
        F = np.array(fid[m]['full'], float)
        A = np.array(fid[m]['randaux'], float)
        Z = np.array(fid[m]['randnobin'], float)
        wp = lambda X, Y: (round(float(np.mean(X) - np.mean(Y)), 4),
                           float(wilcoxon(X, Y).pvalue) if not np.allclose(X, Y) else 1.0)
        row = {'dataset': DATASETS[ds], 'model': mn, 'q': topQ, 'nsplit': NSPLIT, 'n_eval': n_eval,
               'aux_source': aux,
               'full_mean': round(float(np.mean(F)), 4),
               'randaux_mean': round(float(np.mean(A)), 4),
               'randnobin_mean': round(float(np.mean(Z)), 4),
               'gap': wp(F, A)[0], 'gap_p': wp(F, A)[1],                 # explanation's residual value
               'grid_value': wp(A, Z)[0], 'grid_value_p': wp(A, Z)[1],   # self-computed grid's value
               'full_all': [round(x, 4) for x in F],
               'randaux_all': [round(x, 4) for x in A],
               'randnobin_all': [round(x, 4) for x in Z]}
        rows.append(row)
        print('ROW', row['dataset'], row['model'], 'gap(full-randaux)', row['gap'],
              f"(p={row['gap_p']:.3f})", '| grid_value', row['grid_value'], flush=True)
        with open(path, 'w') as f:
            json.dump(rows, f, indent=2)
    return rows


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--ds', type=int, required=True)
    ap.add_argument('--models', type=str, default=','.join(map(str, DEFAULT_MODELS)))
    ap.add_argument('--out', type=str, default='paper_results')
    ap.add_argument('--aux', type=str, default='shadow', choices=['shadow', 'seed'])
    a = ap.parse_args()
    mods = [int(x) for x in a.models.split(',') if x != '']
    run_ds(a.ds, models=mods, out_dir=a.out, aux=a.aux)
