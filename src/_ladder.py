"""Evaluate base-traversal variants and Cutpurse under a paired protocol.

The configurations vary explanation access, grid source, and the added query
generation phases. Per-split fidelity arrays are written as JSON.
"""
import os, json, time, argparse, warnings, traceback
import numpy as np
import random
import lime, lime.lime_tabular
from scipy.stats import wilcoxon
from attack_utils import (load_dataset, load_model, mega_sample_generation, _fit_one_surrogate,
                          traverse_explanations_LIME, traverse_explanations_LIME3)

DATASETS = {1: 'crop', 2: 'adult', 3: 'breast', 4: 'nursery', 5: 'mushroom', 9: 'pendigits'}
MODELS = {0: 'DT', 1: 'LR', 2: 'NB', 3: 'KNN', 4: 'RF'}
DEFAULT_MODELS = [1, 2, 3, 0]              # RF omitted: LIME3 x RF is prohibitively slow
REPS = {'dt': 20, 'rdf': 10, 'nb': 1, 'lr': 1, 'knn': 1}
Q_BY_DS = {1: 1000, 2: 1000, 3: 100, 4: 1000, 5: 1000, 9: 500}
SIZE, NFE, HMS, NSPLIT, DIV_CAP = 1, 3, 1, 10, 15   # LIME main setting (n=1, k=3)

# The `autolycus` arm is NOT recomputed here. `_autolycus_ablation_ms.py`'s `default` arm is the same
# adversary under the same protocol -- same load_dataset(ds, seed=s), same seed set (random.seed(s)),
# same LimeTabularExplainer(X_train.values), same Q_BY_DS, same SIZE/NFE/HMS/NSPLIT, same models and
# same _fid -- so we read its stored per-split array instead of paying for Q x 5000 LIME
# perturbations again. That fixes our attack seed at 1000+s, keeping every arm paired with it.
ARMS = ['blind', 'selfgrid', 'ours']            # all three make ZERO explain_instance calls
BORROWED = 'autolycus'
ATTACK_SEED = 1000

# (label, minuend, subtrahend) -- the contrasts the paper's claim table reports
CONTRASTS = [('c4_ours_vs_autolycus', 'ours', 'autolycus'),
             ('c5_ours_vs_selfgrid', 'ours', 'selfgrid'),
             ('selfgrid_vs_autolycus', 'selfgrid', 'autolycus'),
             ('grid_value', 'selfgrid', 'blind')]


def _borrow_autolycus(ds, out_dir):
    """Load stored Autolycus fidelity arrays for one dataset."""
    p = os.path.join(out_dir, f'autolycus_ablation_ms_ds{ds}.json')
    if not os.path.exists(p):
        print(f'  [warn] {p} missing -- autolycus contrasts will be NaN', flush=True)
        return {}
    out = {}
    for r in json.load(open(p)):
        if 'default_all' in r:
            out[r['model'].lower().replace('rdf', 'rf')] = np.array(r['default_all'], float)
    return out


class _CountingExplainer:
    """Wrap a LIME explainer and count calls forwarded to the target model."""

    def __init__(self, inner):
        self._inner = inner
        self.calls = 0
        self.model_queries = 0

    def __getattr__(self, name):          # discretizer, feature_names, ... pass straight through
        return getattr(self._inner, name)

    def reset(self):
        self.calls = 0
        self.model_queries = 0

    def explain_instance(self, *a, **k):
        self.calls += 1
        self.model_queries += int(k.get('num_samples', 5000))   # LIME's default
        return self._inner.explain_instance(*a, **k)


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
    path = os.path.join(out_dir, f'ladder_ds{ds}.json')
    warnings.filterwarnings('ignore')
    topQ = Q_BY_DS[ds]
    fid = {m: {a: [] for a in ARMS} for m in models}
    cost = {m: {a: [] for a in ARMS} for m in models}      # true target predict_proba spent in LIME
    n_eval = None; t0 = time.time()
    for s in range(NSPLIT):
        a1, a2 = load_dataset(ds, seed=s)
        X_train, X_test, y_train, y_test, X_test_t, X_test_s, y_test_t, y_test_s = a1
        nc = a2[2]; Xt = np.asarray(X_test_t.values, float); n_eval = len(Xt)
        lb = int((topQ // nc) * 0.5 + 1); ub = int((topQ // nc) * (nc + 0.5) + 1)
        random.seed(s); np.random.seed(s)
        mega = mega_sample_generation(X_test_s.to_numpy(), y_test_s, nc, [SIZE], HMS)
        # the adversary's OWN discretizer, fit on its auxiliary pool -- no target access at all
        expl_aux = _CountingExplainer(
            lime.lime_tabular.LimeTabularExplainer(X_test_s.values, discretize_continuous=True))
        for m in models:
            mn = MODELS.get(m, str(m))
            try:
                tm, mn = load_model(m, X_train, y_train)
                for arm in ARMS:
                    random.seed(ATTACK_SEED + s); np.random.seed(ATTACK_SEED + s)   # paired
                    expl_aux.reset()
                    if arm == 'blind':
                        V, P, _ = traverse_explanations_LIME(
                            mega[0][0], expl_aux, tm, lb, ub, topQ, NFE, a2,
                            feature_select='random', use_threshold=False, use_explanation=False)
                    elif arm == 'selfgrid':
                        V, P, _ = traverse_explanations_LIME(
                            mega[0][0], expl_aux, tm, lb, ub, topQ, NFE, a2,
                            feature_select='random', use_threshold=True, use_explanation=False)
                    else:   # ours -- selfgrid plus Phase 1 diverse generation and Phase 2 boundary search
                        V, P, _ = traverse_explanations_LIME3(
                            mega[0][0], expl_aux, tm, lb, ub, topQ, NFE, a2, mn, X_train, y_train,
                            use_diverse=True, diverse_method='manifold', div_cap=DIV_CAP,
                            feature_select='random', use_threshold=True, use_explanation=False)
                    fid[m][arm].append(_fid(mn, V, P, tm, Xt, nc))
                    # sanity guard: these arms must never touch the explanation endpoint
                    assert expl_aux.calls == 0, f'{arm} made {expl_aux.calls} explanation calls'
                    cost[m][arm].append(expl_aux.model_queries)
            except Exception:
                traceback.print_exc()
                for a in ARMS:
                    if len(fid[m][a]) < s + 1:
                        fid[m][a].append(float('nan'))
                    if len(cost[m][a]) < s + 1:
                        cost[m][a].append(float('nan'))
        print(f'  split {s} done ({round(time.time()-t0,1)}s)', flush=True)

    borrowed = _borrow_autolycus(ds, out_dir)
    rows = []
    for m in models:
        mn = MODELS.get(m, str(m))
        A = {a: np.array(fid[m][a], float) for a in ARMS}
        A[BORROWED] = borrowed.get(mn.lower(), np.full(NSPLIT, np.nan))
        row = {'dataset': DATASETS[ds], 'model': mn, 'q': topQ, 'nsplit': NSPLIT, 'n_eval': n_eval}
        for a in ARMS + [BORROWED]:
            row[f'{a}_mean'] = round(float(np.nanmean(A[a])), 4)
            row[f'{a}_all'] = [round(x, 4) for x in A[a]]
        row['borrowed_autolycus'] = f'autolycus_ablation_ms_ds{ds}.json:default_all'
        for lab, x, y in CONTRASTS:
            u, v = A[x], A[y]
            ok = ~(np.isnan(u) | np.isnan(v))
            if ok.sum() < 2:
                d, p, ties = float('nan'), 1.0, 0
            else:
                uu, vv = u[ok], v[ok]
                ties = int(np.sum(uu - vv == 0))
                d = round(float(uu.mean() - vv.mean()), 4)
                p = float(wilcoxon(uu, vv).pvalue) if not np.allclose(uu, vv) else 1.0
            row[lab] = d; row[f'{lab}_p'] = p; row[f'{lab}_ties'] = ties
        rows.append(row)
        print('ROW', row['dataset'], row['model'],
              '| C4', row['c4_ours_vs_autolycus'], f"(p={row['c4_ours_vs_autolycus_p']:.3f})",
              '| C5', row['c5_ours_vs_selfgrid'], f"(p={row['c5_ours_vs_selfgrid_p']:.3f})",
              flush=True)
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
