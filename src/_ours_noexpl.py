"""Does OUR attack still work with no explanation input at all?

E1/E4 report our method (diverse generation + boundary search + traversal). All three phases as
reported consume LIME output: Phase 1 uses diverse_method='lime_threshold', Phases 2-3 use
feature_select='explanation' and use_threshold=True. So we cannot claim the components are
"explanation-free" by construction -- it has to be measured.

Four paired arms, all on traverse_explanations_LIME3 (our method), multi-split:
  ours        = as reported (lime_threshold diverse + LIME ranking + target's LIME bin edge)
  ours_auxbin = geometric diverse + random features + bin edges from the ADVERSARY'S OWN
                discretizer (fit on X_test_s). The realistic explanation-free adversary: per E5
                it cannot read the target's explanation but can self-compute an equivalent grid.
  ours_noexp  = geometric diverse + random features + NO grid (midpoint init + bisection). Floor.
  base        = base Autolycus LIME (reference point)

Deltas:
  gap_vs_auxbin     = ours - ours_auxbin   -> the explanation's residual value over self-supply
  auxbin_grid_value = ours_auxbin - ours_noexp -> what the SELF-COMPUTED grid buys
  cost_of_blinding  = ours - ours_noexp    -> what the explanation buys over no grid at all
  noexp_vs_base     = ours_noexp - base    -> what explanation-free placement buys on its own

If noexp_vs_base is comparable to cost_of_blinding, an adversary can substitute better query
placement for the explanation signal -- the argument that the residual leak is recoverable.

    python _ours_noexpl.py --ds 1 --out paper_results
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
SIZE, NFE, HMS, NSPLIT, DIV_CAP = 1, 3, 1, 10, 15   # LIME main setting (n=1,k=3), fixed diverse budget
ARMS = ['ours', 'ours_auxbin', 'ours_noexp', 'base']


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
    path = os.path.join(out_dir, f'ours_noexpl_ds{ds}.json')
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
        expl = lime.lime_tabular.LimeTabularExplainer(X_train.values, discretize_continuous=True)
        # the adversary's OWN discretizer, fit on its auxiliary pool (self-computed grid, cf. E5)
        expl_aux = lime.lime_tabular.LimeTabularExplainer(X_test_s.values, discretize_continuous=True)
        for m in models:
            mn = MODELS.get(m, str(m))
            try:
                tm, mn = load_model(m, X_train, y_train)
                for arm in ARMS:
                    random.seed(4000 + s); np.random.seed(4000 + s)      # paired across arms
                    if arm == 'base':
                        V, P, _ = traverse_explanations_LIME(mega[0][0], expl, tm, lb, ub, topQ,
                                                             NFE, a2)
                    elif arm == 'ours':
                        V, P, _ = traverse_explanations_LIME3(
                            mega[0][0], expl, tm, lb, ub, topQ, NFE, a2, mn, X_train, y_train,
                            use_diverse=True, diverse_method='lime_threshold', div_cap=DIV_CAP,
                            feature_select='explanation', use_threshold=True)
                    elif arm == 'ours_auxbin':
                        # realistic explanation-free adversary: geometric diverse, random features,
                        # but snapping to bin edges from its OWN discretizer rather than the target's
                        V, P, _ = traverse_explanations_LIME3(
                            mega[0][0], expl_aux, tm, lb, ub, topQ, NFE, a2, mn, X_train, y_train,
                            use_diverse=True, diverse_method='manifold', div_cap=DIV_CAP,
                            feature_select='random', use_threshold=True)
                    else:   # ours_noexp -- same scaffold, no grid at all (the floor)
                        V, P, _ = traverse_explanations_LIME3(
                            mega[0][0], expl, tm, lb, ub, topQ, NFE, a2, mn, X_train, y_train,
                            use_diverse=True, diverse_method='manifold', div_cap=DIV_CAP,
                            feature_select='random', use_threshold=False)
                    fid[m][arm].append(_fid(mn, V, P, tm, Xt, nc))
            except Exception:
                traceback.print_exc()
                for a in ARMS:
                    if len(fid[m][a]) < s + 1:
                        fid[m][a].append(float('nan'))
        print(f'  split {s} done ({round(time.time()-t0,1)}s)', flush=True)

    rows = []
    for m in models:
        mn = MODELS.get(m, str(m))
        O = np.array(fid[m]['ours'], float)
        A = np.array(fid[m]['ours_auxbin'], float)
        N = np.array(fid[m]['ours_noexp'], float)
        B = np.array(fid[m]['base'], float)
        def wp(X, Y):
            ok = ~(np.isnan(X) | np.isnan(Y))
            if ok.sum() < 2: return (float('nan'), 1.0)
            x, y = X[ok], Y[ok]
            return (round(float(x.mean() - y.mean()), 4),
                    float(wilcoxon(x, y).pvalue) if not np.allclose(x, y) else 1.0)
        row = {'dataset': DATASETS[ds], 'model': mn, 'q': topQ, 'nsplit': NSPLIT, 'n_eval': n_eval,
               'ours_mean': round(float(np.nanmean(O)), 4),
               'ours_auxbin_mean': round(float(np.nanmean(A)), 4),
               'ours_noexp_mean': round(float(np.nanmean(N)), 4),
               'base_mean': round(float(np.nanmean(B)), 4),
               'gap_vs_auxbin': wp(O, A)[0], 'gap_vs_auxbin_p': wp(O, A)[1],   # explanation's residual value
               'auxbin_grid_value': wp(A, N)[0], 'auxbin_grid_value_p': wp(A, N)[1],
               'cost_of_blinding': wp(O, N)[0], 'cost_of_blinding_p': wp(O, N)[1],
               'noexp_vs_base': wp(N, B)[0], 'noexp_vs_base_p': wp(N, B)[1],
               'ours_vs_base': wp(O, B)[0], 'ours_vs_base_p': wp(O, B)[1],
               'ours_all': [round(x, 4) for x in O],
               'ours_auxbin_all': [round(x, 4) for x in A],
               'ours_noexp_all': [round(x, 4) for x in N],
               'base_all': [round(x, 4) for x in B]}
        rows.append(row)
        print('ROW', row['dataset'], row['model'],
              'gap_vs_auxbin', row['gap_vs_auxbin'], f"(p={row['gap_vs_auxbin_p']:.3f})",
              '| blind', row['cost_of_blinding'], '| noexp_vs_base', row['noexp_vs_base'], flush=True)
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
