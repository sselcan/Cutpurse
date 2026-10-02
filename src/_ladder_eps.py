"""Run the step-size sensitivity experiment for Cutpurse and Autolycus.

Both methods are evaluated under the unit and feature-scaled perturbation rules
using the paired multi-split protocol.
"""
import os, json, time, argparse, warnings, traceback
import numpy as np
import random
import lime, lime.lime_tabular
from scipy.stats import wilcoxon
from attack_utils import (load_dataset, load_model, mega_sample_generation, _fit_one_surrogate,
                          traverse_explanations_LIME, traverse_explanations_LIME3)

DATASETS = {1: 'crop', 2: 'adult', 3: 'breast', 9: 'pendigits'}
MODELS = {0: 'DT', 1: 'LR', 2: 'NB', 3: 'KNN', 4: 'RF'}
DEFAULT_MODELS = [1, 2, 3, 0, 4]
REPS = {'dt': 20, 'rdf': 10, 'nb': 1, 'lr': 1, 'knn': 1}
Q_BY_DS = {1: 1000, 2: 1000, 3: 100, 9: 1000}
SIZE, NFE, HMS, NSPLIT, DIV_CAP = 1, 3, 1, 10, 15     # matches _ladder.py / _defense_sweep.py
ARMS = ['autolycus', 'autolycus_eps', 'ours', 'ours_eps1']


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


def _wp(u, v):
    u, v = np.asarray(u, float), np.asarray(v, float)
    ok = ~(np.isnan(u) | np.isnan(v))
    if ok.sum() < 2:
        return float('nan'), 1.0
    u, v = u[ok], v[ok]
    return (round(float(u.mean() - v.mean()), 4),
            float(wilcoxon(u, v).pvalue) if not np.allclose(u, v) else 1.0)


def run_ds(ds, models=DEFAULT_MODELS, out_dir='paper_results_eps'):
    os.makedirs(out_dir, exist_ok=True)
    warnings.filterwarnings('ignore')
    path = os.path.join(out_dir, f'ladder_eps_ds{ds}.json')
    topQ = Q_BY_DS[ds]
    fid = {m: {a: [] for a in ARMS} for m in models}
    eps_med = None; n_eval = None; t0 = time.time()
    for s in range(NSPLIT):
        a1, a2 = load_dataset(ds, seed=s)
        X_train, X_test, y_train, y_test, X_test_t, X_test_s, y_test_t, y_test_s = a1
        nc = a2[2]; Xt = np.asarray(X_test_t.values, float); n_eval = len(Xt)
        eps_med = float(np.median(np.asarray(a2[5], float)))
        lb = int((topQ // nc) * 0.5 + 1); ub = int((topQ // nc) * (nc + 0.5) + 1)
        random.seed(s); np.random.seed(s)
        mega = mega_sample_generation(X_test_s.to_numpy(), y_test_s, nc, [SIZE], HMS)
        expl_tgt = lime.lime_tabular.LimeTabularExplainer(X_train.values, discretize_continuous=True)
        expl_aux = lime.lime_tabular.LimeTabularExplainer(X_test_s.values, discretize_continuous=True)
        for m in models:
            try:
                tm, mn = load_model(m, X_train, y_train)
            except Exception:
                traceback.print_exc(); continue
            for arm in ARMS:
                random.seed(1000 + s); np.random.seed(1000 + s)   # paired, matches ladder exactly
                try:
                    if arm in ('autolycus', 'autolycus_eps'):
                        # autolycus_eps = the published baseline given the SAME per-feature step
                        # rule \sys uses, so the margin cannot be credited to the step rule.
                        V, P, _ = traverse_explanations_LIME(
                            mega[0][0], expl_tgt, tm, lb, ub, topQ, NFE, a2,
                            eps_per_feature=(arm == 'autolycus_eps'))
                    else:
                        V, P, _ = traverse_explanations_LIME3(
                            mega[0][0], expl_aux, tm, lb, ub, topQ, NFE, a2, mn,
                            X_train, y_train, use_diverse=True, diverse_method='manifold',
                            div_cap=DIV_CAP, feature_select='random', use_threshold=True,
                            use_explanation=False,
                            eps_override=(1.0 if arm == 'ours_eps1' else None))
                    fid[m][arm].append(_fid(mn, V, P, tm, Xt, nc))
                except Exception:
                    traceback.print_exc(); fid[m][arm].append(float('nan'))
        print(f'  split {s} done ({round(time.time()-t0,1)}s)', flush=True)

    rows = []
    for m in models:
        mn = MODELS.get(m, str(m))
        A = fid[m]['autolycus']; AE = fid[m]['autolycus_eps']
        O = fid[m]['ours']; O1 = fid[m]['ours_eps1']
        row = {'dataset': DATASETS[ds], 'model': mn, 'q': topQ, 'nsplit': NSPLIT, 'n_eval': n_eval,
               'eps_median': eps_med,
               'autolycus_mean': round(float(np.nanmean(A)), 4),
               'autolycus_eps_mean': round(float(np.nanmean(AE)), 4),
               'autolycus_eps_all': [round(x, 4) for x in AE],
               'ours_mean': round(float(np.nanmean(O)), 4),
               'ours_eps1_mean': round(float(np.nanmean(O1)), 4),
               'autolycus_all': [round(x, 4) for x in A],
               'ours_all': [round(x, 4) for x in O],
               'ours_eps1_all': [round(x, 4) for x in O1]}
        row['ours_vs_aut'], row['ours_vs_aut_p'] = _wp(O, A)       # reproduces Table VI
        row['eps1_vs_aut'], row['eps1_vs_aut_p'] = _wp(O1, A)      # step-size-matched margin
        row['eps_effect'], row['eps_effect_p'] = _wp(O, O1)        # step size alone (our side)
        row['ours_vs_autEPS'], row['ours_vs_autEPS_p'] = _wp(O, AE)  # vs the step-matched baseline
        row['autEPS_vs_aut'], row['autEPS_vs_aut_p'] = _wp(AE, A)    # does the step rule help THEM?
        rows.append(row)
        print(f"ROW {row['dataset']} {mn} | ours-aut {row['ours_vs_aut']} (p={row['ours_vs_aut_p']:.3f})"
              f" | ours-autEPS {row['ours_vs_autEPS']} (p={row['ours_vs_autEPS_p']:.3f})"
              f" | autEPS-aut {row['autEPS_vs_aut']} (p={row['autEPS_vs_aut_p']:.3f})"
              f" | eps1-aut {row['eps1_vs_aut']} (p={row['eps1_vs_aut_p']:.3f})", flush=True)
        with open(path, 'w') as f:
            json.dump(rows, f, indent=2)
    return rows


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--ds', type=int, required=True)
    ap.add_argument('--models', type=str, default=','.join(map(str, DEFAULT_MODELS)))
    ap.add_argument('--out', type=str, default='paper_results_eps')
    ap.add_argument('--q', type=int, default=None)
    a = ap.parse_args()
    if a.q:
        Q_BY_DS[a.ds] = [a.q][0]
    run_ds(a.ds, models=[int(x) for x in a.models.split(',') if x != ''], out_dir=a.out)
