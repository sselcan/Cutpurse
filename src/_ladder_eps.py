"""STEP-SIZE CONTROL: is \sys's margin over Autolycus an artefact of a larger perturbation step?

Every baseline arm in _ladder.py and _defense_sweep.py runs traverse_explanations_LIME, which
hardcodes `epsilon = 1` and ignores epsilon_set (as does the original Autolycus, utils.py:113).
\sys runs traverse_explanations_LIME3, which steps by the per-feature epsilon_set. The two are
therefore NOT matched on step size:

    pendigits  epsilon_set median 30.2   -> \sys steps ~30x further than every baseline
    crop                          8.0    -> ~8x
    adult                         2.0    -> ~2x
    breast                        0.073  -> ~14x SHORTER
    nursery, mushroom             1.0    -> identical, no difference at all

The two datasets where \sys shows no gain are exactly the two where it has no step-size advantage.
The paper explains that null mechanistically (no continuous axis, so no bin edges to snap to), and
that explanation is probably right, but the existing experiments cannot separate it from "\sys wins
where it takes bigger steps". This driver separates them.

Four arms, paired within split, identical seeds/budget/evaluation to _ladder.py and
_defense_sweep.py (split seed s, seed set random.seed(s), attack random.seed(1000+s)):

  autolycus      base traversal, target's LIME, eps = 1              (published attack)
  autolycus_eps  the SAME baseline given \sys's per-feature step     (strengthened baseline)
  ours           Phase 1+2+3, own grid, no explanation call, eps = epsilon_set   (as reported)
  ours_eps1      identical to `ours` but eps_override=1.0            (crippled to the baseline)

The step rule is a deliberate design choice, not an accident: a fixed step of 1 is meaningless on a
feature whose range is in the hundreds, and Autolycus already uses epsilon_set for SHAP. But only
\sys got it, so the published contrast credits the phases with whatever the step rule is worth.
There are two ways to match, and the honest claim needs both:

  ours - autolycus       must reproduce the stored Table VI margin (consistency check)
  ours - autolycus_eps   matched at OUR step: does the margin survive a strengthened baseline?
  ours_eps1 - autolycus  matched at THEIR step: does it survive when \sys is crippled instead?
  autolycus_eps - autolycus  is the per-feature step actually better, for them too?
  ours - ours_eps1       the step-size effect on our side alone

Read as: if the margin survives BOTH matchings, the step rule is not the explanation and the phases
carry it. `ours - autolycus_eps` is the one to headline, since strengthening the baseline is the
comparison a reviewer will ask for.

Run on the datasets where the confound exists at all; nursery/mushroom have epsilon_set all-1s and
are already matched.

    python _ladder_eps.py --ds 1 --out paper_results_eps
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
