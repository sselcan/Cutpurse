"""Three-curve budget sweep: Autolycus, Autolycus under the threshold defense, and ODYSSEUS.

The question a single-budget comparison cannot answer is whether withholding LIME's bin edges
*defends* or merely *delays*: if the defended baseline closes the gap as the budget grows, the
defense is a speed bump and the headline number is an artefact of where we stopped measuring.

Three arms, identical protocol, identical seed sets, re-run at each budget (the per-class visit
quotas lb/ub are recomputed per Q, so truncating a single max-Q run is NOT equivalent):

  autolycus  base traversal, target's LIME, bin edges available      (the published attack)
  defended   base traversal, target's LIME, bin edges withheld       (use_threshold=False)
  odysseus   Phase 1 + 2 + 3, own grid, random features, no LIME     (use_explanation=False)

Protocol is the multi-split one used by the ablations, NOT the single-split sweep: NSPLIT
independent train/test splits with the target retrained on each, one seed set per split, arms paired
within a split. Seeds match _ladder.py and _autolycus_ablation_ms.py exactly (split seed s, seed set
random.seed(s), attack random.seed(1000+s)), so at Q = Q_BY_DS[ds] the three curves must reproduce
the stored `ours`, `default` and `nothresh` cells. That is an internal consistency check, and it
means the figure and the defense table report the same quantity rather than two protocols.

    python _defense_sweep.py --ds 1 --out paper_results
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
DEFAULT_MODELS = [1, 2, 3, 0]
REPS = {'dt': 20, 'rdf': 10, 'nb': 1, 'lr': 1, 'knn': 1}
Q_LISTS = {1: (100, 250, 500, 1000), 9: (100, 250, 500),
           2: (100, 250, 500, 1000), 3: (25, 50, 75, 100),
           4: (100, 250, 500, 1000), 5: (100, 250, 500, 1000)}
SIZE, NFE, HMS, NSPLIT, DIV_CAP = 1, 3, 1, 10, 15   # LIME main setting (n=1, k=3)
Q_BY_DS = {1: 1000, 2: 1000, 3: 100, 4: 1000, 5: 1000, 9: 500}   # last sweep point per dataset
ARMS = ['autolycus', 'defended', 'odysseus']


def _fidelity(mname, V, P, t_model, Xt, n_classes):
    if mname == 'nb':
        V = np.clip(np.asarray(V, float), 0, None)
    P = np.asarray(P); ytar = t_model.predict(Xt)
    if len(np.unique(P)) < 2:
        return float(np.mean(ytar == (P[0] if len(P) else 0)))
    fids = []
    for k in range(REPS.get(mname, 1)):
        try:
            sm = _fit_one_surrogate(mname, np.asarray(V, float), P, n_classes, seed=k)
            fids.append(float(np.mean(ytar == sm.predict(Xt))))
        except Exception:
            fids.append(float('nan'))
    return float(np.nanmean(fids)) if fids else float('nan')


def run_ds(ds, models=DEFAULT_MODELS, out_dir='paper_results'):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f'defense_sweep_ds{ds}.json')
    warnings.filterwarnings('ignore')
    q_list = list(Q_LISTS[ds])
    rows = []; t0 = time.time()
    fid = {m: {a: {Q: [] for Q in q_list} for a in ARMS} for m in models}
    n_eval = None
    for s in range(NSPLIT):
        a1, a2 = load_dataset(ds, seed=s)              # independent split, target retrained below
        X_train, X_test, y_train, y_test, X_test_t, X_test_s, y_test_t, y_test_s = a1
        n_classes = a2[2]; Xt = np.asarray(X_test_t.values, float); n_eval = len(Xt)
        random.seed(s); np.random.seed(s)
        mega = mega_sample_generation(X_test_s.to_numpy(), y_test_s, n_classes, [SIZE], HMS)
        expl_tgt = lime.lime_tabular.LimeTabularExplainer(X_train.values, discretize_continuous=True)
        expl_aux = lime.lime_tabular.LimeTabularExplainer(X_test_s.values, discretize_continuous=True)
        for m in models:
            try:
                t_model, mname = load_model(m, X_train, y_train)
            except Exception:
                traceback.print_exc(); continue
            for Q in q_list:
                lb = int((Q // n_classes) * 0.5 + 1)
                ub = int((Q // n_classes) * (n_classes + 0.5) + 1)
                for arm in ARMS:
                    random.seed(1000 + s); np.random.seed(1000 + s)    # paired, matches ladder
                    try:
                        if arm == 'autolycus':
                            V, P, _ = traverse_explanations_LIME(
                                mega[0][0], expl_tgt, t_model, lb, ub, Q, NFE, a2)
                        elif arm == 'defended':
                            V, P, _ = traverse_explanations_LIME(
                                mega[0][0], expl_tgt, t_model, lb, ub, Q, NFE, a2,
                                feature_select='explanation', use_threshold=False)
                        else:
                            V, P, _ = traverse_explanations_LIME3(
                                mega[0][0], expl_aux, t_model, lb, ub, Q, NFE, a2, mname,
                                X_train, y_train, use_diverse=True, diverse_method='manifold',
                                div_cap=DIV_CAP, feature_select='random', use_threshold=True,
                                use_explanation=False)
                        fid[m][arm][Q].append(_fidelity(mname, V, P, t_model, Xt, n_classes))
                    except Exception:
                        traceback.print_exc(); fid[m][arm][Q].append(float('nan'))
        print(f'  split {s} done ({round(time.time()-t0,1)}s)', flush=True)

    for m in models:
        mname = MODELS.get(m, str(m)).lower()
        fidm = fid[m]
        row = {'dataset': DATASETS[ds], 'model': mname, 'q_list': q_list, 'hms': NSPLIT,
               'n_eval': n_eval, 'protocol': 'multi-split, target retrained per split',
               'check_at_Q': Q_BY_DS.get(ds)}
        for a in ARMS:
            row[f'{a}_mean'] = [round(float(np.nanmean(fidm[a][Q])), 4) for Q in q_list]
            row[f'{a}_std'] = [round(float(np.nanstd(fidm[a][Q])), 4) for Q in q_list]
            row[f'{a}_all'] = [[round(x, 4) for x in fidm[a][Q]] for Q in q_list]
        # does the defense hold as the budget grows, or merely delay?
        row['defense_gap'] = [round(float(np.nanmean(fidm['autolycus'][Q])
                                          - np.nanmean(fidm['defended'][Q])) * 100, 2)
                              for Q in q_list]
        row['ody_vs_defended'] = [round(float(np.nanmean(fidm['odysseus'][Q])
                                              - np.nanmean(fidm['defended'][Q])) * 100, 2)
                                  for Q in q_list]
        row['ody_vs_autolycus'] = [round(float(np.nanmean(fidm['odysseus'][Q])
                                               - np.nanmean(fidm['autolycus'][Q])) * 100, 2)
                                   for Q in q_list]
        rows.append(row)
        print('ROW', row['dataset'], row['model'],
              '| defense gap', row['defense_gap'],
              '| ody-def', row['ody_vs_defended'], flush=True)
        with open(path, 'w') as f:
            json.dump(rows, f, indent=2)
    return rows


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--ds', type=int, required=True)
    ap.add_argument('--models', type=str, default=','.join(map(str, DEFAULT_MODELS)))
    ap.add_argument('--out', type=str, default='paper_results')
    ap.add_argument('--qlist', type=str, default=None,
                    help='comma-separated budgets, overrides Q_LISTS for this dataset')
    a = ap.parse_args()
    if a.qlist:
        Q_LISTS[a.ds] = tuple(int(x) for x in a.qlist.split(','))
        Q_BY_DS[a.ds] = Q_LISTS[a.ds][-1]
    run_ds(a.ds, models=[int(x) for x in a.models.split(',') if x != ''], out_dir=a.out)
