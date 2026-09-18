"""
LIME diverse-budget TUNING sweep  --  Autolycus-LIME vs ours at several diverse budgets.

For each (dataset, model) we sweep the query budget Q and, at each Q, compare:
  base          = Autolycus base-LIME                       (traverse_explanations_LIME)
  ours @ cap=c  = LIME3 boundary + LIME-threshold diverse   (traverse_explanations_LIME3,
                  diverse budget = up to c diverse samples; cap=0 => no diverse = boundary only)

Why the CAP is the knob: the Phase-1 diverse budget is n_div = clip(Q*div_frac, 0, div_cap).
At useful budgets div_frac*Q exceeds the cap, so the CAP (max diverse samples) is what actually
binds -- that is the "diverse budget" to tune. cap=0 disables diverse entirely (the old ablation's
'nodiv' endpoint). div_frac is held fixed and only protects very small Q from starvation.

LIME convention (per project): n = size = 3 seeds/class, k = nfe = 1 top feature.

Each traverse returns its ACTUAL query count -> x-axis is real queries, not the cap. Surrogate
fidelity is MEAN over refits (no argmax/test-set selection). Paired sample sets across all arms;
per-set reseeding makes the Q-sweep nested. Fixed seed => reproducible.

Standalone (one process per dataset, run in parallel):
    python lime_paper_run.py --ds 4 --qs 100,250,500,1000 --caps 0,5,15,30 --div_frac 0.3 \
                             --hms 10 --size 3 --nfe 1 --seed 0 --out .
"""
import os, json, time, argparse, warnings, traceback
import numpy as np
import random
import lime, lime.lime_tabular
from scipy.stats import wilcoxon
from attack_utils import (load_dataset, load_model, mega_sample_generation,
                          _fit_one_surrogate, traverse_explanations_LIME,
                          traverse_explanations_LIME3)

DATASETS = {0: 'iris', 1: 'crop', 2: 'adult', 3: 'breast', 4: 'nursery', 5: 'mushroom'}
MODELS   = {0: 'DT', 1: 'LR', 2: 'NB', 3: 'KNN', 4: 'RF', 5: 'MLP'}
DEFAULT_MODELS = [1, 2, 3, 0, 4]         # LR, NB, KNN, DT, RF  (mlp unsupported by _fit_one_surrogate)
DIV_CAPS = [0, 5, 15, 30]                # diverse budget = max diverse samples; 0 = no diverse
DIV_FRAC = 0.3
REPS = {'dt': 20, 'rdf': 10, 'nb': 1, 'lr': 1, 'knn': 1}


def _fidelity(mname, V, P, t_model, Xt, n_classes):
    if mname == 'nb':
        V = np.clip(np.asarray(V, float), 0, None)
    P = np.asarray(P)
    ytar = t_model.predict(Xt)
    if len(np.unique(P)) < 2:
        # only one class among the traversed labels -> a real surrogate degenerates to a
        # constant predictor; report that predictor's agreement (honest degenerate fidelity,
        # and avoids a fit crash that would otherwise drop the whole cell).
        cls = P[0] if len(P) else 0
        return float(np.mean(ytar == cls))
    fids = []
    for k in range(REPS.get(mname, 1)):
        try:
            sm = _fit_one_surrogate(mname, np.asarray(V, float), P, n_classes, seed=k)
            fids.append(float(np.mean(ytar == sm.predict(Xt))))
        except Exception:
            fids.append(float('nan'))
    return float(np.nanmean(fids)) if fids else float('nan')


def _traverse(kind, seed_set, expl, t_model, mname, lb, ub, Q, args2, X_train, y_train, nfe,
              div_cap=10, div_frac=DIV_FRAC):
    """Returns (visited_samples, preds, query_count)."""
    if kind == 'base':
        return traverse_explanations_LIME(seed_set, expl, t_model, lb, ub, Q, nfe, args2)
    return traverse_explanations_LIME3(seed_set, expl, t_model, lb, ub, Q, nfe, args2,
                                       mname, X_train, y_train, use_diverse=(div_cap > 0),
                                       diverse_method='lime_threshold',
                                       div_frac=div_frac, div_cap=div_cap)


def run_dataset_divsweep(ds, models=DEFAULT_MODELS, q_list=(100, 250, 500, 1000),
                         div_caps=DIV_CAPS, div_frac=DIV_FRAC, hms=10, size=3, nfe=1,
                         seed=0, out_dir='.'):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f'lime_divsweep_n{size}_k{nfe}_ds{ds}.json')
    warnings.filterwarnings('ignore')
    q_list = list(q_list); div_caps = list(div_caps)
    args1, args2 = load_dataset(ds)
    X_train, X_test, y_train, y_test, X_test_t, X_test_s, y_test_t, y_test_s = args1
    n_classes = args2[2]
    Xt = np.asarray(X_test_t.values, float)

    rows = []
    for m in models:
        mname = MODELS.get(m, str(m)); t0 = time.time()
        try:
            random.seed(seed); np.random.seed(seed)
            t_model, mname = load_model(m, X_train, y_train)
            expl = lime.lime_tabular.LimeTabularExplainer(X_train.values, discretize_continuous=True)
            mega = mega_sample_generation(X_test_s.to_numpy(), y_test_s, n_classes, [size], hms)

            base_fid = {Q: [] for Q in q_list}; base_q = {Q: [] for Q in q_list}
            ours_fid = {c: {Q: [] for Q in q_list} for c in div_caps}
            ours_q   = {c: {Q: [] for Q in q_list} for c in div_caps}

            for Q in q_list:
                lb = int((Q // n_classes) * 0.5 + 1); ub = int((Q // n_classes) * (n_classes + 0.5) + 1)
                for i in range(hms):
                    random.seed(seed + 1 + i); np.random.seed(seed + 1 + i)
                    V, P, nq = _traverse('base', mega[i][0], expl, t_model, mname, lb, ub, Q,
                                         args2, X_train, y_train, nfe)
                    base_fid[Q].append(_fidelity(mname, V, P, t_model, Xt, n_classes)); base_q[Q].append(int(nq))
                for c in div_caps:
                    for i in range(hms):
                        random.seed(seed + 1 + i); np.random.seed(seed + 1 + i)
                        V, P, nq = _traverse('ours', mega[i][0], expl, t_model, mname, lb, ub, Q,
                                             args2, X_train, y_train, nfe, div_cap=c, div_frac=div_frac)
                        ours_fid[c][Q].append(_fidelity(mname, V, P, t_model, Xt, n_classes)); ours_q[c][Q].append(int(nq))

            def _pack(fd, qd):
                return {'fid_mean': [round(float(np.mean(fd[Q])), 4) for Q in q_list],
                        'fid_std':  [round(float(np.std(fd[Q])), 4) for Q in q_list],
                        'fid_all':  [[round(x, 4) for x in fd[Q]] for Q in q_list],
                        'q_actual': [round(float(np.mean(qd[Q])), 1) for Q in q_list]}

            row = {'dataset': DATASETS[ds], 'model': mname, 'ds': ds, 'm': m,
                   'q_cap': q_list, 'div_caps': div_caps, 'div_frac': div_frac,
                   'hms': hms, 'size': size, 'nfe': nfe, 'seed': seed,
                   'base': _pack(base_fid, base_q),
                   'ours': {str(c): _pack(ours_fid[c], ours_q[c]) for c in div_caps},
                   'secs': round(time.time() - t0, 1)}
            topQ = q_list[-1]
            bt = np.array(base_fid[topQ], float)
            best_c = max(div_caps, key=lambda c: float(np.mean(ours_fid[c][topQ])))
            L = np.array(ours_fid[best_c][topQ], float)
            row['best_div_cap_top'] = best_c
            row['delta_top_bestdiv'] = round(float(L.mean() - bt.mean()), 4)
            row['wilcoxon_p_bestdiv'] = (float(wilcoxon(L, bt).pvalue) if not np.allclose(L, bt) else 1.0)
        except Exception as e:
            traceback.print_exc()
            row = {'dataset': DATASETS[ds], 'model': mname, 'ds': ds, 'm': m,
                   'error': repr(e), 'secs': round(time.time() - t0, 1)}
        rows.append(row)
        print('ROWDONE', row['dataset'], row['model'], 'best_cap', row.get('best_div_cap_top'),
              'delta', row.get('delta_top_bestdiv'), 'p', row.get('wilcoxon_p_bestdiv'),
              'secs', row['secs'], flush=True)
        with open(path, 'w') as f:
            json.dump(rows, f, indent=2)
    return rows


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--ds', type=int, required=True)
    ap.add_argument('--qs', type=str, default='100,250,500,1000')
    ap.add_argument('--caps', type=str, default=','.join(map(str, DIV_CAPS)))
    ap.add_argument('--div_frac', type=float, default=DIV_FRAC)
    ap.add_argument('--hms', type=int, default=10)
    ap.add_argument('--size', type=int, default=3)
    ap.add_argument('--nfe', type=int, default=1)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--models', type=str, default=','.join(map(str, DEFAULT_MODELS)))
    ap.add_argument('--out', type=str, default='.')
    a = ap.parse_args()
    qs = [int(x) for x in a.qs.split(',') if x != '']
    caps = [int(x) for x in a.caps.split(',') if x != '']
    mods = [int(x) for x in a.models.split(',') if x != '']
    run_dataset_divsweep(a.ds, models=mods, q_list=qs, div_caps=caps, div_frac=a.div_frac,
                         hms=a.hms, size=a.size, nfe=a.nfe, seed=a.seed, out_dir=a.out)
