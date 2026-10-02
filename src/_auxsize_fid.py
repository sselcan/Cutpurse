"""Measure how grid-fitting pool size affects RQ3 with quartile discretization.

Only the attacker-side grid-fitting pool varies; the stored no-edge and
service-grid arms provide the paired comparison. Results are written as JSON.
"""
import os, json, time, argparse, warnings, traceback
import numpy as np
import random
import lime, lime.lime_tabular
from scipy.stats import wilcoxon
from attack_utils import (load_dataset, load_model, mega_sample_generation, _fit_one_surrogate,
                          traverse_explanations_LIME)

DATASETS = {1: 'crop', 3: 'breast', 4: 'nursery', 5: 'mushroom', 9: 'pendigits'}
MODELS = {0: 'DT', 1: 'LR', 2: 'NB', 3: 'KNN', 4: 'RF'}
DEFAULT_MODELS = [1, 2, 3, 0, 4]
REPS = {'dt': 20, 'rdf': 10, 'nb': 1, 'lr': 1, 'knn': 1}
Q_BY_DS = {1: 1000, 3: 100, 4: 1000, 5: 1000, 9: 1000}
SIZES = [1, 2, 3, 5, 10]          # auxiliary samples per class fitted to the discretizer
NFE, HMS, NSPLIT = 3, 1, 10
BORROW = {1: 'lime_auxdisc_ds1.json', 3: 'lime_auxdisc_shadow_ds3.json',
          4: 'lime_auxdisc_shadow_ds4.json', 5: 'lime_auxdisc_shadow_ds5.json',
          9: 'lime_auxdisc_ds9.json'}   # stored arms, used only as a cross-check


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


def _borrow(ds, out_dir):
    p = os.path.join(out_dir, BORROW[ds])
    if not os.path.exists(p):
        print(f'  [warn] {p} missing, nothresh/tgt will be NaN', flush=True); return {}
    return {r['model'].upper(): {k: np.array(r[k + '_all'], float) for k in ('nothresh', 'tgt')}
            for r in json.load(open(p))}


def _wp(u, v):
    ok = ~(np.isnan(u) | np.isnan(v))
    if ok.sum() < 2:
        return float('nan'), 1.0
    u, v = u[ok], v[ok]
    return (round(float(u.mean() - v.mean()), 4),
            float(wilcoxon(u, v).pvalue) if not np.allclose(u, v) else 1.0)


def run_ds(ds, models=DEFAULT_MODELS, out_dir='paper_results'):
    os.makedirs(out_dir, exist_ok=True)
    warnings.filterwarnings('ignore')
    path = os.path.join(out_dir, f'auxsize_fid_ds{ds}.json')
    topQ = Q_BY_DS[ds]
    fid = {m: {k: [] for k in ['nothresh', 'tgt'] + SIZES} for m in models}
    n_eval = None; t0 = time.time()
    for s in range(NSPLIT):
        a1, a2 = load_dataset(ds, seed=s)
        X_train, _, y_train, _, X_test_t, X_test_s, _, y_test_s = a1
        nc = a2[2]; Xt = np.asarray(X_test_t.values, float); n_eval = len(Xt)
        lb = int((topQ // nc) * 0.5 + 1); ub = int((topQ // nc) * (nc + 0.5) + 1)
        random.seed(s); np.random.seed(s)
        # one draw: index 0 is the attack's n=1 seed set (identical to the stored runs),
        # the rest are the per-class auxiliary samples the discretizer is fitted on
        mega = mega_sample_generation(X_test_s.to_numpy(), y_test_s, nc, SIZES, HMS)
        seed_set = mega[0][0]
        expl_tgt = lime.lime_tabular.LimeTabularExplainer(X_train.values, discretize_continuous=True)
        for m in models:
            try:
                tm, mn = load_model(m, X_train, y_train)
            except Exception:
                traceback.print_exc(); continue
            # the two reference arms are recomputed here rather than borrowed, so that leak(n) is
            # paired inside this run; the stored _lime_auxdisc.py values are then a cross-check
            for key, expl, ut in ([('nothresh', expl_tgt, False), ('tgt', expl_tgt, True)] +
                                  [(n, lime.lime_tabular.LimeTabularExplainer(
                                      np.asarray(mega[0][i], float), discretize_continuous=True), True)
                                   for i, n in enumerate(SIZES)]):
                random.seed(1000 + s); np.random.seed(1000 + s)     # every arm paired within the split
                try:
                    V, P, _ = traverse_explanations_LIME(seed_set, expl, tm, lb, ub, topQ, NFE, a2,
                                                        feature_select='explanation', use_threshold=ut)
                    fid[m][key].append(_fid(mn, V, P, tm, Xt, nc))
                except Exception:
                    traceback.print_exc(); fid[m][key].append(float('nan'))
        print(f'  split {s} done ({round(time.time()-t0,1)}s)', flush=True)

    borrowed = _borrow(ds, out_dir); rows = []
    for m in models:
        mn = MODELS.get(m, str(m))
        b = borrowed.get(mn, {})
        nt = np.array(fid[m]['nothresh'], float); tg = np.array(fid[m]['tgt'], float)
        row = {'dataset': DATASETS[ds], 'model': mn, 'q': topQ, 'nsplit': NSPLIT, 'n_eval': n_eval,
               'sizes': SIZES, 'aux_pool': int(len(X_test_s)), 'n_classes': nc,
               'nothresh_all': [round(x, 4) for x in nt], 'tgt_all': [round(x, 4) for x in tg],
               'nothresh_mean': round(float(np.nanmean(nt)), 4),
               'tgt_mean': round(float(np.nanmean(tg)), 4),
               'crosscheck_stored_thr_tgt': round(float(
                   np.nanmean(b.get('tgt', [np.nan])) - np.nanmean(b.get('nothresh', [np.nan]))), 4)}
        for n in SIZES:
            a = np.array(fid[m][n], float)
            row[f'aux{n}_all'] = [round(x, 4) for x in a]
            row[f'aux{n}_mean'] = round(float(np.nanmean(a)), 4)
            row[f'thr_aux{n}'], row[f'thr_aux{n}_p'] = _wp(a, nt)
            row[f'leak{n}'], row[f'leak{n}_p'] = _wp(tg, a)
        rows.append(row)
        print('ROW', row['dataset'], mn,
              '| thr_tgt', round(float(np.nanmean(tg) - np.nanmean(nt)), 4),
              '| leak by n', [row[f'leak{n}'] for n in SIZES], flush=True)
        with open(path, 'w') as f:
            json.dump(rows, f, indent=2)
    return rows


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--ds', type=int, required=True)
    ap.add_argument('--models', type=str, default=','.join(map(str, DEFAULT_MODELS)))
    ap.add_argument('--out', type=str, default='paper_results')
    a = ap.parse_args()
    run_ds(a.ds, models=[int(x) for x in a.models.split(',') if x != ''], out_dir=a.out)
