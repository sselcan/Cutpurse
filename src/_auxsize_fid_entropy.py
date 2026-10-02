"""Measure grid-fitting pool-size sensitivity for entropy discretization.

This is the label-dependent counterpart to ``_auxsize_fid.py``. The attacker
uses target predictions for the grid-fitting labels; results are written as JSON.
"""
import os, json, time, argparse, warnings, traceback
import numpy as np
import random
import lime, lime.lime_tabular
from scipy.stats import wilcoxon
from attack_utils import (load_dataset, load_model, mega_sample_generation, _fit_one_surrogate,
                          traverse_explanations_LIME)

DATASETS = {1: 'crop', 2: 'adult', 3: 'breast', 4: 'nursery', 5: 'mushroom', 9: 'pendigits'}
CATEGORICAL_DS = {4, 5}
MODELS = {0: 'DT', 1: 'LR', 2: 'NB', 3: 'KNN', 4: 'RF'}
DEFAULT_MODELS = [1, 2, 3, 0, 4]
REPS = {'dt': 20, 'rdf': 10, 'nb': 1, 'lr': 1, 'knn': 1}
Q_BY_DS = {1: 1000, 2: 1000, 3: 100, 9: 1000}
SIZES = [1, 2, 3, 5, 10]           # auxiliary samples per class fitted to the discretizer
NFE, HMS, NSPLIT = 3, 1, 10


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


def _mk_expl(data, labels, disc):
    """Construct a LIME explainer using the shared experiment RNG."""
    kw = dict(discretize_continuous=True, discretizer=disc)
    if disc == 'entropy':
        kw['training_labels'] = np.asarray(labels)
    return lime.lime_tabular.LimeTabularExplainer(np.asarray(data, float), **kw)


def _cuts(expl):
    d = getattr(expl, 'discretizer', None)
    if d is None or not getattr(d, 'names', None):
        return float('nan')
    return round(float(np.mean([max(len(v) - 1, 0) for v in d.names.values()])), 2)


def _wp(u, v):
    ok = ~(np.isnan(u) | np.isnan(v))
    if ok.sum() < 2:
        return float('nan'), 1.0
    u, v = u[ok], v[ok]
    return (round(float(u.mean() - v.mean()), 4),
            float(wilcoxon(u, v).pvalue) if not np.allclose(u, v) else 1.0)


def run_ds(ds, disc='entropy', models=DEFAULT_MODELS, out_dir='paper_results_disc'):
    os.makedirs(out_dir, exist_ok=True)
    warnings.filterwarnings('ignore')
    path = os.path.join(out_dir, f'auxsize_fid_{disc}_ds{ds}.json')
    topQ = Q_BY_DS[ds]
    fid = {m: {k: [] for k in ['nothresh', 'tgt'] + SIZES} for m in models}
    cuts = {m: {k: [] for k in ['tgt'] + SIZES} for m in models}
    agree = {m: {n: [] for n in SIZES} for m in models}
    n_eval = aux_pool = None; t0 = time.time()
    for s in range(NSPLIT):
        a1, a2 = load_dataset(ds, seed=s)
        X_train, _, y_train, _, X_test_t, X_test_s, _, y_test_s = a1
        nc = a2[2]; Xt = np.asarray(X_test_t.values, float); n_eval = len(Xt)
        aux_pool = int(len(X_test_s))
        lb = int((topQ // nc) * 0.5 + 1); ub = int((topQ // nc) * (nc + 0.5) + 1)
        random.seed(s); np.random.seed(s)
        # index 0 is the attack's n=1 seed set (identical to the stored runs); the rest are the
        # per-class auxiliary samples the discretizer is fitted on
        mega = mega_sample_generation(X_test_s.to_numpy(), y_test_s, nc, SIZES, HMS)
        seed_set = mega[0][0]
        expl_tgt = _mk_expl(X_train.values, np.asarray(y_train), disc)
        for m in models:
            try:
                tm, mn = load_model(m, X_train, y_train)
            except Exception:
                traceback.print_exc(); continue
            arms = [('nothresh', expl_tgt, False), ('tgt', expl_tgt, True)]
            for i, n in enumerate(SIZES):
                data = np.asarray(mega[0][i], float)
                # sample_set_generation walks classes in order taking n each, so the true labels are
                # exactly repeat(arange(nc), n). Assert it, then label by querying the target instead.
                assert len(data) == nc * n, f'expected {nc*n} rows at n={n}, got {len(data)}'
                y_true = np.repeat(np.arange(nc), n)
                y_qry = tm.predict(data)
                agree[m][n].append(float(np.mean(np.asarray(y_qry) == y_true)))
                e = _mk_expl(data, y_qry, disc)
                cuts[m][n].append(_cuts(e))
                arms.append((n, e, True))
            cuts[m]['tgt'].append(_cuts(expl_tgt))
            for key, expl, ut in arms:
                random.seed(1000 + s); np.random.seed(1000 + s)      # every arm paired within the split
                try:
                    V, P, _ = traverse_explanations_LIME(seed_set, expl, tm, lb, ub, topQ, NFE, a2,
                                                         feature_select='explanation', use_threshold=ut)
                    fid[m][key].append(_fid(mn, V, P, tm, Xt, nc))
                except Exception:
                    traceback.print_exc(); fid[m][key].append(float('nan'))
        print(f'  split {s} done ({round(time.time()-t0,1)}s)', flush=True)

    rows = []
    for m in models:
        mn = MODELS.get(m, str(m))
        nt = np.array(fid[m]['nothresh'], float); tg = np.array(fid[m]['tgt'], float)
        row = {'dataset': DATASETS[ds], 'model': mn, 'disc': disc, 'q': topQ, 'nsplit': NSPLIT,
               'n_eval': n_eval, 'sizes': SIZES, 'aux_pool': aux_pool, 'n_classes': nc,
               'pool_exhausted_at': [n for n in SIZES if n * nc >= (aux_pool or 0)],
               'nothresh_all': [round(x, 4) for x in nt], 'tgt_all': [round(x, 4) for x in tg],
               'nothresh_mean': round(float(np.nanmean(nt)), 4),
               'tgt_mean': round(float(np.nanmean(tg)), 4),
               'thr_tgt': round(float(np.nanmean(tg) - np.nanmean(nt)), 4),
               'cuts_tgt': round(float(np.nanmean(cuts[m]['tgt'])), 2)}
        for n in SIZES:
            a = np.array(fid[m][n], float)
            row[f'aux{n}_all'] = [round(x, 4) for x in a]
            row[f'aux{n}_mean'] = round(float(np.nanmean(a)), 4)
            row[f'thr_aux{n}'], row[f'thr_aux{n}_p'] = _wp(a, nt)
            row[f'leak{n}'], row[f'leak{n}_p'] = _wp(tg, a)
            row[f'cuts_aux{n}'] = round(float(np.nanmean(cuts[m][n])), 2)
            row[f'gt_agree{n}'] = round(float(np.nanmean(agree[m][n])), 3)
        rows.append(row)
        print('ROW', row['dataset'], mn, f'[{disc}]', '| thr_tgt', row['thr_tgt'],
              '| thr_aux by n', [row[f'thr_aux{n}'] for n in SIZES],
              '| leak by n', [row[f'leak{n}'] for n in SIZES],
              '| cuts', row['cuts_tgt'], [row[f'cuts_aux{n}'] for n in SIZES], flush=True)
        with open(path, 'w') as f:
            json.dump(rows, f, indent=2)
    return rows


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--ds', type=int, required=True)
    ap.add_argument('--disc', type=str, default='entropy', choices=['entropy', 'decile', 'quartile'])
    ap.add_argument('--models', type=str, default=','.join(map(str, DEFAULT_MODELS)))
    ap.add_argument('--out', type=str, default='paper_results_disc')
    ap.add_argument('--sizes', type=str, default=None,
                    help='comma-separated samples/class, e.g. 1,2,3,5,10,25,50. Crop exhausts its '
                         '170-row pool at n=10 (17 classes); pendigits has room to n=110.')
    ap.add_argument('--allow-categorical', action='store_true')
    a = ap.parse_args()
    if a.sizes:
        SIZES = [int(x) for x in a.sizes.split(',') if x != '']
    if a.ds in CATEGORICAL_DS and not a.allow_categorical:
        raise SystemExit(f'ds {a.ds} ({DATASETS[a.ds]}) is fully categorical; its bin edges are '
                         f'encoding artefacts. Pass --allow-categorical to override.')
    run_ds(a.ds, disc=a.disc, models=[int(x) for x in a.models.split(',') if x != ''], out_dir=a.out)
