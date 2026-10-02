"""Run RQ3 with LIME's label-dependent entropy discretizer.

The paired service- and attacker-grid configurations use target predictions as
the attacker-side grid-fitting labels. Results are written as JSON.
"""
import os, json, time, argparse, warnings
import numpy as np
import random
import lime, lime.lime_tabular
from scipy.stats import wilcoxon
from attack_utils import (load_dataset, load_model, mega_sample_generation, _fit_one_surrogate,
                          traverse_explanations_LIME)

DATASETS = {1: 'crop', 2: 'adult', 3: 'breast', 4: 'nursery', 5: 'mushroom', 9: 'pendigits'}
CATEGORICAL_DS = {4, 5}                 # integer-encoded, no genuine continuous axis (see \S7)
MODELS = {0: 'DT', 1: 'LR', 2: 'NB', 3: 'KNN', 4: 'RF'}
DEFAULT_MODELS = [1, 2, 3, 0, 4]
REPS = {'dt': 20, 'rdf': 10, 'nb': 1, 'lr': 1, 'knn': 1}
# Budgets as used for Table V: pendigits is 1000, NOT the 500 left in _lime_auxdisc.py.
Q_BY_DS = {1: [1000], 2: [1000], 3: [100], 9: [1000]}
SIZE, NFE, HMS, NSPLIT = 1, 3, 1, 10    # n=1 = LIME native, k=3, 10 independent splits


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
    """Construct a LIME tabular explainer for the requested discretizer."""
    kw = dict(discretize_continuous=True, discretizer=disc)
    if disc == 'entropy':
        kw['training_labels'] = np.asarray(labels)
    return lime.lime_tabular.LimeTabularExplainer(np.asarray(data, float), **kw)


def _cuts_per_feature(expl):
    """Return the mean number of interior cuts across continuous features."""
    d = getattr(expl, 'discretizer', None)
    if d is None or not getattr(d, 'names', None):
        return float('nan')
    # BaseDiscretizer.names[f] holds one string per BIN, so cuts = bins - 1.
    return round(float(np.mean([max(len(v) - 1, 0) for v in d.names.values()])), 2)


def run_ds(ds, disc='entropy', models=DEFAULT_MODELS, out_dir='paper_results_disc', aux='shadow'):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f'lime_auxdisc_{disc}_{aux}_ds{ds}.json')
    warnings.filterwarnings('ignore')
    topQ = Q_BY_DS[ds][-1]
    ARMS = ['nothresh', 'tgt', 'aux_gt'] + (['aux_qry'] if disc == 'entropy' else [])
    fid = {m: {a: [] for a in ARMS} for m in models}
    cuts = {a: [] for a in ('tgt', 'aux_gt')}
    n_eval = n_aux = None; t0 = time.time()
    for s in range(NSPLIT):
        a1, a2 = load_dataset(ds, seed=s)
        X_train, X_test, y_train, y_test, X_test_t, X_test_s, y_test_t, y_test_s = a1
        nc = a2[2]; Xt = np.asarray(X_test_t.values, float); n_eval = len(Xt)
        lb = int((topQ // nc) * 0.5 + 1); ub = int((topQ // nc) * (nc + 0.5) + 1)
        random.seed(s); np.random.seed(s)
        mega = mega_sample_generation(X_test_s.to_numpy(), y_test_s, nc, [SIZE], HMS)
        seed_set = np.asarray(mega[0][0], float)   # the attacker's actual seeds, feature-only

        if aux == 'seed':
            # Fitting on the seeds alone costs nothing extra: the attack queries them anyway, so
            # their labels are already paid for. Ground truth is unavailable here, hence tm.predict
            # below -- handled inside the model loop, since it depends on the target.
            aux_data, aux_labels = seed_set, None
        else:
            aux_data, aux_labels = X_test_s.values, np.asarray(y_test_s)
        n_aux = len(aux_data)

        expl_tgt = _mk_expl(X_train.values, np.asarray(y_train), disc)
        expl_aux_gt = _mk_expl(aux_data, aux_labels, disc) if aux_labels is not None else None
        cuts['tgt'].append(_cuts_per_feature(expl_tgt))
        if expl_aux_gt is not None:
            cuts['aux_gt'].append(_cuts_per_feature(expl_aux_gt))

        for m in models:
            mn = MODELS.get(m, str(m))
            tm, mn = load_model(m, X_train, y_train)
            # Target-labelled aux grid. Only entropy reads the labels, so only entropy pays for them.
            expl_aux_qry = _mk_expl(aux_data, tm.predict(aux_data), disc) if 'aux_qry' in ARMS else None
            e_gt = expl_aux_gt if expl_aux_gt is not None else _mk_expl(aux_data, tm.predict(aux_data), disc)
            arm_cfg = {'nothresh': (expl_tgt, False), 'tgt': (expl_tgt, True),
                       'aux_gt': (e_gt, True), 'aux_qry': (expl_aux_qry, True)}
            for arm in ARMS:
                expl, ut = arm_cfg[arm]
                random.seed(1000 + s); np.random.seed(1000 + s)   # paired across arms, and harmonised
                #   with _lime_auxdisc.py so the quartile run reproduces Table V cell for cell
                V, P, _ = traverse_explanations_LIME(mega[0][0], expl, tm, lb, ub, topQ, NFE, a2,
                                                     feature_select='explanation', use_threshold=ut)
                fid[m][arm].append(_fid(mn, V, P, tm, Xt, nc))
        print(f'  split {s} done ({round(time.time()-t0,1)}s)', flush=True)

    rows = []
    wp = lambda X, Y: (round(float(np.mean(X) - np.mean(Y)), 4),
                       float(wilcoxon(X, Y).pvalue) if not np.allclose(X, Y) else 1.0)
    for m in models:
        mn = MODELS.get(m, str(m))
        N = np.array(fid[m]['nothresh'], float); T = np.array(fid[m]['tgt'], float)
        A = np.array(fid[m]['aux_gt'], float)
        row = {'dataset': DATASETS[ds], 'model': mn, 'disc': disc, 'aux_source': aux,
               'q': topQ, 'nsplit': NSPLIT, 'n_eval': n_eval,
               'n_aux_rows': n_aux,   # queries the adversary spends to label the pool in aux_qry
               'cuts_tgt': round(float(np.nanmean(cuts['tgt'])), 2),
               'cuts_aux': round(float(np.nanmean(cuts['aux_gt'])), 2) if cuts['aux_gt'] else None,
               'nothresh_mean': round(float(np.mean(N)), 4), 'tgt_mean': round(float(np.mean(T)), 4),
               'aux_mean': round(float(np.mean(A)), 4),
               'thr_tgt': wp(T, N)[0], 'thr_tgt_p': wp(T, N)[1],
               'thr_aux': wp(A, N)[0], 'thr_aux_p': wp(A, N)[1],
               'leak': wp(T, A)[0], 'leak_p': wp(T, A)[1],
               'nothresh_all': [round(x, 4) for x in N], 'tgt_all': [round(x, 4) for x in T],
               'aux_all': [round(x, 4) for x in A]}
        if 'aux_qry' in ARMS:
            Aq = np.array(fid[m]['aux_qry'], float)
            row.update({'auxq_mean': round(float(np.mean(Aq)), 4),
                        'thr_auxq': wp(Aq, N)[0], 'thr_auxq_p': wp(Aq, N)[1],
                        'leak_q': wp(T, Aq)[0], 'leak_q_p': wp(T, Aq)[1],
                        'auxq_all': [round(x, 4) for x in Aq]})
        rows.append(row)
        msg = (f"ROW {row['dataset']} {row['model']} [{disc}] thr_tgt {row['thr_tgt']} "
               f"thr_aux {row['thr_aux']} leak {row['leak']} (p={row['leak_p']:.3f})")
        if 'leak_q' in row:
            msg += f" | thr_auxq {row['thr_auxq']} leak_q {row['leak_q']} (p={row['leak_q_p']:.3f})"
        print(msg, flush=True)
        with open(path, 'w') as f:
            json.dump(rows, f, indent=2)
    return rows


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--ds', type=int, required=True)
    ap.add_argument('--disc', type=str, default='entropy', choices=['entropy', 'decile', 'quartile'])
    ap.add_argument('--models', type=str, default=','.join(map(str, DEFAULT_MODELS)))
    ap.add_argument('--out', type=str, default='paper_results_disc')
    ap.add_argument('--q', type=int, default=None, help='override the per-dataset query budget')
    ap.add_argument('--aux', type=str, default='shadow', choices=['shadow', 'seed'])
    ap.add_argument('--nsplit', type=int, default=None, help='smoke-test override; paper runs use 10')
    ap.add_argument('--allow-categorical', action='store_true',
                    help='run on nursery/mushroom anyway (their bin edges are encoding artefacts)')
    a = ap.parse_args()
    if a.ds in CATEGORICAL_DS and not a.allow_categorical:
        raise SystemExit(f'ds {a.ds} ({DATASETS[a.ds]}) is fully categorical: its bin edges are '
                         f'quantiles of an integer label encoding, so a different discretizer only '
                         f'recarves an artefact. Pass --allow-categorical to override.')
    if a.q:
        Q_BY_DS[a.ds] = [a.q]
    if a.nsplit:
        NSPLIT = a.nsplit
    mods = [int(x) for x in a.models.split(',') if x != '']
    run_ds(a.ds, disc=a.disc, models=mods, out_dir=a.out, aux=a.aux)
