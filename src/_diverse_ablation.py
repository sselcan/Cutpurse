"""Isolate whether the small-Q dip on nursery is the DIVERSE overhead or the BOUNDARY overhead.
Three arms on the SAME seeded sample sets, swept over Q:
  base   = base LIME (Autolycus, no diverse / no boundary)
  nodiv  = LIME3 use_diverse=False        (boundary search only)
  limediv= LIME3 diverse_method=lime_threshold (diverse + boundary = 'my method')
diverse's marginal effect at each Q = limediv - nodiv ; boundary's = nodiv - base.
Writes diverse_ablation.json.
"""
import json, warnings
import numpy as np, random
import lime, lime.lime_tabular
from attack_utils import (load_dataset, load_model, mega_sample_generation, _fit_one_surrogate,
                          traverse_explanations_LIME, traverse_explanations_LIME3)

WD = 4                       # nursery
MODELS = [0, 1, 2, 3, 4, 5]  # DT LR NB KNN RF P
Q_LIST = [0, 100, 250, 500, 1000]
HMS = 6                      # sample sets
SIZE = 3                     # seeds/class
NFE = 1                      # k
SEED = 0
REPS = {'dt': 5, 'rdf': 5, 'nb': 1, 'lr': 1, 'knn': 1, 'mlp': 1}

warnings.filterwarnings("ignore")
args1, args2 = load_dataset(WD)
X_train, X_test, y_train, y_test, X_test_t, X_test_s, y_test_t, y_test_s = args1
n_classes = args2[2]
Xt = np.asarray(X_test_t.values, dtype=float)


def fidelity(model_name, V, P, t_model):
    if model_name == 'nb':
        V = np.clip(np.asarray(V, dtype=float), 0, None)
    reps = REPS.get(model_name, 1)
    fids = []
    for k in range(reps):
        sm = _fit_one_surrogate(model_name, np.asarray(V, dtype=float), P, n_classes, seed=k)
        fids.append(float(np.mean(t_model.predict(Xt) == sm.predict(Xt))))
    return float(np.mean(fids))


def run_arm(arm, seed_set, expl, t_model, mname, lb, ub, Q):
    if arm == 'base':
        return traverse_explanations_LIME(seed_set, expl, t_model, lb, ub, Q, NFE, args2)
    kw = dict(use_diverse=(arm == 'limediv'))
    if arm == 'limediv':
        kw['diverse_method'] = 'lime_threshold'
    return traverse_explanations_LIME3(seed_set, expl, t_model, lb, ub, Q, NFE, args2,
                                       mname, X_train, y_train, **kw)


out = {}
for m in MODELS:
    random.seed(SEED); np.random.seed(SEED)
    t_model, mname = load_model(m, X_train, y_train)
    expl = lime.lime_tabular.LimeTabularExplainer(X_train.values, discretize_continuous=True)
    mega = mega_sample_generation(X_test_s.to_numpy(), y_test_s, n_classes, [SIZE], HMS)
    res = {a: {} for a in ('base', 'nodiv', 'limediv')}
    for Q in Q_LIST:
        lb = int((Q // n_classes) * 0.5 + 1); ub = int((Q // n_classes) * (n_classes + 0.5) + 1)
        for arm in ('base', 'nodiv', 'limediv'):
            fids = []
            for i in range(HMS):
                V, P, _ = run_arm(arm, mega[i][0], expl, t_model, mname, lb, ub, Q)
                fids.append(fidelity(mname, V, P, t_model))
            res[arm][Q] = fids
    out[mname] = {'q': Q_LIST, **{a: [res[a][Q] for Q in Q_LIST] for a in res}}
    # print a compact per-model summary
    print(f"\n=== nursery / {mname} ===  Q={Q_LIST}")
    for a in ('base', 'nodiv', 'limediv'):
        print(f"  {a:<8} {[round(float(np.mean(res[a][Q])),3) for Q in Q_LIST]}")
    dv = [round(float(np.mean(res['limediv'][Q]) - np.mean(res['nodiv'][Q])), 3) for Q in Q_LIST]
    bd = [round(float(np.mean(res['nodiv'][Q]) - np.mean(res['base'][Q])), 3) for Q in Q_LIST]
    print(f"  diverse  (limediv-nodiv): {dv}")
    print(f"  boundary (nodiv-base)   : {bd}")

json.dump(out, open("diverse_ablation.json", "w"), indent=2)
print("\nwrote diverse_ablation.json")
