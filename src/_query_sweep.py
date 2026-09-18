"""Query-budget sweep: baseline (Autolycus) vs SHAP3 vs SHAP4b(use_diverse=False), for ALL model
types (incl. non-tree), at low->mid query budgets. Answers:
  (1) how does SHAP4b-nodiv do on NON-tree models (LR/NB/KNN), not just trees?
  (2) the low-query regime (Q ~ 50-250), where a query-efficient method should show its edge.
Per dataset the seed sets are generated ONCE and reused across all Q (paired across budget), so a
Q->Q change reflects the budget, not different samples (fixes the earlier scaling confound).
Q ranges follow the Autolycus paper: breast in [50,100,250], others in [100,250,500].
P/MLP excluded here for runtime (20-network surrogate); run separately if needed."""
import random, json, io, traceback
from contextlib import redirect_stdout
import numpy as np
from attack_utils import (load_dataset, load_model, load_explainer, mega_sample_generation,
                          rtest_sim, traverse_explanations_SHAP, traverse_explanations_SHAP3,
                          traverse_explanations_SHAP4b)
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.linear_model import LogisticRegression as LR
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.naive_bayes import MultinomialNB as MNB

random.seed(0); np.random.seed(0)
DNAMES = {1: 'crop', 2: 'adult', 3: 'breast', 5: 'mushroom'}
MNAMES = {0: 'DT', 1: 'LR', 2: 'NB', 3: 'KNN', 4: 'RF'}
DEPTH = 15; HMS = 10
DATASET_Q = {3: [50, 100, 250], 2: [100, 250, 500], 1: [100, 250, 500], 5: [100, 250, 500]}

def reps_for(mn):
    return {'dt': 100, 'rdf': 10}.get(mn, 1)

def make_surrogate(mn, k, n_classes):
    if mn == 'dt':  return DT(random_state=k, max_depth=DEPTH)
    if mn == 'rdf': return RF(max_depth=DEPTH, random_state=k)
    if mn == 'lr':  return LR(max_iter=1000, random_state=k)
    if mn == 'nb':  return MNB()
    if mn == 'knn': return KNN(n_neighbors=n_classes)
    raise ValueError(mn)

def eval_arm(vs, vp, t_model, X_test_t, mn, n_classes):
    sims = []  # MEAN over refits, not max (max = test-set selection bias; see attack_utils fix)
    for k in range(reps_for(mn)):
        m = make_surrogate(mn, k, n_classes)
        try:
            m.fit(vs, vp)
        except Exception:
            continue
        sims.append(rtest_sim(m, t_model, X_test_t.values))
    return float(np.mean(sims)) if sims else -1.0

def quiet(fn, *a, **k):
    with redirect_stdout(io.StringIO()):
        return fn(*a, **k)

METHODS = [
    ('baseline', lambda *a: traverse_explanations_SHAP(*a)),
    ('shap3',    lambda *a: traverse_explanations_SHAP3(*a)),
    ('s4b_nodiv', lambda *a: traverse_explanations_SHAP4b(*a, use_diverse=False)),
]

rows = []
for ds in [2, 1, 5, 3]:  # adult, crop, mushroom, breast
    try:
        args1, args2 = load_dataset(ds)
        Xtr, Xte, ytr, yte, Xtt, Xts, ytt, yts = args1
        classes, features, nc, nf, isCat, eps, canNeg, cposs, dname, franges = args2
        smega = mega_sample_generation(Xts.to_numpy(), yts, nc, [5], HMS)  # once per dataset, reused across Q
    except Exception:
        traceback.print_exc(); continue
    for wm in [0, 1, 2, 3, 4]:
        t_model, mn = load_model(wm, Xtr, ytr)
        expl = load_explainer(1, t_model, mn, Xtr)
        for Q in DATASET_Q[ds]:
            lb = int((Q // nc) * (0.5) + 1); ub = int((Q // nc) * (nc + 0.5) + 1)
            sims = {name: [] for name, _ in METHODS}
            for i in range(HMS):
                seed = smega[i][0]
                for name, fn in METHODS:
                    try:
                        vs, vp, _ = quiet(fn, seed, expl, t_model, lb, ub, Q, 3, args2, mn, Xtr, ytr)
                        sims[name].append(eval_arm(vs, vp, t_model, Xtt, mn, nc))
                    except Exception:
                        traceback.print_exc(); sims[name].append(float('nan'))
            row = {'dataset': DNAMES[ds], 'model': MNAMES[wm], 'Q': Q}
            for name, _ in METHODS:
                row[name] = round(float(np.nanmean(sims[name])), 4)
            row['s4b-base'] = round(row['s4b_nodiv'] - row['baseline'], 4)
            row['s4b-s3'] = round(row['s4b_nodiv'] - row['shap3'], 4)
            rows.append(row)
            print(f"[{DNAMES[ds]:8} {MNAMES[wm]:3} Q={Q:4}] base={row['baseline']:.4f} "
                  f"s3={row['shap3']:.4f} s4b={row['s4b_nodiv']:.4f} "
                  f"(s4b-base {row['s4b-base']:+.4f}, s4b-s3 {row['s4b-s3']:+.4f})", flush=True)
            with open('query_sweep.json', 'w') as f:
                json.dump(rows, f, indent=2)

print("\n=== QUERY SWEEP (10 sets, paired seeds across Q) : s4b_nodiv vs baseline / SHAP3 ===", flush=True)
print(f"{'dataset':9}{'model':4}{'Q':>5}{'baseline':>10}{'shap3':>9}{'s4b_nodiv':>11}{'s4b-base':>10}{'s4b-s3':>9}")
for r in rows:
    print(f"{r['dataset']:9}{r['model']:4}{r['Q']:>5}{r['baseline']:>10.4f}{r['shap3']:>9.4f}"
          f"{r['s4b_nodiv']:>11.4f}{r['s4b-base']:>+10.4f}{r['s4b-s3']:>+9.4f}")
print("done", flush=True)
