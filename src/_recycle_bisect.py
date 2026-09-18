"""Test recycling SHAP3's bisection intermediates into training (recycle_bisection=True).
Every bisection mid is already queried (counted in overhead) but was discarded; now it's added to
training at ZERO extra budget. Compare baseline / SHAP3 / SHAP3+recycle, across Q (incl. low budgets,
where the recovered ~80 free samples are a big fraction). Non-tree models (SHAP3 is their default) + DT.
Paired seeds per dataset across Q. Matched surrogate; NaN on total fit failure (not the -1 sentinel)."""
import random, json, io
from contextlib import redirect_stdout
import numpy as np
from attack_utils import (load_dataset, load_model, load_explainer, mega_sample_generation,
                          rtest_sim, traverse_explanations_SHAP, traverse_explanations_SHAP3)
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.linear_model import LogisticRegression as LR
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.naive_bayes import MultinomialNB as MNB

random.seed(0); np.random.seed(0)
DNAMES = {1: 'crop', 2: 'adult', 3: 'breast', 5: 'mushroom'}
MNAMES = {0: 'DT', 1: 'LR', 2: 'NB', 3: 'KNN'}
DEPTH = 15; HMS = 10; QS = [100, 250, 500]

def reps_for(mn): return 100 if mn == 'dt' else 1
def make(mn, k, nc):
    if mn == 'dt':  return DT(random_state=k, max_depth=DEPTH)
    if mn == 'lr':  return LR(max_iter=1000, random_state=k)
    if mn == 'nb':  return MNB()
    if mn == 'knn': return KNN(n_neighbors=nc)

def eval_arm(vs, vp, t_model, Xtt, mn, nc):
    sims = []  # MEAN over refits, not max (max = test-set selection bias; see attack_utils fix)
    for k in range(reps_for(mn)):
        m = make(mn, k, nc)
        try:
            m.fit(vs, vp)
        except Exception:
            continue
        sims.append(rtest_sim(m, t_model, Xtt.values))
    return float(np.mean(sims)) if sims else float('nan')

def quiet(fn, *a, **k):
    with redirect_stdout(io.StringIO()):
        return fn(*a, **k)

ARMS = [
    ('baseline',   lambda *a: traverse_explanations_SHAP(*a)),
    ('shap3',      lambda *a: traverse_explanations_SHAP3(*a)),
    ('shap3_rec',  lambda *a: traverse_explanations_SHAP3(*a, recycle_bisection=True)),
]
rows = []
for ds in [2, 5, 3]:  # adult, mushroom, breast (skip crop: slow 17-class KernelSHAP)
    a1, a2 = load_dataset(ds)
    Xtr, Xte, ytr, yte, Xtt, Xts, ytt, yts = a1
    classes, features, nc, nf, isCat, eps, canNeg, cposs, dn, fr = a2
    smega = {}
    for wm in [1, 2, 3, 0]:  # LR, NB, KNN, DT
        t_model, mn = load_model(wm, Xtr, ytr); expl = load_explainer(1, t_model, mn, Xtr)
        sm = mega_sample_generation(Xts.to_numpy(), yts, nc, [5], HMS)  # per (ds,model)
        for Q in QS:
            lb = int((Q // nc) * 0.5 + 1); ub = int((Q // nc) * (nc + 0.5) + 1)
            sims = {name: [] for name, _ in ARMS}
            for i in range(HMS):
                seed = sm[i][0]
                for name, fn in ARMS:
                    out = quiet(fn, seed, expl, t_model, lb, ub, Q, 3, a2, mn, Xtr, ytr)
                    sims[name].append(eval_arm(out[0], out[1], t_model, Xtt, mn, nc))
            row = {'dataset': DNAMES[ds], 'model': MNAMES[wm], 'Q': Q}
            for name, _ in ARMS:
                row[name] = round(float(np.nanmean(sims[name])), 4)
            row['rec_gain'] = round(row['shap3_rec'] - row['shap3'], 4)   # recycle vs current SHAP3
            row['rec_vs_base'] = round(row['shap3_rec'] - row['baseline'], 4)
            rows.append(row)
            print(f"[{DNAMES[ds]} {MNAMES[wm]:3} Q={Q:4}] base={row['baseline']:.4f} shap3={row['shap3']:.4f} "
                  f"shap3_rec={row['shap3_rec']:.4f} | recGain={row['rec_gain']:+.4f} recVsBase={row['rec_vs_base']:+.4f}", flush=True)
            with open('recycle_bisect.json', 'w') as f:
                json.dump(rows, f, indent=2)

print("\n=== RECYCLE SHAP3 BISECTION MIDS (10 sets) : shap3_rec vs shap3 / baseline ===", flush=True)
print(f"{'data':9}{'mdl':4}{'Q':>5}{'base':>9}{'shap3':>9}{'shap3_rec':>11}{'recGain':>9}{'recVsBase':>11}")
for r in rows:
    print(f"{r['dataset']:9}{r['model']:4}{r['Q']:>5}{r['baseline']:>9.4f}{r['shap3']:>9.4f}"
          f"{r['shap3_rec']:>11.4f}{r['rec_gain']:>+9.4f}{r['rec_vs_base']:>+11.4f}")
print("done", flush=True)
