"""Boundary-densification (SHAP4b) head-to-head vs Autolycus baseline, SHAP3, and shipping SHAP4.
20 sets, mean±std, equal-budget, surrogate matched to target (DT/RF), reps dt:100 / rdf:10.
Includes adult (mixed continuous/categorical) to exercise SHAP4b's continuous-bisection path."""
import random, json, io
from contextlib import redirect_stdout
import numpy as np
from attack_utils import (load_dataset, load_model, load_explainer, mega_sample_generation,
                          rtest_sim, traverse_explanations_SHAP, traverse_explanations_SHAP3,
                          traverse_explanations_SHAP4, traverse_explanations_SHAP4b)
from sklearn.linear_model import LogisticRegression as lr
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.ensemble import RandomForestClassifier as RF

random.seed(0); np.random.seed(0)
NAMES = {5: 'mushroom', 4: 'nursery', 2: 'adult'}; MODELS = {0: 'DT', 4: 'RF'}
DEPTH = 15; HMS = 20
ARMS = [('baseline', traverse_explanations_SHAP), ('shap3', traverse_explanations_SHAP3),
        ('shap4', traverse_explanations_SHAP4), ('shap4b', traverse_explanations_SHAP4b)]

def reps_for(mn):
    return 100 if mn == 'dt' else 10

def eval_arm(vs, vp, t_model, X_test_t, mn):
    sims = []  # MEAN over refits, not max (max = test-set selection bias; see attack_utils fix)
    for k in range(reps_for(mn)):
        if mn == 'dt':
            m = DT(random_state=k, max_depth=DEPTH)
        elif mn == 'rdf':
            m = RF(max_depth=DEPTH, random_state=k)
        else:
            m = lr(max_iter=1000, random_state=k)
        try:
            m.fit(vs, vp)
        except Exception:
            continue
        sims.append(rtest_sim(m, t_model, X_test_t.values))
    return float(np.mean(sims)) if sims else -1.0

def quiet(fn, *a, **k):
    with redirect_stdout(io.StringIO()):
        return fn(*a, **k)

def run_combo(wd, wm, hms=HMS, Q=500, kf=3):
    args1, args2 = load_dataset(wd)
    Xtr, Xte, ytr, yte, Xtt, Xts, ytt, yts = args1
    classes, features, nc, nf, isCat, eps, canNeg, cposs, dname, franges = args2
    t_model, mn = load_model(wm, Xtr, ytr)
    expl = load_explainer(1, t_model, mn, Xtr)
    relax = 0.5
    lb = int((Q // nc) * (1 - relax) + 1); ub = int((Q // nc) * (nc + relax) + 1)
    smega = mega_sample_generation(Xts.to_numpy(), yts, nc, [5], hms)
    arms = {name: [] for name, _ in ARMS}
    for i in range(hms):
        seed = smega[i][0]
        for name, fn in ARMS:
            vs, vp, _ = quiet(fn, seed, expl, t_model, lb, ub, Q, kf, args2, mn, Xtr, ytr)
            arms[name].append(eval_arm(vs, vp, t_model, Xtt, mn))
        print(f"  [{NAMES[wd]}+{MODELS[wm]}] set {i}: " +
              " ".join(f"{n}={arms[n][-1]:.3f}" for n, _ in ARMS), flush=True)
    res = {'dataset': NAMES[wd], 'model': MODELS[wm]}
    for name, _ in ARMS:
        res[name + '_mean'] = round(float(np.mean(arms[name])), 4)
        res[name + '_std'] = round(float(np.std(arms[name])), 4)
    return res

COMBOS = [(4, 0), (4, 4), (5, 0), (5, 4), (2, 0), (2, 4)]
allres = []
for wd, wm in COMBOS:
    print(f"\n### {NAMES[wd]}+{MODELS[wm]} (20 sets) ###", flush=True)
    allres.append(run_combo(wd, wm))
    r = allres[-1]
    print(f"  -> base {r['baseline_mean']} s3 {r['shap3_mean']} s4 {r['shap4_mean']} s4b {r['shap4b_mean']}", flush=True)
    with open('shap4b_densify.json', 'w') as f:
        json.dump(allres, f, indent=2)

print("\n=== SHAP4b BOUNDARY-DENSIFICATION (20 sets, mean±std, Q=500, k=3, matched surrogate) ===", flush=True)
hdr = f"{'data':9}{'mdl':4}{'baseline':>14}{'shap3':>14}{'shap4':>14}{'shap4b':>14}{'s4b-base':>10}{'s4b-s3':>9}"
print(hdr); print('-' * len(hdr))
for r in allres:
    b = f"{r['baseline_mean']:.4f}±{r['baseline_std']:.3f}"
    s3 = f"{r['shap3_mean']:.4f}±{r['shap3_std']:.3f}"
    s4 = f"{r['shap4_mean']:.4f}±{r['shap4_std']:.3f}"
    s4b = f"{r['shap4b_mean']:.4f}±{r['shap4b_std']:.3f}"
    print(f"{r['dataset']:9}{r['model']:4}{b:>14}{s3:>14}{s4:>14}{s4b:>14}"
          f"{r['shap4b_mean'] - r['baseline_mean']:>+10.4f}{r['shap4b_mean'] - r['shap3_mean']:>+9.4f}")
print("done", flush=True)
