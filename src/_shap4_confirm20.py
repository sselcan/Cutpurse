"""20-set mean±std confirmation of the SHIPPING SHAP4 (train_on_all_queried=True default)
vs SHAP3 vs Autolycus baseline, on the four tree combos. Equal 501-query budget across arms.
Surrogate matches the target family (DT/RF), reps = dt:100 / rdf:10 (matches _build_surrogate_and_eval)."""
import random, json, io
from contextlib import redirect_stdout
import numpy as np
from attack_utils import (load_dataset, load_model, load_explainer, mega_sample_generation,
                          rtest_sim, traverse_explanations_SHAP, traverse_explanations_SHAP3,
                          traverse_explanations_SHAP4)
from sklearn.linear_model import LogisticRegression as lr
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.ensemble import RandomForestClassifier as RF

random.seed(0); np.random.seed(0)
NAMES = {5: 'mushroom', 4: 'nursery'}; MODELS = {0: 'DT', 4: 'RF'}
DEPTH = 15
HMS = 20

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
    buf = io.StringIO()
    with redirect_stdout(buf):
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
    arms = {'baseline': [], 'shap3': [], 'shap4': []}
    for i in range(hms):
        seed = smega[i][0]
        vs, vp, _ = quiet(traverse_explanations_SHAP, seed, expl, t_model, lb, ub, Q, kf, args2, mn, Xtr, ytr)
        arms['baseline'].append(eval_arm(vs, vp, t_model, Xtt, mn))
        vs, vp, _ = quiet(traverse_explanations_SHAP3, seed, expl, t_model, lb, ub, Q, kf, args2, mn, Xtr, ytr)
        arms['shap3'].append(eval_arm(vs, vp, t_model, Xtt, mn))
        # SHAP4 shipping default (train_on_all_queried=True): no extra flags
        vs, vp, _ = quiet(traverse_explanations_SHAP4, seed, expl, t_model, lb, ub, Q, kf, args2, mn, Xtr, ytr)
        arms['shap4'].append(eval_arm(vs, vp, t_model, Xtt, mn))
        print(f"  [{NAMES[wd]}+{MODELS[wm]}] set {i}: base={arms['baseline'][-1]:.3f} "
              f"s3={arms['shap3'][-1]:.3f} s4={arms['shap4'][-1]:.3f}", flush=True)
    res = {'dataset': NAMES[wd], 'model': MODELS[wm]}
    for a in arms:
        res[a + '_mean'] = round(float(np.mean(arms[a])), 4)
        res[a + '_std'] = round(float(np.std(arms[a])), 4)
    res['raw'] = {a: arms[a] for a in arms}
    return res

COMBOS = [(4, 0), (4, 4), (5, 0), (5, 4)]
allres = []
for wd, wm in COMBOS:
    print(f"\n### {NAMES[wd]}+{MODELS[wm]} (20 sets) ###", flush=True)
    allres.append(run_combo(wd, wm))
    r = allres[-1]
    print(f"  -> base {r['baseline_mean']}±{r['baseline_std']} | shap3 {r['shap3_mean']}±{r['shap3_std']} "
          f"| shap4 {r['shap4_mean']}±{r['shap4_std']}", flush=True)
    with open('shap4_confirm20.json', 'w') as f:
        json.dump(allres, f, indent=2)

print("\n=== SHAP4 20-SET CONFIRMATION (mean±std, Q=500, k=3, surrogate matched to target) ===", flush=True)
hdr = f"{'data':9}{'mdl':4}{'baseline':>15}{'shap3':>15}{'shap4(allq)':>15}{'s4-base':>9}{'s4-s3':>8}"
print(hdr); print('-' * len(hdr))
for r in allres:
    b = f"{r['baseline_mean']:.4f}±{r['baseline_std']:.3f}"
    s3 = f"{r['shap3_mean']:.4f}±{r['shap3_std']:.3f}"
    s4 = f"{r['shap4_mean']:.4f}±{r['shap4_std']:.3f}"
    print(f"{r['dataset']:9}{r['model']:4}{b:>15}{s3:>15}{s4:>15}"
          f"{r['shap4_mean'] - r['baseline_mean']:>+9.4f}{r['shap4_mean'] - r['shap3_mean']:>+8.4f}")
print("done", flush=True)
