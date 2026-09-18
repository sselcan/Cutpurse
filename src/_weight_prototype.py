"""Sample-weighting prototype: up-weight the densified boundary samples (is_boundary=True) in the
surrogate .fit(), so a single tree can't dilute them. W=1 == current SHAP4b-nodiv (uniform weights).
Targets DT (the mixed model) + adult/breast RF (confirm weighting doesn't break the RF wins).
The traversal output is reused across all W (cheap re-fits)."""
import random, json, io
from contextlib import redirect_stdout
import numpy as np
from attack_utils import (load_dataset, load_model, load_explainer, mega_sample_generation,
                          rtest_sim, traverse_explanations_SHAP, traverse_explanations_SHAP4b)
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.ensemble import RandomForestClassifier as RF

random.seed(0); np.random.seed(0)
DNAMES = {1: 'crop', 2: 'adult', 3: 'breast', 5: 'mushroom'}; MNAMES = {0: 'DT', 4: 'RF'}
DEPTH = 15; HMS = 10; WEIGHTS = [1, 2, 3, 5, 10]

def reps_for(mn): return 100 if mn == 'dt' else 10
def make(mn, k): return DT(random_state=k, max_depth=DEPTH) if mn == 'dt' else RF(max_depth=DEPTH, random_state=k)

def eval_w(vs, vp, sw, t_model, Xtt, mn):
    sims = []  # MEAN over refits, not max (max = test-set selection bias; see attack_utils fix)
    for k in range(reps_for(mn)):
        m = make(mn, k)
        try:
            m.fit(vs, vp, sample_weight=sw)
        except Exception:
            continue
        sims.append(rtest_sim(m, t_model, Xtt.values))
    return float(np.mean(sims)) if sims else -1.0

def eval_plain(vs, vp, t_model, Xtt, mn):
    sims = []  # MEAN over refits, not max (max = test-set selection bias; see attack_utils fix)
    for k in range(reps_for(mn)):
        m = make(mn, k)
        try:
            m.fit(vs, vp)
        except Exception:
            continue
        sims.append(rtest_sim(m, t_model, Xtt.values))
    return float(np.mean(sims)) if sims else -1.0

def quiet(fn, *a, **k):
    with redirect_stdout(io.StringIO()):
        return fn(*a, **k)

# (dataset, model, Q): DT combos (mixed) + adult/breast RF (wins, check no harm)
COMBOS = [(2, 0, 500), (3, 0, 250), (1, 0, 500), (5, 0, 500), (2, 4, 500), (3, 4, 250)]
rows = []
for ds, wm, Q in COMBOS:
    a1, a2 = load_dataset(ds)
    Xtr, Xte, ytr, yte, Xtt, Xts, ytt, yts = a1
    classes, features, nc, nf, isCat, eps, canNeg, cposs, dn, fr = a2
    t_model, mn = load_model(wm, Xtr, ytr); expl = load_explainer(1, t_model, mn, Xtr)
    lb = int((Q // nc) * 0.5 + 1); ub = int((Q // nc) * (nc + 0.5) + 1)
    sm = mega_sample_generation(Xts.to_numpy(), yts, nc, [5], HMS)
    base = []; w_sims = {W: [] for W in WEIGHTS}; bfrac = []
    for i in range(HMS):
        seed = sm[i][0]
        vsb, vpb, _ = quiet(traverse_explanations_SHAP, seed, expl, t_model, lb, ub, Q, 3, a2, mn, Xtr, ytr)
        base.append(eval_plain(vsb, vpb, t_model, Xtt, mn))
        vs, vp, _, mask = quiet(traverse_explanations_SHAP4b, seed, expl, t_model, lb, ub, Q, 3, a2, mn, Xtr, ytr,
                                use_diverse=False, return_boundary_mask=True)
        mask = np.array(mask); bfrac.append(float(mask.mean()))
        for W in WEIGHTS:
            sw = np.where(mask, float(W), 1.0)
            w_sims[W].append(eval_w(vs, vp, sw, t_model, Xtt, mn))
    row = {'dataset': DNAMES[ds], 'model': MNAMES[wm], 'Q': Q,
           'baseline': round(float(np.mean(base)), 4), 'bfrac': round(float(np.mean(bfrac)), 3)}
    for W in WEIGHTS:
        row[f'W{W}'] = round(float(np.mean(w_sims[W])), 4)
    rows.append(row)
    print(f"[{DNAMES[ds]} {MNAMES[wm]} Q={Q}] base={row['baseline']:.4f} bfrac={row['bfrac']:.2f} " +
          " ".join(f"W{W}={row[f'W{W}']:.4f}" for W in WEIGHTS), flush=True)
    with open('weight_prototype.json', 'w') as f:
        json.dump(rows, f, indent=2)

print("\n=== SAMPLE-WEIGHTING (W=1 == current SHAP4b-nodiv); similarity, 10 sets ===", flush=True)
print(f"{'data':9}{'mdl':4}{'Q':>5}{'base':>9}{'bfrac':>7}" + "".join(f"{'W'+str(W):>9}" for W in WEIGHTS))
for r in rows:
    print(f"{r['dataset']:9}{r['model']:4}{r['Q']:>5}{r['baseline']:>9.4f}{r['bfrac']:>7.2f}" +
          "".join(f"{r['W'+str(W)]:>9.4f}" for W in WEIGHTS))
print("done", flush=True)
