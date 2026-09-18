"""Exp 3.1 (does SHAP help?) + 3.2 (can densification replace the traversal?).
Arms (all use_diverse=False, matched DT/RF surrogate, Q=500, 10 paired sets):
  baseline       : Autolycus (traverse_explanations_SHAP)
  s4b_shap       : SHAP4b full, SHAP-guided        <- reference
  s4b_rand       : SHAP4b full, RANDOM features    (3.1: shap vs rand on the full method)
  densonly_shap  : SHAP4b, skip Phase-3 traversal, SHAP   (3.2: is the traversal needed?)
  densonly_rand  : SHAP4b, skip traversal, RANDOM         (3.1 within densify-only)
Reads:
  3.1  -> s4b_shap vs s4b_rand  (and densonly_shap vs densonly_rand): if ~equal, SHAP isn't pulling weight
  3.2  -> s4b_shap vs densonly_shap: if ~equal, SHAP-guided densification replaces the Autolycus traversal
"""
import random, json, io
from contextlib import redirect_stdout
import numpy as np
from attack_utils import (load_dataset, load_model, load_explainer, mega_sample_generation,
                          rtest_sim, traverse_explanations_SHAP, traverse_explanations_SHAP4b)
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.ensemble import RandomForestClassifier as RF

random.seed(0); np.random.seed(0)
DNAMES = {1: 'crop', 2: 'adult', 3: 'breast', 5: 'mushroom'}; MNAMES = {0: 'DT', 4: 'RF'}
DEPTH = 15; HMS = 10; Q = 500
ARMS = [
    ('baseline',      lambda *a: traverse_explanations_SHAP(*a)),
    ('s4b_shap',      lambda *a: traverse_explanations_SHAP4b(*a, use_diverse=False)),
    ('s4b_rand',      lambda *a: traverse_explanations_SHAP4b(*a, use_diverse=False, use_shap_flip=False, use_shap_traverse=False)),
    ('densonly_shap', lambda *a: traverse_explanations_SHAP4b(*a, use_diverse=False, skip_traversal=True)),
    ('densonly_rand', lambda *a: traverse_explanations_SHAP4b(*a, use_diverse=False, skip_traversal=True, use_shap_flip=False, use_shap_traverse=False)),
]

def reps_for(mn): return 100 if mn == 'dt' else 10
def make(mn, k): return DT(random_state=k, max_depth=DEPTH) if mn == 'dt' else RF(max_depth=DEPTH, random_state=k)

def eval_arm(vs, vp, t_model, Xtt, mn):
    sims = []  # MEAN over refits, not max (max = test-set selection bias; see attack_utils fix)
    for k in range(reps_for(mn)):
        m = make(mn, k)
        try:
            m.fit(vs, vp)
        except Exception:
            continue
        sims.append(rtest_sim(m, t_model, Xtt.values))
    return float(np.mean(sims)) if sims else float('nan')  # NaN (not -1) on total fit failure

def quiet(fn, *a, **k):
    with redirect_stdout(io.StringIO()):
        return fn(*a, **k)

COMBOS = [(2, 0), (1, 0), (5, 0), (3, 0), (2, 4), (1, 4), (5, 4), (3, 4)]
rows = []
for ds, wm in COMBOS:
    a1, a2 = load_dataset(ds)
    Xtr, Xte, ytr, yte, Xtt, Xts, ytt, yts = a1
    classes, features, nc, nf, isCat, eps, canNeg, cposs, dn, fr = a2
    t_model, mn = load_model(wm, Xtr, ytr); expl = load_explainer(1, t_model, mn, Xtr)
    lb = int((Q // nc) * 0.5 + 1); ub = int((Q // nc) * (nc + 0.5) + 1)
    sm = mega_sample_generation(Xts.to_numpy(), yts, nc, [5], HMS)
    sims = {name: [] for name, _ in ARMS}
    for i in range(HMS):
        seed = sm[i][0]
        for name, fn in ARMS:
            out = quiet(fn, seed, expl, t_model, lb, ub, Q, 3, a2, mn, Xtr, ytr)
            vs, vp = out[0], out[1]
            sims[name].append(eval_arm(vs, vp, t_model, Xtt, mn))
    row = {'dataset': DNAMES[ds], 'model': MNAMES[wm]}
    for name, _ in ARMS:
        row[name] = round(float(np.nanmean(sims[name])), 4)
    row['shap_gain'] = round(row['s4b_shap'] - row['s4b_rand'], 4)        # 3.1
    row['traversal_gain'] = round(row['s4b_shap'] - row['densonly_shap'], 4)  # 3.2
    rows.append(row)
    print(f"[{DNAMES[ds]}+{MNAMES[wm]}] base={row['baseline']:.4f} s4b={row['s4b_shap']:.4f} "
          f"rand={row['s4b_rand']:.4f} densonly={row['densonly_shap']:.4f} densonlyRand={row['densonly_rand']:.4f} "
          f"| shapGain(3.1)={row['shap_gain']:+.4f} traversalGain(3.2)={row['traversal_gain']:+.4f}", flush=True)
    with open('shap_ablation.json', 'w') as f:
        json.dump(rows, f, indent=2)

print("\n=== SHAP-ABLATION (3.1) + DENSIFY-ONLY (3.2)  [Q=500, 10 sets, matched surrogate] ===", flush=True)
hdr = f"{'data':9}{'mdl':4}{'base':>9}{'s4b':>9}{'rand':>9}{'densO':>9}{'densORand':>11}{'shapGain':>10}{'travGain':>10}"
print(hdr); print('-' * len(hdr))
for r in rows:
    print(f"{r['dataset']:9}{r['model']:4}{r['baseline']:>9.4f}{r['s4b_shap']:>9.4f}{r['s4b_rand']:>9.4f}"
          f"{r['densonly_shap']:>9.4f}{r['densonly_rand']:>11.4f}{r['shap_gain']:>+10.4f}{r['traversal_gain']:>+10.4f}")
print("\n3.1: shapGain = s4b_shap - s4b_rand (>0 => SHAP helps).  3.2: travGain = s4b_shap - densonly_shap (~0 => traversal not needed).")
print("done", flush=True)
