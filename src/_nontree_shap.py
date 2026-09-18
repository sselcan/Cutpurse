"""SHAP-contribution ablation on NON-TREE models, split into boundary-generation vs traverse.
Non-tree models have large gains over Autolycus, so the SHAP contribution (if any) is most visible here.
SHAP4b (use_diverse=True) 2x2 on (use_shap_flip x use_shap_traverse), LR/NB/KNN x adult/mushroom/breast,
Q=500, 10 paired sets, matched surrogate. + baseline (Autolycus) + shap3 (recycled default) for reference.
  flipGain     = SHAP in boundary generation  = avg[(TT-FT),(TF-FF)]  (the question of interest)
  traverseGain = SHAP in the traversal        = avg[(TT-TF),(FT-FF)]  (Autolycus critique: expect ~0)
"""
import random, json, io
from contextlib import redirect_stdout
import numpy as np
from attack_utils import (load_dataset, load_model, load_explainer, mega_sample_generation,
                          rtest_sim, traverse_explanations_SHAP, traverse_explanations_SHAP3,
                          traverse_explanations_SHAP4b)
from sklearn.linear_model import LogisticRegression as LR
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.naive_bayes import MultinomialNB as MNB

random.seed(0); np.random.seed(0)
DNAMES = {2: 'adult', 3: 'breast', 5: 'mushroom'}; MNAMES = {1: 'LR', 2: 'NB', 3: 'KNN'}
HMS = 10; Q = 500

def make(mn, nc):
    if mn == 'lr':  return LR(max_iter=1000)
    if mn == 'nb':  return MNB()
    if mn == 'knn': return KNN(n_neighbors=nc)

def eval_arm(vs, vp, t_model, Xtt, mn, nc):
    m = make(mn, nc)
    try:
        m.fit(vs, vp)
    except Exception:
        return float('nan')
    return rtest_sim(m, t_model, Xtt.values)

def quiet(fn, *a, **k):
    with redirect_stdout(io.StringIO()):
        return fn(*a, **k)

S4 = traverse_explanations_SHAP4b
ARMS = [
    ('baseline', lambda *a: traverse_explanations_SHAP(*a)),
    ('shap3',    lambda *a: traverse_explanations_SHAP3(*a)),
    ('s4b_TT',   lambda *a: S4(*a, use_diverse=True)),
    ('s4b_TF',   lambda *a: S4(*a, use_diverse=True, use_shap_traverse=False)),
    ('s4b_FT',   lambda *a: S4(*a, use_diverse=True, use_shap_flip=False)),
    ('s4b_FF',   lambda *a: S4(*a, use_diverse=True, use_shap_flip=False, use_shap_traverse=False)),
]
rows = []
for ds in [2, 5, 3]:  # adult, mushroom, breast (binary, KernelSHAP-friendly)
    a1, a2 = load_dataset(ds)
    Xtr, Xte, ytr, yte, Xtt, Xts, ytt, yts = a1
    classes, features, nc, nf, isCat, eps, canNeg, cposs, dn, fr = a2
    for wm in [1, 2, 3]:  # LR, NB, KNN
        t_model, mn = load_model(wm, Xtr, ytr); expl = load_explainer(1, t_model, mn, Xtr)
        lb = int((Q // nc) * 0.5 + 1); ub = int((Q // nc) * (nc + 0.5) + 1)
        sm = mega_sample_generation(Xts.to_numpy(), yts, nc, [5], HMS)
        sims = {name: [] for name, _ in ARMS}
        for i in range(HMS):
            seed = sm[i][0]
            for name, fn in ARMS:
                out = quiet(fn, seed, expl, t_model, lb, ub, Q, 3, a2, mn, Xtr, ytr)
                sims[name].append(eval_arm(out[0], out[1], t_model, Xtt, mn, nc))
        r = {'dataset': DNAMES[ds], 'model': MNAMES[wm]}
        for name, _ in ARMS:
            r[name] = round(float(np.nanmean(sims[name])), 4)
        r['flipGain'] = round(((r['s4b_TT'] - r['s4b_FT']) + (r['s4b_TF'] - r['s4b_FF'])) / 2, 4)
        r['traverseGain'] = round(((r['s4b_TT'] - r['s4b_TF']) + (r['s4b_FT'] - r['s4b_FF'])) / 2, 4)
        rows.append(r)
        print(f"[{DNAMES[ds]} {MNAMES[wm]:3}] base={r['baseline']:.4f} shap3={r['shap3']:.4f} "
              f"TT={r['s4b_TT']:.4f} TF={r['s4b_TF']:.4f} FT={r['s4b_FT']:.4f} FF={r['s4b_FF']:.4f} "
              f"| flipGain={r['flipGain']:+.4f} traverseGain={r['traverseGain']:+.4f}", flush=True)
        with open('nontree_shap.json', 'w') as f:
            json.dump(rows, f, indent=2)

print("\n=== NON-TREE SHAP CONTRIBUTION (boundary-gen vs traverse) [Q=500, 10 sets] ===", flush=True)
hdr = f"{'data':9}{'mdl':4}{'base':>8}{'shap3':>8}{'TT':>8}{'TF':>8}{'FT':>8}{'FF':>8}{'flipGain':>10}{'travGain':>10}"
print(hdr); print('-' * len(hdr))
for r in rows:
    print(f"{r['dataset']:9}{r['model']:4}{r['baseline']:>8.4f}{r['shap3']:>8.4f}{r['s4b_TT']:>8.4f}"
          f"{r['s4b_TF']:>8.4f}{r['s4b_FT']:>8.4f}{r['s4b_FF']:>8.4f}{r['flipGain']:>+10.4f}{r['traverseGain']:>+10.4f}")
print("\nflipGain>0 => SHAP in boundary generation helps.  traverseGain~0 => SHAP in traverse is inert (Autolycus critique).")
print("done", flush=True)
