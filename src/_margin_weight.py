"""Margin-weighting (no densification): spend the WHOLE budget on traversal, then up-weight
near-boundary samples in the surrogate fit via w = 1 + lam*(1 - max predict_proba). The margin is
free (the traversal already queried predict_proba to label each sample), so no extra budget.
Question: can baseline/SHAP3 + margin-weights match densification (SHAP4b-nodiv) without spending
budget on synthetic boundary samples? Tested on DT/RF tree combos. lam=0 == unweighted."""
import random, json, io
from contextlib import redirect_stdout
import numpy as np
from attack_utils import (load_dataset, load_model, load_explainer, mega_sample_generation,
                          rtest_sim, traverse_explanations_SHAP, traverse_explanations_SHAP3,
                          traverse_explanations_SHAP4b)
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.ensemble import RandomForestClassifier as RF

random.seed(0); np.random.seed(0)
DNAMES = {1: 'crop', 2: 'adult', 3: 'breast', 5: 'mushroom'}; MNAMES = {0: 'DT', 4: 'RF'}
DEPTH = 15; HMS = 10; LAMS = [0, 3, 5, 10, 20]

def reps_for(mn): return 100 if mn == 'dt' else 10
def make(mn, k): return DT(random_state=k, max_depth=DEPTH) if mn == 'dt' else RF(max_depth=DEPTH, random_state=k)

def eval_margin(vs, vp, t_model, Xtt, mn, lam):
    arr = np.asarray(vs, dtype=float)
    conf = t_model.predict_proba(arr).max(axis=1)          # free at query-time (predict_proba already called)
    sw = 1.0 + lam * (1.0 - conf)                          # up-weight low-margin (near-boundary) samples
    sims = []  # MEAN over refits, not max (max = test-set selection bias; see attack_utils fix)
    for k in range(reps_for(mn)):
        m = make(mn, k)
        try:
            m.fit(vs, vp, sample_weight=sw)
        except Exception:
            continue
        sims.append(rtest_sim(m, t_model, Xtt.values))
    return float(np.mean(sims)) if sims else -1.0

def quiet(fn, *a, **k):
    with redirect_stdout(io.StringIO()):
        return fn(*a, **k)

COMBOS = [(2, 0, 500), (1, 0, 500), (5, 0, 500), (3, 0, 250),
          (2, 4, 500), (1, 4, 500), (5, 4, 500), (3, 4, 250)]
rows = []
for ds, wm, Q in COMBOS:
    a1, a2 = load_dataset(ds)
    Xtr, Xte, ytr, yte, Xtt, Xts, ytt, yts = a1
    classes, features, nc, nf, isCat, eps, canNeg, cposs, dn, fr = a2
    t_model, mn = load_model(wm, Xtr, ytr); expl = load_explainer(1, t_model, mn, Xtr)
    lb = int((Q // nc) * 0.5 + 1); ub = int((Q // nc) * (nc + 0.5) + 1)
    sm = mega_sample_generation(Xts.to_numpy(), yts, nc, [5], HMS)
    base = {L: [] for L in LAMS}; s3 = {L: [] for L in LAMS}; s4b = []
    for i in range(HMS):
        seed = sm[i][0]
        vsb, vpb, _ = quiet(traverse_explanations_SHAP, seed, expl, t_model, lb, ub, Q, 3, a2, mn, Xtr, ytr)
        vs3, vp3, _ = quiet(traverse_explanations_SHAP3, seed, expl, t_model, lb, ub, Q, 3, a2, mn, Xtr, ytr)
        vs4, vp4, _ = quiet(traverse_explanations_SHAP4b, seed, expl, t_model, lb, ub, Q, 3, a2, mn, Xtr, ytr, use_diverse=False)
        for L in LAMS:
            base[L].append(eval_margin(vsb, vpb, t_model, Xtt, mn, L))
            s3[L].append(eval_margin(vs3, vp3, t_model, Xtt, mn, L))
        s4b.append(eval_margin(vs4, vp4, t_model, Xtt, mn, 0))   # densification, unweighted reference
    row = {'dataset': DNAMES[ds], 'model': MNAMES[wm], 'Q': Q,
           's4b_nodiv': round(float(np.mean(s4b)), 4)}
    for L in LAMS:
        row[f'base_L{L}'] = round(float(np.mean(base[L])), 4)
        row[f's3_L{L}'] = round(float(np.mean(s3[L])), 4)
    rows.append(row)
    bb = max(LAMS, key=lambda L: row[f'base_L{L}']); b3 = max(LAMS, key=lambda L: row[f's3_L{L}'])
    print(f"[{DNAMES[ds]} {MNAMES[wm]} Q={Q}] s4b={row['s4b_nodiv']:.4f} | "
          f"base L0={row['base_L0']:.4f}->bestL{bb}={row[f'base_L{bb}']:.4f} | "
          f"s3 L0={row['s3_L0']:.4f}->bestL{b3}={row[f's3_L{b3}']:.4f}", flush=True)
    with open('margin_weight.json', 'w') as f:
        json.dump(rows, f, indent=2)

print("\n=== MARGIN-WEIGHTING vs DENSIFICATION (10 sets); s4b_nodiv = densification reference ===", flush=True)
hdr = f"{'data':9}{'mdl':4}{'Q':>5}{'s4b':>8}" + "".join(f"{'baseL'+str(L):>9}" for L in LAMS) + "".join(f"{'s3L'+str(L):>9}" for L in LAMS)
print(hdr)
for r in rows:
    print(f"{r['dataset']:9}{r['model']:4}{r['Q']:>5}{r['s4b_nodiv']:>8.4f}" +
          "".join(f"{r['base_L'+str(L)]:>9.4f}" for L in LAMS) + "".join(f"{r['s3_L'+str(L)]:>9.4f}" for L in LAMS))
print("done", flush=True)
