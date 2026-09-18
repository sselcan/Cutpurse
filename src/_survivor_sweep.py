"""Paired Q-sweep on the two combos that survived the mean-eval re-check (adult+RF, breast+DT),
plus crop+DT as a 'dissolved' control. FOUR arms so we also settle div-vs-nodiv honestly:
  base   = Autolycus (traverse_explanations_SHAP)
  shap3  = traverse_explanations_SHAP3
  s4b_nd = SHAP4b(use_diverse=False)   [notes' 'best' tree config]
  s4b_dv = SHAP4b(use_diverse=True)    [full SHAP4b]
Seeds generated once per combo and REUSED across all Q (paired design, fixes the §3.5 confound).
Metric: MEAN over refits (the fixed, unbiased metric). dt reps=100, rdf reps=10.
"""
import random, json, io
from contextlib import redirect_stdout
import numpy as np
from attack_utils import (load_dataset, load_model, load_explainer, mega_sample_generation,
                          rtest_sim, traverse_explanations_SHAP, traverse_explanations_SHAP3,
                          traverse_explanations_SHAP4b)
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.ensemble import RandomForestClassifier as RF

random.seed(0); np.random.seed(0)
DNAMES = {1: 'crop', 2: 'adult', 3: 'breast'}; MNAMES = {0: 'DT', 4: 'RF'}
DEPTH = 15; HMS = 20; KF = 3
# (dataset, model, [Q values]); breast train set is small -> cap Q
COMBOS = [(2, 4, [100, 250, 500, 1000]),   # adult+RF  (flagship survivor)
          (3, 0, [100, 250, 500]),         # breast+DT (surprise survivor)
          (1, 0, [100, 250, 500, 1000])]   # crop+DT   (dissolved control)

def reps_for(mn): return 100 if mn == 'dt' else 10
def make(mn, k): return DT(random_state=k, max_depth=DEPTH) if mn == 'dt' else RF(max_depth=DEPTH, random_state=k)

def meaneval(vs, vp, t_model, Xtt, mn):
    sims = []
    for k in range(reps_for(mn)):
        m = make(mn, k)
        try: m.fit(vs, vp)
        except Exception: continue
        sims.append(rtest_sim(m, t_model, Xtt.values))
    return float(np.mean(sims)) if sims else float('nan')

def quiet(fn, *a, **k):
    with redirect_stdout(io.StringIO()): return fn(*a, **k)

ARMS = [('base', traverse_explanations_SHAP, {}),
        ('shap3', traverse_explanations_SHAP3, {}),
        ('s4b_nd', traverse_explanations_SHAP4b, {'use_diverse': False}),
        ('s4b_dv', traverse_explanations_SHAP4b, {'use_diverse': True})]

allres = []
for wd, wm, QS in COMBOS:
    a1, a2 = load_dataset(wd)
    Xtr, Xte, ytr, yte, Xtt, Xts, ytt, yts = a1
    classes, features, nc, nf, isCat, eps, canNeg, cposs, dn, fr = a2
    t_model, mn = load_model(wm, Xtr, ytr); expl = load_explainer(1, t_model, mn, Xtr)
    sm = mega_sample_generation(Xts.to_numpy(), yts, nc, [5], HMS)  # ONE seed set, reused across Q
    for Q in QS:
        lb = int((Q // nc) * 0.5 + 1); ub = int((Q // nc) * (nc + 0.5) + 1)
        acc = {a: [] for a, _, _ in ARMS}
        for i in range(HMS):
            seed = sm[i][0]
            for name, fn, kw in ARMS:
                vs, vp, _ = quiet(fn, seed, expl, t_model, lb, ub, Q, KF, a2, mn, Xtr, ytr, **kw)
                acc[name].append(meaneval(vs, vp, t_model, Xtt, mn))
        row = {'dataset': DNAMES[wd], 'model': MNAMES[wm], 'Q': Q}
        for name, _, _ in ARMS:
            row[name] = round(float(np.nanmean(acc[name])), 4)
        row['nd-base'] = round(row['s4b_nd'] - row['base'], 4)
        row['dv-base'] = round(row['s4b_dv'] - row['base'], 4)
        row['nd-shap3'] = round(row['s4b_nd'] - row['shap3'], 4)
        row['nd-dv'] = round(row['s4b_nd'] - row['s4b_dv'], 4)
        allres.append(row)
        print(f"[{DNAMES[wd]}+{MNAMES[wm]} Q={Q:<4}] base={row['base']:.4f} shap3={row['shap3']:.4f} "
              f"nd={row['s4b_nd']:.4f} dv={row['s4b_dv']:.4f} | nd-base={row['nd-base']:+.4f} "
              f"nd-shap3={row['nd-shap3']:+.4f} nd-dv={row['nd-dv']:+.4f}", flush=True)
        with open('survivor_sweep.json', 'w') as f:
            json.dump(allres, f, indent=2)

print("\n=== SURVIVOR Q-SWEEP (mean-eval, 20 paired sets) ===", flush=True)
hdr = f"{'data':8}{'mdl':4}{'Q':>6}{'base':>8}{'shap3':>8}{'nd':>8}{'dv':>8}{'nd-base':>9}{'nd-shap3':>10}{'nd-dv':>8}"
print(hdr); print('-' * len(hdr))
for r in allres:
    print(f"{r['dataset']:8}{r['model']:4}{r['Q']:>6}{r['base']:>8.4f}{r['shap3']:>8.4f}{r['s4b_nd']:>8.4f}"
          f"{r['s4b_dv']:>8.4f}{r['nd-base']:>+9.4f}{r['nd-shap3']:>+10.4f}{r['nd-dv']:>+8.4f}", flush=True)
print("done", flush=True)
