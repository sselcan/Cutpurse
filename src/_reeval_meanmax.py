"""Re-evaluation under the FIXED metric (mean over refits) vs the OLD leaky metric (max over
refits selected on X_test_t), side by side, on the headline tree combos.

Each traversal is run ONCE and scored two ways, so the ONLY difference between the mean and max
columns is the aggregation -- an apples-to-apples measurement of how much the test-set selection
bias was inflating each arm, and whether the densification win survives under the unbiased mean.

Arms: baseline (Autolycus SHAP) / SHAP3 / SHAP4b(use_diverse=False)  [the shipping tree config].
Paired seed sets per combo (same sets across arms). Q=500, 20 sets, k=3, surrogate matched to target.
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
DNAMES = {1: 'crop', 2: 'adult', 3: 'breast', 5: 'mushroom'}
MNAMES = {0: 'DT', 4: 'RF'}
DEPTH = 15; HMS = 20; Q = 500; KF = 3
COMBOS = [(2, 4), (1, 0), (3, 4), (5, 4), (2, 0), (1, 4), (3, 0), (5, 0)]  # RF wins first, then rest

def reps_for(mn): return 100 if mn == 'dt' else 10
def make(mn, k): return DT(random_state=k, max_depth=DEPTH) if mn == 'dt' else RF(max_depth=DEPTH, random_state=k)

def eval_both(vs, vp, t_model, Xtt, mn):
    """Return (mean, max) similarity over the refits -- the fixed metric and the old leaky one."""
    sims = []
    for k in range(reps_for(mn)):
        m = make(mn, k)
        try:
            m.fit(vs, vp)
        except Exception:
            continue
        sims.append(rtest_sim(m, t_model, Xtt.values))
    if not sims:
        return float('nan'), float('nan')
    return float(np.mean(sims)), float(np.max(sims))

def quiet(fn, *a, **k):
    with redirect_stdout(io.StringIO()):
        return fn(*a, **k)

ARMS = [('base', traverse_explanations_SHAP, {}),
        ('shap3', traverse_explanations_SHAP3, {}),
        ('s4b', traverse_explanations_SHAP4b, {'use_diverse': False})]

allres = []
for wd, wm in COMBOS:
    a1, a2 = load_dataset(wd)
    Xtr, Xte, ytr, yte, Xtt, Xts, ytt, yts = a1
    classes, features, nc, nf, isCat, eps, canNeg, cposs, dn, fr = a2
    t_model, mn = load_model(wm, Xtr, ytr); expl = load_explainer(1, t_model, mn, Xtr)
    lb = int((Q // nc) * 0.5 + 1); ub = int((Q // nc) * (nc + 0.5) + 1)
    sm = mega_sample_generation(Xts.to_numpy(), yts, nc, [5], HMS)
    acc = {a: {'mean': [], 'max': []} for a, _, _ in ARMS}
    for i in range(HMS):
        seed = sm[i][0]
        for name, fn, kw in ARMS:
            vs, vp, _ = quiet(fn, seed, expl, t_model, lb, ub, Q, KF, a2, mn, Xtr, ytr, **kw)
            me, mx = eval_both(vs, vp, t_model, Xtt, mn)
            acc[name]['mean'].append(me); acc[name]['max'].append(mx)
        print(f"  [{DNAMES[wd]}+{MNAMES[wm]}] set {i}: " +
              " ".join(f"{n}={np.mean(acc[n]['mean'][-1:]):.3f}" for n, _, _ in ARMS), flush=True)
    row = {'dataset': DNAMES[wd], 'model': MNAMES[wm], 'Q': Q, 'n_sets': HMS}
    for name, _, _ in ARMS:
        row[f'{name}_mean'] = round(float(np.mean(acc[name]['mean'])), 4)
        row[f'{name}_max'] = round(float(np.mean(acc[name]['max'])), 4)
        row[f'{name}_mean_std'] = round(float(np.std(acc[name]['mean'])), 4)
    row['gap_mean_s4b_base'] = round(row['s4b_mean'] - row['base_mean'], 4)
    row['gap_max_s4b_base'] = round(row['s4b_max'] - row['base_max'], 4)   # what the OLD notes reported
    row['gap_mean_s4b_shap3'] = round(row['s4b_mean'] - row['shap3_mean'], 4)
    allres.append(row)
    print(f"### {DNAMES[wd]}+{MNAMES[wm]}: base={row['base_mean']:.4f} shap3={row['shap3_mean']:.4f} "
          f"s4b={row['s4b_mean']:.4f} | MEAN gap s4b-base={row['gap_mean_s4b_base']:+.4f} "
          f"(OLD max gap={row['gap_max_s4b_base']:+.4f})", flush=True)
    with open('reeval_meanmax.json', 'w') as f:
        json.dump(allres, f, indent=2)

print("\n=== RE-EVAL: fixed MEAN metric vs old leaky MAX metric (Q=500, 20 sets) ===", flush=True)
hdr = f"{'data':9}{'mdl':4}{'base':>8}{'shap3':>8}{'s4b':>8}{'s4b-base(mean)':>16}{'s4b-base(OLDmax)':>17}{'s4b-shap3':>11}"
print(hdr); print('-' * len(hdr))
for r in allres:
    print(f"{r['dataset']:9}{r['model']:4}{r['base_mean']:>8.4f}{r['shap3_mean']:>8.4f}{r['s4b_mean']:>8.4f}"
          f"{r['gap_mean_s4b_base']:>+16.4f}{r['gap_max_s4b_base']:>+17.4f}{r['gap_mean_s4b_shap3']:>+11.4f}", flush=True)
print("done", flush=True)
