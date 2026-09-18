"""Ablation answering two questions on the two winning combos (adult+RF, crop+DT):
  (1) does dropping Phase-1 diverse generation + reallocating budget to boundary search help trees?
      -> arm 'shap4b_nodiv' (use_diverse=False)
  (2) is SHAP4b's gain from densification or just the shared SHAP3 scaffolding (Phase1+Phase3)?
      -> arm 'scaffold' (densify_n_pairs=0 => no Phase 2 at all). If scaffold ~= shap4b, Phase 2 is inert.
30 sets, mean±std, equal-budget, surrogate matched to target."""
import random, json, io, functools
from contextlib import redirect_stdout
import numpy as np
from attack_utils import (load_dataset, load_model, load_explainer, mega_sample_generation,
                          rtest_sim, traverse_explanations_SHAP, traverse_explanations_SHAP3,
                          traverse_explanations_SHAP4b)
from sklearn.linear_model import LogisticRegression as lr
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.ensemble import RandomForestClassifier as RF

random.seed(0); np.random.seed(0)
NAMES = {2: 'adult', 1: 'crop'}; MODELS = {0: 'DT', 4: 'RF'}
DEPTH = 15; HMS = 30
ARMS = [
    ('baseline', traverse_explanations_SHAP),
    ('shap3', traverse_explanations_SHAP3),
    ('shap4b', traverse_explanations_SHAP4b),
    ('shap4b_nodiv', functools.partial(traverse_explanations_SHAP4b, use_diverse=False)),  # Q1
    ('scaffold', functools.partial(traverse_explanations_SHAP4b, densify_n_pairs=0)),       # Q2 (Phase1+Phase3 only)
]

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

COMBOS = [(2, 4), (1, 0)]  # adult+RF, crop+DT (the two winning combos)
allres = []
for wd, wm in COMBOS:
    print(f"\n### {NAMES[wd]}+{MODELS[wm]} (30 sets) ###", flush=True)
    allres.append(run_combo(wd, wm))
    r = allres[-1]
    print("  -> " + " ".join(f"{n}={r[n + '_mean']}" for n, _ in ARMS), flush=True)
    with open('shap4b_ablate.json', 'w') as f:
        json.dump(allres, f, indent=2)

print("\n=== SHAP4b ABLATION (30 sets, mean±std, Q=500): diverse-drop + scaffold control ===", flush=True)
hdr = f"{'data':7}{'mdl':4}{'baseline':>13}{'shap3':>13}{'shap4b':>13}{'s4b_nodiv':>13}{'scaffold':>13}"
print(hdr); print('-' * len(hdr))
for r in allres:
    cells = "".join(f"{r[n + '_mean']:.4f}±{r[n + '_std']:.3f}".rjust(13) for n in ['baseline', 'shap3', 'shap4b', 'shap4b_nodiv', 'scaffold'])
    print(f"{r['dataset']:7}{r['model']:4}{cells}")
print("\nReading it:")
print("  Q1 (drop diverse): compare shap4b_nodiv vs shap4b")
print("  Q2 (is it Phase 2?): compare shap4b vs scaffold (if ~equal, densification adds nothing)")
print("done", flush=True)
