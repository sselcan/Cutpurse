"""SHAP3 (shipping non-tree) gain attribution on ALL non-tree datasets, incl. the big-gain crop/nursery.
Answers: does SHAP's feature choice help (gen vs traverse), and is the gain from the boundary method or
the diverse generation? Arms (leave-one-out + SHAP gen/traverse 2x2), LR/NB/KNN, Q=500, 10 paired sets:
  baseline  Autolycus
  TT        full shipping SHAP3 (diverse + SHAP gen + SHAP traverse)
  TF/FT/FF  SHAP gen/traverse 2x2  -> genGain, traverseGain
  no_diverse (use_diverse=False)   -> div_contrib      = TT - no_diverse
  no_middle  (n_middle=0)          -> boundary_contrib = TT - no_middle  (whole middle-sample method incl. recycling)
"""
import random, json, io
from contextlib import redirect_stdout
import numpy as np
from attack_utils import (load_dataset, load_model, load_explainer, mega_sample_generation,
                          rtest_sim, traverse_explanations_SHAP, traverse_explanations_SHAP3)
from sklearn.linear_model import LogisticRegression as LR
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.naive_bayes import MultinomialNB as MNB

random.seed(0); np.random.seed(0)
DNAMES = {1: 'crop', 2: 'adult', 3: 'breast', 4: 'nursery', 5: 'mushroom'}; MNAMES = {1: 'LR', 2: 'NB', 3: 'KNN'}
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

S3 = traverse_explanations_SHAP3
ARMS = [
    ('baseline',   lambda *a: traverse_explanations_SHAP(*a)),
    ('s3_TT',      lambda *a: S3(*a)),                                              # shipping (diverse + SHAP gen + SHAP traverse)
    ('s3_TF',      lambda *a: S3(*a, use_shap_traverse=False)),                     # SHAP gen, random traverse
    ('s3_FT',      lambda *a: S3(*a, use_shap_gen=False)),                          # random gen, SHAP traverse
    ('s3_FF',      lambda *a: S3(*a, use_shap_gen=False, use_shap_traverse=False)), # random both
    ('no_diverse', lambda *a: S3(*a, use_diverse=False)),                           # Phase-1 diverse OFF
    ('no_middle',  lambda *a: S3(*a, n_middle=0)),                                  # middle-sample method OFF
]
rows = []
for ds in [1, 4, 2, 5, 3]:  # crop, nursery, adult, mushroom, breast
    a1, a2 = load_dataset(ds)
    Xtr, Xte, ytr, yte, Xtt, Xts, ytt, yts = a1
    classes, features, nc, nf, isCat, eps, canNeg, cposs, dn, fr = a2
    for wm in [1, 2, 3]:
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
        r['full_vs_base'] = round(r['s3_TT'] - r['baseline'], 4)
        r['div_contrib'] = round(r['s3_TT'] - r['no_diverse'], 4)
        r['boundary_contrib'] = round(r['s3_TT'] - r['no_middle'], 4)
        r['genGain'] = round(((r['s3_TT'] - r['s3_FT']) + (r['s3_TF'] - r['s3_FF'])) / 2, 4)
        r['traverseGain'] = round(((r['s3_TT'] - r['s3_TF']) + (r['s3_FT'] - r['s3_FF'])) / 2, 4)
        rows.append(r)
        print(f"[{DNAMES[ds]} {MNAMES[wm]:3}] base={r['baseline']:.4f} full={r['s3_TT']:.4f} (vsBase {r['full_vs_base']:+.3f}) "
              f"| div={r['div_contrib']:+.4f} boundary={r['boundary_contrib']:+.4f} "
              f"genSHAP={r['genGain']:+.4f} travSHAP={r['traverseGain']:+.4f}", flush=True)
        with open('nontree_shap3.json', 'w') as f:
            json.dump(rows, f, indent=2)

print("\n=== SHAP3 NON-TREE GAIN ATTRIBUTION (leave-one-out from full) [Q=500, 10 sets] ===", flush=True)
hdr = f"{'data':8}{'mdl':4}{'base':>8}{'full':>8}{'vsBase':>8}{'div':>9}{'boundary':>10}{'genSHAP':>9}{'travSHAP':>10}"
print(hdr); print('-' * len(hdr))
for r in rows:
    print(f"{r['dataset']:8}{r['model']:4}{r['baseline']:>8.4f}{r['s3_TT']:>8.4f}{r['full_vs_base']:>+8.3f}"
          f"{r['div_contrib']:>+9.4f}{r['boundary_contrib']:>+10.4f}{r['genGain']:>+9.4f}{r['traverseGain']:>+10.4f}")
print("\nEach = full - (factor removed). Positive => factor helps. boundary=whole middle-sample method (incl. recycling).")
print("done", flush=True)
