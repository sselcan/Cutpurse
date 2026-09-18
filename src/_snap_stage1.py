"""Stage 1 - threshold snapping. Fit a DT on densified samples (keeps full depth/structure), then
overwrite each CONTINUOUS split threshold with the nearest bisection-recovered true threshold f*
(returned by densification). Tests whether residual split-POSITION error after densification is worth
correcting for DT targets. Continuous datasets only (snapping needs numeric thresholds): adult, breast, crop.
Arms: baseline (Autolycus) / plain (= SHAP4b-nodiv) / snap_all / snap_close (snap only if within 0.1*range)."""
import random, json, io
from contextlib import redirect_stdout
import numpy as np
from attack_utils import (load_dataset, load_model, load_explainer, mega_sample_generation,
                          rtest_sim, traverse_explanations_SHAP, traverse_explanations_SHAP4b)
from sklearn.tree import DecisionTreeClassifier as DT

random.seed(0); np.random.seed(0)
DNAMES = {1: 'crop', 2: 'adult', 3: 'breast'}; DEPTH = 15; HMS = 10; REPS = 30

class SnappedTree:
    """Wrap a fitted DT; snap continuous split thresholds to nearest recovered f*; predict via traversal."""
    def __init__(self, clf, recovered, isCat, feature_ranges, tol_frac=None):
        t = clf.tree_
        self.feature = t.feature.copy(); self.threshold = t.threshold.copy().astype(float)
        self.left = t.children_left.copy(); self.right = t.children_right.copy()
        self.value = t.value.copy(); self.classes_ = clf.classes_
        self.n_splits = int((self.feature >= 0).sum()); self.n_snapped = 0
        for node in range(len(self.feature)):
            f = int(self.feature[node])
            if f >= 0 and not isCat[f] and f in recovered and recovered[f]:
                cur = float(self.threshold[node])
                nearest = min(recovered[f], key=lambda v: abs(v - cur))
                ok = True
                if tol_frac is not None:
                    lo, hi = feature_ranges[f]; rng = (hi - lo) if hi > lo else 1.0
                    ok = abs(nearest - cur) <= tol_frac * rng
                if ok:
                    self.threshold[node] = nearest; self.n_snapped += 1
    def predict_proba(self, X):
        X = np.asarray(X, dtype=float); out = np.empty((len(X), self.value.shape[2]))
        for r in range(len(X)):
            node = 0
            while self.feature[node] >= 0:
                node = self.left[node] if X[r, int(self.feature[node])] <= self.threshold[node] else self.right[node]
            v = self.value[node][0]; s = v.sum(); out[r] = v / s if s > 0 else v
        return out
    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

def quiet(fn, *a, **k):
    with redirect_stdout(io.StringIO()):
        return fn(*a, **k)

rows = []
for ds in [2, 3, 1]:  # adult, breast, crop (continuous)
    a1, a2 = load_dataset(ds)
    Xtr, Xte, ytr, yte, Xtt, Xts, ytt, yts = a1
    classes, features, nc, nf, isCat, eps, canNeg, cposs, dn, fr = a2
    t_model, mn = load_model(0, Xtr, ytr); expl = load_explainer(1, t_model, mn, Xtr)  # DT target
    sm = mega_sample_generation(Xts.to_numpy(), yts, nc, [5], HMS)
    Qs = [100, 250] if ds == 3 else [250, 500, 1000]  # breast: Autolycus low-Q range
    for Q in Qs:
        lb = int((Q // nc) * 0.5 + 1); ub = int((Q // nc) * (nc + 0.5) + 1)
        base = []; plain = []; snapA = []; snapC = []; nsn = []; nsp = []
        for i in range(HMS):
            seed = sm[i][0]
            vsb, vpb, _ = quiet(traverse_explanations_SHAP, seed, expl, t_model, lb, ub, Q, 3, a2, mn, Xtr, ytr)
            bb = float(np.mean([rtest_sim(DT(random_state=k, max_depth=DEPTH).fit(vsb, vpb), t_model, Xtt.values) for k in range(REPS)]))  # MEAN not max
            base.append(bb)
            vs, vp, _, rec = quiet(traverse_explanations_SHAP4b, seed, expl, t_model, lb, ub, Q, 3, a2, mn, Xtr, ytr,
                                   use_diverse=False, return_thresholds=True)
            plain_r = []; snapA_r = []; snapC_r = []; sn = sp = 0  # MEAN over refits, not max
            for k in range(REPS):
                clf = DT(random_state=k, max_depth=DEPTH).fit(vs, vp)
                plain_r.append(rtest_sim(clf, t_model, Xtt.values))
                sa = SnappedTree(clf, rec, isCat, fr, tol_frac=None); snapA_r.append(rtest_sim(sa, t_model, Xtt.values))
                sc = SnappedTree(clf, rec, isCat, fr, tol_frac=0.1); snapC_r.append(rtest_sim(sc, t_model, Xtt.values))
                sn += sa.n_snapped; sp += sa.n_splits
            plain.append(float(np.mean(plain_r))); snapA.append(float(np.mean(snapA_r))); snapC.append(float(np.mean(snapC_r))); nsn.append(sn / REPS); nsp.append(sp / REPS)
        row = {'dataset': DNAMES[ds], 'Q': Q, 'baseline': round(float(np.mean(base)), 4),
               'plain': round(float(np.mean(plain)), 4), 'snap_all': round(float(np.mean(snapA)), 4),
               'snap_close': round(float(np.mean(snapC)), 4),
               'n_snapped': round(float(np.mean(nsn)), 1), 'n_splits': round(float(np.mean(nsp)), 1)}
        rows.append(row)
        print(f"[{DNAMES[ds]} DT Q={Q}] base={row['baseline']:.4f} plain={row['plain']:.4f} "
              f"snapAll={row['snap_all']:.4f} snapClose={row['snap_close']:.4f} "
              f"(snapped {row['n_snapped']}/{row['n_splits']})", flush=True)
        with open('snap_stage1.json', 'w') as f:
            json.dump(rows, f, indent=2)

print("\n=== STAGE 1: THRESHOLD SNAPPING (DT, continuous datasets, 10 sets) ===", flush=True)
print(f"{'data':8}{'Q':>5}{'base':>9}{'plain':>9}{'snapAll':>10}{'snapClose':>11}{'snap/splits':>14}")
for r in rows:
    print(f"{r['dataset']:8}{r['Q']:>5}{r['baseline']:>9.4f}{r['plain']:>9.4f}{r['snap_all']:>10.4f}"
          f"{r['snap_close']:>11.4f}{(str(r['n_snapped'])+'/'+str(r['n_splits'])):>14}")
print("done", flush=True)
