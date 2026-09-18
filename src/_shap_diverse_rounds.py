"""Test the supervisor's proposal: SHAP-vector-distance seed diversification.

Reframes SHAP from "which FEATURE to perturb" (which ties random -- see
lime-mechanism / shap-normal notes) to "which already-queried REGION to revisit
next". A round-based extraction loop that, each round:
  1. keeps queried points whose TARGET prediction is near 0.5 (boundary-adjacent),
  2. selects k diverse SEEDS from that pool,
  3. grows RANDOM-feature epsilon perturbations around them, queries the target,
  4. adds them to the training set; their SHAP vectors feed the NEXT round's seed pick.

The FOUR arms differ ONLY in the round-2 seed-selection diversity metric:
  random -- k random seeds from the near-0.5 pool            (null: diversity off)
  input  -- farthest-first in INPUT space (categorical-aware)(the existing 'diverse region' idea)
  shap   -- farthest-first over SHAP explanation vectors     (the supervisor's new axis)
  both   -- farthest-first over normalized input + SHAP      (his 'cover regions AND behaviours')

Everything else (seeds, near-0.5 filter, random perturbation, budget, surrogate)
is identical and PAIRED across arms on the same seed sets. SHAP never chooses a
direction here -- it only supplies a behavioural fingerprint whose DISTANCES
diversify seeds, a far weaker (and more robust) demand than direction recovery.

Key prediction from coverage-signature note: on a LINEAR target (lr) diverse gen
adds ~0 (boundary pins interiors) -- so lr is the NULL control; the real test is
the NON-LINEAR targets (knn, rdf) where interiors are NOT determined by the boundary.

Writes shap_diverse_rounds.json.
"""
import json, time, warnings
import numpy as np
from attack_utils import (load_dataset, load_model, load_explainer,
                          mega_sample_generation, _fit_one_surrogate)

warnings.filterwarnings("ignore")

# ---- experiment grid ----
DATASETS = [2, 3]                 # adult (mixed, binary), breast (continuous, binary)
MODELS   = [1, 0, 3, 4]           # lr (linear NULL), dt (piecewise) first for fast validation; then knn, rdf (NON-LINEAR: the real test)
ARMS     = ['random', 'input', 'shap', 'both']
Q_TOTAL      = 300
CHECKPOINTS  = [100, 200, 300]
N_SEEDSETS   = 4
SIZE         = 5                  # seeds/class (paper n=5) -> 10 seeds (binary)
K_SEEDS      = 8                  # diverse seeds selected per round
CHILDREN     = 4                  # random-perturbation children per seed
NFE          = 3                  # features perturbed per child
BAND         = 0.25              # near-0.5 band on p(class 1)
POOL_CAP     = 50                # cap candidate pool (lowest-margin points) for selection speed
NSAMPLES_SHAP = 64              # KernelExplainer nsamples for the fingerprint (fast, not exact)
REPS = {'dt': 2, 'rdf': 2, 'knn': 1, 'lr': 1, 'nb': 1}
EVAL_CAP = 2000                  # subsample the target test set for fidelity (speed)
SEED = 0


# ---------- SHAP fingerprint (predicted-class attribution vector) ----------
def shap_fingerprint(explainer, X, model_name, model, n_classes):
    """Return (n, d) SHAP vectors: the attribution of the point's predicted class.
    Trees use the TreeExplainer; others the KernelExplainer with l1_reg=0 (the
    memory's pipeline fix -- default l1_reg='auto' SPARSIFIES and collapses vectors)."""
    X = np.asarray(X, dtype=float)
    if model_name in ('dt', 'rdf'):
        sv = np.asarray(explainer.shap_values(X, check_additivity=False))
        preds = model.predict(X).astype(int)
        if sv.ndim == 3 and sv.shape[0] == n_classes and sv.shape[1] == len(X):   # (C,n,d)
            out = np.stack([sv[preds[i], i] for i in range(len(X))])
        elif sv.ndim == 3 and sv.shape[-1] == n_classes:                          # (n,d,C)
            out = np.stack([sv[i, :, preds[i]] for i in range(len(X))])
        else:
            out = sv.reshape(len(X), -1)
    else:
        sv = explainer.shap_values(X, nsamples=NSAMPLES_SHAP, l1_reg=0, silent=True)
        out = np.asarray(sv).reshape(len(X), -1)
    return np.asarray(out, dtype=float)


# ---------- distance matrices + farthest-first selection ----------
def input_dist(X, isCat, feature_ranges):
    X = np.asarray(X, float); n, d = X.shape
    D = np.zeros((n, n))
    for j in range(d):
        col = X[:, j]
        if isCat[j]:
            D += (col[:, None] != col[None, :]).astype(float)
        else:
            lo, hi = feature_ranges[j]; r = (hi - lo) if hi > lo else 1.0
            D += ((col[:, None] - col[None, :]) / r) ** 2
    return np.sqrt(D)


def shap_dist(S):
    S = np.asarray(S, float)
    sd = S.std(axis=0); sd[sd == 0] = 1.0
    Sn = S / sd
    diff = Sn[:, None, :] - Sn[None, :, :]
    return np.sqrt((diff ** 2).sum(-1))


def _norm(D):
    pos = D[D > 0]
    return D / pos.mean() if pos.size else D


def farthest_first(D, k, start):
    n = D.shape[0]
    sel = [start]
    while len(sel) < min(k, n):
        mind = D[:, sel].min(axis=1)
        mind[sel] = -np.inf
        nxt = int(np.argmax(mind))
        if not np.isfinite(mind[nxt]):
            break
        sel.append(nxt)
    return sel


def select_seeds(arm, cand_X, cand_S, margins, k, isCat, feature_ranges, rng):
    """Return LOCAL indices (into cand_*) of the chosen diverse seeds."""
    n = len(cand_X)
    if n <= k:
        return list(range(n))
    start = int(np.argmin(margins))              # anchor on the most boundary-adjacent point
    if arm == 'random':
        others = [i for i in range(n) if i != start]
        return [start] + list(rng.choice(others, size=k - 1, replace=False))
    if arm == 'input':
        return farthest_first(input_dist(cand_X, isCat, feature_ranges), k, start)
    if arm == 'shap':
        return farthest_first(shap_dist(cand_S), k, start)
    # both
    D = _norm(input_dist(cand_X, isCat, feature_ranges)) + _norm(shap_dist(cand_S))
    return farthest_first(D, k, start)


# ---------- random-feature perturbation (SHAP does NOT pick the feature) ----------
def perturb(x, rng, nfe, epsilon_set, isCat, canNegative, classPossibilities, n_features):
    child = np.array(x, dtype=float)
    feats = rng.choice(n_features, size=min(nfe, n_features), replace=False)
    for j in feats:
        if isCat[j]:
            step = 1 if rng.random() < 0.5 else -1               # +-1 category (on-manifold)
            val = child[j] + step
            val = min(max(val, 0), classPossibilities[j] - 1)
        else:
            step = epsilon_set[j] * (1 if rng.random() < 0.5 else -1)
            val = child[j] + step
            if val < 0 and not canNegative[j]:
                val = child[j] + abs(step)
        child[j] = val
    return child


# ---------- one arm on one seed set ----------
def run_arm(arm, seed_rows, t_model, model_name, explainer, n_classes, args2, rng):
    (classes, features, _, n_features, isCat, epsilon_set, canNegative,
     classPossibilities, dataset_name, feature_ranges) = args2
    uses_shap = arm in ('shap', 'both')

    seeds = np.asarray(list(seed_rows), dtype=float)   # harness seed rows are already feature-only
    pool_X = list(seeds)
    p1 = t_model.predict_proba(seeds)[:, 1]
    pool_p1 = list(p1)
    pool_lab = list(t_model.predict(seeds).astype(int))
    pool_shap = [None] * len(pool_X)                              # lazy: only near-0.5 points get SHAP
    train_X = list(seeds); train_y = list(pool_lab)
    seen = set(tuple(np.round(s, 6)) for s in seeds)
    q_used = len(seeds)

    checkpoints = {}
    ci = 0

    def record(q):
        Xt = np.asarray(train_X, float); yt = np.asarray(train_y, int)
        if len(np.unique(yt)) < 2:
            return float('nan')
        reps = REPS.get(model_name, 1)
        fids = []
        for kk in range(reps):
            if model_name == 'nb':
                Xt = np.clip(Xt, 0, None)
            sm = _fit_one_surrogate(model_name, Xt, yt, n_classes, seed=kk)
            fids.append(float(np.mean(t_model.predict(EVAL_X) == sm.predict(EVAL_X))))
        return float(np.mean(fids))

    while q_used < Q_TOTAL:
        margins_all = np.abs(np.asarray(pool_p1) - 0.5)
        order = np.argsort(margins_all)
        cand_idx = [i for i in order if margins_all[i] <= BAND][:POOL_CAP]
        if len(cand_idx) < K_SEEDS:                              # fallback: take the k lowest-margin points
            cand_idx = list(order[:max(K_SEEDS, POOL_CAP // 2)])

        # lazily compute SHAP for candidate points that don't have it yet
        if uses_shap:
            missing = [i for i in cand_idx if pool_shap[i] is None]
            if missing:
                sv = shap_fingerprint(explainer, [pool_X[i] for i in missing],
                                      model_name, t_model, n_classes)
                for loc, i in enumerate(missing):
                    pool_shap[i] = sv[loc]

        cand_X = [pool_X[i] for i in cand_idx]
        cand_S = [pool_shap[i] for i in cand_idx] if uses_shap else None
        cand_m = margins_all[cand_idx]
        loc_sel = select_seeds(arm, cand_X, cand_S, cand_m, K_SEEDS,
                               isCat, feature_ranges, rng)
        chosen = [cand_idx[i] for i in loc_sel]

        # grow random-perturbation children around chosen seeds; query the target
        new = []
        for gi in chosen:
            for _ in range(CHILDREN):
                c = perturb(pool_X[gi], rng, NFE, epsilon_set, isCat,
                            canNegative, classPossibilities, n_features)
                key = tuple(np.round(c, 6))
                if key not in seen:
                    seen.add(key); new.append(c)
        if not new:
            break
        new = new[:max(0, Q_TOTAL - q_used)]
        Xn = np.asarray(new, float)
        pn1 = t_model.predict_proba(Xn)[:, 1]
        yn = t_model.predict(Xn).astype(int)
        for r in range(len(new)):
            pool_X.append(new[r]); pool_p1.append(float(pn1[r]))
            pool_lab.append(int(yn[r])); pool_shap.append(None)
            train_X.append(new[r]); train_y.append(int(yn[r]))
        q_used += len(new)

        while ci < len(CHECKPOINTS) and q_used >= CHECKPOINTS[ci]:
            checkpoints[CHECKPOINTS[ci]] = record(q_used); ci += 1

    while ci < len(CHECKPOINTS):                                 # budget exhausted before last cp
        checkpoints[CHECKPOINTS[ci]] = record(q_used); ci += 1
    return checkpoints


# ---------- main sweep ----------
out = {}
t0 = time.time()
for wd in DATASETS:
    args1, args2 = load_dataset(wd)
    X_train, X_test, y_train, y_test, X_test_t, X_test_s, y_test_t, y_test_s = args1
    n_classes = args2[2]; dataset_name = args2[8]
    EVAL_X = np.asarray(X_test_t.values, dtype=float)[:EVAL_CAP]
    out[dataset_name] = {}
    for m in MODELS:
        np.random.seed(SEED)
        t_model, model_name = load_model(m, X_train, y_train)
        explainer = load_explainer(1, t_model, model_name, X_train)
        mega = mega_sample_generation(X_test_s.to_numpy(), y_test_s, n_classes, [SIZE], N_SEEDSETS)
        res = {a: {cp: [] for cp in CHECKPOINTS} for a in ARMS}
        for a in ARMS:
            ta = time.time()
            for si in range(N_SEEDSETS):
                rng = np.random.RandomState(1000 * ARMS.index(a) + si)
                cps = run_arm(a, mega[si][0], t_model, model_name, explainer,
                              n_classes, args2, rng)
                for cp in CHECKPOINTS:
                    res[a][cp].append(cps.get(cp, float('nan')))
            print(f"    [{dataset_name}/{model_name}] arm {a:>7} done in {time.time()-ta:.0f}s", flush=True)
        out[dataset_name][model_name] = {
            a: {cp: res[a][cp] for cp in CHECKPOINTS} for a in ARMS}

        # per-cell summary: mean fidelity + the two decisive deltas
        print(f"\n=== {dataset_name} / {model_name} ===   (elapsed {time.time()-t0:.0f}s)")
        print(f"  {'Q':>4}  " + "  ".join(f"{a:>7}" for a in ARMS) +
              "  | shap-input  both-input  input-rand")
        for cp in CHECKPOINTS:
            mean = {a: np.nanmean(res[a][cp]) for a in ARMS}
            print(f"  {cp:>4}  " + "  ".join(f"{mean[a]:7.3f}" for a in ARMS) +
                  f"  |   {mean['shap']-mean['input']:+.3f}     {mean['both']-mean['input']:+.3f}"
                  f"     {mean['input']-mean['random']:+.3f}")

json.dump(out, open("shap_diverse_rounds.json", "w"), indent=2)
print(f"\nwrote shap_diverse_rounds.json  (total {time.time()-t0:.0f}s)")
