# Finding: Batched SHAP Reconstructs the Boundary Normal (Diagnostic Report, 2026-07)

**TL;DR.** A *local batch* of SHAP explanations reconstructs the target's local
decision-boundary normal to cos ≈ 1.0 where a *single* SHAP vector cannot — but only
inside a specific box (margin/non-saturated output space, smooth model, continuous
features, local batch of ~10–50). The recovered normal is orthogonal to the direction the current attack perturbs along (`cos(g,p)≈0`), i.e. it is *new* geometry. Whether it *improves extraction* is untested (geometry result, not yet an attack result).

Code: `run_shap_normal_diagnostic` + helpers in `attack_utils.py` (read-only; no attack path touched). Drivers: `_shap_normal_diag.py` → `shap_normal_diag.json` / `_shap_normal_diag.log`; `_shap_normal_smallk.py` → `shap_normal_smallk.json`. Design: `SPEC_shap_normal_diagnostic.md`.

## 1. Question
Individual SHAP vectors don't help the attack (DEVELOPMENT_NOTES §3.13: the gain is boundary *sampling*, not SHAP feature-ranking). Gradient-based extractors use the gradient as a direct pointer to the boundary. **Can a *collection* of SHAP vectors serve the same role — reconstruct the local boundary normal — where a single one fails?**

## 2. Method (one sentence each)
- **Reconstruction identity**: for a locally-linear target, interventional SHAP with a single baseline `x0` satisfies `s_i = w_i·(x_i − x0_i)`, so `w_i = s_i/(x_i−x0_i)`.
- **single normal** `w_single`: from one point's SHAP (0/0-guarded).
- **batch normal** `w_batch(k)`: per-feature no-intercept OLS of `s_i` on `d_i=(x_i−x0_i)` over the `k` nearest neighbors (a "collection"), unidentified coords flagged.
- **true normal** `g`: finite-difference gradient of the model output (no SHAP).
- **attack direction** `p`: the current attack's top-k |SHAP| axes.
- Points = a class-agnostic random bank of 350 `X_train` rows; 150 eval points; kNN in standardized input space; SHAP cached once with **`l1_reg=0`** (see §6). Binary targets only (class-1-only SHAP is invalid for multiclass).

## 3. Correctness anchor (Exp 1, margin space) — pipeline is exact
`w_single` recovers the closed-form linear weights on LR/NB across adult/breast/mushroom: **cos(w_single, w_true) = 1.0000, max|err| 1e-14 … 1e-19.** The reconstruction is exact when the SHAP output space is linear (margin/log-odds).

## 4. Recovery + redundancy (Exp 2, probability space)
`ws`=single, `wb200`=batch@200, `g`=true normal, `p`=attack direction.

| combo | cos(ws,g) | cos(wb200,g) | cos(g,p) | verdict |
|---|---|---|---|---|
| adult/mlp | 0.967 | **0.999** | 0.018 | ✅ recovers normal; new direction |
| adult/lr  | 0.848 | **0.999** | 0.066 | ✅ recovers normal; new direction |
| adult/nb  | 0.408 | 0.467 | −0.208 | ⚠️ partial (nonlinear in prob space) |
| adult/knn | 0.064 | 0.027 | −0.194 | ❌ piecewise-constant |
| breast/lr | 0.015 | 0.063 | −0.025 | ❌ prob-space saturation |
| breast/nb | 0.001 | −0.132 | 0.036 | ❌ prob-space saturation |
| breast/mlp| 0.028 | 0.263 | −0.006 | ❌ prob-space saturation (noisy) |
| breast/knn| nan | nan | nan | ❌ g degenerate everywhere |
| mushroom/* | nan | nan | nan | ❌ all-categorical, g undefined |

- **collection beats single** where it works (adult lr/mlp: ~0.85/0.97 → 0.999).
- **not redundant**: wherever `g` is defined, `cos(g,p)≈0` — the attack's direction is ~orthogonal to the true normal.

## 5. Small-batch locality sweep — how many SHAP vectors do you need?
`cos(w_batch@k, g)`, probability space, k down to 5:

| combo | single | k5 | k10 | k20 | k30 | k50 | k100 | k200 |
|---|---|---|---|---|---|---|---|---|
| adult/lr  | 0.85 | 0.96 | 0.99 | 0.99 | 1.00 | 1.00 | 1.00 | 1.00 |
| adult/mlp | 0.97 | 0.99 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| adult/nb  | 0.41 | 0.46 | 0.50 | 0.56 | **0.63** | 0.62 | 0.54 | 0.47 |
| adult/knn | 0.06 | −0.02 | −0.00 | 0.02 | −0.00 | 0.09 | 0.01 | 0.03 |
| breast/mlp| −0.02 | −0.04 | −0.06 | −0.07 | −0.08 | −0.12 | −0.19 | −0.23 |

- **Smooth targets (lr/mlp): a batch of 5–10 already suffices** (k5=0.96–0.99), flat after.
- **Nonlinear target (nb): a sweet spot at k≈20–50** (peak 0.63 @k30). Too few (5–10) = underidentified/noisy OLS; too many (100–200) = pools distinct normals → decays. This is the bias–variance / locality tradeoff made visible.
- **Failures (knn, breast) are not fixed by any k** — the walls are structural, not batch-size.

*(Caveat: at k5–k10 many coords are zeroed by the identifiability guard, so a high cosine means "the identifiable part aligns," not "all coords recovered.")*

## 6. The failures and why (the boundary of applicability)

### breast — cos ≈ 0 in probability space (worse with k)
**Cause: probability-space SHAP through a saturated sigmoid with a distant baseline.**
The identity is exact only for a linear output; `p(x)=σ(margin)` is linear only if the SHAP path from `x0` to `x` stays near margin 0. breast is nearly separable, so it doesn't:

| | adult (works) | breast (fails) |
|---|---|---|
| p(median baseline) | 0.17 (near-linear) | **0.95 (saturated)** |
| median \|margin\| over data | 2.13 | **6.43** (σ≈0.998) |
| frac conf > 0.9 | 0.49 | **0.87** |
| frac near-boundary (<0.7) | 0.22 | **0.05** |

Coalition paths on breast run tail→center→tail across the sigmoid; the probability change is dominated by the nonlinear crossing, whose per-feature credit is *not* ∝ `w` → wrong direction. Because the culprit is the **fixed distant baseline**, even near-boundary eval points fail (confidence-binning didn't rescue it). It gets **worse with k** because the per-neighbor bias is systematic, not noise, so pooling compounds it.
*Not scale:* adult's feature-range spread (104,000×) and breast's (143,000×) are both
extreme — scale is not the differentiator; **saturation is.**
**Fixes:** margin/log-odds SHAP (verified cos=1.0 on breast/LR), or a baseline at p≈0.5.

### KNN — cos ≈ 0, or NaN
**Cause: `predict_proba` is piecewise-constant.** The true normal `g` (a gradient) is
~zero almost everywhere and spikes at cell edges → its *direction* is numerical noise.
`breast/knn = NaN` because `g` is a near-zero vector at essentially every eval point, so
the cosine is undefined. Same structural reason trees fail: no meaningful ∇f.

### mushroom — all NaN
**Cause: all-categorical features.** The finite-difference normal `g` requires continuous
axes; there are none, so `g` (and every `g`-comparison) is undefined. This is a limit of
the *diagnostic*, not necessarily of the idea — but there is no continuous boundary normal
to reconstruct on a purely categorical target.

## 6b. Margin/log-odds SHAP — fixes the breast failure (confirms the mechanism)
Re-running the continuous datasets with `output_space='margin'` (LR: `decision_function`; NB: log-odds; MLP: `logit(p)` pseudo-margin). Driver `_shap_normal_margin.py` → `shap_normal_margin.json`. `cos(w_batch@k, g)`:

| combo | space | single | k5 | k30 | k200 | one-shot |
|---|---|---|---|---|---|---|
| breast/lr | **margin** | **1.00** | 1.00 | 1.00 | 1.00 | 1.000 |
| breast/lr | prob | 0.02 | – | 0.04 | 0.06 | – |
| breast/nb | **margin** | **1.00** | 1.00 | 1.00 | 1.00 | 1.000 |
| breast/nb | prob | 0.00 | – | −0.08 | −0.13 | – |
| breast/mlp | margin | 0.06 | 0.08 | 0.20 | **0.40** | – |
| breast/mlp | prob | 0.03 | – | 0.09 | 0.26 | – |
| adult/nb | **margin** | **0.84** | 0.97 | **1.00** | **1.00** | 1.000 |
| adult/nb | prob | 0.41 | – | 0.63 | 0.47 | – |

- **breast/LR & breast/NB: 0 → perfect 1.00 at every k.** The breast failure was entirely a probability-space saturation artifact; margin SHAP is exact for linear targets. Confirms §6.
- **adult/NB's "partial + sweet-spot" was also a prob-space artifact** — flat 1.00 in margin space (**revises §5**: the k≈20–50 sweet spot is a property of *intrinsic model nonlinearity*, not of NB, which is linear in log-odds).
- **breast/MLP: improved (0.26 → 0.40) but NOT fixed.** Margin removes the *output* sigmoid, but an MLP is *internally* nonlinear (`logit(p)` still isn't linear in x), so a genuinely nonlinear model on a curved boundary stays partial — locality helps, exactness doesn't return.
- `cos(g,p)` stays ≈0 in margin space → the recovered normal is still not redundant with the attack direction, independent of output space.

**Takeaway:** margin/log-odds SHAP is the correct output space and reclaims breast for the linear targets; the remaining wall is *intrinsic model nonlinearity* (MLP), not the sigmoid.

## 6c. Why a high recovery cosine ≠ attack gain (recoverability vs usefulness)
A high `cos(w_single, g)` (adult/mlp 0.97, adult/lr 0.85) does **not** imply the attack
improves. The cosine measures whether the normal is *recoverable*, not whether it is *used*
or *useful* — three different things:

1. **The attack never steps along the normal.** Autolycus/SHAP3 uses SHAP only for top-k
   feature *selection* + fixed ±ε axis perturbation; it never moves along `w`. Proof: `cos(g,p)≈0`
   — the direction the attack actually perturbs along (`p`) is orthogonal to the true normal (`g`).
   So the recovered direction is information the attack *discards*.
2. **Direction isn't the bottleneck; sampling/coverage is** (DEVELOPMENT_NOTES §3.13). Surrogate
   fidelity comes from many well-labelled boundary points, not from knowing the boundary's
   orientation. One perfect direction is an arrow, not a covered manifold of training points.
3. **"High" isn't exact, and it's on a flattering subset.** cos 0.85 ≈ 32° off, 0.97 ≈ 14° off —
   enough to point boundary-ward, but the existing ±ε already wanders there, so the marginal gain
   is small (and angular error compounds over a multi-step walk). Also the single cosine is scored
   only over the ~half of coords that are identifiable (`single_ident_frac≈0.52`, i.e. `x_i≠median`)
   — the easy directions — so it over-states usefulness.

**Meta-point:** this diagnostic answers "can SHAP recover the normal?" (geometry). "Does it help the
attack?" is the separate question "does stepping along it produce better labelled coverage than ±ε?"
A high cosine is a promising *precondition* (the raw material is present), not a result. The test that
would settle it: wire the normal into query synthesis (`x_new = x − η·signed_dist·ŵ`, query, add to
transfer set) on adult/MLP and plot queries-to-fidelity vs the current top-k perturbation.

## 7. Conclusion
**Claimable:** a local collection of SHAP explanations recovers the decision-boundary normal that a single explanation cannot — in the regime {smooth target, continuous features, non-saturated (margin) output space, local batch ~10–50} — and it is a direction the current attack does not already exploit. Demonstrator: **adult/MLP**.

**Not claimable (yet):** (a) a blanket "SHAP batches find the direction" — it fails on KNN/trees (degenerate ∇f), categorical targets (no ∇f), and saturated probability-space SHAP (breast); (b) that this direction *improves extraction* — only geometric recovery was measured, not surrogate fidelity. Open next step: normal-step query synthesis on adult/MLP, measured on the queries-to-fidelity curve.

## 8. Methodological catches (affect the whole SHAP pipeline)
1. **Default SHAP is L1-sparsified.** `KernelExplainer.shap_values` defaults to
   `l1_reg="auto"`, which zeroes features and breaks the reconstruction (breast/NB stuck at cos≈0.5 until `l1_reg=0`). The shipping attack (`load_explainer`) uses this default.
2. **Output space matters.** `load_explainer` explains `predict_proba[:,1]` (probability, class-1 only). Probability-space SHAP is unusable for direction on saturated targets; margin space is exact. Class-1-only is valid only for binary targets.

## 9. Reproduce
- `python _shap_normal_diag.py`  → `shap_normal_diag.json`, `_shap_normal_diag.log`
- `python _shap_normal_smallk.py` → `shap_normal_smallk.json`
- Note: KernelSHAP sampling isn't fully deterministic even at fixed seed; the *failure-band* rows (breast) wobble run-to-run — raise `shap_nsamples` if reproducibility there matters.
