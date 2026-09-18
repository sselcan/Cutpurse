# Spec: SHAP-Normal Reconstruction Diagnostic (2026-07)

Scoped-down slice of `shap_normal_query_generation_brief.md`. Two **read-only** diagnostics that reuse the existing harness loaders (`load_dataset`, `load_model`, `load_explainer`). **No attack path is touched.** Goal: test whether a SHAP vector (single, or a *collection*) reconstructs the local decision-boundary normal, and whether that normal is *redundant* with the direction the attack already perturbs along.

## Why this and not the full brief
- The full brief's "exact" regime (linear target + margin SHAP) is **degenerate**: one sample recovers `w` exactly (Milli et al. 2019), so LR/NB can only be a *correctness anchor*, not a demonstration that a collection helps.
- The real, untested question (from `DEVELOPMENT_NOTES.md` §3.13: "gain is boundary *sampling*, not SHAP feature-ranking") is whether SHAP used as a **dense geometric normal** — reconstructed from a *neighborhood* — recovers the local boundary where a *single* SHAP fails. That is only testable on **nonlinear** targets (MLP, KNN).
- Excluded per decision: ElasticNet + λ knobs, `eta` schedules, straddle emission, query synthesis, full target sweep.

## Setup facts (verified in code)
- Non-tree explainer = `KernelExplainer(lambda x: predict_proba(x)[:,1], median)`
  (`attack_utils.py:2815`). **Probability space, class-1 only, single median baseline.**
- Class-1-only ⇒ valid only for **binary** targets. Restrict to **adult(2), breast(3),
  mushroom(5)** × **LR(1), NB(2), KNN(3), MLP(5)**.
- Single median baseline `x0` is exactly the fixed reference the identity needs — the
  batched-regression "same baseline" assumption holds by construction.

## Definitions (one eval point `x`, one fixed output space `S`)
- `x0` = median baseline (same as the attack's).
- `s`  = SHAP vector of `x` in space `S`.
- `w_single` = `s_i / (x_i - x0_i)`, with a 0/0 guard: `|x_i-x0_i| < eps` → `w_i=0` + mask.
- `w_batch(k)` = per-feature **plain OLS** slope of `s_i` on `d_i=(x_i-x0_i)` over the
  `k` nearest neighbors (no intercept, no regularization). Reports per-feature
  `d_i`-variance; low-variance / near-zero-denominator coords → `w_i=0` + flagged
  (unidentified coords are **visible, not silently noisy**).
- `g` = true local normal = central finite-difference gradient of `S` at `x`
  (categorical axes → NaN, excluded from `g`-comparisons).
- `p` = attack's perturbation direction = `s` masked to its top-`k` `|s|` features
  (`top_k` = `n_f_e`, default 3). Derived from the **same** `s`, so same space as `w`.

**Output-space rule (mandatory):** `w_hat`, `g`, and `p` must all be computed in the
same `S`. Reason: probability↔margin SHAP differ by a *per-feature, non-uniform* factor
(the sigmoid), which **rotates** the vector — it is not a uniform scale, so it changes
the cosine. Per model: LR/NB support `margin` (exact) and `prob`; MLP `prob` (+logit
pseudo-margin); KNN `prob` only.

## Experiment 1 — one-shot correctness anchor
Targets LR, NB in **margin** space (LR: `LinearExplainer`/`decision_function`, exact;
NB: KernelExplainer on log-odds `logP1-logP0`, linear). Assert `w_single ≈ w_true`
(LR `coef_`; NB `feature_log_prob_[1]-[0]`) and `cos(w_single, w_true) ≈ 1.000` on the
identifiable coords. Validates the whole reconstruction pipeline end-to-end.

## Experiment 2 — cosine redundancy / locality (headline)
Targets LR, NB, KNN, MLP × {adult, breast, mushroom}, fixed `S` per model, `n_eval=150`.
Per eval point compute:
- `cos(w_single, g)`, `cos(w_batch(k), g)`  — does SHAP recover the true normal?
- `cos(w_single, p)`, `cos(w_batch(k), p)`  — **redundancy headline**: is the normal
  the same direction the attack already perturbs along?
- `cos(w_batch(k), w_single)`               — does the collection change the direction?

**k-sweep** `{30, 50, 100, 200}` (clamped to bank size): report `cos(w_hat, g)` vs `k`
to map the locality regime. **Bin by boundary distance** (`|max proba - 0.5|`) to show
the "best near the boundary, degrades in high-confidence" prediction. Log per-feature
`d_i`-variance / fraction-identifiable, and `k / delta / output_space / top_k`.

Reading: `cos(w_batch,g)` high while `cos(w_single,g)` low ⇒ collection-beats-single
holds (chase MLP). `cos(w_hat,p)` high ⇒ SHAP-normal redundant with the shipping attack.
Both low ⇒ SHAP carries no usable boundary direction here (clean negative boundary).

## Hooks / deliverables
- `attack_utils.py` (appended, new functions only — no edits to attack code):
  `reconstruct_normal_single`, `reconstruct_normal_batch`, `finite_diff_normal`,
  `run_shap_normal_diagnostic(which_dataset, which_model, output_space, k_sweep,
   n_eval, top_k, ...)`.
- Driver `_shap_normal_diag.py` → `shap_normal_diag.json` (Exp 1 asserts + Exp 2 table).
- Notes on assumptions kept in this file.

## Guardrails
- Multiclass raises (binary-only). 0/0 and low-`d`-variance guarded & surfaced.
- KNN `predict_proba` is piecewise-constant ⇒ `g` is degenerate; reported, not hidden.
- Mushroom is all-categorical ⇒ `g`-comparisons N/A there (only `cos(w_hat,p)` meaningful).
