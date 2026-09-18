# Why LIME Beats SHAP/Random in Explanation-Guided Extraction — Mechanism Study

**Date:** 2026-07-16
**Scope:** offensive (model-extraction) — explaining an empirical ablation, no attack code changed.
**One-line result:** Explanations help extraction by steering perturbations *onto the decision
boundary on categorical data*, **not** by recovering boundary geometry. LIME's advantage is
model-dependent: feature *selection* for smooth models, the bin-edge *threshold* for piecewise ones.

---

## 0. The empirical result we are explaining

Ablation (from the attack harness): run the Autolycus-style attack twice — once choosing the features
to perturb by **LIME** importance, once choosing them **at random** — and take the difference in
surrogate fidelity (surrogate-vs-target test agreement). Positive = the LIME explanation helped.

| Dataset | Model | LIME − random fidelity |
|---|---|---|
| Crop | LR | **−0.0042** |
| Adult Income | LR | **+0.0078** |
| Nursery | NB | **+0.0191** |
| Nursery | LR | **+0.0198** |
| Mushroom | LR | **+0.0352** |
| Nursery | KNN | **+0.0442** |

**Pattern:** the gain is concentrated on **categorical data + non-tree models**; it is ~0 on
continuous **crop**. The question: *why*, and *what predicts the size of the gain?*

We ran three read-only diagnostics. None touches the attack path; all reuse the harness loaders.

---

## 1. Diagnostic A — does LIME recover the boundary *geometry* (normal) better than SHAP?

**Idea being tested:** the original hypothesis was "LIME ≈ the local boundary normal (a gradient),
SHAP ≈ global attribution orthogonal to it," which would make LIME's directions more useful.

**Metrics**
- **`g` (true local normal):** the direction the decision boundary faces at a point, computed by
  *central finite differences* of the target's class-1 probability. Only defined for **continuous**
  features (a discrete feature has no gradient) → categorical datasets are N/A.
- **`w_single` (SHAP normal):** boundary normal reconstructed from one SHAP vector, `w_i = s_i /
  (x_i − x0_i)` against the median baseline `x0`.
- **`w_lime` (LIME normal):** LIME's local linear-surrogate coefficients converted to raw feature
  space (i.e. the gradient of LIME's local model).
- **`cos(w, g)`:** cosine similarity between a reconstructed normal and the true normal. **1 = perfect
  recovery, 0 = unrelated.** This is the headline "does the explanation capture local geometry?"
- **`hit`:** fraction of the explanation's top-k features that are in `g`'s top-k (feature-level agreement).

**Results** (prob space, paired eval points; `nan` = all-categorical, `g` undefined)

| combo | cos(SHAP, g) | cos(LIME, g) | note |
|---|---|---|---|
| adult/lr | 0.852 | **0.998** | LR is linear (easy case) |
| adult/nb | 0.423 | 0.485 | both mediocre |
| adult/knn | 0.298 | **0.087** | LIME worse; `g` degenerate (KNN proba is piecewise-constant) |
| adult/mlp | **0.970** | 0.655 | **SHAP wins the nonlinear case** |
| breast/* | ~0.01 | ~0.02 | both fail entirely |
| mushroom/* | nan | nan | categorical → `g` undefined |

**Reading — hypothesis REFUTED.** LIME does **not** recover the boundary normal better than SHAP. In
the one clean nonlinear continuous case (adult/MLP) SHAP is decisively better (0.97 vs 0.66); on breast
neither works; on categorical data the whole notion is undefined. Geometry is **not** the channel.

---

## 2. Diagnostic B — does LIME *select* more locally-influential features?

**Idea being tested:** maybe LIME simply picks better features to perturb than SHAP/random.

**Metrics**
- **True local influence:** ground truth for "which features matter *here*" — for each feature, how
  much perturbing it actually moves the target's prediction at that point (measured by enumeration).
- **`hit_L / hit_S / hit_R`:** top-k selection agreement with true local influence, for
  **L**IME / **S**HAP / **R**andom. Higher = picks the features that really control the local prediction.
- **`capt_L / capt_S`:** *captured influence share* — the true influence of the top-k features the
  method **selected**, divided by the influence of the actual top-k most-influential features.
  **1.0 = selected the optimal k.** A softer, magnitude-weighted version of `hit` (it still gives
  credit for picking a near-miss feature that is almost as influential).
- **`hit_R`:** the random baseline, computed analytically as `k / n_features` (expected overlap of a
  random top-k), not a sampled value.
- **`L−S`:** LIME's hit minus SHAP's hit. **`abl`:** the attack ablation from §0, for comparison.

**Results**

| combo | type | hit_L | hit_S | hit_R | capt_L | capt_S | L−S | abl |
|---|---|---|---|---|---|---|---|---|
| crop/lr | continuous | 0.53 | 0.53 | 0.43 | 0.66 | 0.51 | 0.01 | −0.004 |
| adult/lr | mixed | 0.79 | 0.44 | 0.27 | 0.94 | 0.52 | **0.34** | +0.008 |
| nursery/nb | categorical | 0.76 | 0.58 | 0.38 | 0.95 | 0.70 | 0.18 | +0.019 |
| nursery/lr | categorical | 0.78 | 0.70 | 0.38 | 0.95 | 0.81 | 0.08 | +0.020 |
| mushroom/lr | categorical | 0.65 | 0.46 | 0.14 | 0.77 | 0.58 | 0.19 | +0.035 |
| nursery/knn | categorical | 0.61 | 0.66 | 0.38 | 0.83 | 0.83 | **−0.04** | +0.044 |

**Reading — a real positive, but it does not explain the *magnitude*.**
- **Positive and solid:** LIME beats random at selection in **all 6** combos, and beats SHAP in **4/6**.
  So "LIME selects locally-influential features" is true and backs the ablation qualitatively.
- **But the selection *gap* does not track the attack *gain*.** adult/lr has the biggest LIME−SHAP gap
  (0.34) yet almost no attack effect (+0.008); nursery/knn has the biggest attack effect (+0.044) yet
  LIME selects *worse* than SHAP there (−0.04). And both LIME and SHAP beat random at selection, even
  though SHAP ties random *in the attack*. So picking good features is **not** what separates them.

---

## 3. Diagnostic C — magnitude decomposition (feature channel vs threshold channel)

**Idea being tested:** decompose LIME's actual attack move into its two parts and see which drives the
gain size. The LIME attack does two things at once: (i) it *chooses* the top-k features, and (ii) it
*snaps* each to its discretization **bin edge** before stepping ±ε.

**Metric — boundary-crossing rate.** Fraction of generated perturbations whose predicted class differs
from the source point — i.e. the sample lands on the **other side of the boundary**, which is a
boundary-informative training example. We measure three variants and difference them:
- **`bin`** = LIME features, snapped to the bin edge ±ε — *the real LIME move*.
- **`limeN`** = LIME features, plain ±ε from the current value — *feature choice only, no threshold*.
- **`rand`** = random features, plain ±ε — *the random ablation baseline*.
- **`total` = bin − rand** (mirrors the LIME-vs-random ablation, in crossing-rate terms).
- **`feat_ch` = limeN − rand** — value of LIME's **feature selection**.
- **`thr_ch` = bin − limeN** — value of LIME's **bin-edge threshold**.

**Results**

| combo | type | bin | limeN | rand | total | feat_ch | thr_ch | abl |
|---|---|---|---|---|---|---|---|---|
| crop/lr | continuous | 0.110 | 0.000 | 0.000 | +0.110 | +0.000 | **+0.110** | −0.004 |
| adult/lr | mixed | 0.149 | 0.036 | 0.030 | +0.119 | +0.006 | +0.112 | +0.008 |
| nursery/nb | categorical | 0.410 | 0.330 | 0.152 | +0.258 | +0.178 | +0.080 | +0.019 |
| nursery/lr | categorical | 0.432 | 0.346 | 0.148 | +0.284 | +0.198 | +0.086 | +0.020 |
| mushroom/lr | categorical | 0.416 | 0.443 | 0.062 | +0.354 | **+0.381** | −0.027 | +0.035 |
| nursery/knn | categorical | 0.441 | 0.321 | 0.232 | +0.209 | +0.089 | **+0.120** | +0.044 |

**Correlation with the ablation:** across all 6, `total` r ≈ **+0.66** (carried by `feat_ch` +0.59;
`thr_ch` −0.37). Within categorical only (n=4) all correlations collapse to ≈ 0.

**Reading**
1. **Feature type gates the magnitude.** On categorical data LIME's perturbations cross the boundary
   **3–7× more** than random (e.g. mushroom 0.44 vs 0.06), producing on-manifold, boundary-informative
   samples (±1 = an adjacent valid category). On continuous **crop**, plain ±ε *never* crosses and the
   only "crossings" come from ε=1 bin jumps that land **off-manifold** — so they don't help fidelity,
   and the gain is ~0. This is what the r ≈ 0.66 really is: the categorical-vs-continuous split.
2. **The winning channel is model-dependent** (the genuinely new structural finding):
   - **Smooth models (LR/NB) → feature selection.** mushroom/LR: `feat_ch` +0.38, `thr_ch` ~0/negative
     (the feature is so influential that any step crosses; the bin edge adds nothing).
   - **Piecewise models (KNN) → the bin-edge threshold.** nursery/KNN: `thr_ch` +0.12 > `feat_ch` +0.09
     (you must land in the correct Voronoi cell, and the bin edge tells you where).
3. **The fine ordering *within* categorical is not explained** by crossing rate (corr ≈ 0) and is small
   enough to be seed noise (cf. `SESSION_NOTES_CORRECTION_2026-07.md`'s winner's-curse caution). The one
   unmeasured factor that plausibly drives it is **headroom** (how far the random baseline sits below
   ceiling) — nursery/KNN topping the list fits "KNN is harder to extract → more room."

---

## 4. Conclusion

**Explanations help extraction because they steer perturbations onto the decision boundary — but not by
leaking boundary *geometry*.** The mechanism is discrete/threshold-based, not gradient-based:

- **Geometry (normals) is not the channel** (Diagnostic A): LIME does not recover the boundary normal
  better than SHAP, and normals are undefined exactly where the gain lives (categorical data).
- **Feature selection is real but not the whole story** (Diagnostic B): LIME picks locally-influential
  features better than random/SHAP, yet the selection gap does not predict the gain magnitude.
- **The gain is gated by categorical feature type and split across two channels** (Diagnostic C):
  on categorical data, LIME points the attack at features whose ±1 flip crosses the boundary. For
  **smooth** targets the value is the **feature choice**; for **piecewise** targets (KNN) it is the
  **bin-edge threshold** LIME hands over for free.
- **Why SHAP underperforms:** SHAP is *magnitude-only* attribution. It identifies influential features
  moderately well but provides **no threshold**, and — decisively for piecewise models — it cannot tell
  the attacker *where* the boundary along a feature is. To match LIME, a SHAP-based attack would have to
  *search* for the threshold (bisection), which is exactly the SHAP4b / TRA boundary-densification line.
- **Why continuous data shows ~0:** a fixed ε=1 step on a continuous axis is off-manifold, so
  explanation-guided selection buys nothing.

**Headline for the paper:** *explanation-guided extraction leaks decision **thresholds**, not geometry —
LIME's discretization gives them for free (most valuable on categorical + piecewise models); SHAP,
being thresholdless, must reconstruct them by search.*

---

## 5. Caveats & next steps

- **Small design.** 6 combos; crossing rate is a *proxy* for fidelity, not fidelity itself.
- **Within-categorical ordering is open.** Needs the definitive experiment: a **paired, mean-aggregated
  (not max — see the CORRECTION note) LIME-vs-random *fidelity* re-run** that logs the random baseline,
  so gain can be regressed on headroom + crossing-total.
- **Untested idea (raised 2026-07-16):** does **averaging same-class SHAP** (global class-average, or a
  local same-class neighborhood average) give SHAP a cleaner signal? Prior: it should denoise the
  *direction* (neighborhood-averaged SHAP already recovers the normal: 0.85→0.999) but still yields **no
  threshold**, so it is unlikely to close the LIME gap on categorical data. Cheap to check.

---

## 6. Reproduction

All functions appended to `attack_utils.py`; drivers write JSON + `.log` alongside.

| Diagnostic | Function | Driver | Output |
|---|---|---|---|
| A — geometry | `run_shap_normal_diagnostic(..., include_lime=True)` | `_lime_normal_diag.py` | `lime_normal_diag.json` |
| B — selection | `run_lime_shap_selection_diagnostic` | `_lime_selection_diag.py` | `lime_selection_diag.json` |
| C — threshold decomposition | `run_lime_threshold_diagnostic` | `_lime_threshold_diag.py` | `lime_threshold_diag.json` |
