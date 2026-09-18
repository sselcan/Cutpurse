# Boundary-Targeted Methods for Tree-Based Model Extraction — Session Report (June 2026)

> ⚠️ **CORRECTION (2026-07-03): the tree-target numbers below are affected by a test-set
> model-selection leak** (surrogate refits were selected by `max` similarity on the eval set, then that
> max was reported — biased most for dt reps=100 / rdf reps=10). After fixing it (aggregate refits by
> `mean`), most tree "wins" do **not** survive: densification ≈ Autolycus on average and trails SHAP3;
> only **adult+RF** holds up (robust across Q=100→1000 and 4 independent seed draws). crop+DT, breast+RF,
> mushroom+RF, breast+DT all dissolve or reverse. Non-tree results (reps=1) are unaffected.
> **See `SESSION_NOTES_CORRECTION_2026-07.md` for the corrected tables and verdict.**

## 1. Goal & context

The SHAP-guided extraction attack (built on Autolycus) already improves surrogate fidelity substantially for the **non-tree** interpretable models (LR, NB, KNN, perceptron). The **tree-based targets (Decision Tree, Random Forest) showed little to no gain** over the Autolycus baseline. This session designed, implemented, and empirically tested several **boundary-targeted** Phase-2 strategies aimed specifically at closing that gap for tree models, inspired by the counterfactual / threshold-reconstruction logic of TRA (Khouna et al., *From Counterfactuals to Trees*, NeurIPS 2025).

**Evaluation protocol (held fixed throughout):**
- Attack budget `Q` (queries), `n = 5` seed samples per class, `k = 3` features explored.
- **Surrogate matches the target family**: DT target → DT surrogate (`max_depth=15`), RF → RF, LR → LR, NB → MultinomialNB, KNN → KNN, P → MLP ensemble. The surrogate selection (`repetition`, "best similarity over re-fits") mirrors `_build_surrogate_and_eval` in `attack_utils.py`. 
- Metric: **prediction agreement** (`rtest_sim`) between surrogate and target on a held-out test set.
- Each result is a **mean over N independently generated seed sets** (N = 10/20/30 as noted), with the *same* seed sets shared across all compared methods within a run (paired comparison).

All methods are implemented in `src/attack_utils.py`; experiment drivers and raw outputs (`*.log`, `*.json`) are kept in `src/`.

---

## 2. Methods implemented (technical detail)

The attack traverses the target's decision space in three phases. 
**Phase 1** generates diverse, confidently-classified seed samples. 
**Phase 2** is the boundary-targeting step (the part varied in this session).
 **Phase 3** is the iterative ε-perturbation traversal inherited from the original method. 
 The variants below differ **only in Phase 2**, so any difference vs the SHAP3 control isolates the Phase-2 strategy.

### 2.1 `shap_guided_counterfactual_flip(model, explainer, s_a, s_b, model_name)`

A SHAP-guided counterfactual walk from a source sample `s_a` (class A) toward a target sample `s_b` (class B), used to locate a single decision boundary.

- Compute SHAP values of the current sample for its **current predicted class**.
- **Flip the single feature with the highest positive SHAP value** (the feature most strongly holding the prediction in place) to the corresponding value from `s_b`. This monotonically weakens the current class's attributed margin.
- Repeat until the target model's prediction flips.
- **Output**: a *straddle pair* `(pre, post)` — the last sample before the flip (confidently class A) and the first after (confidently class B) — which **differ in exactly one feature**, i.e. the single axis controlling the local boundary.
- **Cost**: ~8–12 target queries per pair. Principled for categorical features (each flip is a valid category, not a synthetic hybrid).

### 2.2 `expand_confident_descendants(model, explainer, pre, post, ...)`

Given a straddle pair, generate **clean-labelled** training samples by walking *away* from the boundary (the boundary pair itself has ambiguous labels).

- For each side, ε-perturb the features that already **support** that side's class (SHAP-guided),  **rejection-sampling** on `predict_proba.max() >= 0.8`, and chain accepted candidates progressively deeper into the class interior.
- **Output**: confidently-labelled samples whose existence is informed by the boundary but whose labels are unambiguous (avoids on-boundary label noise that hurts simple surrogates).

### 2.3 `generate_counterfactual_confident_samples(...)` — orchestrator (used by **SHAP4**)

Per cross-class seed pair: run the flip (2.1), confirm the boundary was crossed, run descendant expansion (2.2). Confident descendants → traversal queue; for tree targets the straddle pair itself is injected directly as training data (known labels from the flip).

### 2.4 The key fix: **recycle all queried samples** (`train_on_all_queried`)

Phase 2 *queries* many samples against the target (every flip step; every rejection-sampled descendant candidate, including rejected ones). Originally these were discarded — only the final "kept" samples entered training. **Every queried sample is target-labelled, hence valid training data.** Recycling them (via a `query_log`) turns the ~100-query Phase-2 overhead into ~100 extra labelled training points at the **same** budget. *This is what rescued the counterfactual variant from clearly-worse to competitive* (Section 3.1). Reusable principle: **every target query — even an intermediate one — is free labelled training data.**

### 2.5 **Boundary densification** — `densify_boundary_pairs` / `traverse_explanations_SHAP4b`

The central new idea. The earlier methods *sprinkle* a few boundary points into training, where they are **diluted** among ~400 samples — a `DecisionTree.fit` induces its own splits from the bulk distribution and barely notices 10 boundary points. Densification instead makes boundary-straddling pairs the **bulk** of the training set, so the surrogate's splits are forced onto the true thresholds.

**How a boundary is densified (per cross-class pair):**

1. **Locate the boundary axis.** Run the SHAP-guided flip (2.1) → straddle pair `(pre, post)` differing in exactly one feature `f`. `f` is the boundary axis; `pre[f]`/`post[f]` bracket it.
2. **Localise the threshold.**
   - *Continuous `f`*: **deep bisection** along `f` (holding all other features at `pre`) — classify the midpoint, move the bracket toward the side that changed class, repeat (6 steps). This pins the split value `f*` to a tight interval `[lo, hi]`. The bisection midpoints (which sit on the boundary) are themselves kept as training samples.
   - *Categorical `f`*: no bisection — the flip already gives the exact category boundary
     (`pre[f]` vs `post[f]`).
3. **Densify both sides.** Emit `n_dense` *sibling* pairs that differ **only in `f`** (`lo` side vs `hi` side) while applying small random perturbations to a couple of **other** features. Each sibling is queried and target-labelled. This piles up many labelled points tightly bracketing the *same* boundary, across different settings of the other features.
4. **Recycle everything queried** (flip walk + bisection midpoints + brackets) into the training set.

**Why this should help a tree surrogate (the mechanism):** a tree split is chosen to minimise impurity. With Autolycus's diffuse ε-perturbed samples the split lands *somewhere in the margin* between the nearest class-A and class-B points — imprecise. With many samples bracketing the true threshold, the impurity-optimal split **snaps to it**. Axis-aligned splits are exactly what dense straddling pairs pin down. This is DualCF's "decision-boundary pairs" idea (Wang et al., FAccT 2022), and a SHAP-only stand-in for TRA's counterfactual-oracle threshold reconstruction.

**Knobs:** `n_pairs` (boundaries probed), `n_dense` (siblings per boundary), bisection depth, and `use_diverse` (Section 3.4). Budget for Phase 2 is capped at `min(400, 0.7·Q)` (or `0.85·Q` when diverse generation is dropped).

### 2.6 Naming summary

| Variant | Phase 2 | Notes |
|---|---|---|
| **Autolycus baseline** (`traverse_explanations_SHAP`) | none (original ε-perturbation only) | reference |
| **SHAP3** (`traverse_explanations_SHAP3`) | cross-class feature mixing + 2-step bisection | the existing "main" method |
| **SHAP4** (`traverse_explanations_SHAP4`) | counterfactual flip + confident descendants + straddle injection + recycle-all-queried | §2.1–2.4 |
| **SHAP4b** (`traverse_explanations_SHAP4b`) | boundary densification | §2.5 |
| **SHAP4b-nodiv** (`SHAP4b(use_diverse=False)`) | densification + Phase-1 budget reallocated to boundary search | best tree config found |

### 2.7 Ablation / variant knobs on `traverse_explanations_SHAP4b`

Flags added to support the experiments in §3.6–§3.10 (all default to the shipping behaviour, so existing
callers are unaffected):

| flag | default | effect |
|---|---|---|
| `use_diverse` | True | Phase-1 diverse generation on/off (off → that budget reallocated to densification) |
| `densify_n_pairs` | 12 | # boundaries probed (0 → scaffold control: Phase 1 + Phase 3 only) |
| `densify_cap_frac` | None | fraction of budget spent on densification (None → 0.7/0.85; lower → more budget to traversal) |
| `use_shap_flip` / `use_shap_traverse` (SHAP4b) | True / True | SHAP vs **random** feature choice in the boundary **flip** vs the Phase-3 **traverse** (each False = random, no SHAP query) — isolates the SHAP contribution per stage (§3.12) |
| `use_shap_gen` / `use_shap_traverse` (SHAP3) | True / True | same split for SHAP3: **boundary generation** (`generate_shap_informed_samples_new` top-k) vs **traverse** (§3.13) |
| `recycle_bisection` (SHAP3) | **True** (now default) | recycle SHAP3's discarded bisection mids into training — free low-Q win (§3.11) |
| `skip_traversal` | False | True → skip the Phase-3 ε-perturbation traversal entirely (densification-only attack) |
| `return_thresholds` / `return_boundary_mask` | False | surface the bisection-recovered thresholds (§3.9) / boundary-sample mask (§3.7) for eval-side use |

---

## 3. Experiments & results

### 3.1 Counterfactual flip (SHAP4) vs SHAP3 vs baseline — categorical tree targets

*Driver:* `_shap4_confirm20.py` → `shap4_confirm20.json` (20-set). The 10-set noise finding, straddle
ablation, budget-matched and all-queried diagnostics came from `_shap4_investigate.py` and
`_shap4_nodesc.py` (removed during code cleanup; results kept in `shap4_investigation.json` /
`shap4_nodesc.json`).

10-set runs *looked* like SHAP4 beat SHAP3 everywhere; **at 20 sets the effect disappeared** — SHAP4 ≈ SHAP3 (within noise). Two diagnostic findings:

- **Straddle pairs are inert.** Ablating the direct straddle injection moved similarity < 0.5pp in every combo — even with a tree surrogate. A handful of boundary points cannot reshape what a sample-trained surrogate learns.
- **The original underperformance was wasted budget**, not boundary noise: Phase 2 spent ~100 of 500
  queries and discarded the samples, shrinking the training set (~410 vs ~515). Fixed by
  recycle-all-queried (§2.4), which restored SHAP4 to parity with SHAP3.

20-set means (Q=500, surrogate matched):

| dataset | model | baseline | SHAP3 | SHAP4 |
|---|---|---|---|---|
| nursery | DT | 0.9002 | 0.8920 | 0.8856 |
| nursery | RF | 0.9088 | 0.8980 | 0.9064 |
| mushroom | DT | 0.9184 | 0.9359 | 0.9357 |
| mushroom | RF | 0.9296 | 0.9248 | 0.9337 |

**Conclusion:** the counterfactual flip is *on par* with SHAP3 on trees — no reliable gain. The
useful artefact is the recycle-all-queried principle.

### 3.2 Boundary densification (SHAP4b) — first head-to-head (20 sets)

*Driver:* `_shap4b_densify.py` → `shap4b_densify.json`.

| dataset | model | baseline | SHAP3 | SHAP4 | SHAP4b |
|---|---|---|---|---|---|
| nursery | DT | 0.9019 | 0.8943 | 0.8858 | 0.8889 |
| nursery | RF | 0.9031 | 0.9004 | 0.8988 | 0.9024 |
| mushroom | DT | 0.9362 | 0.9332 | 0.9368 | **0.9457** |
| mushroom | RF | 0.9285 | 0.9123 | 0.9257 | 0.9139 |
| adult | DT | 0.8157 | 0.8316 | 0.8412 | 0.8265 |
| adult | RF | 0.8996 | 0.9012 | 0.9143 | **0.9235** |

**adult+RF is a clear win** (+2.4pp vs baseline, +2.2pp vs SHAP3, ~8 SE, *lower* variance). The win
appears on a target with **continuous features** (adult), where the bisection step localises a real
numeric threshold — exactly the TRA-style logic. Purely categorical targets (nursery, mushroom) show
no reliable gain (no numeric threshold to pin).

### 3.3 Continuous-dataset confirmation (20 sets)

*Driver:* `_shap4b_continuous.py` → `shap4b_continuous.json`.

| dataset | model | baseline | SHAP3 | SHAP4b | s4b−base |
|---|---|---|---|---|---|
| crop | DT | 0.9202 | 0.9218 | **0.9324** | +1.2pp |
| crop | RF | 0.9692 | 0.9775 | 0.9712 | +0.2pp |
| breast | DT | 0.9341 | 0.9412 | 0.9371 | +0.3pp |
| breast | RF | 0.9776 | 0.9806 | 0.9806 | +0.3pp |

Densification beats baseline on **all** continuous combos, but only *materially* where the target has
headroom: **crop+DT (+1.2pp, best arm)** joins adult+RF as a clear win; near-ceiling targets
(breast 0.93–0.98, crop+RF 0.97) show within-noise margins.

### 3.4 Ablation (30 sets, adult+RF & crop+DT) — two design questions

*Driver:* `_shap4b_ablate.py` → `shap4b_ablate.json`.

| dataset | model | baseline | SHAP3 | SHAP4b | SHAP4b-nodiv | scaffold (no Phase 2) |
|---|---|---|---|---|---|---|
| adult | RF | 0.9058 | 0.9018 | 0.9185 | **0.9195** | 0.8988 |
| crop | DT | 0.9183 | 0.9243 | 0.9195 | **0.9357** | 0.9182 |

- **Q1 — drop Phase-1 diverse generation, reallocate budget to boundary search (`use_diverse=False`).**
  Best arm in both combos. crop+DT 0.9195 → **0.9357** (+1.6pp vs default SHAP4b); adult+RF neutral.
  *Diffuse diverse sampling was starving the boundary search of query budget.*
- **Q2 — is the gain densification or shared scaffolding?** The scaffold control (Phase 1 + Phase 3,  no Phase 2) sits **at baseline** (adult+RF 0.8988, crop+DT 0.9182); adding densification lifts it  **+2.0pp (adult+RF) / +1.8pp (crop+DT)**. The gain is **genuinely the densification**, not the shared parts. (On crop+DT the *default* SHAP4b looked scaffold-equal precisely because diverse was starving densification — Q1 and Q2 are the same story.)

### 3.5 Full 6×6 table + budget scaling (10 sets, Q=500 and Q=1000)

*Driver:* `_final_table.py` → `final_table_Q500.json`, `final_table_Q1000.json`.

Tree models ran `SHAP4b(use_diverse=False)`, others ran SHAP3; all vs Autolycus. **% change vs
baseline.** (iris omitted — its test split is too small to draw 5 samples/class.)

**Q = 500**

| dataset | DT | LR | NB | KNN | RF | P |
|---|---|---|---|---|---|---|
| crop | +1.95 | +9.88 | +3.75 | +2.84 | +0.20 | +0.24 |
| adult | −0.55 | +0.16 | +2.03 | +1.72 | +1.89 | +0.08 |
| breast | −0.38 | +1.36 | +0.13 | +1.40 | −0.48 | +0.48 |
| nursery | −1.42 | +13.19 | +1.88 | +4.06 | −0.96 | +12.08 |
| mushroom | +5.62 | +8.94 | +1.21 | +2.76 | −2.22 | +1.51 |

**Q = 1000**

| dataset | DT | LR | NB | KNN | RF | P |
|---|---|---|---|---|---|---|
| crop | −0.12 | +6.25 | +5.74 | +1.51 | −0.36 | +3.10 |
| adult | −1.59 | +0.93 | +5.48 | +1.32 | +1.02 | +11.79 |
| breast | +3.51 | +0.13 | +0.76 | +1.19 | +0.72 | +0.00 |
| nursery | −0.61 | +11.27 | +3.65 | +3.41 | −0.57 | +13.22 |
| mushroom | −0.19 | +7.88 | −0.77 | +1.50 | −0.39 | +5.84 |

- **Non-tree models are a robust, large win at both budgets** (SHAP3): LR +6..13%, NB +2..6%,
  KNN +1..4%, P up to +12%.
- **Tree models are inconsistent at this 10-set scale** and the gains **do not scale with budget** —
  several flip sign between Q=500 and Q=1000 (crop+DT +1.95→−0.12; mushroom+DT +5.62→−0.19). The only
  tree combo positive at both budgets is **adult+RF (+1.9% → +1.0%)**.
- **Caveat:** 10 sets, single unseeded run, and the two budgets drew *different* sample sets, so this
  conflates budget with sampling noise. The focused 20–30-set runs (§3.2–3.4) are more reliable than
  this breadth sweep for any individual combo.

### 3.6 Query-budget sweep + SHAP4b-nodiv on non-tree models (10 sets, **paired seeds across Q**)

*Driver:* `_query_sweep.py` → `query_sweep.json`.

Seed sets generated once per dataset and reused across all Q (fixes §3.5's unpaired confound).
SHAP4b-nodiv was run on **all** model types. P/MLP excluded for runtime; breast uses the Autolycus low-Q range [50,100,250].

**Non-tree models (LR/NB/KNN): densification is NOT a general win — keep SHAP3.** s4b_nodiv beats SHAP3 on only a minority of non-tree combos. Notable points:
- **mushroom+LR: large, growing win** (s4b vs baseline +2.5pp@Q100, +10.4pp@Q250, +14.0pp@Q500) — oncentrating training data at the boundary helps an LR separator here.
- **Multiclass non-tree breaks it** (crop+KNN -14.6pp@Q100): the KernelExplainer explains class-1 only, so the SHAP-guided flip is mis-guided for >2 classes.
- **breast+NB excluded**: intermittent `MultinomialNB.fit` failure on SHAP4b output (NB out of scope;
  SHAP4b is a tree method). Driver eval should return NaN, not the -1.0 sentinel, on total fit failure.

**Tree models (use_diverse=False, paired seeds): RF improves robustly.** s4b vs Autolycus baseline:

| combo | low Q | mid Q | high Q |
|---|---|---|---|
| **adult+RF** | +1.3pp (100) | +1.0pp (250) | +1.9pp (500) |
| **breast+RF** | +2.4pp (50) | +1.1pp (100) | +3.1pp (250) |
| mushroom+RF | +1.7pp (100) | +1.0pp (250) | -0.4pp (500) |
| crop+RF | ~0 (near-ceiling 0.97) | | |
| DT adult/breast | mostly < baseline | | crop/mushroom DT win at Q=500 |

- **adult+RF and breast+RF beat Autolycus at every budget.** breast+RF wins inside the **paper's low-Q range** (Q=50–250, +2.4..+3.1pp) — a clean, paper-relevant result. breast+RF was "no gain" in §3.3 only because that run used the budget-starved default (use_diverse=True) — confirms starvation.
- **Budget scaling, paired, REVERSES §3.5's pessimism**: wins HOLD or GROW with Q (adult+RF +1.3→+1.9pp; crop+DT -3.7→+1.6pp), they do not shrink. The earlier "shrinks with budget" was the unpaired-seed artifact. **RF benefits more consistently than DT.**

#### Q1 — full results: non-tree models (LR / NB / KNN), similarity (mean over 10 paired sets)

| dataset | model | Q | baseline | SHAP3 | SHAP4b-nodiv | s4b−base |
|---|---|---|---|---|---|---|
| adult | LR | 100 | 0.8650 | 0.7928 | 0.8107 | −0.0543 |
| adult | LR | 250 | 0.8582 | 0.8452 | 0.8338 | −0.0244 |
| adult | LR | 500 | 0.8464 | 0.8670 | 0.8431 | −0.0033 |
| adult | NB | 100 | 0.9460 | 0.9353 | 0.9486 | +0.0026 |
| adult | NB | 250 | 0.9412 | 0.9565 | 0.9551 | +0.0139 |
| adult | NB | 500 | 0.9284 | 0.9451 | 0.9428 | +0.0144 |
| adult | KNN | 100 | 0.8402 | 0.7869 | 0.8321 | −0.0081 |
| adult | KNN | 250 | 0.8493 | 0.8520 | 0.8490 | −0.0003 |
| adult | KNN | 500 | 0.8540 | 0.8604 | 0.8546 | +0.0006 |
| crop | LR | 100 | 0.7345 | 0.7318 | 0.7043 | −0.0302 |
| crop | LR | 250 | 0.7455 | 0.7918 | 0.7373 | −0.0082 |
| crop | LR | 500 | 0.7631 | 0.8075 | 0.7804 | +0.0173 |
| crop | NB | 100 | 0.6388 | 0.6384 | 0.6510 | +0.0122 |
| crop | NB | 250 | 0.7443 | 0.7176 | 0.6820 | −0.0623 |
| crop | NB | 500 | 0.7376 | 0.7714 | 0.7494 | +0.0118 |
| crop | KNN | 100 | 0.6094 | 0.4475 | 0.4635 | −0.1459 |
| crop | KNN | 250 | 0.6490 | 0.6243 | 0.5541 | −0.0949 |
| crop | KNN | 500 | 0.6710 | 0.6808 | 0.6843 | +0.0133 |
| mushroom | LR | 100 | 0.7871 | 0.7751 | 0.8123 | +0.0252 |
| mushroom | LR | 250 | 0.7867 | 0.8444 | **0.8904** | +0.1037 |
| mushroom | LR | 500 | 0.7591 | 0.8319 | **0.8994** | +0.1403 |
| mushroom | NB | 100 | 0.8667 | 0.8736 | 0.8686 | +0.0019 |
| mushroom | NB | 250 | 0.8782 | 0.8865 | 0.8725 | −0.0057 |
| mushroom | NB | 500 | 0.8808 | 0.8845 | 0.8871 | +0.0063 |
| mushroom | KNN | 100 | 0.7878 | 0.7732 | 0.7701 | −0.0177 |
| mushroom | KNN | 250 | 0.8002 | 0.7981 | 0.8229 | +0.0227 |
| mushroom | KNN | 500 | 0.8234 | 0.8133 | 0.8324 | +0.0090 |
| breast | LR | 50 | 0.9553 | 0.8976 | 0.9141 | −0.0412 |
| breast | LR | 100 | 0.9447 | 0.8976 | 0.9176 | −0.0271 |
| breast | LR | 250 | 0.9400 | 0.9141 | 0.9212 | −0.0188 |
| breast | NB | 50 | 0.9506 | 0.9682 | fit-fail | n/a |
| breast | NB | 100 | 0.9141 | 0.9706 | fit-fail | n/a |
| breast | NB | 250 | 0.9141 | 0.9365 | fit-fail | n/a |
| breast | KNN | 50 | 0.9176 | 0.9247 | 0.9141 | −0.0035 |
| breast | KNN | 100 | 0.9318 | 0.9071 | 0.9141 | −0.0177 |
| breast | KNN | 250 | 0.9271 | 0.9188 | 0.9153 | −0.0118 |

*(breast+NB: `MultinomialNB.fit` intermittently rejects SHAP4b output — out of scope, excluded.)*

#### Q2 — full results: tree models (DT / RF), similarity (mean over 10 paired sets)

| dataset | model | Q | baseline | SHAP3 | SHAP4b-nodiv | s4b−base |
|---|---|---|---|---|---|---|
| adult | DT | 100 | 0.8212 | 0.7681 | 0.7912 | −0.0300 |
| adult | DT | 250 | 0.8260 | 0.7927 | 0.7915 | −0.0345 |
| adult | DT | 500 | 0.8336 | 0.8229 | 0.8298 | −0.0038 |
| adult | RF | 100 | 0.8706 | 0.8434 | **0.8839** | +0.0133 |
| adult | RF | 250 | 0.8924 | 0.8758 | **0.9028** | +0.0104 |
| adult | RF | 500 | 0.8970 | 0.9049 | **0.9157** | +0.0187 |
| crop | DT | 100 | 0.8953 | 0.8353 | 0.8588 | −0.0365 |
| crop | DT | 250 | 0.9055 | 0.8910 | 0.9000 | −0.0055 |
| crop | DT | 500 | 0.9157 | 0.9129 | **0.9314** | +0.0157 |
| crop | RF | 100 | 0.9690 | 0.9686 | 0.9686 | −0.0004 |
| crop | RF | 250 | 0.9671 | 0.9784 | 0.9635 | −0.0036 |
| crop | RF | 500 | 0.9714 | 0.9769 | 0.9690 | −0.0024 |
| mushroom | DT | 100 | 0.8966 | 0.8308 | 0.8511 | −0.0455 |
| mushroom | DT | 250 | 0.9124 | 0.9185 | 0.9039 | −0.0085 |
| mushroom | DT | 500 | 0.9137 | 0.9365 | **0.9388** | +0.0251 |
| mushroom | RF | 100 | 0.8873 | 0.8628 | **0.9040** | +0.0167 |
| mushroom | RF | 250 | 0.9077 | 0.9076 | **0.9179** | +0.0102 |
| mushroom | RF | 500 | 0.9437 | 0.9198 | 0.9393 | −0.0044 |
| breast | DT | 50 | 0.9118 | 0.8647 | 0.8965 | −0.0153 |
| breast | DT | 100 | 0.9141 | 0.8647 | 0.8859 | −0.0282 |
| breast | DT | 250 | 0.9188 | 0.9212 | 0.9024 | −0.0164 |
| breast | RF | 50 | 0.9447 | 0.9447 | **0.9682** | +0.0235 |
| breast | RF | 100 | 0.9471 | 0.9412 | **0.9576** | +0.0105 |
| breast | RF | 250 | 0.9529 | 0.9694 | **0.9835** | +0.0306 |

**Takeaway:** boundary densification (`use_diverse=False`) is an **RF/tree** tool — a robust, budget-stable win on adult / breast / mushroom RF — and should **not** replace SHAP3 for non-tree models. This is the strongest, cleanest tree result of the project so far.

### 3.7 Sample-weighting prototype (up-weight densified boundary samples in the surrogate fit)

*Driver:* `_weight_prototype.py` → `weight_prototype.json` (needs `SHAP4b(return_boundary_mask=True)`).

Hypothesis: a single DT dilutes boundary points, so up-weighting them via `.fit(sample_weight=...)`
should help DT. Tested W ∈ {1,2,3,5,10} (W=1 == current SHAP4b-nodiv), reusing each traversal across W.
`bfrac` = fraction of the training set that is densified boundary samples.

| combo | bfrac | W=1 | W=2 | W=3 | W=5 | W=10 | best |
|---|---|---|---|---|---|---|---|
| adult+DT | 0.50 | 0.8240 | 0.8189 | 0.8202 | 0.8264 | 0.8262 | ~W1 (noise) |
| breast+DT | 0.81 | **0.9200** | 0.9165 | 0.9165 | 0.9165 | 0.9165 | W=1 |
| crop+DT | 0.51 | **0.9424** | 0.9220 | 0.9176 | 0.9157 | 0.9122 | W=1 (↑W hurts) |
| mushroom+DT | 0.44 | 0.9169 | 0.9241 | 0.9237 | **0.9259** | 0.9219 | W=5 (+0.9pp) |
| adult+RF | 0.50 | **0.9213** | 0.9213 | 0.9208 | 0.9200 | 0.9190 | W=1 |
| breast+RF | 0.81 | 0.9765 | 0.9812 | 0.9824 | **0.9835** | 0.9824 | W=5 (+0.7pp) |

**Result: sample-weighting does NOT help — densification already un-dilutes by volume.** The densified
boundary samples are **already 44–81% of the training set** (`bfrac`), so they are not diluted;
up-weighting them further starves the class interiors and mostly hurts (crop+DT 0.9424 → 0.9122 as
W→10). The two mild gains (mushroom+DT, breast+RF at W=5) are within noise; there is no consistent
sweet spot. **W=1 (current SHAP4b-nodiv) is the right operating point.**

This sharpens the mechanism: `use_diverse=False` densification works for RF precisely *because* it makes
boundary samples the majority of training — leaving nothing for weighting to fix. **DT's remaining
weakness is therefore NOT dilution**, so the lever for DT is the untried **structural split
reconstruction** (build tree splits directly from SHAP+bisection thresholds rather than `.fit()` on
samples — §5 item 1), not sample weighting.

### 3.8 Margin-weighting WITHOUT densification (spend all budget on traversal, then emphasise boundary)

*Driver:* `_margin_weight.py` → `margin_weight.json`.

Idea (no extra budget): skip densification, spend the whole budget on traversal coverage, then
up-weight near-boundary samples via `w = 1 + λ·(1 − max predict_proba)`. The margin is free — the
traversal already queried `predict_proba` to label each sample. Tested baseline+margin and
SHAP3+margin (λ ∈ {0,3,5,10,20}) vs densification (SHAP4b-nodiv) on tree combos.

| combo | densif (s4b) | base best (λ) | SHAP3 best (λ) | winner |
|---|---|---|---|---|
| adult+DT | 0.8312 | 0.8281 (L20) | **0.8416 (L5)** | SHAP3+margin |
| crop+DT | **0.9435** | 0.9341 (L0) | 0.9314 (L0) | densif |
| mushroom+DT | **0.9406** | 0.9366 (L0) | 0.9178 (L0) | densif |
| breast+DT | 0.9424 | **0.9447 (L0)** | 0.9282 (L0) | baseline (unweighted) |
| adult+RF | **0.9172** | 0.9006 (L0) | 0.9009 (L0) | densif (+1.6pp) |
| crop+RF | **0.9816** | 0.9792 (L3) | 0.9812 (L3) | densif |
| mushroom+RF | **0.9488** | 0.9320 (L0) | 0.9355 (L5) | densif (+1.3pp) |
| breast+RF | **0.9847** | 0.9812 (L0) | 0.9847 (L0) | densif / tie |

**Verdict: margin-weighting does NOT match densification — densification wins 6/8** (clearly on RF:
adult+RF +1.6pp, mushroom+RF +1.3pp). On most combos the best λ is **0** (weighting inert or harmful).
The reason is a hard limit of the idea: **you cannot weight boundary samples that do not exist** — a baseline/SHAP3 traversal has only a few near-boundary points, so emphasising them cannot reproduce densification's *dense* threshold brackets. **This confirms densification's value is the synthetic boundary COVERAGE it generates, not mere reweighting** (a positive for the method's story).
One exception: **adult+DT** (the hardest DT combo), where SHAP3+margin (λ=5) edges densification (0.8416 vs 0.8312) — a small, combo-specific win.

---

### 3.9 Stage 1 — threshold snapping (DT, continuous datasets)

*Driver:* `_snap_stage1.py` → `snap_stage1.json`. Needs `densify_boundary_pairs` returning recovered
thresholds and `SHAP4b(return_thresholds=True)`.

Fit DT on densified samples (full depth/structure), then overwrite each **continuous** split threshold
with the nearest bisection-recovered true threshold `f*`. `snapAll` = always; `snapClose` = only if
within 0.1×feature-range. `snapped/splits` = avg # thresholds changed / total splits.

| data | Q | baseline | plain (densif) | snapAll | snapClose | snapped/splits |
|---|---|---|---|---|---|---|
| adult | 250 | 0.8341 | 0.8080 | 0.7880 | 0.8036 | 7.6/37 |
| adult | 500 | 0.8305 | 0.8214 | 0.7084 | 0.7575 | 25.7/77 |
| adult | 1000 | 0.8297 | 0.8218 | 0.7183 | 0.7788 | 46.9/140 |
| breast | 100 | 0.9647 | 0.9588 | 0.9624 | 0.9612 | 2.7/4.3 |
| breast | 250 | 0.9482 | 0.9753 | 0.9671 | 0.9753 | 4.1/7.1 |
| crop | 250 | 0.8902 | 0.9067 | 0.6424 | 0.8902 | 16.9/24 |
| crop | 500 | 0.9098 | 0.9118 | 0.7404 | 0.9145 | 23.3/33.6 |

**Verdict: snapping does NOT help.** `snapAll` is destructive (crop 0.91→0.64; adult −11pp); `snapClose`
is neutral-to-noise (best: crop Q500 +0.3pp). Why, and what it teaches:

- **No node↔threshold correspondence.** The surrogate has many splits (37–140 on adult, 24–34 on crop),
  the same feature split at many nodes (distinct true thresholds), but densification recovers only ~12
  thresholds. Snapping each node to the *nearest* recovered value collapses distinct splits onto wrong
  thresholds → `snapAll` blows up. `snapClose` only snaps when the fit was already ≈right, so it does
  little; where a big correction is needed there is no nearby recovered threshold to snap to.
- **Split POSITION is not the DT bottleneck.** Correcting thresholds doesn't help → DT's weakness is
  the **number of splits / single-tree variance**, not threshold precision. breast (4–7 splits) is
  already near-ceiling under densification; adult/crop (37–140 splits) are the hard, high-variance ones.
- **Implication for Stage 2 (full reconstruction):** budget yields a *shallow* tree (~15–30 splits),
  but adult/crop need far more, so shallow reconstruction would **underfit exactly the targets it must
  win on**, while breast is already solved. Full reconstruction is therefore unlikely to beat a
  sample-trained DT on these targets — DT's gap is variance/complexity, which an ensemble (RF) absorbs
  but a single reconstructed tree cannot.

---

### 3.10 SHAP-contribution + densification-only ablation (2×2 factorial)

*Driver:* `_shap_ablation.py` → `shap_ablation.json`.

Motivated by the observation that Autolycus's gain comes largely from its *traversal* strategy (with SHAP ≈ random feature choice gives similar results). Two questions, run as one 2×2 on SHAP4b (`use_diverse=False`), DT/RF × {adult, crop, mushroom, breast}, Q=500, 10 paired sets, matched surrogate:

| | traversal ON | traversal OFF (densify-only) |
|---|---|---|
| **SHAP** | `s4b_shap` (ref) | `densonly_shap` |
| **random** | `s4b_rand` | `densonly_rand` |

- **3.1 — does SHAP pull weight?** `shapGain = s4b_shap − s4b_rand` (and `densonly_shap − densonly_rand`).
  ≈0 ⇒ SHAP guidance is *not* the source of the gain — the flip/densify/traversal machinery is.
- **3.2 — can densification replace the traversal?** `traversalGain = s4b_shap − densonly_shap`.
  ≈0 ⇒ SHAP-guided densification carries the attack on its own (a genuinely SHAP-oriented attack with no
  borrowed Autolycus traversal). The `densonly_rand` cell closes the 2×2 so the SHAP×traversal interaction
  is identifiable.

Still queued: **(1)** SHAP4b-full (`use_diverse=True`) on non-tree LR/NB/KNN vs SHAP3 (fair diversity
comparison); **(2)** low-Q budget reallocation (`densify_cap_frac` sweep at Q=100/250 — does spending less
on diversity/boundary and more on traversal recover Autolycus's low-budget edge?).

---

**Results (Q=500, 10 sets, matched surrogate):**

| combo | base | s4b_shap | s4b_rand | densOnly | shapGain (3.1) | travGain (3.2) |
|---|---|---|---|---|---|---|
| adult+DT | 0.8254 | 0.8336 | 0.8137 | 0.7698 | +0.0199 | +0.0638 |
| crop+DT | 0.9090 | 0.9255 | 0.9239 | 0.9016 | +0.0016 | +0.0239 |
| mushroom+DT | 0.9323 | 0.9433 | 0.8725 | 0.8892 | **+0.0708** | +0.0541 |
| breast+DT | 0.9094 | 0.9082 | 0.9047 | 0.9259 | +0.0035 | −0.0177 |
| adult+RF | 0.9049 | 0.9103 | 0.9058 | 0.8985 | +0.0045 | +0.0118 |
| crop+RF | 0.9765 | 0.9675 | 0.9710 | 0.9643 | −0.0035 | +0.0032 |
| mushroom+RF | 0.9250 | 0.9064 | 0.8860 | 0.9195 | +0.0204 | −0.0131 |
| breast+RF | 0.9212 | 0.9318 | 0.9318 | 0.9329 | +0.0000 | −0.0011 |

- **3.1 (does SHAP help?) — dataset-dependent.** SHAP clearly helps on **mushroom** (DT +7.1pp, RF
  +2.0pp) and **adult+DT** (+2.0pp); it is **≈ random** on crop, breast, and most RF. So the SHAP
  guidance earns its keep on the categorical-heavy target but is no better than random feature choice
  elsewhere — partially confirming the Autolycus observation that the *machinery* does most of the work.
- **3.2 (can densification replace the traversal?) — not for DT.** Dropping Phase-3 costs DT +2.4..+6.4pp
  (adult/crop/mushroom). On **RF and breast**, densify-only is tied-or-better, so a densification-only
  attack is viable for those targets.

---

### 3.11 Recycling SHAP3's bisection intermediates (free low-budget training data)

*Driver:* `_recycle_bisect.py` → `recycle_bisect.json`. Needs
`generate_shap_informed_samples_new(mid_log=...)` + `traverse_explanations_SHAP3(recycle_bisection=True)`.

`generate_shap_informed_samples_new` runs an 8-step bisection per middle sample and **discards 7 of the
8 queried mids** — ~80 target-labelled, boundary-concentrated samples that were *paid for* (counted in
overhead) but thrown away. Recycling adds them to training at **zero extra budget** (verified: +80
samples, identical query count). Same principle as SHAP4b's train-on-all-queried. Tested
baseline / SHAP3 / SHAP3+recycle on LR/NB/KNN/DT × adult/mushroom/breast, Q ∈ {100,250,500}, paired seeds.

| Q | mean recGain (rec − SHAP3) | wins |
|---|---|---|
| **100** | **+0.0203** | **12 / 12** |
| 250 | +0.0076 | 6 / 12 |
| 500 | +0.0078 | 7 / 12 |

- **Clear, universal win at low Q** (+2.0pp, *all 12 combos* at Q=100; up to +5.5pp on breast+LR) — the
  recovered samples are a large fraction of a small budget. Directly recovers the low-budget gap where
  Autolycus was winning ("exhausting the budget too early").
- Shrinks/mixes at higher Q (free samples become a small fraction). Notable loss: **mushroom+NB** −3.5pp
  at Q≥250 (MultinomialNB dislikes extra near-boundary points once it already has enough data).
- Standout: **mushroom+LR** recycled-SHAP3 beats Autolycus by **+15.5pp @Q=500** (recycling adds +6.0pp).
- vs Autolycus at low Q: recycling narrows/flips the gap on most combos; still trails on a few
  (adult+LR/KNN, mushroom+DT) → the *fixed diversity front-load* (`num_desired_samples=10`) is the
  remaining low-Q drag (exp 2).

**Recommendation:** make `recycle_bisection` the **default** — it's free, net-positive, and dominant at
low Q (NB the lone high-Q exception).

---

### 3.12 SHAP contribution on NON-tree models — SHAP4b (boundary flip vs traverse)

*Driver:* `_nontree_shap.py` → `nontree_shap.json`. SHAP4b (`use_diverse=True`) 2×2 on
(`use_shap_flip` × `use_shap_traverse`), LR/NB/KNN × adult/mushroom/breast, Q=500, 10 sets.
TT = SHAP both; TF = SHAP flip / random traverse; FT = random flip / SHAP traverse; FF = random both.
`flipGain` = SHAP in boundary generation; `travGain` = SHAP in traverse.

| data | mdl | base | shap3 | TT | TF | FT | FF | flipGain | travGain |
|---|---|---|---|---|---|---|---|---|---|
| adult | LR | 0.8442 | 0.8461 | 0.8620 | 0.8958 | 0.8546 | 0.8910 | +0.006 | −0.035 |
| adult | NB | 0.9360 | 0.9648 | 0.9589 | 0.9727 | 0.9674 | 0.9700 | −0.003 | −0.008 |
| adult | KNN | 0.8587 | 0.8719 | 0.8670 | 0.8610 | 0.8658 | 0.8598 | +0.001 | +0.006 |
| mushroom | LR | 0.7442 | 0.8613 | 0.9035 | 0.9195 | 0.9204 | 0.9526 | −0.025 | −0.024 |
| mushroom | NB | 0.8468 | 0.8631 | 0.8558 | 0.8453 | 0.8885 | 0.8565 | −0.022 | +0.021 |
| mushroom | KNN | 0.7971 | 0.8012 | 0.8127 | 0.7989 | 0.8091 | 0.8204 | −0.009 | +0.001 |
| breast | LR | 0.9506 | 0.9518 | 0.9494 | 0.9482 | 0.9247 | 0.9353 | +0.019 | −0.005 |
| breast | NB | 0.8471 | 0.9141 | 0.8706 | fail | fail | fail | — | — |
| breast | KNN | 0.8871 | 0.9435 | 0.8882 | 0.8812 | 0.8624 | 0.8812 | +0.013 | −0.006 |

- **SHAP in the traverse is inert-to-harmful** (travGain avg ≈ −0.5pp; **adult+LR −3.5pp**, where
  *random* traverse beats SHAP by ~3pp — best cell is `TF` = SHAP-flip + random traverse). Strongly
  confirms the Autolycus observation (SHAP-in-traverse ≈ random) for SHAP4b.
- **SHAP in the boundary flip does not reliably help either** (flipGain mixed: +breast, −mushroom,
  ~0 adult). On the biggest-gain case **mushroom+LR, all-random `FF` is best (0.9526)** — SHAP hurts.
- **On non-tree, the boundary-generation *machinery* (flip + densify) carries the gain, not the SHAP
  feature choice.** Caveats: (i) SHAP4b, not the shipping SHAP3 (see §3.13); (ii) non-tree SHAP is a
  KernelExplainer that only explains class-1, so part of "SHAP ≈ random" may be low SHAP *quality*.

---

### 3.13 SHAP contribution on the SHIPPING non-tree method (SHAP3) — the actionable one

*Driver:* `_nontree_shap3.py` → `nontree_shap3.json`. **Gain attribution** on ALL non-tree datasets
(crop, nursery, adult, mushroom, breast) × LR/NB/KNN, Q=500, 10 sets. Leave-one-out from full SHAP3:
each column = `full − (factor removed)`, so **positive = that factor helps**. `boundary` = the whole
middle-sample method (`n_middle=0`, incl. recycling); `div` = Phase-1 diverse; `genSHAP` / `travSHAP` =
SHAP's feature choice in generation / traverse.

| data | mdl | base | full | vsBase | div | boundary | genSHAP | travSHAP |
|---|---|---|---|---|---|---|---|---|
| crop | LR | 0.7663 | 0.8204 | +0.054 | +0.026 | **+0.037** | −0.002 | −0.033 |
| crop | NB | 0.6576 | 0.7776 | +0.120 | +0.067 | **+0.061** | −0.040 | +0.057 |
| crop | KNN | 0.6435 | 0.6686 | +0.025 | −0.004 | +0.030 | +0.019 | −0.008 |
| nursery | LR | 0.7617 | 0.9123 | +0.151 | −0.001 | **+0.081** | +0.000 | −0.067 |
| nursery | NB | 0.5534 | 0.6386 | +0.085 | +0.011 | **+0.095** | −0.008 | **−0.257** |
| nursery | KNN | 0.5597 | 0.5988 | +0.039 | +0.013 | +0.018 | +0.001 | −0.109 |
| adult | LR | 0.8491 | 0.8656 | +0.017 | −0.001 | −0.004 | −0.003 | −0.047 |
| adult | NB | 0.9152 | 0.9504 | +0.035 | +0.029 | +0.011 | +0.012 | +0.026 |
| adult | KNN | 0.8662 | 0.8706 | +0.004 | −0.000 | +0.008 | −0.007 | +0.011 |
| mushroom | LR | 0.7538 | 0.8318 | +0.078 | +0.010 | **+0.056** | −0.011 | −0.075 |
| mushroom | NB | 0.8718 | 0.8613 | −0.011 | −0.021 | −0.022 | +0.001 | −0.017 |
| mushroom | KNN | 0.7746 | 0.7878 | +0.013 | +0.014 | +0.002 | +0.008 | +0.010 |
| breast | LR | 0.9471 | 0.9153 | −0.032 | −0.011 | −0.008 | +0.008 | +0.002 |
| breast | NB | 0.8176 | 0.9224 | +0.105 | +0.000 | **+0.039** | +0.017 | −0.019 |
| breast | KNN | 0.9106 | 0.9024 | −0.008 | −0.001 | +0.004 | −0.014 | +0.007 |

**Averages (15 combos):** boundary **+2.7pp (13/15 positive)**, diverse +0.9pp, **genSHAP ≈ 0
(−0.1pp)**, **travSHAP −3.5pp**.

- **The boundary/middle-sample method IS the source of the non-tree gain** — +2.7pp avg, robust, *large*
  exactly where the big gains are: crop (+3–6pp), nursery (+2–9.5pp), mushroom+LR (+5.6pp). Removing it
  (`n_middle=0`) kills most of the gain. **Confirms the "gain comes from the boundary method" hypothesis.**
- **SHAP's feature *ranking* contributes essentially nothing** (genSHAP ≈ 0, even −4pp on crop+NB). It is
  the boundary-**sampling mechanism** (cross-class mixing + bisection), not the SHAP importances.
- **SHAP in the traverse is strongly harmful** (−3.5pp avg; **nursery+NB −25.7pp**, nursery+KNN −10.9pp,
  LR broadly −5..−7.5pp). Random traverse is dramatically better.
- **Diverse generation is a minor helper** (+0.9pp; notable only on crop+NB +6.7pp).
- **Thesis:** the non-tree power is a **boundary-sampling attack** (generate cross-class boundary
  samples), not a SHAP-feature-ranking attack — precisely quantifying the Autolycus observation.
- **Big free win:** turn SHAP **off** in the Phase-3 traverse — recovers nursery+NB +25.7pp, nursery+KNN
  +10.9pp, +5..7.5pp on LR — at zero cost. Keep the boundary method + diverse.

---

## 4. What we learned about why tree targets are hard

1. **Boundary information is diluted when used as training samples** (the central obstacle). A
   sample-trained tree induces its own splits from the bulk distribution; a few boundary points don't move it (confirmed: straddle ablation inert). **Densification addresses this directly** — make boundary pairs the *bulk* — and that is exactly where the gains (adult+RF, crop+DT) come from (scaffold control confirms it).
2. **SHAP gives the boundary *axis*, not the *threshold*, for trees.** TreeSHAP magnitude flags which feature the local split uses, but not its value; the threshold must be found by search (bisection). The "SHAP ≈ boundary normal" intuition holds for smooth models but breaks for piecewise-constant trees.
3. **Threshold localisation only helps when there is a numeric threshold** — i.e. **continuous
   features**. This is the clearest pattern in the data: the wins are adult+RF and crop+DT (continuous targets with headroom); purely categorical targets (nursery, mushroom) gain nothing.
4. **Budget must be spent on the boundary, not on diffuse coverage.** Diverse-sample generation *starved* the boundary search; `use_diverse=False` is what converted crop+DT from null to a win.
5. **Headroom matters.** Near-ceiling targets (breast, crop+RF at 0.97–0.98) leave nothing to gain.

---

## 5. Conclusion & the path forward for tree-based extraction

**Status.** Boundary densification (`SHAP4b(use_diverse=False)`) is the first method this session that **beats Autolycus on tree targets with a mechanism we understand and verified** (the scaffold control proves the lift is the densification, not shared scaffolding). The effect is **real on continuous- feature targets with headroom** — **adult+RF (+1–2.4pp) and crop+DT (+1.2–1.6pp)** — but is **not yet consistent across all tree datasets**, and the categorical targets (nursery, mushroom) remain unsolved because there is no numeric threshold to localise.

The session's diagnosis points to concrete, **untried** directions to make tree extraction work
more broadly:

1. **Structural reconstruction instead of surrogate training (highest potential).** The root cause is   gap-3 dilution. TRA avoids it entirely by *reconstructing the tree's split thresholds directly* rather than fitting a surrogate on samples. We can approximate this **without a CF oracle**: SHAP picks the split axis, bisection localises the threshold (both already implemented in `densify_boundary_pairs`), and we **build the surrogate's splits from those thresholds** rather than calling `.fit()`. This sidesteps dilution and is the natural way to get a large tree gain.
2. **Sample weighting / replication (cheap, try first).** If `.fit()` dilutes boundary points,
   **up-weight them**: pass `sample_weight` to the surrogate so the bracketed boundary samples
   dominate the impurity calculation. This un-dilutes densification's output with a one-line change to the surrogate fit, no change to the attack.
3. **A categorical analogue of threshold localisation.** Categorical splits are subset memberships, not thresholds. Densify by holding the boundary feature at each side's category while varying others (partially done), but additionally **enumerate the categorical split** (which values map to which side) to teach the surrogate the exact membership test — the categorical counterpart of bisection.
4. **Push densification harder on continuous trees** where it already works: more `n_pairs`, deeper bisection, and **active targeting of the surrogate's most-uncertain regions** (find where the
   current surrogate disagrees with the target and densify *there*), rather than uniform boundary
   coverage.
5. **Confirm and harden the existing wins.** A **paired, fixed-seed** budget sweep on adult+RF and
   crop+DT at 30+ sets (same seed sets across Q) to cleanly isolate the budget effect that the
   unseeded 10-set sweep could not.

**Immediate next step:** prototype (2) sample-weighting on top of `SHAP4b(use_diverse=False)` for
DT/RF — lowest effort, directly attacks the dilution mechanism — then (1) structural split
reconstruction for the larger gain.

---

*Code: `src/attack_utils.py` (`shap_guided_counterfactual_flip`, `expand_confident_descendants`,
`generate_counterfactual_confident_samples`, `densify_boundary_pairs`, `traverse_explanations_SHAP4`,
`traverse_explanations_SHAP4b`; routing via `run_attack_auto_compare(..., main_variant='best')`).
Drivers & raw outputs: `_shap4_confirm20.py`, `_shap4b_densify.py`, `_shap4b_continuous.py`,
`_shap4b_ablate.py`, `_final_table.py` and their `*.log` / `*.json`. Method catalogue & running log:
`DEVELOPMENT_NOTES.md` (#12, boundary-methods catalogue #4–#6).*
