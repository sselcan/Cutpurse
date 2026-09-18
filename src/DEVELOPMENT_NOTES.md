# Development Notes: Improving Autolycus Method

## Overview

`traverse_explanations_SHAP3` is the primary explanation traversal method in the attack utility that explores a model's decision space by generating diverse samples, computing SHAP explanations, and strategically perturbing important features to discover new samples for explanation attacks.

---

## Core Algorithm: traverse_explanations_SHAP3
### Observation: 
We observed the surrogate model differentiate from the target model under two conditions: 
    1. Original Autolycus method had limited sample diversity. 
    2. It also has inefficient boundary exploration 
   This version aims to enhance both aspects while maintaining query efficiency.
### Purpose
This function simulates an adaptive attack that traverses the explanation landscape of a target model. The difference from the original attack is that we improve the auxiliary dataset. It maintains sample diversity while exploring class boundaries and generating new candidate samples based on SHAP-identified important features.

### Algorithm Flow

1. **Initialization**
   - Takes initial `sample_set` and computes initial predictions
   - Tracks visited samples and predictions across all classes
   - Sets up per-class visit counters (`n_visits`) with lower/upper bounds
   - This part is same as original method, but we will generate more diverse samples in the next step to ensure better coverage of the feature space.

2. **Phase 1: Diverse Sample Generation**
   - Calls one of the diverse sample generation methods (currently `create_manifold_aware_diverse_samples_corrected()`)
   - Generates new samples and picks those with most distinctive features (outliers) compared to existing samples
   - Filters samples with confidence > 0.8 (high prediction certainty)
   - Adds filtered samples to the working pool
   - **Purpose**: Ensures initial exploration covers a wide feature space

3. **Phase 2: SHAP-Informed Sample Generation**
   - Calls one of the SHAP-informed sample generation methods (currently `generate_shap_informed_samples_new()`)
   - Mixes features from samples of different classes based on SHAP importance
   - These samples are likely to be near decision boundaries
   - **Purpose**: Target high-gradient regions where explanations change significantly

4. **Phase 3: Iterative Traversal**
   - While query budget not exceeded and classes not yet sufficiently visited:
     - Pop next sample from queue
     - Compute SHAP explanation for this sample
     - Select top-k features by absolute SHAP values
     - Generate two perturbations: ±epsilon on each feature
     - Apply bounds checking to ensure valid feature values
     - Add valid new samples to queue

5. **Output**
   - Returns: `(visited_samples, preds, query_count)`
   - These are the visited samples, their predictions, and total queries used that will be used to train the surrogate model and evaluate attack success

### Key Parameters
- `n_visits_lb/ub`: Lower/upper bounds on samples to explore per class
- `upper_limit`: Total query budget
- `n_f_e`: Number of features to explore (k in paper)
- `explanation_type`: Which explanation variant to use (not currently enabled in v3)
- `num_exp`: Number of top features in some explanation variants

---

## Diverse Sample Generation Methods

### 1. **create_copula_diverse_samples()** 🚫 ABANDONED

**Location**: Line 815
**Status**: Tested and abandoned — did not generate good results in practice

**Algorithm**:
1. Fit a Gaussian Copula model to training samples using SDV library
2. Generate 100 synthetic samples from the copula
3. Standardize both real and synthetic data
4. Find nearest neighbor distance from each synthetic sample to real data
5. Select top 10 synthetic samples with largest distance (most diverse/outliers)

**Advantages**:
- Respects feature correlations and distributions
- Generates realistic synthetic data
- Samples at the edge of the known distribution (boundary candidates)
- Works well for both categorical and continuous features

**Disadvantages**:
- Requires SDV library dependency
- Computationally more expensive (fitting + sampling)
- May miss modes in distribution if sample set is skewed
- Fixed at generating exactly 10 samples

---

### 2. **create_diverse_samples_hybrid()** ⚠️ NOT USED
**Location**: 
**Status**: Not used, generates gibrish diverse samples that has high confidence for some models and dataset (breastcancer).

**Algorithm**:
1. **Phase 1: Random Search**
   - Generate `pool_size` (default 1000) random candidates
   - Calculate mixed distance metric:
     - Hamming distance for categorical features
     - Squared Euclidean for continuous features
   - Select candidate with maximum minimum distance to existing pool

2. **Phase 2: Local Refinement**
   - Hill climbing refinement for `refinement_steps` iterations
   - Try local mutations on each feature
   - For categorical: try all possible values
   - For continuous: try 10 random values in valid range
   - Accept mutations that improve diversity

**Advantages**:
- Combines global search (random pool) with local optimization
- Handles mixed data types well
- Tunable refinement steps for quality vs. speed trade-off
- More transparent/explainable than copula

**Disadvantages**:
- Slower due to two-phase approach
- Discrete local search may miss better continuous values
- Generate gibrish diverse samples that has high confidence
- Parameter tuning needed (pool_size, refinement_steps)

**Why not used**: Generate gibrish diverse samples that has high confidence for some models and dataset (breastcancer) the similarity sinks.

---

### 3. **create_diverse_samples()** ⚠️ NOT USED (Enumeration-based)
**Location**:   
**Status**: Legacy method for fully categorical datasets

**Algorithm**:
1. Enumerate all possible feature combinations using `itertools.product()`
2. For each desired sample:
   - Calculate Hamming distance from all combinations to existing pool
   - Select combination with maximum minimum distance
   - Add to pool and repeat

**Advantages**:
- Optimal diversity for small enumerable spaces
- Mathematically guaranteed to find max-min diverse samples
- Simple and deterministic

**Disadvantages**:
- Exponential complexity: O(∏claspossibilities) feature combinations
- Only works with purely categorical data
- Fails for continuous features
- Memory prohibitive for features with > 10 unique values each

**When to consider**: Only for very small categorical datasets (e.g., < 4 features with < 10 categories each)

---

### 4. **create_diverse_samples_optimized()** ⚠️ NOT USED
**Location**:
**Status**: Optimized alternative to enumeration

**Algorithm**:
1. For 1000 iterations:
   - Generate `candidate_pool_size` (default 5000) random candidates
   - Calculate distance matrix efficiently using broadcasting:
     - Hamming distance: count feature differences
     - Euclidean distance: calculate norms
   - Find candidate with maximum minimum distance to pool
   - Add to pool

**Advantages**:
- Vectorized/broadcasting for speed
- No enumeration needed (scales to any dataset)
- Can handle mixed categorical/continuous
- Similar quality to enumeration approach

**Disadvantages**:
- Fixed to 1000 iterations regardless of desired samples
- Very high computational cost (1000 × 5000 comparisons)
- Still loses diversity after ~50-100 generated samples
- Why this was replaced by copula (much faster)

**When to consider**: When copula fitting fails (e.g., insufficient samples for fit)

---

### 5. **create_manifold_aware_diverse_samples()** ⚠️ NOT USED (commented out)

**Location**: Line 1042
**Status**: Commented out at line 1528 in traverse_explanations_SHAP3 — superseded by `create_manifold_aware_diverse_samples_corrected`

**Algorithm**:
1. For each desired sample:
   - **Parent Selection**: Randomly select parent from existing pool
   - **Mutation**: Apply mutation with probability `mutation_rate` (0.7)
     - Categorical: random value from 0 to class_possibility
     - Continuous: add Gaussian noise (std=10% of range), clip to bounds
   - **Candidate Pool**: Generate `pool_size` mutated candidates
   - **Diversity Selection**: Max-min selection among candidates
   - Add best candidate to pool

**Advantages**:
- Generates locally-realistic samples (near parents)
- Respects feature ranges and types
- Handles NaN values with sanitization
- Good for manifold-aware exploration

**Disadvantages**:
- Offspring might stay close to parents (limited global exploration)
- May get stuck in local modes
- Mutation parameters need tuning
- Bias toward high-density regions

**When to consider**: When `_corrected`'s plausibility filter is too restrictive for very small datasets

---

### 6. **create_manifold_aware_diverse_samples_corrected()** ✅ LATEST -- USED

**Location**: Line 924
**Status**: Active — called at line 1530 in traverse_explanations_SHAP3

**Algorithm**:

1. Precompute plausibility threshold from nearest-neighbor distances in original data (`plausibility_percentile`=90 by default)
2. For each desired sample:
   - **Crossover Candidates** (50% of pool): pick two random parents, mix features 50/50, apply light mutation (rate × 0.3)
   - **Mutation Candidates** (remaining 50%): pick a random parent, mutate each feature with probability `mutation_rate` (0.3)
     - Categorical: random value from 0 to class_possibility
     - Continuous: Gaussian noise (std=10% of range), clip to bounds
   - **Plausibility Filter**: reject NaN candidates; keep only candidates within threshold distance from original data
   - **Diversity Selection**: Max-min selection among plausible candidates
   - Add best candidate to pool

**Advantages**:

- Crossover preserves inter-feature correlations (unlike pure mutation)
- Plausibility threshold keeps candidates on the data manifold
- Handles NaN values with sanitization

**Disadvantages**:

- More complex than base manifold method
- Plausibility threshold tuned by percentile — may filter too aggressively on small datasets
- Crossover from a small pool can re-use similar parents

**When to consider**: Default choice — best balance of realism and diversity

---

### 7. **create_manifold_aware_diverse_samples_knn()** ⚠️ NOT USED

**Location**: Line 987
**Status**: Available but commented out in traverse_explanations_SHAP3 (line 1175)

**Algorithm**:

1. Same mutation-based parent/offspring approach as `create_manifold_aware_diverse_samples`
2. Key difference: uses `sklearn.neighbors.NearestNeighbors` with `StandardScaler` instead of a manual distance matrix
3. Scales features before distance computation → better handles mixed feature magnitudes

**Advantages over base manifold method**:

- Properly normalised distances across features with different scales
- sklearn KNN is faster for large pools (ball-tree/kd-tree indexing)

**Disadvantages**:

- Adds sklearn dependency (already present, so low cost)
- Slightly more overhead per call due to scaler fitting

**When to consider**: When feature scales vary widely and diversity selection is drifting toward high-magnitude features

---

### 7. **create_copula_diverse_samples_distance_based()** 🚫 ABANDONED

**Location**: Line 854
**Status**: Abandoned along with the base copula method — copula-based generation produced poor results

**Algorithm**:

1. Fits a Gaussian Copula to existing samples (SDV)
2. Generates `pool_size` (default 100) synthetic samples
3. Selects `num_desired_samples` by max-min distance to existing samples (same selection as optimized method)
4. Combines copula realism with explicit diversity maximization

**Advantages over plain copula**:

- Better diversity: explicitly maximizes distance rather than relying on copula spread
- Same realism guarantees (respects correlations)

**Disadvantages**:

- Still requires SDV; copula fitting cost unchanged
- Distance matrix over 100 candidates is light, but adds a step

**When to consider**: When `create_copula_diverse_samples` produces too many near-duplicate samples

---

### 10. **create_shap_directed_diverse_samples()** ⚠️ NOT USED

**Location**: Line 1106
**Status**: Implemented but never called in any active code path

**Algorithm**:

1. Compute SHAP values for all existing samples
2. Compute distance band [d_lower, d_upper] from nearest-neighbor distances in original data (10th–95th×2 percentile)
3. **Strategy A — SHAP-direction walk**: step from each sample along its signed SHAP vector (and negation) at 5 magnitudes (0.1–1.0); flips important categorical features at larger steps
4. **Strategy B — Single-feature sweep**: for each sample, vary each of its top-k SHAP features across their range (5 α values; all categorical values enumerated)
5. **Distance-band filter**: keep candidates with min distance to originals in [d_lower, d_upper] — too close = redundant, too far = off-manifold
6. **Diversity selection**: max-min distance from filtered candidates

**Advantages**:

- Explores along directions the model is actually sensitive to (SHAP-guided)
- Distance-band filter prevents both gibberish and redundant samples
- Two complementary strategies cover global direction and local feature sensitivity

**Disadvantages**:

- Requires SHAP computation upfront
- Strategy B generates O(n_samples × top_k × alphas) candidates — can be large
- Band thresholds based on percentiles — sensitive to skewed data

**When to consider**: When probing model-sensitive directions rather than random manifold exploration

---

## Boundary Sample Finding Methods

### 1. **find_boundary_point()** ⚠️ IMPLEMENTED BUT COMMENTED OUT

**Location**: Line 1063
**Status**: Available but disabled in traverse_explanations_SHAP3

**Algorithm** (Binary Search):
1. Given two points `x_a` (class A) and `x_b` (class B)
2. Use binary search on interpolation parameter λ ∈ [0, 1]
3. Each iteration: compute `x_mid = x_a + λ(x_b - x_a)`
4. Classify `x_mid`:
   - If class(x_mid) == class_a: move toward B (low = λ)
   - If class(x_mid) == class_b: move toward A (high = λ)
5. Continue for fixed iterations (detailed parameter), return refined boundary point

**Why Commented Out**:
```python
# for i in range(len(classes)):
#     # generate target-model-confident samples near the decision boundary...
#     # TODO: increase query also in find_boundary_point
```

**Advantages if enabled**:
- Mathematically grounded (binary search minimizes distance to decision boundary)
- Generates exact boundary-crossing points
- Useful for adversarial XAI attacks (boundary = highest explanation gradient)

**Disadvantages**:
- Query cost: Each call costs ~10 queries (one per binary search iteration)
- Combinatorial: For N samples of class A and M of other classes, costs O(N×M) queries
- May not find actual boundary if model is non-smooth
- Requires pairs from different classes

**Why disabled in traverse_explanations_SHAP3**:
- **Query budget exhaustion**: Binary search method alone could use 10-20% of query budget
- **Redundant with SHAP perturbations**: Current method already explores boundaries via feature perturbations
- **Computational complexity**: With 50+ samples, boundary search becomes O(2500+) comparisons
- **Note in code**: "todo: increase query also in find_boundary_point" suggests integration issue

**Potential Re-enablement**:
- Could be enabled selectively for high-value sample pairs (e.g., farthest pairs)
- Could use informed binary search (fewer iterations, e.g., 3-5 instead of 10)
- Would need query budget adjustment: `upper_limit` should be increased

---

### 2. **generate_shap_informed_samples()** ⚠️ NOT USED (superseded)

**Location**: Line 1355
**Status**: Superseded by `generate_shap_informed_samples_new` — no longer called in traverse_explanations_SHAP3

**Algorithm**:

1. For each new sample to generate:
   - **Pair Selection**: Pick two random samples from different classes
   - **Feature Importance**: Compute SHAP values for both samples
   - **Top Features**: Identify top-k features by combined importance
   - **Feature Mixing**:
     - Top-k features: randomly pick value from either parent
     - Other features: 50/50 chance for either parent's value
   - Add mixed offspring to sample set

2. Returns list of offspring samples

**Advantages**:

- No additional queries (uses cached SHAP values)
- Explicitly targets decision boundaries (mixing opposite classes)
- Lightweight and fast

**Disadvantages**:

- Not guaranteed to land exactly on boundary
- Random pair selection may repeat pairs — no diversity guarantee
- Uses `predict` instead of `predict_proba` — hard class assignments only
- Superseded by `generate_shap_informed_samples_new`

---

### 3. **generate_shap_informed_samples_new()** ✅ CURRENTLY USED

**Location**: Line 1407
**Status**: Called at line 1541 in traverse_explanations_SHAP3; results ARE added to queue (line 1542: `samples += middle_samples`)

**Algorithm**:

1. Compute `predict_proba` and SHAP values for all existing samples
2. Pre-enumerate all cross-class pairs, shuffle (exhaustive pool)
3. For each new sample:
   - **Pair selection**: pop next pair from pre-shuffled pool (guarantees all pairs used before repeating)
   - **Feature mixing**:
     - Top-k SHAP features: midpoint for continuous, random pick for categorical
     - Other features: random pick for categorical, uniform random between parents for continuous
   - **Adaptive boundary search**: 2 bisection steps between child and s_b to nudge toward decision boundary
   - Add final child to output

**Advantages over `generate_shap_informed_samples`**:

- Pre-enumerated cross-class pairs → diversity guarantee before any pair repeats
- Uses `predict_proba` → softer class assignments
- Handles 1-output vs multi-output SHAP shapes explicitly
- 2-step bisection nudges offspring closer to decision boundary

**Disadvantages**:

- 2 bisection steps cost ~2 model queries per sample
- Midpoint on top-k features may overshoot boundary for non-linear models

**When to consider**: Always prefer over `generate_shap_informed_samples` — strictly better

---

### 4. **shap_guided_counterfactual_flip()** ⚠️ TRIED — used by SHAP4, on par with SHAP3 (see #12)

**Location**: `attack_utils.py` ~line 1550
**Status**: Phase-2 building block of `traverse_explanations_SHAP4` (the counterfactual variant).
Not used by SHAP3. Verdict from the 20-set study (#12): no reliable tree-model gain.

**Algorithm**: greedy SHAP-guided categorical walk from `s_a` toward `s_b`. At each step flips the
single feature with the highest positive SHAP value for the *current* predicted class (the feature
most strongly holding the prediction in place) to `s_b`'s value, monotonically weakening the margin
until the prediction flips. Returns a straddle pair `(pre, post)` differing in exactly one feature —
the axis controlling the local boundary — plus the query count.

**Advantages**: principled for categorical data (each flip is a valid category, not a synthetic
hybrid); ~8–12 queries per pair; the straddle pair localises one axis-parallel boundary.

**Disadvantages / findings**: SHAP identifies the boundary *axis*, not the *threshold* (must search);
the "SHAP ≈ boundary normal" intuition is weak for piecewise-constant trees; and — decisively — a
handful of straddle pairs are **diluted** when fed as training samples (ablating the straddle
injection moved similarity <0.5pp). See #12.

---

### 5. **expand_confident_descendants()** ⚠️ TRIED — used by SHAP4, ~neutral (see #12)

**Location**: `attack_utils.py` ~line 1632
**Status**: Phase-2 building block of `traverse_explanations_SHAP4`, orchestrated together with
`shap_guided_counterfactual_flip` by `generate_counterfactual_confident_samples` (~line 1739).

**Algorithm**: given a straddle pair, ε-walks *away* from the boundary on each side (SHAP-guided:
push features that already support that side's class), rejection-sampling on `predict_proba.max() >=
0.8`, chaining accepted candidates deeper into the class interior. Produces confidently-labelled
descendants whose labels are clear (avoids on-boundary label noise).

**Findings (20-set ablation, #12)**: turning it off changed similarity within noise (helped nursery
DT +0.9pp, hurt mushroom DT/RF ~0.6pp) → **roughly neutral**. Like the straddle pairs, it doesn't
reshape the bulk training distribution, so it can't move a sample-trained surrogate much.

**Key fix that made SHAP4 competitive**: recycle *every* sample queried during Phase 2 (flip walk +
rejected descendant candidates) into the training set (`query_log`), so the ~100-query overhead
becomes labelled training data instead of wasted budget. Baked in as SHAP4's default behaviour.

---

### 6. **densify_boundary_pairs() / traverse_explanations_SHAP4b()** 🧪 NEW — under test

**Location**: `attack_utils.py` (`densify_boundary_pairs` + `traverse_explanations_SHAP4b`, after SHAP4)
**Status**: New variant motivated by the #12 finding that *sprinkling* boundary points is inert
(gap 3): a few straddle pairs are diluted among ~400 training samples, so they never reshape what a
sample-trained surrogate learns.

**Idea (DualCF-style, borrowing TRA's logic without an oracle)**: instead of adding a few boundary
points, make boundary-straddling pairs the **bulk** of the training set. For many cross-class pairs:
1. SHAP-guided flip → single-axis straddle pair `(pre, post)`, identifying the boundary feature `f`.
2. If `f` is continuous: deep-bisect along `f` to localise the threshold; if categorical: use the
   flip's exact category boundary.
3. **Densify**: emit many sibling pairs that differ only in `f` (lo vs hi side) while perturbing a
   couple of *other* features — piling up labelled points on both sides of that one boundary.
4. Recycle the flip-walk queries too. Phase 1 (diverse) and Phase 3 (ε-perturbation) unchanged.

**Why it might beat Autolycus on trees**: a `DecisionTree`'s split is chosen to minimise impurity;
with Autolycus's diffuse ε-samples the split lands somewhere in the margin, but with dense points
bracketing the true threshold the impurity-optimal split *snaps to it*. Axis-aligned splits are
exactly what dense straddling pairs pin down. This is the first variant that changes the *shape* of
the training distribution rather than adding a few points to it.

Densification spends most of the query budget at the boundary (capped at `min(300, 0.7·Q)`),
recycling all of it as training data.

**Result (20-set, mean±std, Q=500, surrogate matched to target):** densification helps specifically
**where there are continuous features to localise a threshold** (the bisection path = TRA's logic):

| dataset | model | baseline | SHAP3 | SHAP4 | SHAP4b | s4b−base |
|---|---|---|---|---|---|---|
| adult | RF | 0.8996±.019 | 0.9012±.021 | 0.9143±.023 | **0.9235±.013** | **+2.4pp** |
| adult | DT | 0.8157±.034 | 0.8316±.025 | 0.8412±.021 | 0.8265±.032 | +1.1pp (but < SHAP3/4) |
| mushroom | DT | 0.9362 | 0.9332 | 0.9368 | **0.9457**±.048 | +1.0pp |
| mushroom | RF | 0.9285 | 0.9123 | 0.9257 | 0.9139 | −1.5pp |
| nursery | DT | 0.9019 | 0.8943 | 0.8858 | 0.8889 | −1.3pp |
| nursery | RF | 0.9031 | 0.9004 | 0.8988 | 0.9024 | ~0 |

- **adult+RF is a real, significant win** (+2.4pp vs baseline, +2.2pp vs SHAP3, ~8 SE, *lower*
  variance) — pinning continuous split thresholds via bisection is where it pays off.
- **Purely categorical data (nursery, mushroom) sees no reliable gain** — no numeric threshold to
  localise; the flip already lands on the category boundary. (mushroom DT wins, RF/nursery don't.)

**Continuous-dataset confirmation (20-set; iris failed — test split too small for n=5/class):**

| dataset | model | baseline | SHAP3 | SHAP4 | SHAP4b | s4b−base | best arm |
|---|---|---|---|---|---|---|---|
| crop | DT | 0.9202 | 0.9218 | 0.9216 | **0.9324**±.028 | +1.2pp | **SHAP4b** |
| crop | RF | 0.9692 | 0.9775 | 0.9669 | 0.9712 | +0.2pp | SHAP3 |
| breast | DT | 0.9341 | 0.9412 | 0.9418 | 0.9371 | +0.3pp | SHAP4 |
| breast | RF | 0.9776 | 0.9806 | 0.9835 | 0.9806 | +0.3pp | SHAP4 |
| adult | RF | 0.8996 | 0.9012 | 0.9143 | **0.9235**±.013 | +2.4pp | **SHAP4b** |

- **Takeaway (final)**: densification's gain is **real but narrow and headroom-dependent**. It never
  underperforms Autolycus on continuous data (6/6 ≥ baseline), but only *materially* where the target
  isn't already saturated: **adult+RF (+2.4pp)** and **crop+DT (+1.2pp, best arm)** are the clear wins;
  near-ceiling targets (breast 0.93–0.98, crop+RF 0.97) show within-noise margins and SHAP3/SHAP4 tie
  or beat it. It is the best arm in only 2 of the continuous combos, and does **not** move categorical
  datasets. Paper framing: a SHAP-only, TRA-style threshold densification that helps continuous-feature
  extraction most where the target has room to improve: a conditional improvement, not a blanket one.

**Ablation (30-set, adult+RF & crop+DT) - two design questions, both resolved:**

- **Drop Phase-1 diverse generation, reallocate its budget to boundary search (`use_diverse=False`)**:
  a clear win - `shap4b_nodiv` is the **best arm in both combos**. crop+DT 0.9195 -> **0.9357** (+1.6pp
  vs default SHAP4b); adult+RF 0.9185 -> 0.9195 (neutral). vs Autolycus baseline **+1.4pp (adult RF) /
  +1.7pp (crop DT)**; vs SHAP3 **+1.8pp / +1.1pp**; *lower* variance. Diffuse diverse sampling was
  **starving the boundary search of query budget**.
- **Scaffold control (Phase 1 + Phase 3, no Phase 2; `densify_n_pairs=0`)**: confirms the gain is the
  densification, **not** shared SHAP3 scaffolding. scaffold ~= baseline (adult+RF 0.8988, crop+DT
  0.9182); adding densification lifts it **+2.0pp (adult+RF)** / **+1.8pp (crop+DT, once diverse is
  dropped)**. On crop+DT the *default* SHAP4b looked scaffold-equal precisely because diverse starved
  densification - Q1 and Q2 are the same story.
- **Best config found: `traverse_explanations_SHAP4b(use_diverse=False)`** (pure boundary
  densification). **Next**: confirm across the full continuous set (the earlier continuous run used the
  budget-starved default, so breast / crop+RF "no gain" may also be starvation) before baking
  `use_diverse=False` in as the default and writing it up.

**Full 6x6 table + budget scaling (FINAL; how_many_sets=10, single unseeded run, Q=500 & Q=1000):**
Tree models (DT/RF) ran `SHAP4b(use_diverse=False)`, others ran SHAP3, all vs Autolycus.

- **Non-tree models are the robust, large win at both budgets** (SHAP3): LR +6..13%, NB +2..6%,
  KNN +1..4%, P up to +12% across datasets. This is the project's real result.
- **Tree models do NOT generalize or scale.** DT/RF densification gains swing by dataset and flip
  sign between budgets: crop+DT +1.95% @Q500 -> -0.12% @Q1000; mushroom+DT +5.62 -> -0.19;
  breast+DT -0.38 -> +3.51. The only tree combo positive at BOTH budgets is **adult+RF
  (+1.9% @Q500, +1.0% @Q1000)** -- and it *shrinks* with budget, the opposite of the scaling
  prediction (denser boundaries with more queries should *widen* the gap).
- **Caveat**: 10 sets, single unseeded run, and the two budgets drew *different* sample sets, so this
  conflates budget with sampling noise -> the scaling check is **inconclusive but unencouraging**, not
  a clean refutation.
- **Honest final conclusion**: boundary densification is at best a **small, inconsistent, non-robust**
  tree effect (adult+RF the lone survivor; even it doesn't scale). The two 30-set ablation wins
  (adult+RF, crop+DT) were real for those settings but do **not** hold across datasets/budgets. Trees
  are near-ceiling and not reliably improvable by this approach. **Recommendation**: frame the paper
  around the robust non-tree gains (LR/NB/KNN/MLP via SHAP3); if trees are pursued, run a *paired*
  fixed-seed budget sweep on adult+RF at 30+ sets to confirm the one surviving effect before claiming it.

---

## Current Workflow in traverse_explanations_SHAP3

> **Last verified against code**: March 2026

```
Input: initial sample_set

    ↓
[Phase 1] create_manifold_aware_diverse_samples_corrected()   ← ACTIVE (line 1530)
    → Crossover + mutation with plausibility filtering
    → Filter with confidence > 0.8
    → Add to sample queue
    → Query cost: ~10 (one per diverse sample)
    → NOTE: copula / knn / base manifold variants available but commented out (lines 1527-1532)

    ↓
[Phase 2] generate_shap_informed_samples_new()      ← ACTIVE, results ADDED (lines 1541-1542)
    → Mix features from cross-class pairs + 2-step bisection toward boundary
    → offspring ARE added to queue (samples += middle_samples)
    → Query cost: ~2 per sample (bisection steps)
    → NOTE: find_boundary_point() block still fully commented out (lines 1545-1561)

    ↓
[Phase 3] Iterative SHAP perturbation loop
    → For each sample in queue:
        - Compute SHAP
        - Find top-k features
        - Generate ±epsilon perturbations (fixed step size)
        - Add valid new samples
    → Continue until query budget or convergence
    → NOTE: explanation_type branching fully commented out (lines 1226-1244)

Output: visited_samples, predictions, total_queries
```

---

## Attack Sample Perturbation: generateNewSamples

### **generateNewSamples()** ✅ USED IN run_attack_auto_v3

**Location**: Line 2929
**Status**: Called at line 3106 in `run_attack_auto_v3` as the iterative sample expansion loop

**Algorithm**:

For each sample in sample_set:

1. Compute SHAP values (local explainer), extract robust 1D importance vector
2. Identify `orig_top` — feature with highest absolute SHAP value
3. Select top-k features (`nfe`) to perturb
4. For each top feature:
   - Categorical: change by +1 mod number of categories
   - Continuous: generate +epsilon and −epsilon variants
5. For each perturbed variant:
   - Recompute SHAP explanation
   - Keep if `new_top != orig_top` (top feature shifted — explanation-boundary crossed)
6. Return kept samples, labels, and total query count

**Key property**: Only retains perturbations that cause the highest-importance feature to change — targets regions where the model's explanation shifts, i.e., near decision boundaries.

**Advantages**:

- Tracks query count accurately (increments per explanation call)
- Handles labelled (with appended class column) and unlabelled sample sets
- Robust SHAP extraction across 1D/2D/3D output shapes

**Disadvantages**:

- Categorical perturbation only tries +1 mod — may miss valid alternatives
- Strict top-feature-change criterion may discard many valid perturbations on smooth models

**When to consider**: Used in `run_attack_auto_v3` pipeline — replaces `traverse_explanations_SHAP3` for that experiment variant

---

## TODO List

### High Priority
- [ ] **Confidence Threshold Tuning**
  - Current: 0.8 threshold hardcoded (line 1018)
  - Make configurable parameter
  - Test sensitivity across datasets
  - Document recommended values by dataset type

### Medium Priority

- [ ] **SHAP Variant Flexibility**
  - Currently uses raw SHAP values
  - Could support: base SHAP, Kernel SHAP, sampling-based variants
  - Parametrize explainer selection
  - Add timing benchmarks

- [ ] **Feature Perturbation Strategy**
  - Lines 1091-1130: Current dual-feature random perturbation
  - Consider alternatives:
    - Multi-feature simultaneous perturbation (current is 2)
    - Adaptive epsilon based on feature importance magnitude
    - Directional perturbations (toward/away from feature means)

### Low Priority (Experimentation)

- [ ] **Manifold Awareness**:
  - [ ] Compare generated samples to real distribution (KL divergence)
  - [ ] Test on low-dimensional datasets (e.g., Iris)
  - [ ] Benchmark `create_manifold_aware_diverse_samples_knn` vs base manifold method on mixed-scale datasets

- [ ] **Adaptive Visit Bounds**
  - Current: fixed `n_visits_lb`, `n_visits_ub`
  - Idea: Adapt based on class prediction frequency
  - Rare classes could have lower bounds automatically

---

## Analysis & Improvement Opportunities

### 1. **Inefficiency in Copula-Based Diverse Sample Generation**

**Current Issue** (Lines 1015-1021):
```python
diverse_samples = create_copula_diverse_samples(...)
while len(diverse_samples) > 0:
    current_diverse = diverse_samples.pop(0)
    query += 1
    if model.predict_proba([current_diverse]).max() > 0.8:
        samples += [current_diverse]
```

**Problems**:
- Generates fixed 10 samples, but filters out ~30-50% with low confidence
- Returns ~5 samples on average
- Inefficient: generates fixed amount regardless of quality distribution
- No feedback loop to regenerate diverse samples if many fail filtering

**Recommended Improvement**:
```python
# Option A: Adaptive generation until N confident samples collected
confident_samples = 0
target_confident_samples = 5
while confident_samples < target_confident_samples:
    new_diverse = create_copula_diverse_samples(samples, ..., num_desired=1)
    if model.predict_proba([new_diverse[0]]).max() > 0.8:
        samples += new_diverse
        confident_samples += 1
    query += 1

# Option B: Reduce filtering threshold adaptively
threshold = 0.8
while len(diverse_samples) > 0 and num_added < 5:
    current_diverse = diverse_samples.pop(0)
    if model.predict_proba([current_diverse]).max() > threshold:
        samples += [current_diverse]
    # After processing all: if < 5 added, reduce threshold and regenerate
```

---

### 3. **Redundant SHAP Computation in SHAP-Informed Sampling**

**Current Issue** (Lines 1023-1024):
```python
middle_samples = generate_shap_informed_samples(model, samples, explainer, ...)
samples += middle_samples
```

**Problem**:
- `generate_shap_informed_samples()` calls `explainer.shap_values(samples_array)` on entire array
- This re-computes SHAP for all existing samples (already computed in main loop!)
- Wasteful computation

**Current Code** (Line 945):
```python
def generate_shap_informed_samples(model, samples, explainer, ...):
    shap_results = explainer.shap_values(samples_array)  # ← Recomputes all!
    # Uses shap_results to pick pairs...
```

**Recommended Fix**:
```python
# Option A: Cache SHAP values and pass to function
shap_cache = {}

# After computing SHAP in main loop:
with warnings.catch_warnings():
    # ... existing SHAP computation ...
    exp = explainer.shap_values(curr)
    shap_cache[tuple(curr)] = exp

# Then pass cache to function:
middle_samples = generate_shap_informed_samples(
    model, samples, explainer, 
    shap_cache=shap_cache,  # Add this parameter
    ...
)

# Option B: Compute once at the start
shap_results_all = explainer.shap_values(np.array(sample_set))
# Pass to both places that need it
```

**Expected Benefit**: 50-70% reduction in explanation computation time

---

### 4. **Binary Search Boundary Method Not Integrated with Query Limit**

**Current Issue** (Lines 1053-1073):
- Code is commented out with note "todo: increase query also in find_boundary_point"
- `find_boundary_point()` doesn't return query count (Line 930 returns only sample)

**Problem**:
- Can't safely enable boundary search without query tracking
- Missing ~10 queries per boundary point causes silent budget overrun

**Recommended Fix**:
```python
# Modify find_boundary_point signature:
def find_boundary_point(target_model, x_a, x_b, iterations=5):  # Reduce to 5
    # ... binary search ...
    return boundary_sample, query_count  # Already does this! ✓
    
# Then in traverse_explanations_SHAP3, re-enable with controls:
if query < upper_limit - 100:  # Reserve budget for boundary search
    for i in range(len(classes)):
        class_i_samples = [s for s, p in zip(samples, preds) if p == i]
        other_samples = [s for s, p in zip(samples, preds) if p != i]
        
        # Limit pairs: only search between extremes (farthest apart)
        if len(class_i_samples) > 2:
            class_i_samples = [class_i_samples[0], class_i_samples[-1]]
        if len(other_samples) > 2:
            other_samples = [other_samples[0], other_samples[-1]]
            
        for s_a in class_i_samples:
            for s_b in other_samples:
                if query > upper_limit - 50:
                    break
                try:
                    boundary_sample, qc = find_boundary_point(model, s_a, s_b, iterations=5)
                    query += qc
                    if model.predict_proba([boundary_sample]).max() > 0.8:
                        samples += [boundary_sample]
                except ValueError:
                    continue
```

---

### 5. **Feature Perturbation Strategy Could Be More Adaptive**

**Current Strategy** (`attack_utils.py:1982-2018`):
- Always selects exactly top-k features by SHAP value magnitude
- Always perturbs by ±epsilon (fixed step size)
- Always generates exactly 2 perturbations (±)
- **Dead knob (verified June 2026):** `mult = random.randint(1, 1)` (line 1988) is always 1,
  despite the comment "apply 1, 2 or 3 epsilons randomly". Changing it to `randint(1, 3)` is the
  cheapest form of the adaptive-magnitude idea below. (Same dead knob in SHAP/SHAP2 at lines 467, 622.)

**Issues**:
- Ineffective for features with naturally large SHAP values (not relative importance)
- Fixed epsilon ignores feature scale/type
- 2 perturbations may miss optimal direction for continuous features

**Recommended Improvements**:

```python
# Adaptive epsilon selection based on feature type and SHAP magnitude
def compute_adaptive_epsilon(feature_idx, current_val, shap_val, 
                            isCat, feature_ranges, shap_mag_percentile):
    if isCat[feature_idx]:
        # For categorical: move to nearby category if possible
        return 1.0  # Move by 1 category value
    else:
        # For continuous: scale by percentile of SHAP magnitude
        range_size = feature_ranges[feature_idx][1] - feature_ranges[feature_idx][0]
        normalized_eps = range_size * 0.05  # 5% of range
        # Scale by SHAP magnitude (higher SHAP = larger perturbation expected)
        return normalized_eps * max(1.0, abs(shap_val) / shap_mag_percentile)

# Directional perturbation: bias toward feature mean
class_feature_means = compute_class_feature_distributions(samples, preds, classes)
optimal_direction = np.sign(class_feature_means[target_class][f_idx] - curr[f_idx])

# More than 2 perturbations for continuous:
if not isCat[sort_index[i]]:
    cpys = [curr.copy()]  # Base
    for direction in [-1, 1]:  # Toward and away
        for magnitude in [0.5, 1.0, 1.5]:
            c = curr.copy()
            c[sort_index[i]] += epsilon * magnitude * direction
            cpys.append(c)
```

---


### 7. **Feature Validity Checking Could Be Optimized**

**Current Check** (Lines 1113-1115):
```python
tmp = (any((cpys[i] == x).all() for x in visited_samples) or
       (any((cpys[i] == x).all() for x in samples)) or
       (cpys[i][sort_index[ind_i]] < 0) or
       (cpys[i][sort_index[ind_i]] >= classPossibilities[sort_index[ind_i]]))
if not tmp:
    samples += [cpys[i]]
```

**Problems**:
- Exact equality check on floats (continuous features) → always fails
- `visited_samples` and `samples` lists: O(n) lookup per check
- Redundant: both lists check if sample already seen

**Recommended Fix**:
```python
# Use set with tolerance for near-duplicates
visited_set = {}  # Map: rounded_sample → True

def is_duplicate(sample, tolerance=1e-6):
    key = tuple(np.round(sample, decimals=6))
    return key in visited_set

def is_valid(sample, sort_index, class_idx, isCat, feature_ranges):
    # Check duplicates (O(1) with hashing)
    if is_duplicate(sample):
        return False
    
    # Check bounds efficiently
    for i, feat_idx in enumerate(sort_index):
        if isCat[feat_idx]:
            if sample[feat_idx] < 0 or sample[feat_idx] >= classPossibilities[feat_idx]:
                return False
        else:
            low, high = feature_ranges[feat_idx]
            if sample[feat_idx] < low or sample[feat_idx] > high:
                return False
    
    return True

# Usage in loop:
for i in range(2):
    if is_valid(cpys[i], sort_index, class_index, isCat, feature_ranges):
        samples += [cpys[i]]
        visited_set[tuple(np.round(cpys[i], decimals=6))] = True
```

---

### 8. **Explanation Type Variants Are Disabled — Intentionally (Abandoned Defense Method)**

**Status**: The explanation type branching (`topk`, `random`, `zero`, `hybrid`) in `traverse_explanations_SHAP3` (lines 1226-1244) is commented out. This is intentional — these variants were part of an abandoned defense-side experiment and are not relevant to the current attack focus. The `explanation_type` parameter is kept in the signature for legacy compatibility but is effectively ignored. No action needed.

---

## Summary Table: Diverse Sample Methods

| Method | Currently Used | Query Cost | Speed | Handles Mixed Data | Quality | When to Use |
|--------|---|---|---|---|---|---|
| **create_manifold_aware_diverse_samples_corrected** | ✅ Yes (line 1530) | ~10 | Fast | ✓ | Good | Default — on-manifold + plausibility filtering |
| **create_manifold_aware_diverse_samples** | ❌ No (line 1528) | ~10 | Fast | ✓ | Fair | If `_corrected` plausibility filter is too strict |
| **create_manifold_aware_diverse_samples_knn** | ❌ No (line 1529) | ~10 | Fast | ✓ | Fair | When feature scales vary widely |
| **create_shap_directed_diverse_samples** | ❌ No (never called) | SHAP + low | Medium | ✓ | Unknown | When probing model-sensitive directions |
| **create_copula_diverse_samples** | 🚫 Abandoned (poor results) | ~20-30 | Very Fast | ✓ | Poor in practice | — |
| **create_copula_diverse_samples_distance_based** | 🚫 Abandoned (poor results) | ~20-30 | Fast | ✓ | Poor in practice | — |
| **create_diverse_samples_hybrid** | ❌ No | ~50-100 | Slow | ✓ | Good | Fallback if manifold fails |
| **create_diverse_samples** | ❌ No | 0 | Ultra-slow | ✗ (categorical) | Optimal | Only pure categorical, < 4 features |
| **create_diverse_samples_optimized** | ❌ No | 0 | Slow | ✓ | Good | When speed > memory matters |

---

## Summary Table: Boundary Methods

| Method | Currently Used | Query Cost | Integration | Quality | When to Use |
|--------|---|---|---|---|---|
| **generate_shap_informed_samples_new** | ✅ Yes (line 1541) | ~2 per sample | Inline | Good | Default — boundary-targeted cross-class mixing |
| **generate_shap_informed_samples** | ❌ No (superseded) | 0 | — | Fair | Use `_new` instead |
| **find_boundary_point** | ❌ Fully commented (lines 1545-1561) | ~10 per pair | Optional | Excellent | When budget permits, < 10% pairs |


---


### 10. **create_manifold_aware_diverse_samples_knn Is Ready to Test**

**Context**: This KNN variant (line 987) was implemented alongside the base manifold method but has never been benchmarked. It normalises distances with `StandardScaler` before selection, which should improve diversity on datasets with mixed-scale features (e.g., credit/income datasets where one feature spans thousands and another spans 0-1).

**Recommended Action**: Run a side-by-side on adult/credit datasets. Switch by uncommenting line 1175 and commenting line 1174. Measure: average pairwise distance of generated samples and downstream surrogate similarity.

---

### 11. **No Ablation Between Diverse Sample Methods in Notebook**

**Current State**: `combined_attack.ipynb` sweeps `top_exp` values and model/dataset combinations, but never swaps the diverse sample method. All results use `create_manifold_aware_diverse_samples` exclusively.

**Recommended Action**: Add a cell that compares `manifold` vs `manifold_knn` by temporarily routing a `diverse_method` parameter into `traverse_explanations_SHAP3`. Copula variants are excluded — they were tested and produced poor results.

---

### 12. **New SHAP-Guided Counterfactual Flip & Confident Descendant Expansion**

> 📄 **Standalone write-up of this whole line of work** (methods, results, conclusions, next steps —
> supervisor-facing): [`SESSION_NOTES_tree_extraction_2026-06.md`](SESSION_NOTES_tree_extraction_2026-06.md)


**Problem Addressed**

The original SHAP3 approach used categorical bisection (random midpoint / `random.choice` on features) to find decision boundaries, which is mathematically invalid for categorical features — "bisection" on random categorical choices is just random crossover, not boundary convergence. Additionally, deliberately placing training samples ON the boundary (where labels are noisy) hurts simple surrogates like LR, even though boundary information is valuable for understanding the target.

## Two New Functions

### `shap_guided_counterfactual_flip(model, explainer, s_a, s_b, model_name, max_flips=None)`

**Location**: `attack_utils.py` line ~1551

- **Core Idea**: Walk greedily from source sample `s_a` toward target `s_b` using SHAP as a boundary-search guide. At each step, flip the feature with the **highest positive SHAP value** for the current class — the feature most strongly holding the prediction where it is. This monotonically weakens the SHAP-attributed margin.
- **Termination**: Guaranteed to flip the prediction in ≤ `num_differing_features` steps, returning two straddle samples: one confidently in the source class, one in the target class, differing in exactly one feature (the one controlling the local boundary).
- **Principled for Categorical Data**: Unlike interpolation, each flip is a **valid category assignment** (no synthetic hybrids), and the SHAP-guided direction is the **axis-aligned boundary normal** in label-encoded space.
- **Overhead**: ~8–12 target-model queries per source-target pair (one per flip + SHAP evaluation).

#### `expand_confident_descendants(model, explainer, pre_boundary, post_boundary, isCat, classPossibilities, epsilon_set, model_name, num_per_side=3, ...)`
**Location**: `attack_utils.py` line ~1638

- **Core Idea**: The boundary pair itself has ambiguous labels. This function uses the boundary pair as *localization*, not training data. For each side of the boundary, it ε-perturbs the straddle sample AWAY from the boundary (guided by SHAP: increase features that already support that class), **rejection-samples by confidence** (`predict_proba.max() >= 0.8`), and chains successful candidates (walk progressively deeper into the class interior).
- **Output**: Confidently-labeled samples whose existence is informed by the boundary, but whose labels are clear (not ambiguous).
- **Model-Agnostic Labeling**: The samples retain the straddle side's class label, not a noisy intermediate label. This solves the "boundary noise" problem for LR/MLP.
- **Overhead**: ~6–30 target-model queries total, highly variable because of rejection sampling. On average ~2–3 successful candidates per side × 2 sides.

**Integration Pattern** (not yet wired into SHAP3, but ready):
```python
# In traverse_explanations_SHAP3, replace the categorical bisection branch with:
all_boundary_pairs = []
for s_a, s_b in cross_class_pairs:
    pre, post, q1 = shap_guided_counterfactual_flip(model, explainer, s_a, s_b, model_name)
    overhead_queries += q1
    confident, labels, q2 = expand_confident_descendants(model, explainer, pre, post, isCat, ...)
    overhead_queries += q2
    samples += confident
    # pre/post can optionally be added for tree-based models (they directly reveal splits)
    # For LR/MLP, skip them and use only confident descendants
```

**Expected Impact**:
- On categorical datasets (mushroom, nursery), should replace invalid "bisection" with principled boundary search.
- On LR specifically, confident descendants should improve surrogate fit versus raw boundary samples.
- On trees (dt/rdf), adding pre/post straddle pairs in addition to confident descendants might yield even stronger results (axis-aligned splits are tree-native).

**Trade-offs**:
- Higher overhead than the current `generate_shap_informed_samples_new` (80 vs. ~30–50 queries per run).
- Requires tuning: `confidence_threshold`, `num_per_side`, `max_attempts_per_side`.
- Only beneficial if the target model is:
  - **Categorical data** (labels, ordinals) where bisection is invalid.
  - **LR/MLP** where boundary noise is problematic (less of an issue for trees).

**Empirical Results (June 2026; 10-set means, Q=500, k=3, surrogate matched to target family):**

Wired as `traverse_explanations_SHAP4` (clone of SHAP3 with Phase 2 swapped for the counterfactual
flip) and exposed via `run_attack_auto_compare(..., run_shap4=True)`. Tested on the four tree combos
(nursery/mushroom × DT/RF). Key flags on SHAP4: `inject_straddle`, `subtract_overhead`, `train_on_all_queried`.

- ❌ **The "tree-native straddle pairs help" hypothesis is refuted.** Ablating the direct straddle
  injection (`inject_straddle` on/off) moved similarity by <0.5pp in every combo — the pre/post
  pairs are *inert*, even with a DT/RF surrogate. (≈10 boundary pairs are negligible against a
  ~400-sample training set.)
- 🔍 **Root cause of underperformance = wasted query budget, not boundary noise.** The counterfactual
  Phase 2 spends ~100–140 of the 500 queries on the flip walk + rejection sampling, but the original
  SHAP4 discarded those samples — so Phase 3 produced a *smaller* training set (~410 vs baseline ~515).
  For these surrogates, training *volume* dominates boundary precision, so SHAP4-as-first-written lost
  to both SHAP3 and the Autolycus baseline.
- ✅ **Fix — `train_on_all_queried` (now the SHAP4 default).** Every sample queried during Phase 2
  (flip walk + rejected descendant candidates) is target-labelled, so it is recycled into the training
  set (size ~410→~515) at the *same* 501-query budget. This rescued SHAP4 from clearly-worse to
  competitive with SHAP3. (A 10-set run looked like it beat SHAP3 everywhere — that did NOT replicate.)
- ⚠️ **20-set mean±std confirmation (the honest result).** s4 / s3 / base:
    - nursery  DT: 0.8856±.015 / 0.8920±.016 / 0.9002±.013  → s4−s3 −0.6pp, s4−base −1.5pp
    - nursery  RF: 0.9064±.008 / 0.8980±.009 / 0.9088±.008  → s4−s3 +0.8pp, s4−base −0.2pp
    - mushroom DT: 0.9357±.036 / 0.9359±.035 / 0.9184±.035  → s4−s3 ~0,     s4−base +1.7pp
    - mushroom RF: 0.9337±.032 / 0.9248±.043 / 0.9296±.041  → s4−s3 +0.9pp, s4−base +0.4pp
  Deltas are **within one σ** in nearly every case → SHAP4 is **on par with SHAP3** (faint hint it
  helps RF, hurts DT), and beats the Autolycus baseline only on mushroom. No reliable tree-model win.
- 📊 **Budget-matched diagnostic (`subtract_overhead=False`)** gives the strongest raw numbers
  (mushroom DT 0.959, RF 0.946) but uses ~600–640 queries (+20–28%) — *not* a fair-budget claim;
  it only confirms the counterfactual samples help when Phase-3 volume is not sacrificed.

**Takeaway (after the 20-set run):** the counterfactual flip is **not a reliable win on tree models** —
at equal budget it is on par with SHAP3 (differences mostly within noise) and beats Autolycus only on
mushroom. The `train_on_all_queried` fix is still worth keeping: it made SHAP4 competitive and is a
clean, reusable idea — *every target query, even intermediate flip/rejection samples, is free labelled
training data* (could also be retro-fitted to SHAP3/baseline). Trees are near a ceiling (~0.90–0.95);
the project's real gains remain on the non-tree models (LR/NB).

**Remaining:**
- [x] Re-confirmed at 20 sets with mean ± std — tempered the 10-set optimism (see above): on par with SHAP3.
- [ ] Decide if SHAP4 earns a paper slot or stays a documented ablation. Current evidence leans **ablation / negative** for trees; if used, frame as "competitive at equal budget" + the free-training-data insight, not a win.
- [x] Wired `traverse_explanations_SHAP4`, ran the ablation, profiled overhead, found + fixed the lost-volume bug (`train_on_all_queried`).

---

## Related Literature

> Verified June 2026. Full BibTeX in `paper/references.bib` (keys in brackets).
> Several entries in the original bib were wrong — see "Citation fixes" below.

### Directly competing / foundational — read these first
- **Autolycus** [`oksuz2024autolycus`] — Oksuz, Halimi, Ayday, *PoPETs 2024* (arXiv:2302.02162).
  The method we extend. Uses LIME/SHAP as importance *rankings* + fixed ±ε perturbations;
  no explicit boundary targeting. Our improvement = treat SHAP as boundary geometry.
- **Explanation Leaks / XaMEA** [`yan2023explanationleaks`] — Yan et al., *Information Sciences 632 (2023)*.
  Closest explanation-guided extraction competitor; model-agnostic, exploits "spatial knowledge"
  from explanations. Should be a baseline/related-work anchor and a similarity comparison.
- **Model Reconstruction from Model Explanations** [`milli2019model`] — Milli et al., *FAccT 2019*.
  Gradient explanations → exact recovery of linear models. Origin of "explanations leak geometry".

### Counterfactual-based extraction — the boundary-pair lineage (relevant to the NEW functions)
- **Model Extraction from Counterfactual Explanations** [`aivodji2020model`] — Aïvodji, Bolot, Gambs, 2020 (arXiv:2009.01884).
  First CF-based extraction. Foundational.
- **DualCF** [`wang2022dualcf`] — Wang, Qian, Miao, *FAccT 2022* (arXiv:2205.06504).
  **Most relevant to our `shap_guided_counterfactual_flip`**: trains on CF + counter-CF *pairs*
  that straddle the boundary to fix the "decision-boundary shift" caused by on-boundary samples.
  This is exactly the pre/post straddle-pair idea + the "boundary noise hurts LR" motivation in #12.
- **From Counterfactuals to Trees (TRA)** [`khouna2025tra`] — Khouna, Ferry, Vidal, *NeurIPS 2025* (arXiv:2502.05325).
  **Most relevant to the TREE problem**: provably-perfect reconstruction of trees/RF/GBM via
  axis-parallel binary search on counterfactuals. Confirms that trees need single-axis straddle
  pairs — our SHAP-guided single-feature flip is the cheap, SHAP-only analogue. Their assumption
  (restricted nearest-CF oracle) is our differentiator: we only need SHAP + predict_proba.
- **Linear Model Extraction via Factual/Counterfactual Queries** [`otto2026linear`] — Otto, Kurtz, den Hertog, Birbil, 2026 (arXiv:2602.09748).
  Query-complexity bounds for linear models; 1 CF query can suffice. Theory anchor.

### Privacy implications / downstream use
- **Survey of Privacy-Preserving Model Explanations** [`nguyen2024survey`] — Nguyen et al., 2024 (arXiv:2404.00673).
  Use to position related work and find countermeasures (defense framing for FaithfulDefense).
- **Amplification of risks by post-hoc explanations** [`quan2022amplification`] — Quan et al., 2022 (arXiv:2206.14004).
- **MEGEX** [`miura2024megex`] — Miura et al., 2024 (arXiv:2107.08909). Data-free extraction using
  gradient explanations to train a generator (DNN/image setting; orthogonal modality).
- **Explanations Leak → MIA** [`ezzeddine2026membership`] — Ezzeddine et al., 2026 (arXiv:2602.03611). CF → membership inference + DP/active-learning defense.
- **LoMime** [`oksuz2026lomime`] — Oksuz, Halimi, Ayday, 2026 (arXiv:2602.18934). Surrogate extraction → label-only MIA. Our fidelity gains feed this downstream.

### SHAP tooling
- **KernelSHAP** [`lundberg2017shap`] and **TreeSHAP** [`lundberg2020treeshap`] — basis for the
  "SHAP ≈ boundary normal" claim in `paper/sections/problem_formulation.tex`.
- **SHAP docs**: https://shap.readthedocs.io/  • **SDV Copula** (abandoned path): https://docs.sdv.dev/sdv/

### Citation fixes applied to `paper/references.bib` (June 2026)
- `luo2023autolycus` → **`oksuz2024autolycus`**: original entry had wrong authors ("Luo, Yi")
  and a wrong title ("...Backdoor Attacks"). arXiv id 2302.02162 was correct.
- `biradar2026linear` → **`otto2026linear`**: wrong author ("Biradar"); real authors are Otto et al.
- `khouna2025tra`: first author was "Abdallah" → corrected to **Awa Khouna**; added Ferry & Vidal + arXiv id.
- `somepaper2026leakage` → **`ezzeddine2026membership`**: was "Anonymous" placeholder; real paper exists.
- `lomime2026` → **`oksuz2026lomime`**: was "Anonymous" placeholder; confirmed = Autolycus authors.
- Added: `aivodji2020model`, `wang2022dualcf`, `yan2023explanationleaks`, `miura2024megex`,
  `quan2022amplification`, `nguyen2024survey`.

**Gap to fill next**: get XaMEA [`yan2023explanationleaks`] and DualCF [`wang2022dualcf`] numbers
as similarity baselines if their code/setup is reproducible on our datasets.

---

**Last Updated**: June 2026 (verified + corrected literature, added Related Literature section;
prior: March 2026 — verified workflow, corrected method statuses, added improvements #9-13,
added SHAP-guided counterfactual flip & confident descendants #12)
**File**: `/Users/sesame/FaithfulDefense/src/DEVELOPMENT_NOTES.md`
