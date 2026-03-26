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
   - Calls one of the diverse sample generation methods (currently `create_copula_diverse_samples()`)
   - Generates new samples and picks those with most distinctive features (outliers) compared to existing samples
   - Filters samples with confidence > 0.8 (high prediction certainty)
   - Adds filtered samples to the working pool
   - **Purpose**: Ensures initial exploration covers a wide feature space

3. **Phase 2: SHAP-Informed Sample Generation**
   - Calls one of the SHAP-informed sample generation methods (currently `generate_shap_informed_samples()`)
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
- [ ] **Boundary Search Integration**: Re-enable `find_boundary_point()` with reduced iterations (3-5)
  - Estimate additional query cost
  - Adjust `upper_limit` accordingly
  - Test on sample datasets
  
- [ ] **Query Counting Consistency**: Verify all query costs are tracked
  - [ ] Boundary search queries are counted
  - [ ] SHAP computation queries are counted
  - Add assertions to validate query_count accuracy

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

### 2. **Missing Query Cost in Diverse Sample Generation**

**Current Issue** (Line 1015):
- `create_copula_diverse_samples()` itself uses queries (copula sampling, model predictions)
- Not counted in the overall query budget
- Only filtering step increments `query` counter

**Impact**:
- Actual query count exceeds reported count
- Could violate `upper_limit` constraint
- Makes results non-reproducible across runs

**Recommended Fix**:
```python
# Modify create_copula_diverse_samples to return query_count
def create_copula_diverse_samples(...):
    query_count = 0
    # ... existing code ...
    # In the filtering loop:
    for sample in synthetic_data_copula:
        predictions = model.predict(sample)  # This is a query
        query_count += len(sample) if batched else 1
    # ...
    return diverse_samples, query_count

# Then in traverse_explanations_SHAP3:
diverse_samples, query_inc = create_copula_diverse_samples(...)
query += query_inc
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

**Current Strategy** (Lines 1091-1130):
- Always selects exactly top-k features by SHAP value magnitude
- Always perturbs by ±epsilon (fixed step size)
- Always generates exactly 2 perturbations (±)

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

### 6. **Class Visit Balance Not Enforced**

**Current Logic** (Lines 1038-1039):
```python
n_visits = np.zeros(len(classes))
isPassed = [n_visits[i] >= n_v_lb[i] for i in range(len(n_v_lb))]
# ...
while ... and not all(isPassed) and not query > upper_limit:
```

**Issue**:
- Once ANY class reaches lower bound, loop continues
- No guarantee all classes are equally visited
- Biased toward discovering explanations of majority/easiest classes

**Impact**:
- Minority classes get few exploration samples
- Their explanations may be underdiscovered
- Attack effectiveness varies by class

**Recommended Fix**:
```python
# Track percentiles instead of binary passed/failed
min_target_visits = min(n_v_lb)  # Minimum across all classes

while query < upper_limit:
    curr_min_visits = min(n_visits)
    
    if all(n_visits >= n_v_ub):
        break  # All classes have upper bound samples
    
    # Prioritize classes below lower bound
    needs_more = [i for i in range(len(classes)) if n_visits[i] < n_v_lb[i]]
    if needs_more:
        # Preferentially sample from minority classes
        samples = biased_sample_selection(samples, preds, needs_more)
    
    # ... continue main loop ...
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

### 9. **Two Bugs in run_attack_auto / run_attack_auto_v2**

**Bug 1 — Escape sequence** (Line 2523):

```python
print(f"\Training a surrogate model...")
```

`\T` is not a recognised escape sequence; should be `\\Training` or a plain string.

**Bug 2 — Parameter order** (Line 2176 in run_attack_auto):

```python
top_exp, X_train,   # ← wrong order
```

Should be `X_train, y_train, explanation_type, num_exp` to match the function signature of `traverse_explanations_SHAP3`.

---

### 10. **create_manifold_aware_diverse_samples_knn Is Ready to Test**

**Context**: This KNN variant (line 987) was implemented alongside the base manifold method but has never been benchmarked. It normalises distances with `StandardScaler` before selection, which should improve diversity on datasets with mixed-scale features (e.g., credit/income datasets where one feature spans thousands and another spans 0-1).

**Recommended Action**: Run a side-by-side on adult/credit datasets. Switch by uncommenting line 1175 and commenting line 1174. Measure: average pairwise distance of generated samples and downstream surrogate similarity.

---

### 11. **No Ablation Between Diverse Sample Methods in Notebook**

**Current State**: `combined_attack.ipynb` sweeps `top_exp` values and model/dataset combinations, but never swaps the diverse sample method. All results use `create_manifold_aware_diverse_samples` exclusively.

**Recommended Action**: Add a cell that compares `manifold` vs `manifold_knn` by temporarily routing a `diverse_method` parameter into `traverse_explanations_SHAP3`. Copula variants are excluded — they were tested and produced poor results.

---

## References

- **Original Paper**: Check implementation in `run_attack_auto_v2()` for experimental setup
- **SHAP Documentation**: https://shap.readthedocs.io/
- **SDV Copula**: https://docs.sdv.dev/sdv/
- **Binary Search for Boundaries**: Inspired by adversarial perturbation literature

---

**Last Updated**: March 2026 (verified workflow, corrected method statuses, added improvements #9-13)
**File**: `/Users/sesame/FaithfulDefense/ssm/DEVELOPMENT_NOTES.md`
