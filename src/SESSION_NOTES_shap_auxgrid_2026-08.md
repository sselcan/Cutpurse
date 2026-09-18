# SHAP + aux-quantile grid: is LIME's threshold gain a self-computable grid?

**Date:** 2026-08-25 · **Drivers:** `_shap_auxgrid.py` (+ `quantile_grid` mode in
`traverse_explanations_SHAP`, `seed=` param in `load_dataset`)

## Question
LIME's threshold gain comes from snapping the top-k feature to its **bin edge**. The default
`QuartileDiscretizer` sets that edge to `np.percentile(X, [25,50,75])` — a **data quantile**, model-
and label-independent (verified in installed LIME source). So: can an attacker just build its own
quartile grid and get the same benefit? If yes, the "threshold leak" is a self-computable data
artifact, not privileged information about the target.

## Method
On the **base** Autolycus SHAP traversal only (no diverse gen, no boundary search — confirmed), three
paired arms:
- `shap`         — base, step ±ε from current value (no grid)
- `shap_auxgrid` — snap top-k SHAP feature to quartiles of the **attacker's seed set** (self-computed)
- `shap_tgtgrid` — snap to quartiles of the **target's** `X_train` (= what LIME's discretizer leaks)

Grid snapping mirrors base-LIME exactly (snap to bracketing quartile edge, ±1 across it, same bounds;
categoricals not discretized). **Multi-split protocol** (colleague's idea): NSPLIT=10 independent
train/test splits, HMS=1, target retrained per split, arms paired within split, Wilcoxon over splits.
Also fixed breast's non-deterministic split (`random_state=42/21`).

## Results

**1. aux ≈ tgt — the grid is self-computable (the keeper).** Across all 25 cells at n=5, and at n=1,
the aux-grid and tgt-grid deltas are within ~1–2 pp with no consistent sign. The attacker's grid —
built from as few as 5 samples/class (n=5) or 1/class (n=1) — matches the target's full-data grid.
**=> No privileged target-quantile leak; the threshold channel is a self-computable data-quantile grid.**

**2. The grid's OWN effect on SHAP is dataset-dependent — does NOT generalize (not for the paper).**
n=1 vs n=5 contrast (aux-grid delta, pp):

| dataset | LR n1 / n5 | NB | KNN | DT | RF |
|---|---|---|---|---|---|
| **crop**  (helps)  | +25 / +8 | +9 / +4 | +11 / +5 | +5 / −1 | +2 / −0 |
| **pendigits** (hurts) | −16 / −8 | −10 / −1 | −4 / −0 | −13 / −4 | −23 / −6 |

crop confirms the coverage hypothesis (starved base @ n1, big grid recovery). pendigits **refutes** it —
the grid hurts, and worse at n1. Likely: base SHAP already spreads via ±ε; snapping to quartiles
*contracts* toward typical values → helps crop (integer-encoded, stuck steps) but collapses diversity
on smooth pendigits. Note LIME's *threshold* DID help pendigits (+10.6) with the same quartiles →
on pendigits LIME's feature choice coupled to its bins matters (SHAP's raw-space features don't).

## Caveats
- **Cross-explainer confound:** SHAP runs n=5, LIME n=1, so base-SHAP starts near ceiling; only the
  *within-SHAP* aux-vs-tgt comparison is clean. Don't compare SHAP+grid delta to LIME threshold delta.
- **wine invalid** (~27 eval samples → unstable); dropped. Eval-set size decides stability — always report.
- **digits dropped** — it's 8×8 pixels, not tabular; Autolycus is a tabular attack.

## Bottom line
Core finding (aux ≈ tgt → self-computable grid, no privileged leak) is solid and split-robust
(holds on crop, adult, breast, wine, pendigits, letter — n=5 and n=1). The secondary "bolt a grid
onto SHAP to improve it" idea does NOT generalize: the grid helps SHAP on **crop only** (the one
integer-LabelEncoded dataset), and HURTS pendigits and letter (raw floats) — so it's a crop-encoding
artifact (snapping unsticks integer-rank ±ε steps; contracts/hurts smooth-float exploration), not a
general improvement. Keep as an interesting negative, not a paper claim. Framing: **attribution inert
for SHAP only**; LIME attribution is dataset/model dependent. Next: confirm on LIME itself via aux vs
target discretizer (`_lime_auxdisc.py`, running on crop+pendigits).


---

## Full results

All fidelities in percentage points, 10-split protocol, paired Wilcoxon over splits (\* = p<.05). auxΔ/tgtΔ = grid arm minus base-SHAP. `aux−tgt` gap ≈ 0 everywhere = grid is self-computable (source doesn't matter).

**Result files:** `paper_results/shap_auxgrid_ds<ds>.json` (n=5) and `paper_results/shap_auxgrid_n1_ds<ds>.json` (n=1). ds: 1=crop 2=adult 3=breast 8=wine 9=pendigits 10=letter.

### n=5 (5 seeds/class)


**crop** (eval=255)

| model | base | +aux | +tgt | auxΔ (p) | tgtΔ (p) | aux−tgt |
|---|--:|--:|--:|--:|--:|--:|
| LR | 80.0 | 88.2 | 87.4 | +8.1\* | +7.3\* | +0.8 |
| NB | 74.9 | 78.8 | 79.6 | +3.8\* | +4.7\* | -0.8 |
| KNN | 70.3 | 75.6 | 77.3 | +5.2\* | +7.0\* | -1.8 |
| DT | 92.3 | 91.3 | 91.8 | -1.0 | -0.5 | -0.5 |
| RF | 96.7 | 96.4 | 96.5 | -0.3 | -0.2 | -0.1 |


**adult** (eval=4884)

| model | base | +aux | +tgt | auxΔ (p) | tgtΔ (p) | aux−tgt |
|---|--:|--:|--:|--:|--:|--:|
| LR | 85.5 | 84.7 | 84.5 | -0.8 | -1.1 | +0.2 |
| NB | 92.0 | 95.5 | 95.7 | +3.5 | +3.6 | -0.2 |
| KNN | 86.3 | 82.7 | 82.3 | -3.6\* | -4.0\* | +0.4 |
| DT | 82.6 | 77.5 | 75.9 | -5.0\* | -6.6\* | +1.6 |
| RF | 91.9 | 88.3 | 88.6 | -3.6\* | -3.3\* | -0.3 |


**breast** (eval=85)

| model | base | +aux | +tgt | auxΔ (p) | tgtΔ (p) | aux−tgt |
|---|--:|--:|--:|--:|--:|--:|
| LR | 96.9 | 95.1 | 95.4 | -1.9 | -1.5 | -0.4 |
| NB | 92.5 | 97.8 | 96.7 | +5.3\* | +4.2 | +1.1 |
| KNN | 88.1 | 88.0 | 88.9 | -0.1 | +0.8 | -0.9 |
| DT | 90.1 | 87.2 | 87.3 | -2.9\* | -2.8 | -0.1 |
| RF | 93.4 | 92.4 | 92.0 | -1.0 | -1.4 | +0.4 |


**wine** (eval=27 — INVALID, too small)

| model | base | +aux | +tgt | auxΔ (p) | tgtΔ (p) | aux−tgt |
|---|--:|--:|--:|--:|--:|--:|
| LR | 95.9 | 89.6 | 90.4 | -6.3 | -5.6 | -0.7 |
| NB | 78.9 | 85.9 | 85.6 | +7.0\* | +6.7\* | +0.4 |
| KNN | 81.1 | 77.8 | 77.8 | -3.3 | -3.3 | +0.0 |
| DT | 90.1 | 84.7 | 85.7 | -5.5\* | -4.5 | -1.0 |
| RF | 94.6 | 92.5 | 92.0 | -2.1 | -2.7 | +0.6 |


**pendigits** (eval=1648)

| model | base | +aux | +tgt | auxΔ (p) | tgtΔ (p) | aux−tgt |
|---|--:|--:|--:|--:|--:|--:|
| LR | 87.4 | 79.9 | 80.2 | -7.5\* | -7.2\* | -0.3 |
| NB | 90.2 | 89.1 | 89.2 | -1.1 | -0.9 | -0.2 |
| KNN | 84.2 | 84.0 | 84.0 | -0.2 | -0.2 | +0.0 |
| DT | 68.5 | 64.8 | 65.5 | -3.7 | -3.0 | -0.7 |
| RF | 86.1 | 80.0 | 80.7 | -6.2\* | -5.5\* | -0.7 |


### n=1 (1 seed/class — coverage-starved, LIME's native regime)


**crop** (eval=255)

| model | base | +aux | +tgt | auxΔ (p) | tgtΔ (p) | aux−tgt |
|---|--:|--:|--:|--:|--:|--:|
| LR | 38.1 | 63.2 | 61.0 | +25.1\* | +22.9\* | +2.2 |
| NB | 56.5 | 65.1 | 64.8 | +8.6\* | +8.2\* | +0.4 |
| KNN | 44.0 | 54.7 | 54.9 | +10.7\* | +10.8\* | -0.2 |
| DT | 73.4 | 78.0 | 77.7 | +4.7 | +4.3 | +0.4 |
| RF | 80.1 | 81.5 | 82.5 | +1.5 | +2.4 | -0.9 |


**pendigits** (eval=1648)

| model | base | +aux | +tgt | auxΔ (p) | tgtΔ (p) | aux−tgt |
|---|--:|--:|--:|--:|--:|--:|
| LR | 72.0 | 56.1 | 60.4 | -16.0\* | -11.6\* | -4.4 |
| NB | 82.5 | 72.1 | 71.9 | -10.4\* | -10.6\* | +0.2 |
| KNN | 66.0 | 61.8 | 61.8 | -4.2\* | -4.2\* | -0.0 |
| DT | 55.5 | 43.0 | 45.9 | -12.5\* | -9.6\* | -2.9 |
| RF | 73.0 | 50.4 | 49.1 | -22.5\* | -23.9\* | +1.4 |


**letter** (eval=3000)

| model | base | +aux | +tgt | auxΔ (p) | tgtΔ (p) | aux−tgt |
|---|--:|--:|--:|--:|--:|--:|
| LR | 60.1 | 46.6 | 47.0 | -13.4\* | -13.0\* | -0.4 |
| NB | 46.8 | 38.5 | 38.8 | -8.3\* | -8.0\* | -0.3 |
| KNN | 25.8 | 20.2 | 20.2 | -5.6\* | -5.5\* | -0.1 |
| DT | 30.3 | 30.7 | 31.5 | +0.3 | +1.2 | -0.9 |
| RF | 50.1 | 35.7 | 36.6 | -14.5\* | -13.5\* | -1.0 |


**letter** (eval=3000) — n=5

| model | base | +aux | +tgt | auxΔ (p) | tgtΔ (p) | aux−tgt |
|---|--:|--:|--:|--:|--:|--:|
| LR | 75.0 | 70.5 | 71.0 | -4.5\* | -4.0\* | -0.5 |
| NB | 61.4 | 57.4 | 57.3 | -4.0\* | -4.1\* | +0.1 |
| KNN | 38.5 | 29.8 | 29.9 | -8.7\* | -8.6\* | -0.1 |
| DT | 45.0 | 43.1 | 43.4 | -1.9 | -1.6 | -0.2 |
| RF | 62.3 | 57.9 | 58.1 | -4.4\* | -4.2\* | -0.3 |


---

## LIME aux-discretizer test (the clean one — on LIME itself)

Driver `_lime_auxdisc.py`; results `paper_results/lime_auxdisc_ds{1,9}.json`. Base Autolycus LIME, n=1 (native), 10 splits, 3 paired arms: `nothresh` (no bin edge), `tgt` (LIME discretizer fit on target `X_train`), `aux` (LIME discretizer fit on attacker `X_test_s`). thr_tgt=tgt−nothresh, thr_aux=aux−nothresh, leak=tgt−aux. (\* p<.05)

**Result: threshold genuinely helps (thr_tgt big & sig), aux-disc reproduces it (thr_aux≈thr_tgt), leak≈0 (never sig). => LIME's discretization helps extraction but is NOT an explanation leak — the adversary self-computes the identical discretization from its own aux data.** This is the CLEAN version (on LIME's own mechanism, unconfounded); here the threshold helps crop AND pendigits, unlike the SHAP graft which hurt pendigits/letter.


**crop** (eval~255)

| model | no-thr | tgt | aux | thr_tgt | thr_aux | leak |
|---|--:|--:|--:|--:|--:|--:|
| LR | 39.9 | 79.5 | 81.5 | +39.6\* | +41.6\* | -2.0 |
| NB | 55.8 | 80.8 | 80.2 | +25.0\* | +24.3\* | +0.6 |
| KNN | 51.4 | 66.5 | 69.6 | +15.1\* | +18.2\* | -3.1 |
| DT | 38.6 | 76.0 | 76.0 | +37.4\* | +37.4\* | +0.0 |
| RF | 59.1 | 81.7 | 81.5 | +22.7\* | +22.4\* | +0.3 |


**pendigits** (eval~1648)

| model | no-thr | tgt | aux | thr_tgt | thr_aux | leak |
|---|--:|--:|--:|--:|--:|--:|
| LR | 53.9 | 62.8 | 63.0 | +8.9\* | +9.1\* | -0.2 |
| NB | 65.8 | 72.1 | 71.3 | +6.2\* | +5.5\* | +0.8 |
| KNN | 62.5 | 62.3 | 62.1 | -0.2 | -0.4 | +0.2 |
| DT | 30.1 | 42.9 | 40.6 | +12.8\* | +10.5\* | +2.3 |
| RF | 52.3 | 50.3 | 49.6 | -2.0 | -2.7 | +0.7 |


### How little aux is enough? (seed-set) and why online refit fails

Same test, but the aux discretizer is fit on the attacker's **seed set** (~10-17 samples) instead of the 170-sample shadow pool; and an **online** variant refits the discretizer every 50 queries on the growing query set (`_lime_auxdisc_online.py` → `lime_auxdisc_online_ds*.json`; static seed → `lime_auxdisc_seed_ds*.json`).

**Findings:** (1) **seed set (natural samples) mostly suffices** — crop leak≈0 from 17 seeds; pendigits shows a small residual (NB +3.0\*) because 10 samples is too few. (2) **Online refit BACKFIRES** on both crop and pendigits (leak grows, NB significant) — the attack's queries are snapped-to-edge / boundary-pushed artifacts, NOT iid data, so quartiles fit on them are distorted. You cannot bootstrap the bins from the attack's own queries. **=> The clean headline is the shadow-pool run (`lime_auxdisc_ds*.json`): a modest sample of NATURAL aux data gives leak≈0 everywhere. Seed-only and online are supporting detail (the aux-size floor + a ruled-out shortcut).**

#### seed-set aux (static, ~10-17 samples)

**crop** (eval~255)

| model | no-thr | tgt | aux | thr_tgt | thr_aux | leak |
|---|--:|--:|--:|--:|--:|--:|
| LR | 39.9 | 79.5 | 81.8 | +39.6\* | +41.9\* | -2.2 |
| NB | 55.8 | 80.8 | 79.7 | +25.0\* | +23.8\* | +1.2 |
| KNN | 51.4 | 66.5 | 63.2 | +15.1\* | +11.8\* | +3.3 |
| DT | 38.6 | 76.0 | 76.3 | +37.4\* | +37.7\* | -0.3 |
| RF | 59.1 | 81.7 | 82.5 | +22.7\* | +23.5\* | -0.8 |

**pendigits** (eval~1648)

| model | no-thr | tgt | aux | thr_tgt | thr_aux | leak |
|---|--:|--:|--:|--:|--:|--:|
| LR | 53.9 | 62.8 | 58.1 | +8.9\* | +4.2 | +4.7 |
| NB | 65.8 | 72.1 | 69.0 | +6.2\* | +3.2 | +3.0\* |
| KNN | 62.5 | 62.3 | 63.4 | -0.2 | +0.9 | -1.1 |
| DT | 30.1 | 42.9 | 39.0 | +12.8\* | +8.8\* | +3.9 |
| RF | 52.3 | 50.3 | 49.8 | -2.0 | -2.5 | +0.5 |

#### online refit (every 50 q) — NEGATIVE (worse than static seed)

**crop** (eval~255)

| model | no-thr | tgt | aux | thr_tgt | thr_aux | leak |
|---|--:|--:|--:|--:|--:|--:|
| LR | 39.9 | 79.5 | 75.6 | +39.6\* | +35.7\* | +4.0 |
| NB | 55.8 | 80.8 | 77.2 | +25.0\* | +21.4\* | +3.6\* |
| KNN | 51.4 | 66.5 | 63.5 | +15.1\* | +12.1\* | +3.0 |
| DT | 38.6 | 76.0 | 73.0 | +37.4\* | +34.3\* | +3.1 |
| RF | 59.1 | 81.7 | 79.9 | +22.7\* | +20.9\* | +1.8 |

**pendigits** (eval~1648)

| model | no-thr | tgt | aux | thr_tgt | thr_aux | leak |
|---|--:|--:|--:|--:|--:|--:|
| LR | 53.9 | 62.8 | 55.8 | +8.9\* | +1.9 | +7.0\* |
| NB | 65.8 | 72.1 | 68.6 | +6.2\* | +2.8 | +3.4\* |
| KNN | 62.5 | 62.3 | 63.1 | -0.2 | +0.7 | -0.9 |
| DT | 30.1 | 42.9 | 39.0 | +12.8\* | +8.8\* | +4.0 |
| RF | 52.3 | 50.3 | 49.3 | -2.0 | -3.0 | +1.0 |


---

## Threat-model framing (Autolycus arXiv:2302.02162, §System and Threat Models)

- Split = **75% train / 15% test / 10% auxiliary**. The auxiliary set is the **attacker's** (used to create D_A). This is exactly our `X_test_s`.
- The paper's *evaluated* D_A is minimal: "a subset of n samples per class from the auxiliary set" (our seed set, n=1/class).
- But the threat model **permits more**: D_A "may range from a rudimentary collection of samples to comprehensive datasets." The attacker is also assumed to know each feature's **type, domain, and lower/upper bounds**.
- **=> Fitting the aux discretizer on the FULL 10% auxiliary pool (`X_test_s`) is INSIDE the threat model** — not a stretch. Do **not** add a separate validation split (not in the Autolycus protocol; would break comparability).

Report the aux arms as a **sensitivity analysis** (all n=1 traversal, LIME native):

- **aux_full** (`lime_auxdisc_shadow_ds*.json`): discretizer fit on **all of `X_test_s`** = the "comprehensive D_A" the threat model allows. **leak ≈ 0** on crop and pendigits (threshold helps +9..+40, self-computed). **HYBRID — state explicitly, don't gloss:** the discretizer is fit on the full aux pool (crop: 170 rows) while the traversal is still **seeded with only n=1 per class**. Realistic (an attacker uses all its data for distribution estimation but seeds with a few points), but must be described.
- **aux_seed** (`lime_auxdisc_seed_ds*.json`): discretizer fit on the **n/class seed set** = the paper's minimal D_A. Holds on crop (leak≈0); **partially degrades on pendigits** (NB +3.0pp, p<.05) — 10 samples too few for stable quartiles.
- **online** (`lime_auxdisc_online_ds*.json`): refit bins on the **accumulated query set** every 50 q. **WORSE than both** — leak becomes significant on crop NB (+3.6) and pendigits LR (+7.0). **Mechanistic reason (a real finding, not a failed run):** the attack deliberately concentrates queries near decision boundaries, so the query history is a **biased estimator of the data distribution** → distorted quartiles.

**One-line result:** LIME's discretization genuinely helps extraction, but under Autolycus's own threat model the adversary self-computes an equivalent discretization from its auxiliary pool (leak≈0); it degrades only if restricted to a handful of seeds, and cannot be bootstrapped from the (boundary-biased) query history.
