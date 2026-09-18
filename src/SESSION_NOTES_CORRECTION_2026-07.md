# CORRECTION to the June-2026 Tree-Extraction Session Notes (dated 2026-07-03)

**This supersedes the tree-target numbers and conclusions in  `SESSION_NOTES_tree_extraction_2026-06.md`.**
A test-set model-selection leak was found in the evaluation harness. After fixing it, most of the tree "wins" do not survive. The non-tree results are unaffected.

---

## 1. The bug: test-set model selection (winner's curse)

`_build_surrogate_and_eval` (and every standalone driver's `eval_*`) fit `repetition` surrogates (**dt: 100, rdf: 10, lr/nb/knn: 1**), scored **each** with `rtest_sim` on `X_test_t`, and returned the **`max`** — then reported that same max. `X_test_t` was thus used **twice**: to *select* the best refit and to *report* it. That is selection-on-the-test-set, which is upward-biased (you preferentially keep the refit whose noise on `X_test_t` was luckiest). For `n` refits the inflation ≈ `+c·σ`, with `c ≈ 2.5 (n=100)`, `1.5 (n=10)`, `0 (n=1)`, and `σ` = std across refits.

The bias **does not cancel** in a paired A−B comparison unless `σ_A ≈ σ_B`. Densification changes the training-set size/composition → changes `σ` → the reported gap carried an uncancelled `c·(σ_A−σ_B)` term of the same magnitude (±1–3 pp) as the claimed wins.

**Diagnostic that confirms it:** the models the notes call "noisy / sign-flipping" are exactly the high-`reps` ones. reps=1 (LR/NB/KNN) = "robust, clean"; reps=10 (RF) = "more consistent"; reps=100 (DT) = "inconsistent." That ordering *is* the leak.

## 2. The fix

Aggregate the `repetition` refits by **`mean`**, not `max` (mean = unbiased fidelity estimate, and variance reduction is what `repetition` was for). Applied to `attack_utils.py` (`run_attack_auto_compare`, `run_attack_auto`; MLP left as architecture-selection, out of scope) and to all standalone drivers. lr/nb/knn (reps=1) are unchanged by construction → **all non-tree results stand as reported.** Aggregation does not touch the RNG, so re-runs reproduce the *same* traversals — the only thing that changes is max→mean (clean before/after).

## 3. Corrected head-to-head (Q=500, 20 paired sets, MEAN metric)

Driver: `_reeval_meanmax.py` → `reeval_meanmax.json`. Each traversal scored under both metrics.

| combo | base | SHAP3 | s4b(nodiv) | **s4b−base** | s4b−SHAP3 | SHAP3−base | OLD max gap | s4b std |
|---|---|---|---|---|---|---|---|---|
| adult+RF | 0.8992 | 0.8964 | 0.9127 | **+1.35** | +1.63 | −0.28 | +1.11 | 0.016 |
| breast+DT | 0.9648 | 0.9786 | 0.9824 | **+1.76** | +0.38 | +1.38 | +1.94 | 0.018 |
| crop+DT | 0.9108 | 0.9196 | 0.9090 | −0.18 | −1.06 | +0.38 | +0.38 | 0.040 |
| breast+RF | 0.9572 | 0.9673 | 0.9568 | −0.04 | −1.05 | +1.01 | +0.00 | 0.021 |
| mushroom+RF | 0.9340 | 0.9382 | 0.9219 | −1.21 | −1.63 | +0.42 | −1.12 | 0.048 |
| adult+DT | 0.8189 | 0.8023 | 0.8111 | −0.78 | +0.88 | −1.66 | −0.83 | 0.031 |
| crop+RF | 0.9656 | 0.9716 | 0.9621 | −0.35 | −0.95 | +0.60 | −0.17 | 0.013 |
| mushroom+DT | 0.9128 | 0.9100 | 0.9103 | −0.25 | +0.03 | −0.28 | −0.18 | 0.056 |

(pp; s4b = `SHAP4b(use_diverse=False)`, the notes' "best" tree config.)

**Averages across the 8 tree combos:** s4b−base **+0.04 pp (≈ 0)**, s4b−SHAP3 **−0.22 pp**, SHAP3−base **+0.26 pp**.

## 3b. Confirmation runs (survivor Q-sweep + regenerated drivers, all MEAN metric)

**Survivor Q-sweep** (`_survivor_sweep.py` → `survivor_sweep.json`, 20 paired sets, seeds reused across Q):

| combo | metric | Q100 | Q250 | Q500 | Q1000 |
|---|---|---|---|---|---|
| adult+RF | nd−base | +1.20 | +1.68 | +1.38 | +1.67 |
| adult+RF | nd−SHAP3 | +2.10 | +2.05 | +2.52 | +2.25 |
| breast+DT | nd−base | +0.90 | +1.70 | **+0.17** | — |
| breast+DT | nd−SHAP3 | +1.95 | +2.00 | **−1.06** | — |
| crop+DT | nd−base | −1.05 | −1.26 | +1.25 | +0.19 |
| nd−dv (nodiv−div) | avg over all rows | \~+0.5 pp, noisy (breast Q100 −0.41) | | | |

**Regenerated drivers** (mean metric): densify adult+RF **+2.10 vs base / +2.61 vs SHAP3** (std .013); ablate (30 sets) adult+RF nodiv **+1.56 vs base, +2.33 vs scaffold** (scaffold≈base → the lift *is* the densification); ablate crop+DT nodiv +1.23 vs base but nodiv−div +2.16 (combo-specific). shap_ablation:
SHAP feature choice helps on mushroom (DT +8.1, RF +3.6) / adult+DT (+2.1), ≈random elsewhere; dropping the Phase-3 traversal costs DT +5–8 pp (DT still needs it).

## 4. Revised conclusions (after all confirmation runs)

1. **Densification is NOT a general tree win.** Across the 8 combos it matches Autolycus on average (net +0.04 pp) and is slightly *worse* than SHAP3 (−0.22 pp). SHAP3 is the better general tree method.
2. **adult+RF is the ONE robust win.** Positive in **every** independent measurement — reeval +1.35, sweep +1.38, densify +2.10, ablate(30) +1.56 (all vs base); **+2.0–2.6 pp vs SHAP3**; holds across **Q = 100→1000**; scaffold control confirms the lift is the densification (+2.33 pp over scaffold); lowest variance of any arm (std .013–.016). This is real and publishable on its own.
3. **breast+DT is DOWNGRADED — not reliable.** It swung +1.76 (reeval) / +0.17 (sweep Q500) / −1.06 (continuous driver) across different seed draws of the same combo → seed-lucky, low-Q-only. My earlier "second survivor" call was premature; the sweep exposes it as noise.
4. **The June headline wins were max artifacts:** crop+DT (notes +1.2–1.6) is noisy around 0 and Q-unstable (−1.3 → +1.25); breast+RF → −0.04; mushroom+RF → −1.21 (a loss). These were the highest-σ arms — exactly where `max` had the most to cherry-pick.
5. **`use_diverse=False` (nodiv) is only weakly better than div** (~+0.5 pp avg, noisy) — NOT the +1.6 pp of §3.4. For adult+RF, div≈nodiv (+0.1 pp); nodiv's edge is real only on the unreliable crop+DT.
6. **Meta-lesson: results are seed-draw sensitive.** adult+RF's magnitude ranged +1.4→+2.1 pp with RNG/combo order; only combos that stay positive across *seed draws* (adult+RF) should be trusted. A fixed-seed, multi-draw protocol is needed for any future claim.
7. **DT stays the highest-variance family** (std up to .056) — the most leak-contaminated, least reliable.

## 5. What is invalidated vs. what still stands

**Invalidated / needs the mean number** (all tree, reps>1):
- §3.2 (densify): the crop+DT and mushroom-DT "wins" and the adult+RF magnitude.
- §3.3 (continuous): crop+DT +1.2 pp and the breast rows.
- §3.4 (ablate): the scaffold/nodiv comparison was under `max` (re-running).
- §3.5 / §3.6: tree rows; the "RF wins hold/grow with Q" claim (re-checking via the survivor sweep).
- §3.7–§3.10: all tree, measured under `max` (re-running the core ones).

**Still valid (leak-free, reps=1):**
- §3.11 (recycle_bisection) non-tree rows, §3.12 and **§3.13** (non-tree SHAP attribution) — the
  "gain is boundary-**sampling**, not SHAP feature-ranking; SHAP-in-traverse is harmful" thesis is
  untouched. This remains the project's strongest, cleanest result.
- The two free-lunch principles (recycle-all-queried, recycle-bisection).

## 6. Status (2026-07-04)

**Done:** eval leak fixed in `attack_utils.py` + all standalone drivers; re-eval (`_reeval_meanmax`), survivor Q-sweep (`_survivor_sweep`), and core driver regen (`_shap4b_densify`, `_shap4b_continuous`, `_shap4b_ablate`, `_shap_ablation`) all complete under the mean metric. Regenerated JSONs written.

**Not re-run** (low value): secondary/negative-result drivers (`_weight_prototype`, `_margin_weight`, `_snap_stage1`, `_shap4_confirm20`, `_recycle_bisect`) — their conclusions were negative and won't improve under mean; and the full `final_table` / `query_sweep` (non-tree rows are leak-free/identical, tree rows are covered above). Run on request.

**Bottom line:** the tree line of work reduces to a single defensible result — **adult+RF densification (+1.4–2.1 pp vs Autolycus, +2–2.6 pp vs SHAP3, budget-stable, scaffold-confirmed)**. The broad "boundary densification helps tree extraction" claim does not survive. The project's strength remains the leak-free **non-tree boundary-sampling** result (§3.13).

**Caveats:** single seeded run per script (though adult+RF reproduces across 4 independent draws); paired SEs not formally computed, but adult+RF's consistency across draws/budgets puts it well clear of the noise while every other combo sits within it.
