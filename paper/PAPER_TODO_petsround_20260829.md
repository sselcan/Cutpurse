# Paper TODO

Tracks the PETS submission. `[x]` done, `[ ]` open, `[~]` in progress.
Review item codes (M1..M13, RE1..RE5) refer to `PETS_review_Cutpurse.md`.

Last updated 2026-08-29.

---

## 1. Done

### Phrasing / framing fixes from the review (no compute)
- [x] **M4 — the disclosure comparison leads.** Abstract and contribution bullet 2 now lead with the
  E5 leak (`tgt - aux`, no significant residual in any of the 12 cells where the threshold operates,
  mean -0.4pp against effects up to +41pp) and cite the ladder's `selfgrid - autolycus = -0.3pp over
  24 cells` as corroboration, both before Cutpurse appears.
  *Note:* `selfgrid - autolycus` alone is NOT a clean isolation. `_ladder.py` gives `selfgrid`
  `feature_select='random'` while `autolycus` uses the target's ranking, so that contrast varies the
  grid source AND the feature selection, and it is noisy per cell (mean -0.3pp but 3 significant
  positive, 4 significant negative; crop/NB -8.2**, adult/LR +5.6**). E5 varies only the discretizer's
  fitting data and is the statistic to headline.
  Bullet 3 labels the Cutpurse numbers "a claim about attack strength rather than about disclosure".
  New paragraph in the ladder subsection tells the reader which contrast answers which question.
- [x] **M1 — narrow the claims.** Title scoped to "on Tabular Data". Abstract's closing scope clause
  now names the auxiliary-data assumption. Intro Scope paragraph carries an explicit
  established / not-established split (reproducible bin edges, costless substitution, attack wins vs.
  arbitrary explainers, high-dimensional/neural targets, weak-auxiliary adversary).
- [x] **M9 — drop "inert".** H1 renamed "attribution is not systematically advantageous"; body now
  says the adversary gains nothing it can count on in advance, and points at the TreeSHAP exception
  rather than talking around it. All five occurrences of "inert" removed.
- [x] **M13 — soften "counterproductive" / "worse than useless" / "no privacy gained".** Every claim
  about the defense is now indexed to *this* adversary, and each site says the defense may still be
  worth something against an adversary with weaker auxiliary data.
- [x] **M12 — confidentiality vs privacy.** New paragraph in Problem Formulation declares the primary
  property (model confidentiality, measured as agreement) and the secondary implication (downstream
  membership/attribute inference, inherited from prior work, *not* measured here).
- [x] **M5 — what is novel.** Method section states that diverse synthesis and boundary bisection are
  established techniques, and that the contribution is the analysis specifying them plus the
  self-computability of the coordinates they need.

- [x] Regenerated `figures/defense_sweep.png` with mushroom included (all 6 datasets).
- [x] **M4, visible in the table.** Added `sg-aut` as the first contrast column of Table 7
  (`tab:ladder`), so the disclosure comparison is now readable off the table instead of only in prose.
  Caption names it as the disclosure comparison. This also fixed a caption/text mismatch ("two
  contrasts" vs "three contrasts").
- [x] **Deleted a phantom arm.** Section 6.4 promised a fifth arm, `\sys`+LIME, that does not exist in
  `ladder_ds*.json` (only blind/selfgrid/ours/autolycus) and was not in Table 7.
- [x] **Corrected the LIME attribution aggregate.** The paper claimed "mean +0.6pp, median -0.3pp",
  which reproduces from neither the multi-split nor the single-split data. Recomputed from the stored
  per-split arrays over all 24 cells: **mean -0.33, median -0.55, 3 sig+ / 2 sig-**. The threshold
  aggregate (+6.81, 11 sig+, 0 sig-) reproduces exactly and was already over 24 cells, which is how
  the two tables ended up on different cell sets.
- [x] **Filled the mushroom column of `tab:lime-attr`** (LR -2.9, NB -15.4*, KNN +6.1*, DT +5.3).
  Verified the regeneration reproduces all 20 pre-existing cells exactly, including both daggers.
- [x] **Page limit.** Official template confirms 12 typeset pages for a normal submission (13 for a
  revision); ethics / open-science / AI sections excluded. Main body now ends on p12. Achieved by
  compacting `tab:shap-attr` from a full-width `table*` to a single-column diff grid matching its two
  sibling tables, shrinking the pipeline figure to 0.78\textwidth, and trimming four passages that
  restated results given elsewhere (all listed in the diff against `main_pre_review_fixes.tex`).

Build after these edits: 19 pages, 0 overfull boxes, no undefined references.
Backup of the pre-edit version: `main_pre_review_fixes.tex`.

**Page budget regression.** The main body now ends part-way into p13 (Discussion p11, Conclusion and
Appendix A both start p13, References p14). It ended on p12 before this pass; the M1
established/not-established list, the M12 paragraph and the M4 rewrite cost roughly half a page. Needs
half a page back: candidates are shrinking the full-width pipeline `figure*` to one column, tightening
the Scope paragraph, or compressing the M12 paragraph.

### Earlier passes
- [x] Removed the wine dataset (never used).
- [x] Rewrote the Conclusion around the attack.
- [x] Added the TreeSHAP split-node explanation for the positive SHAP cells.
- [x] Reconciled "order of magnitude weaker" with the +5/+6pp figures.
- [x] Relaxed appendix float placement to `[tbp]`.

---

## 2. Open, no compute needed

- [ ] **Abstract rewrite.** Deferred. Now partly overtaken by the M1/M4 edits above; re-read it end to
  end once the run list below settles.
- [x] **Renamed to `\textsc{Cutpurse}`** (Cutpurse was taken by other work). One line in the preamble;
  `_defense_sweep_plot.py` legend updated and both sweep figures regenerated. No literal "Cutpurse"
  survives in the built PDF.
- [x] **Removed `\usepackage{bm}`** (would not compile in Overleaf). The two `\bm{\phi}` uses became
  `\boldsymbol{\phi}`, which amsmath provides and acmart already loads.
- [ ] **M8 — statistics.** Reanalysis of the stored per-split arrays, no re-running:
  - [ ] bootstrap 95% CIs on every headline contrast;
  - [ ] win rate (k/10 splits) alongside the mean;
  - [ ] rank-biserial effect size instead of bare p-values;
  - [ ] check whether the large threshold effects survive Holm across 24 cells. Note for the text:
    Holm needs p < .05/24 = .00208 and the minimum attainable Wilcoxon p at n=10 is 2/2^10 = .00195,
    so the correction is decidable only for a perfect 10/10 split. Say this rather than ignoring it.
- [ ] **Reproducibility.** Reviewer wants the artifact at review time, not on acceptance. Prepare an
  anonymised repo (code + stored `paper_results/*.json` + table/figure scripts).
- [x] **RF added to the LIME channel ablations.** All six datasets, merged into
  `paper_results/autolycus_ablation_ms_ds*.json` (pre-merge backup in `paper_results_backup_pre_rf/`).
  Both LIME tables are now 30 cells, matching the SHAP table. New aggregates: attribution mean
  $-0.59$, median $-1.29$, **4 sig+ / 4 sig-**; threshold mean $+6.23$, **12 sig+ / 0 sig-**.
  RF is the most negative-leaning family for attribution (pendigits $-5.8$**, adult $-3.7$**,
  neither direction significant-positive), which is the strongest single piece of H1 evidence in the
  paper. Cross-check passed: crop/RF threshold $+22.7$ here independently reproduces `tab:auxdisc`'s
  stored `thr_tgt` of $+22.7$. Setup and Limitations updated to say five families, with the ladder and
  sweep still at four.
- [x] **Figure 2 split by feature type.** The four continuous-feature datasets stay in
  Figure~2; nursery and mushroom move to an appendix companion, organised by the scope condition the
  paper already declares. Main text states plainly that \sys\ does not win there, and reframes the
  panels: the defense barely moves Autolycus on those datasets either, because the withheld channel
  does not exist. `_defense_sweep_plot.py` gained `--only continuous|categorical`.
- [ ] **Figure typo.** `figures/extraction_pipeline-Page-2.jpg` has an arrow labelled "Predicttion
  Explanation" (double t). Must be fixed in the source drawing; not editable from the repo.
- [ ] **Nursery.** Section 7 currently gives the wrong mechanism for nursery (blames spurious bins;
  true for mushroom, not nursery). Either diagnose it or cut the explanation and report the cell.
  Known: grid alone helps (+6.4/+7.5), Phases 1+2 hurt (-9.5/-5.5), cause unidentified.

---

## 3. Open, needs a run — cheap

- [ ] **RF target family.** "RF omitted, prohibitively slow" is now false. Timing on crop, Q=1000,
  per split: Autolycus 95.8s, Cutpurse 14.7s. Six datasets x 10 splits is roughly 1.6h for the
  expensive arm. Adding the row deletes a limitation the reviewer counted against us (S4/M-side).
- [ ] **M7 / RE-categorical — declared categorical features.** Re-run mushroom and nursery with
  `categorical_features` passed to `LimeTabularExplainer`. Prediction to state *before* running: with
  categoricals declared, LIME rules become `feature=value` with no bin edge, so the threshold channel
  cannot exist by construction, the threshold ablation should go to ~0, and whatever remains is pure
  attribution. Turns a disclosed weakness into a confirmatory result and retro-explains the negative
  categorical cells we already report.
- [ ] **M3 part (a) — quantile error vs auxiliary sample size.** No model queries at all, pure numpy:
  for n_aux in {5, 10, 25, 50, 100, 500, all} per class, compare the attacker's 25/50/75 percentiles
  against the target's, per feature, per dataset. Cheap and directly addresses the reviewer's RE1.

---

## 4. Open, needs a run — moderate, and one is existential

- [ ] **M10 — prediction-only extraction baselines. RUN THIS FIRST.** Uncertainty sampling plus
  k-center greedy over the same auxiliary pool, same budget, same seeds, same surrogate family, same
  evaluation set. Roughly 100 lines and cheap, because there is no traversal.
  **Why first:** it decides the paper's shape. If it loses, Section 5 is validated and the review's
  worst gap becomes a strength. If it wins, the "our attack" framing collapses, but the privacy claim
  survives and sharpens to "explanations add nothing over standard active extraction", which is
  arguably the better PETS paper and reuses 90% of the experiments. Either way, do not invest further
  in polishing Cutpurse until this is known.
- [ ] **M3 part (b) — fidelity vs auxiliary sample size.** Re-run `selfgrid` and `ours` at a few n_aux
  values and plot n_aux -> quantile error -> fidelity, with the target-grid arm as the ceiling. This
  is the experiment the whole disclosure claim rests on.
- [ ] **M6 — alternative discretizers.** Swap `discretizer` in `LimeTabularExplainer` for `decile`
  (still label-free) and `entropy` (label-dependent, fit on training labels). Redo target-grid vs
  self-grid for each. This is the difference between "LIME's default is reconstructible" and "the
  discretization channel is reconstructible". Note: LIME's entropy discretizer is fit on ground-truth
  labels, not on model outputs, so an attacker holding labelled auxiliary data can plausibly fit it
  too. Worth confirming rather than assuming.
- [ ] **M2/M6 — auxiliary distribution shift.** Fit the attacker's grid on a shifted population rather
  than a disjoint sample of the same one. Currently our auxiliary pool is a random split of the same
  dataset, the most generous assumption available, which is what makes M2 and M4 land hard.

---

## 5. Push back / not doing

- [ ] **`\sys`+LIME: our attack handed the target's attributions.** Discussed and deliberately deferred,
  not rejected. \sys\ uses `feature_select='random'` in both the ladder and the sweep, which is what
  makes it explanation-free, and H1 justifies it (attribution's sign is unpredictable, so an adversary
  cannot know when consulting it would help). An arm that reads the target's ranking on top of our own
  grid would measure what explanation access is still worth once the grid is self-supplied. It is
  consistent with H1 rather than in tension with it. Two reasons it is not in this submission:
  it reintroduces the explanation-access requirement the paper argues is unnecessary, and the
  regression below already predicts the answer. `_ladder.py`'s docstring describes this arm
  (`ours_lime`); it was never run, and the sentence promising it has been removed from Section 6.4.
  - Predicted result: `ours+LIME - ours` $\approx$ +attribution, so $\approx 0$ on average and
    positive in the 8 cells where LIME's ranking helps. Do not report a predicted number as a result.
  - Cost: the expensive LIME path (Phase 3 calls `explain_instance`).
- [ ] **M11 — full downstream MIA programme.** Model extraction is in scope at PETS as model
  confidentiality; the fix for the reviewer's discomfort is the M12 framing, which is now in. If we
  want insurance, one shadow-free threshold MIA comparing target / Autolycus-surrogate /
  Cutpurse-surrogate is a figure rather than a project, but it ranks last.
- **M8 multiple comparisons.** See the Holm arithmetic under section 2. Report CIs and win rates and
  explain why a correction is uninformative at n=10, rather than applying one for appearance.
- **On disclosing limitations (M7).** Keep doing it. Being explicit earned strength S4, but disclosure
  buys goodwill, not immunity: when a disclosed flaw affects cells we actually report, a reviewer will
  still discount those cells. The fix is to run the thing, not to stop saying it.

---

## 6. Pre-existing items, not from the review

- [x] **Mushroom defense sweep** finished 2026-08-29 01:38; figure regenerated. Nothing is running now.
- [ ] **pendigits Q=500 -> 1000.** Proven not to saturate (500 -> 501, 1000 -> 1001), so Q=500 is
  currently unjustified.
- [ ] **HMS=2 with 10 splits.** Suggested re-run for statistical stability. Combine with RF and the
  pendigits budget change in a single overnight pass.

---

## 7. Strategic

**Decision: submit this cycle** on the free fixes, withdraw and retarget if it does not land.

Consequence for M10: **do not run it before submitting.** On a short clock the downside is asymmetric.
If a prediction-only baseline loses, we gain a paragraph we cannot use in time; if it wins, Section 5
has to be gutted days before a deadline. Run it immediately *after* submission, so the answer informs
either the rebuttal or the next venue.

Worth remembering: the reviewer's own bottom line is close to our thesis. The 4/10 is driven by
framing plus two genuinely missing experiments, not by a result they think is wrong.
