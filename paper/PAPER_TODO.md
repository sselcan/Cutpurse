# Paper TODO — SaTML resubmission

Target: SaTML (deadline TBC). Previous: PoPETs 2027.2, quick-rejected 2026-09-06 on scope
(chairs: "does not contain a privacy violation or privacy evaluation"). No criticism of the
results. Venue change means the privacy framing becomes a secondary implication and the channel
decomposition leads.

Last updated 2026-09-15. Supersedes `PAPER_TODO_petsround_20260829.md`.

---

## 1. Open — correctness, found in the 2026-09-01 read

These are self-contradictions a reviewer can catch by comparing the text to our own tables.

- [ ] **"The two explanation-free arms"** (§6.5). There are **three**: the bullet list directly above
  says `blind` makes "no explanation call", and `src/_ladder.py:49` confirms all of blind, self-grid
  and Cutpurse issue zero `explain_instance` calls. The sentence contradicts the list one paragraph
  earlier.
- [ ] **"On the two datasets whose evaluation sets can resolve it"** (§6.3). Wrong as a resolution
  claim: mushroom has 1218 evaluation points and nursery 1944, both resolve fine and both show
  significant threshold effects. Separately, the quoted ranges cover only *significant* cells without
  saying so — pendigits actually spans $+0.4$ (KNN) to $+18.6$ (LR), not "+8 to +19".
- [ ] **§3 threshold definition is the unhedged wording** we removed from the abstract, intro and §4:
  "where along a feature axis the prediction changes, i.e. the coordinate of the decision boundary."
  This is the formal definition the paper hangs on, and §6.4 concludes the bin edge is a background
  quantile rather than a boundary coordinate. Coauthor flagged the same thing.
- [ ] **§6.1 list grammar**: "the channel ablations of §6.2 and §6.3, the self-computability test of
  §6.4, **in** the adversary comparison of §6.5 and **in** the budget sweep." Mixed structure.

## 2. Open — author decisions

- [ ] **Figure 3 vs Figure 4.** Fig 3 (`decomposition.png`) is the aggregate of Fig 4
  (`decomposition_per_dataset.png`), and its two numbers ($+1.6$/$+1.8$pp) appear in the text twice.
  Cutting it is the last obvious trim with no information loss. Needs two `\ref{fig:decomp}`
  rewordings in Appendix A.
- [x] **Figures 5 and 7 (LIME/SHAP matched-access grids).** Decision: **keep both**. Fig 7 is the only
  SHAP attack-performance figure, and the "same machinery helps under both explainers" claim needs
  the pair.
- [ ] **Anonymised artifact link** for the open-science block (`% BEFORE SUBMISSION` comment still in
  place). Check SaTML's artifact policy, which differs from PoPETs'.
- [ ] Remove the unused `\TODO` macro and its comment before submission.

## 3. Open — coauthor review of chapters 4–7

`~/Downloads/paper_review_chapters_4_7.md`. Substantial line-level review. Its own priority list:

- [ ] Replace "leak" with "exploitable signal" / "disclosure" where appropriate.
- [ ] Do not call LIME bin edges a "direct estimate" of the boundary — candidate coordinates.
  (Partly done: abstract, intro and §4 already softened; §3 and §5.2 still say "direct estimate".)
- [ ] Remove rhetorical language: "for free", "hopeless", "rung below", "speed bump", "the floor".
- [ ] Don't claim the refit mean is "unbiased" — say it avoids selection on the evaluation metric.
- [ ] Shorten §7; it re-proves §6 rather than interpreting it.
- [ ] Move the categorical-encoding detail out of §7 into §6.1 or an appendix.
- [ ] "percentage points" consistently rather than "points".
- [ ] Consistent SHAP/LIME capitalisation and consistent use of `\sys`.

## 4. Venue reframing for SaTML

- [ ] Rewrite abstract and intro to lead with the channel decomposition and self-computability.
  Privacy becomes an implication, not the frame. Desk decisions appear to turn heavily on framing
  (cf. the PETS title/abstract precedent), so this deserves real effort rather than a find-replace.
- [ ] Revisit the title. "Survives Withholding the Explanation" is good but was written for a privacy
  venue.
- [ ] Reformat to SaTML style; confirm page limit (currently 12pp ACM two-column, 18pp total).

## 5. Verified, do not re-check

All reproduce from the stored per-split arrays:

- Cutpurse − defended, continuous: mean $+16.29$pp, median $+10.50$, 13 sig+ / 0 sig−, n=20.
  Matches the typeset Table 6 column exactly ($+16.30$ / $+10.50$).
- Cutpurse − Autolycus: $+7.0$pp continuous, $-2.5$pp categorical.
- Cutpurse − self-grid: $+3.3$pp all, $+5.9$pp continuous (9 sig+, 0 sig−), $-1.7$pp categorical.
- self-grid − Autolycus: $+0.5$pp over 30 cells (7 sig+, 4 sig−).
- Table 5 leak column: mean $-0.70$pp over 12 cells, range $-3.4$ to $+1.1$, 7 of 12 favour the adversary.
- Appendix D seed/budget numbers: $+4.51$ ($k$=1,$n$=3), $+1.88$ ($k$=3,$n$=3), $+3.41$ (headline);
  baseline spends 25.6% vs 88.6% of top budget.
- Appendix B: LIME 18/25 wins, 10 sig+ / 2 sig−, $+3.41$pp. SHAP 18/25, 8 sig+ / 1 sig−, $+2.27$pp.
- `thr_tgt` in Tables 5 and 7 reproduces Table 4 cell for cell, including a third independent
  mushroom run (KNN $+13.90$).

## 6. Not doing before submission

- Prediction-only extraction baselines (Knockoff Nets / ActiveThief). Different data assumption
  (large transfer pool vs $n\le5$ seeds per class). The `blind` arm is already a prediction-only
  adversary; its bullet says so.
- Alternative discretizers (decile, entropy). See `MSC_PROJECT.md` — handed to the MSc student.
- Membership inference on the extracted surrogates. Out of scope for this paper.
