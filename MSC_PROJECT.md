# MSc project: when is the discretization grid *not* self-computable?

## Background

Our paper shows that withholding LIME's bin edges fails as a defence, because the edges are
quartiles of the explainer's background data and an adversary with representative auxiliary data
reconstructs an equivalent grid (mean difference $-0.7$pp across the twelve cells where the channel
operates, and the one significant difference favours the adversary).

That result is bounded by two assumptions we made in the adversary's favour, and both are stated as
limitations in the paper. The project is to remove them and see whether the defence survives.

**Thesis question: under what conditions is the grid not reconstructible, and does withholding it
protect the model there?**

A defence that works only for weak adversaries is still a defence, provided the regime is stated.
A defence that fails even there is a stronger negative result than ours. Either outcome is a result.

## Axis 1 (start here): adversary data quality

Our auxiliary pool is a uniform random split of the *same* dataset — the most generous assumption
available. Real adversaries hold something worse. Vary:

1. **Size.** Already swept at 1, 2, 3, 5 per class on continuous datasets (Appendix C,
   `src/_auxsize_fid.py`). Extend downward and to the categorical datasets, where we found the small-$n$
   grid is unstable rather than merely imprecise (nursery/KNN at $n{=}1$ is $-14.3$pp against a target
   of $+1.0$, recovering only at $n{=}10$). That instability is unexplained and worth a section.
2. **Distribution shift.** Fit the adversary's discretizer on a *shifted* population rather than a
   disjoint sample of the same one: a subpopulation, a covariate shift, a different collection period.
   This is the untested assumption that matters most.
3. **Locate the crossover.** As auxiliary quality degrades, at what point does `tgt − aux` become
   significantly positive? That crossover point *is* the defence's operating regime, and nobody has
   measured it.

Deliverable: a curve of leak against auxiliary-data quality, with the regime where withholding the
grid actually protects marked on it.

## Axis 2: label-dependent discretizers

LIME's default `QuartileDiscretizer` is model- and label-independent, which is *why* it is
reconstructible. LIME also ships an **entropy discretizer**, which fits cut points using the labels.

Hypothesis: a label-dependent grid is not a pure marginal statistic, so the self-computability
argument should not apply, and serving entropy-discretized explanations could be a defence that
survives an adaptive adversary.

Counter-hypothesis to test in the same experiment: an adversary with *labelled* auxiliary data can
fit the same entropy discretizer. Then the defence fails for the same reason as before, and the paper's
claim generalises beyond the default configuration.

Run the three-arm design of §6.4 (`nothresh` / `tgt` / `aux`) once per discretizer
(quartile, decile, entropy). Both answers are publishable.

## Why these two compose

They are the only two ways the grid can fail to be self-computable: either the adversary's data is
unrepresentative (Axis 1) or the grid is not a property of the data distribution (Axis 2). Together
they answer the paper's closing sentence, which is currently an open problem:

> Characterising what a defender can withhold that an adaptive adversary cannot reconstruct remains open.

## Alternative, if she prefers detection to prevention

Don't hide the grid, notice it being used. Cutpurse leaves a distinctive query trace: coordinates
snapped to a quartile grid, and 8-step bisection chains between oppositely-classified points. Build a
detector over the query stream and measure detection rate against false positives on benign traffic.
Different skill set, more open-ended, less anchored to our results.

## Practical notes

- Reuse, do not rebuild: `src/_lime_auxdisc.py` (three-arm design), `src/_auxsize_fid.py` (per-class
  sweep), `src/_defense_sweep.py` (budget sweep), `src/_ladder.py` (adversary arms).
- Protocol to match ours: 10 independent splits, target retrained per split, arms paired within
  split, paired Wilcoxon, mean over refits. See §6.1.
- Watch the evaluation-set size trap: breast has 85 held-out points, so one point is 1.18pp. Don't
  rest a claim on it.
- Give her the paper and the code up front. The project is to extend the result, not re-derive it.
- If one semester only: Axis 1, distribution shift. It is the most direct continuation of the original
  defence topic and the least likely to run out of road.
