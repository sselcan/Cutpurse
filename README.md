# Cutpurse

Artifact for a paper on **what explanations actually leak to a model-extraction adversary** on
tabular classifiers. An end-to-end attack comparison cannot tell whether an explanation revealed
something about the model or merely handed the attacker a convenient place to put its next query, so
this work separates the explanation into two channels and ablates them independently:

- **Attribution** — the importance ranking, i.e. which features to perturb.
- **Bin edges** — the discretization thresholds LIME reports in its rule conditions
  (`age <= 27.0`), i.e. which feature *values* to perturb to.

The useful channel turns out to be the bin edges, and those are quantiles of the explainer's
background data rather than properties of the target model. **Cutpurse** is the attack that follows:
a self-computed grid, diverse query generation and boundary search, making no explanation API calls
at all. The published explanation-guided attack of Oksuz, Halimi and Ayday
([Autolycus](https://arxiv.org/abs/2302.02162), PoPETs 2024) is the baseline the analysis is run on
and the one Cutpurse is measured against.

## Layout

    src/
      attack_utils.py      traversal, seed generation, boundary search, dataset loaders
      <driver>.py          experiment drivers, one or two per table or figure
      *_table.py           assemblers that turn stored results into the paper's tables
      paper_results*/      stored per-split results (JSON)
    data/                  the three CSV-backed datasets (crop, nursery, mushroom)

## Running

```bash
pip install -r src/requirements.txt
```

[REPRODUCE.md](REPRODUCE.md) maps every table and figure to the command that produces it and names
the result directory it reads. Every driver stores its per-split fidelity arrays, so the tables can
be rebuilt from `src/paper_results*/` without rerunning an attack.

`breast` comes from scikit-learn, `adult` from the `shap` package and `pendigits` from OpenML, so
those need network access on first run. Set `CUTPURSE_DATA` to relocate the CSV root.

## Scope

Tabular classifiers, default LIME and SHAP configurations, and the auxiliary-data condition
inherited from the baseline. The results do not extend to other explainers or discretizers, to
neural or high-dimensional targets, or to adversaries whose auxiliary pool is not representative of
the target's data.
