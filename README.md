# Cutpurse

Research code for **explanation-guided model extraction**: how much of a deployed classifier an
adversary can reconstruct when the model serves feature-attribution explanations (LIME, SHAP)
alongside its predictions.

The attack extends [Autolycus](https://arxiv.org/abs/2302.02162) (Oksuz, Halimi and Ayday, PoPETs
2024), which walks a target model's decision space by perturbing the features its explanations mark
as locally important.

## What is being tested

The guiding question is *which* part of an explanation actually leaks. The work separates two
channels that are usually conflated:

- **Attribution** — the importance scores themselves (which features matter, and how much).
- **Discretization thresholds** — the bin edges LIME reports in its rule conditions
  (`age <= 27.0`), which hand the attacker candidate split points directly.

Each is ablated independently against the same target models and query budgets, so the gain from
one is not credited to the other.

## Layout

    src/
      attack_utils.py         core attack: explanation traversal, sample generation, boundary search
      generative_attack.py    generative / synthetic-query variants
      max_cover.py            coverage-based selection
      min_feat.py             minimal-feature-set utilities
      *_paper_run.py          experiment drivers that produce the paper's numbers
      *_paper_results.ipynb   analysis and figures for those runs
      _*.py                   one-off ablations, sweeps and diagnostics
      DEVELOPMENT_NOTES.md    design notes on the traversal methods

    paper/                    manuscript sources (untracked until you choose to commit them)

## Running

```bash
pip install -r src/requirements.txt
```

The drivers under `src/` are standalone scripts; each writes its own results and is run directly,
e.g. `python src/lime_paper_run.py`. Datasets are fetched by the loaders in `attack_utils.py`
rather than vendored into the repo.

## Status

Active research code accompanying a paper in preparation. Interfaces change between experiments and
the `_*.py` drivers are snapshots of specific runs, not a maintained API.
