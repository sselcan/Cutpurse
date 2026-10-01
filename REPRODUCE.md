# Reproducing the paper

All commands run from `src/`. Pin BLAS to one thread per process, or parallel runs thrash:

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
pip install -r requirements.txt
```

Dataset IDs: `1` crop, `2` adult, `3` breast, `4` nursery, `5` mushroom, `9` pendigits.
Every driver takes `--ds`, most take `--models` (`0` DT, `1` LR, `2` NB, `3` KNN, `4` RF) and `--out`.
All drivers write one JSON per dataset holding the per-split fidelity arrays, so every table can be
rebuilt without rerunning an attack.

## Read this before trusting an output directory

Results live in several directories because pendigits was re-run at Q=1000 and the random-forest
rows were re-run separately. **The same cell can exist in four directories with different values, and
only one is the published one.** Example: `pendigits/RF` in Tables III and IV appears in
`paper_results/` (Q=500), `paper_results_rf/` (Q=500), `paper_results_rf_ladder/` (Q=500) and
`paper_results_q1000_rf/` (Q=1000). The published values are the Q=1000 ones.

`_ci_table.py` encodes the authoritative precedence for every table and **asserts each recomputed
point estimate against the value printed in the .tex source**, so run it first if you are unsure
whether you are looking at the right file:

```bash
python3 _ci_table.py
```

It prints `all N cells reproduce the published values` per contrast, or names the cells that disagree.

## Tables

| paper | command | output |
|---|---|---|
| **II** RQ1 SHAP | `for ds in 1 2 3 4 5; do python3 _autolycus_ablation_shap_ms.py --ds $ds --out paper_results; done`<br>`python3 _autolycus_ablation_shap_ms.py --ds 9 --q 1000 --out paper_results_q1000` | `autolycus_ablation_shap_ms_ds*.json` |
| **III** RQ1 LIME<br>**IV** RQ2 bin edges | one run produces both:<br>`for ds in 1 2 3 4 5; do python3 _autolycus_ablation_ms.py --ds $ds --out paper_results; done`<br>`python3 _autolycus_ablation_ms.py --ds 9 --q 1000 --out paper_results_q1000`<br>`python3 _autolycus_ablation_ms.py --ds 9 --q 1000 --models 4 --out paper_results_q1000_rf` | `autolycus_ablation_ms_ds*.json` |
| **V** RQ3 grid source | `for ds in 1 3 4 5; do python3 _lime_auxdisc.py --ds $ds --out paper_results_e5fresh; done`<br>`python3 _lime_auxdisc.py --ds 9 --q 1000 --out paper_results_q1000` | `lime_auxdisc_shadow_ds*.json` |
| **VI**, **XI** grid-data sensitivity | `for ds in 1 9; do python3 _auxsize_fid.py --ds $ds --out paper_results; done` | `auxsize_fid_ds*.json` |
| **VII**, **XII**, **XIII** class support | see *Class-support restriction* below | `paper_results_shift/` |
| **VIII** entropy discretizer | `for ds in 1 2 3 9; do python3 _lime_auxdisc_entropy.py --ds $ds --disc entropy --out paper_results_disc; done`<br>`for ds in 1 9; do python3 _auxsize_fid_entropy.py --ds $ds --disc entropy --sizes 1,2,3,5,10 --out paper_results_disc; done` | `lime_auxdisc_entropy_ds*.json` |
| **IX** adversary ladder | `for ds in 1 2 3 4 5; do python3 _ladder.py --ds $ds --out paper_results; done`<br>`python3 _ladder.py --ds 9 --q 1000 --out paper_results_q1000`<br>RF rows: `bash _rf_ladder_then_sweep.sh`<br>assemble: `python3 _ladder_table.py` | `ladder_ds*.json` |
| **X** step-size sensitivity | `for ds in 1 2 3 9; do python3 _ladder_eps.py --ds $ds --out paper_results_eps2; done`<br>`python3 _eps_table.py --out paper_results_eps2` | `ladder_eps_ds*.json` |

`_eps_table.py` refuses to report unless its re-run `autolycus` arm reproduces the stored one element
for element, since it pairs against `defended`/`self-grid` arrays from a different run.

## Figures

| paper | command |
|---|---|
| **2** defense sweep, continuous | `for ds in 1 2 3 9; do python3 _defense_sweep.py --ds $ds --out paper_results; done`<br>`python3 _defense_sweep_plot.py --only continuous --out figures/defense_sweep.png` |
| **7** defense sweep, categorical | `for ds in 4 5; do python3 _defense_sweep.py --ds $ds --out paper_results; done`<br>`python3 _defense_sweep_plot.py --only categorical --out figures/defense_sweep_categorical.png` |
| **4** per-dataset decomposition | needs the LIME sweep below, then `python3 _e1_perdataset.py` |
| **6** LIME budget grid | `for ds in 1 2 3 4 5; do python3 lime_paper_run.py --ds $ds --qs 100,250,500,1000 --caps 0,5,15,30 --div_frac 0.3 --hms 10 --size 1 --nfe 3 --seed 0 --out paper_results; done`<br>`python3 _regen_grids.py` |
| **8** SHAP budget grid | `for ds in 1 2 3 4 5; do python3 shap_paper_run.py --ds $ds --qs 100,250,500,1000 --hms 10 --size 5 --nfe 3 --seed 0 --main shap3 --out paper_results; done`<br>`python3 _regen_grids.py` |
| **1** pipeline diagram | hand-drawn (drawio), not generated |

`_e1_perdataset.py` and `_regen_grids.py` read `lime_divsweep_n1_k3_ds*.json` and
`shap_sweep_n5_k3_ds*.json`; the filename encodes `--size`/`--nfe`, so those flags must match.

**Known gap:** `figures/decomposition.png` (Figure 5, the aggregate decomposition) has no generating
script in this repository. `_e1_perdataset.py` produces the per-dataset version (Figure 4) from the
same `lime_divsweep` data, and the aggregate is the mean over datasets of the same two quantities.

## Class-support restriction (Tables VII, XII, XIII)

Three drivers over the same class-holdout pools. Held-out classes are drawn from
`default_rng(7000+s)` with an identical call order in all three, so the pools are byte-identical
across runs and the arms stay paired.

```bash
# grid displacement only, no target queries, seconds to run
python3 _aux_shift_grid.py

# (a) shift the grid only, seeds intact
python3 _lime_auxshift.py --ds 1 --models 1,2,3   # and --models 0 / --models 4
python3 _lime_auxshift.py --ds 9 --models 1,2     # and --models 0

# (b) shift the seeds too: the end-to-end restricted attacker
python3 _lime_auxshift_e2e.py --ds 1 --models 1,2,3   # and 0 / 4
python3 _lime_auxshift_e2e.py --ds 9 --models 1,2     # and 0

# (c) restricted seeds with the SERVICE's grid: the RQ3 contrast under shift
python3 _lime_auxshift_tgt.py --ds 1 --models 1,2,3   # and 0 / 4
python3 _lime_auxshift_tgt.py --ds 9 --models 1,2     # and 0

python3 _auxshift_table.py     # Table XII
python3 _shift_e2e_table.py    # Table XIII
```

`nothresh` and `tgt` are invariant to this manipulation and are read from the stored Table V run
rather than recomputed. Both assemblers verify that the re-run `aux_full` arm reproduces the stored
RQ3 `aux` arm element for element and refuse to report if it does not.

Shard by `--models` to parallelise; each shard writes its own file. The restricted-seed arms cannot
satisfy the traversal's per-class visit quota, so they spend the whole budget and take roughly four
times longer than the grid-only arms.

## Statistics

```bash
python3 _ci_table.py --margin 3.0   # cross-cell CIs, BH-adjusted p, verification against the .tex
python3 _recompute_stats.py         # per-cell Wilcoxon with tied-split handling
```

Ties matter: where two arms give identical fidelity on a split the signed-rank test discards the
pair, so a cell with 8 tied splits of 10 is tested on 2 observations. `_recompute_stats.py`
suppresses the marker below 6 untied pairs.

## Protocol

Ten independent splits (`load_dataset(ds, seed=s)`), target retrained per split. Within a split the
seed set is drawn under `random.seed(s)` and every arm is reseeded to `random.seed(1000+s)`, so arms
are paired. Surrogate refits: DT 20, RF 10, LR/NB/KNN 1, aggregated by mean. Query accounting follows
the baseline: the explanation is bundled with the prediction and not charged separately.
