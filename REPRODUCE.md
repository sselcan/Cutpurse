# Reproducing the paper

Run all commands from `src/`.

```bash
python3 -m pip install -r requirements.txt
```

Dataset IDs are `1` crop, `2` adult, `3` breast, `4` nursery, `5` mushroom, and `9` pendigits.
Stored JSON files contain the per-split arrays and the summary values reported in the tables, so any
result can be checked without rerunning an attack.

## Tables

| paper result | command |
|---|---|
| RQ1 SHAP (Table II) | `for ds in 1 2 3 4 5; do python3 _autolycus_ablation_shap_ms.py --ds $ds --out paper_results; done`<br>`python3 _autolycus_ablation_shap_ms.py --ds 9 --q 1000 --out paper_results_q1000` |
| RQ1 LIME and RQ2 (Tables III--IV) | `for ds in 1 2 3 4 5; do python3 _autolycus_ablation_ms.py --ds $ds --out paper_results; done`<br>`python3 _autolycus_ablation_ms.py --ds 9 --q 1000 --out paper_results_q1000`<br>`python3 _autolycus_ablation_ms.py --ds 9 --q 1000 --models 4 --out paper_results_q1000_rf` |
| RQ3 grid source (Table V) | `for ds in 1 3 4 5; do python3 _lime_auxdisc.py --ds $ds --out paper_results_e5fresh; done`<br>`python3 _lime_auxdisc.py --ds 9 --q 1000 --out paper_results_q1000` |
| Grid-data sensitivity (Tables VI, XI) | `for ds in 1 9; do python3 _auxsize_fid.py --ds $ds --out paper_results; done` |
| Entropy discretizer (Table VIII) | `for ds in 1 2 3 9; do python3 _lime_auxdisc_entropy.py --ds $ds --disc entropy --out paper_results_disc; done`<br>`for ds in 1 9; do python3 _auxsize_fid_entropy.py --ds $ds --disc entropy --sizes 1,2,3,5,10 --out paper_results_disc; done` |
| Adversary ladder (Table IX) | `for ds in 1 2 3 4 5; do python3 _ladder.py --ds $ds --out paper_results; done`<br>`python3 _ladder.py --ds 9 --q 1000 --out paper_results_q1000`<br>`bash _rf_ladder_then_sweep.sh`<br>`python3 _ladder_table.py` |
| Step-size sensitivity (Table X) | `for ds in 1 2 3 9; do python3 _ladder_eps.py --ds $ds --out paper_results_eps2; done`<br>`python3 _eps_table.py --out paper_results_eps2` |

## Figures

| figure | command |
|---|---|
| 2: continuous defense sweep | `for ds in 1 2 3 9; do python3 _defense_sweep.py --ds $ds --out paper_results; done`<br>`python3 _defense_sweep_plot.py --only continuous --out ../figures/defense_sweep.png` |
| 3: aggregate decomposition | `python3 _e1_aggregate.py` |
| 4: per-dataset decomposition | `python3 _e1_perdataset.py` |
| 5: LIME budget grid | `for ds in 1 2 3 4 5; do python3 lime_paper_run.py --ds $ds --qs 100,250,500,1000 --caps 0,5,15,30 --div_frac 0.3 --hms 10 --size 1 --nfe 3 --seed 0 --out paper_results; done`<br>`python3 _regen_grids.py` |
| 6: categorical defense sweep | `for ds in 4 5; do python3 _defense_sweep.py --ds $ds --out paper_results; done`<br>`python3 _defense_sweep_plot.py --only categorical --out ../figures/defense_sweep_categorical.png` |
| 7: SHAP budget grid | `for ds in 1 2 3 4 5; do python3 shap_paper_run.py --ds $ds --qs 100,250,500,1000 --hms 10 --size 5 --nfe 3 --seed 0 --main shap3 --out paper_results; done`<br>`python3 _regen_grids.py` |

## Class-support restriction

Tables VII, XII, and XIII use the following three paired conditions. Run each command for the listed
models (`--models 0` and `--models 4` are separate crop runs). All three drivers draw the held-out
classes from `default_rng(7000+s)` in the same call order, so the pools are identical across runs and
the conditions stay paired.

```bash
# grid restriction only
python3 _lime_auxshift.py --ds 1 --models 1,2,3
python3 _lime_auxshift.py --ds 9 --models 1,2

# restricted grid and seeds
python3 _lime_auxshift_e2e.py --ds 1 --models 1,2,3
python3 _lime_auxshift_e2e.py --ds 9 --models 1,2

# restricted seeds with the service grid
python3 _lime_auxshift_tgt.py --ds 1 --models 1,2,3
python3 _lime_auxshift_tgt.py --ds 9 --models 1,2

python3 _auxshift_table.py
python3 _shift_e2e_table.py
```

`nothresh` and `tgt` do not depend on the grid or seed source, so both assemblers read them from the
stored Table V run rather than recomputing them.

The paper uses ten paired train/test splits with a retrained target on each split. The stored results
correspond to the paper's budgets, including the final `Q=1000` pendigits runs.

**Note.** Separate directories hold runs at different budgets. Where a dataset and model appears in
two of them, the `q` field in each JSON identifies the budget, and the paper's is the larger one.
