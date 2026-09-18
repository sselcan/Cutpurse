#!/bin/bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cd "$(dirname "$0")"
for ds in 1 2 3 4 5 9; do
  python3 _autolycus_ablation_ms.py --ds $ds --models 4 --out paper_results_rf > _rf_ablation_ds$ds.log 2>&1 &
done
wait
echo "ALL RF ABLATION DONE"
