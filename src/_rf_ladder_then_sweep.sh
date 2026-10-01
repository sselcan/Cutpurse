#!/bin/bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
cd "$(dirname "$0")"
echo "STAGE 1: ladder, RF only (borrows the autolycus arm; 3 cheap arms per split)"
for ds in 1 2 3 4 5; do
  python3 _ladder.py --ds $ds --models 4 --out paper_results_rf_ladder > _rfladder_ds$ds.log 2>&1 &
done
python3 _ladder.py --ds 9 --models 4 --q 1000 --out paper_results_rf_ladder > _rfladder_ds9.log 2>&1 &
wait
echo "STAGE 1 DONE"
echo "STAGE 2: defense sweep, RF only (expensive: explain_instance across 4 budgets)"
for ds in 1 2 3 4 5; do
  python3 _defense_sweep.py --ds $ds --models 4 --out paper_results_rf_sweep > _rfsweep_ds$ds.log 2>&1 &
done
python3 _defense_sweep.py --ds 9 --models 4 --qlist 100,250,500,1000 --out paper_results_rf_sweep > _rfsweep_ds9.log 2>&1 &
wait
echo "STAGE 2 DONE"
