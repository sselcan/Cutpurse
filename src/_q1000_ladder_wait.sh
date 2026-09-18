#!/bin/bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
cd "$(dirname "$0")"
# wait on the ARTEFACT, not on a process pattern: pgrep -f also matches any shell whose
# command line happens to quote the pattern, which is what stalled the previous attempt.
until [ -s paper_results_q1000/autolycus_ablation_ms_ds9.json ]; do sleep 30; done
sleep 5
python3 _ladder.py --ds 9 --q 1000 --out paper_results_q1000 > _q1000_ladder_ds9.log 2>&1
echo "LADDER Q1000 DONE"
