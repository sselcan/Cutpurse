#!/bin/bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
cd "$(dirname "$0")"
# the ladder borrows `default_all` from the LIME ablation, so it must run after it
while pgrep -f '_autolycus_ablation_ms.py --ds 9' > /dev/null 2>&1; do sleep 30; done
cp paper_results_q1000/autolycus_ablation_ms_ds9.json paper_results_q1000/ 2>/dev/null
python3 _ladder.py --ds 9 --q 1000 --out paper_results_q1000 > _q1000_ladder_ds9.log 2>&1
echo "LADDER Q=1000 DONE"
