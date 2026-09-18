"""Final results table + budget scaling.
Main method per model: tree targets (DT, RF) -> SHAP4b(use_diverse=False) [boundary densification];
other models (LR, NB, KNN, P/MLP) -> SHAP3. Baseline = Autolycus (traverse_explanations_SHAP).
Surrogates handled by run_attack_auto_compare's _build_surrogate_and_eval (correct per model type).
Run at Q=500 and Q=1000. how_many_sets=10 (matches the existing all-combos table)."""
import json, traceback
import numpy as np
from attack_utils import run_attack_auto_compare

DATASETS = {0: 'iris', 1: 'crop', 2: 'adult', 3: 'breast', 4: 'nursery', 5: 'mushroom'}
MODELS = {0: 'DT', 1: 'LR', 2: 'NB', 3: 'KNN', 4: 'RF', 5: 'P'}
HMS, SSS, NFE = 10, [5], [3]

def flat(L):
    if L and isinstance(L[0], list):
        return [s for b in L for s in b]
    return L

for Q in [500, 1000]:
    print(f"\n########## BUDGET Q={Q} ##########", flush=True)
    rows = []
    for ds in range(6):
        for m in range(6):
            try:
                out = run_attack_auto_compare(ds, m, 1, HMS, SSS, NFE, [Q], False, main_variant='best')
                main = float(np.mean(flat(out[1])))
                base = float(np.mean(flat(out[3])))
            except Exception:
                traceback.print_exc()
                main, base = float('nan'), float('nan')
            pct = round((main - base) / base * 100, 2) if (base == base and base > 0) else None
            rows.append({'dataset': DATASETS[ds], 'model': MODELS[m],
                         'main': round(main, 4), 'baseline': round(base, 4), 'pct': pct})
            print(f"[Q={Q}] {DATASETS[ds]:8} {MODELS[m]:3} main={main:.4f} base={base:.4f} pct={pct}", flush=True)
            with open(f'final_table_Q{Q}.json', 'w') as f:
                json.dump(rows, f, indent=2)

    # pivot: % change vs baseline (tree models DT/RF use the densification method)
    print(f"\n=== FINAL TABLE Q={Q}: % change vs Autolycus (DT/RF=SHAP4b-nodiv, others=SHAP3) ===", flush=True)
    hdr = f"{'dataset':9}" + "".join(f"{MODELS[m]:>9}" for m in range(6))
    print(hdr); print('-' * len(hdr))
    grid = {(r['dataset'], r['model']): r['pct'] for r in rows}
    for ds in range(6):
        line = f"{DATASETS[ds]:9}"
        for m in range(6):
            p = grid.get((DATASETS[ds], MODELS[m]))
            line += (f"{p:>+9.2f}" if isinstance(p, (int, float)) else f"{'N/A':>9}")
        print(line, flush=True)
    print(f"done Q={Q}", flush=True)

print("\nALL DONE", flush=True)
