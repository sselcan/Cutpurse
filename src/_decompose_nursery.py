"""Which half of SHAP3 fixes the nursery/lr coverage gap: Phase-1 diverse or Phase-2 boundary
bisection? Runs three arms on the SAME sample sets, across 2 seeds:

  base         : neither (base Autolycus, traverse_explanations_SHAP)
  shap3_nodiv  : boundary bisection only (SHAP3 use_diverse=False)
  shap3        : diverse + bisection (full SHAP3)

Attribution:  bisection contribution = shap3_nodiv - base ; diverse contribution = shap3 - shap3_nodiv
(on both the agreed-minus-disagreed confidence GAP and on fidelity). Writes decompose_nursery.json.
"""
import json
from attack_utils import run_confidence_gap_diagnostic

WD, WM, NAME = 4, 1, 'nursery/lr'
SEEDS = [0, 1]
METHODS = ('base', 'shap3_nodiv', 'shap3')
HOW_MANY_SETS = 10

per_seed = []
for sd in SEEDS:
    r = run_confidence_gap_diagnostic(WD, WM, how_many_sets=HOW_MANY_SETS, query_limit=500,
                                      nfe=3, set_size=5, seed=sd, methods=METHODS, verbose=False)
    per_seed.append({'seed': sd, 'avg': {m: r['methods'][m]['avg'] for m in METHODS}})

with open('decompose_nursery.json', 'w') as fh:
    json.dump(per_seed, fh, indent=2, default=str)

print("=" * 84)
print(f"{NAME}: diverse vs boundary decomposition  (gap = agreed - disagreed; higher=better)")
print("=" * 84)
print(f"{'seed':>4}{'arm':<14}{'gap':>9}{'fidelity':>10}{'disagreed':>11}{'neg-sets':>10}")
print("-" * 58)
for ps in per_seed:
    for m in METHODS:
        a = ps['avg'][m]
        print(f"{ps['seed']:>4}{m:<14}{a['gap']:>+9.4f}{a['similarity']:>10.4f}"
              f"{a['disagreed_conf']:>11.4f}{a['n_neg_gap']:>7}/10")
    b, nd, s = ps['avg']['base'], ps['avg']['shap3_nodiv'], ps['avg']['shap3']
    print(f"    {'-> bisection':<14}{nd['gap']-b['gap']:>+9.4f}{nd['similarity']-b['similarity']:>+10.4f}"
          f"{'':>11}   (nodiv-base)")
    print(f"    {'-> diverse':<14}{s['gap']-nd['gap']:>+9.4f}{s['similarity']-nd['similarity']:>+10.4f}"
          f"{'':>11}   (shap3-nodiv)")
    print("-" * 58)

print("\nWhichever contribution (bisection vs diverse) carries the gap-flip / fidelity lift is the")
print("half doing the coverage fix. Large 'diverse' -> Goal 2; large 'bisection' -> Goal 1.")
print("Wrote decompose_nursery.json")
