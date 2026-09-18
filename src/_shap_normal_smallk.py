"""Small-batch sweep: how well does the collection normal recover the true normal g
when the batch is only 5-10 neighbors? Extends k_sweep down to 5/10 and prints the
cos(w_batch, g) locality curve.  Prob space; adult (where recovery works) + breast/mlp
for contrast.
"""
import json
from attack_utils import run_shap_normal_diagnostic

KS = (5, 10, 20, 30, 50, 100, 200)
COMBOS = [(2, 'adult', 1, 'lr'), (2, 'adult', 2, 'nb'),
          (2, 'adult', 3, 'knn'), (2, 'adult', 5, 'mlp'),
          (3, 'breast', 5, 'mlp')]

rows = []
for wd, dn, wm, mn in COMBOS:
    r = run_shap_normal_diagnostic(wd, wm, output_space='prob', k_sweep=KS,
                                   n_eval=150, aux_size=350, shap_nsamples=128, verbose=False)
    rows.append((f"{dn}/{mn}", r))

json.dump([r for _, r in rows], open('shap_normal_smallk.json', 'w'), indent=2, default=str)

hdr = "combo".ljust(14) + "single" + "".join(f"k{k}".rjust(8) for k in KS)
print(hdr)
print("-" * len(hdr))
for name, r in rows:
    ks = r['k_sweep']
    line = name.ljust(14) + f"{r['cos_single_g'][0]:>6.2f}"
    for k in KS:
        v = r['cos_batch_g'][k][0] if k in ks else float('nan')
        line += f"{v:>8.2f}"
    print(line)
print("\n(single = cos(w_single,g); kN = cos(w_batch@N, g); NaN g => not shown)")
print("Wrote shap_normal_smallk.json")
