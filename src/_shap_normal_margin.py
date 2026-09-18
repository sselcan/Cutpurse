"""Margin/log-odds SHAP vs probability SHAP: does explaining the *margin* fix the breast
failure? Runs output_space='margin' on the continuous datasets (breast, adult) x LR/NB/MLP
and prints cos(w_batch@k, g) next to the probability-space numbers.
"""
import json
from attack_utils import run_shap_normal_diagnostic

KS = (5, 10, 30, 50, 100, 200)
COMBOS = [(3, 'breast', 1, 'lr'), (3, 'breast', 2, 'nb'), (3, 'breast', 5, 'mlp'),
          (2, 'adult', 1, 'lr'), (2, 'adult', 2, 'nb'), (2, 'adult', 5, 'mlp')]

margin = {}
for wd, dn, wm, mn in COMBOS:
    r = run_shap_normal_diagnostic(wd, wm, output_space='margin', k_sweep=KS,
                                   n_eval=150, aux_size=350, shap_nsamples=200, verbose=False)
    margin[f"{dn}/{mn}"] = r
json.dump(margin, open('shap_normal_margin.json', 'w'), indent=2, default=str)

# probability-space numbers for the same combos (from the main diagnostic run)
prob = {f"{r['dataset']}/{r['model']}": r for r in json.load(open('shap_normal_diag.json'))['exp2_redundancy']}

print("MARGIN vs PROBABILITY SHAP  —  cos(w_batch@k, g)  [recovery of true normal]")
print(f"{'combo':<12}{'space':<7}{'single':>7}" + "".join(f'k{k}'.rjust(7) for k in KS)
      + f"{'cos(g,p)':>10}{'1shot':>7}")
print("-" * 92)
for _, dn, _, mn in COMBOS:
    key = f"{dn}/{mn}"
    for space, r in [('margin', margin.get(key)), ('prob', prob.get(key))]:
        if r is None:
            print(f"{key:<12}{space:<7}   (n/a)"); continue
        d = r['cos_batch_g']
        def _cg(k):
            v = d.get(k, d.get(str(k)))
            return v[0] if v is not None else float('nan')
        cells = "".join(f"{_cg(k):>7.2f}" for k in KS)
        os = r.get('oneshot_cos_w_true')
        os = f"{os[0]:>7.3f}" if os else "     - "
        print(f"{key:<12}{space:<7}{r['cos_single_g'][0]:>7.2f}{cells}{r['cos_g_p'][0]:>10.3f}{os}")
    print()
print("Wrote shap_normal_margin.json")
