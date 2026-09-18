"""SHAP-normal reconstruction diagnostic (see SPEC_shap_normal_diagnostic.md).

Exp 1 — one-shot correctness: LR/NB in MARGIN space, single-sample reconstruction must
        recover the closed-form weights (cos ~= 1, max|err| ~= 0).
Exp 2 — cosine redundancy/locality: LR/NB/KNN/MLP x {adult,breast,mushroom} in PROB space
        (the attack's actual setup); cos(w_hat, g) [recovers true normal?] and
        cos(w_hat, p) [redundant with attack direction?], with a k-sweep.

Binary targets only:  datasets adult=2, breast=3, mushroom=5 ; models lr=1,nb=2,knn=3,mlp=5.
Writes shap_normal_diag.json.
"""
import json
from attack_utils import run_shap_normal_diagnostic

DATASETS = {2: 'adult', 3: 'breast', 5: 'mushroom'}
MODELS = {1: 'lr', 2: 'nb', 3: 'knn', 5: 'mlp'}

results = {'exp1_oneshot': [], 'exp2_redundancy': []}

print("=" * 78)
print("EXP 1 — one-shot correctness (margin space, LR/NB)")
print("=" * 78)
for wd, dname in DATASETS.items():
    for wm in (1, 2):  # lr, nb
        try:
            r = run_shap_normal_diagnostic(wd, wm, output_space='margin',
                                           n_eval=30, aux_size=200, shap_nsamples=500)
            results['exp1_oneshot'].append(r)
        except Exception as e:
            print(f"  [skip {dname}/{MODELS[wm]}] {type(e).__name__}: {e}")

print("\n" + "=" * 78)
print("EXP 2 — cosine redundancy / locality (prob space)")
print("=" * 78)
for wd, dname in DATASETS.items():
    for wm, mname in MODELS.items():
        try:
            r = run_shap_normal_diagnostic(wd, wm, output_space='prob',
                                           k_sweep=(30, 50, 100, 200),
                                           n_eval=150, aux_size=350, shap_nsamples=128)
            results['exp2_redundancy'].append(r)
        except Exception as e:
            print(f"  [skip {dname}/{mname}] {type(e).__name__}: {e}")

with open('shap_normal_diag.json', 'w') as fh:
    json.dump(results, fh, indent=2, default=str)

print("\n" + "=" * 78)
print("SUMMARY — Exp2  (mean cosines; g=true FD normal, p=attack top-k direction)")
print("=" * 78)
print(f"{'combo':<16}{'cos(ws,g)':>10}{'cos(wb200,g)':>14}{'cos(g,p)':>10}   (g=NaN => all-categorical)")
for r in results['exp2_redundancy']:
    kmax = max(r['k_sweep'])
    print(f"{r['dataset']+'/'+r['model']:<16}"
          f"{r['cos_single_g'][0]:>10.3f}{r['cos_batch_g'][kmax][0]:>14.3f}"
          f"{r['cos_g_p'][0]:>10.3f}")
print("\nWrote shap_normal_diag.json")
