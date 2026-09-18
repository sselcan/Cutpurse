"""Does LIME's feature SELECTION beat SHAP's (and random's) at picking locally-influential
features? Mirrors the exact combos from the Autolycus LIME-vs-random ablation, so the
mechanism metric lines up row-for-row with the attack outcome.

Hypothesis (after the geometry check came back null): the LIME-beats-random effect is a
SELECTION effect on CATEGORICAL data — LIME picks features that actually control the local
prediction; SHAP-magnitude ties random-ish. Expect: hit_lime > hit_shap, gap largest on the
categorical combos (nursery, mushroom), ~nil on continuous crop — tracking the ablation diff.
Writes lime_selection_diag.json.
"""
import json
from attack_utils import run_lime_shap_selection_diagnostic

# (which_dataset, which_model, name, LIME-minus-random from the ablation image)
COMBOS = [
    (1, 1, 'crop/lr',     -0.0042),
    (2, 1, 'adult/lr',    +0.0078),
    (4, 2, 'nursery/nb',  +0.0191),
    (4, 1, 'nursery/lr',  +0.0198),
    (5, 1, 'mushroom/lr', +0.0352),
    (4, 3, 'nursery/knn', +0.0442),
]

N_EVAL = 60

results = []
print("=" * 96)
print("LIME vs SHAP feature-SELECTION vs true local influence  (paired, class-1 proba)")
print("=" * 96)
for wd, wm, name, abl in COMBOS:
    try:
        r = run_lime_shap_selection_diagnostic(wd, wm, n_eval=N_EVAL, aux_size=350,
                                               shap_nsamples=128, lime_num_samples=1000)
        r['ablation_lime_minus_random'] = abl
        results.append(r)
    except Exception as e:
        print(f"  [skip {name}] {type(e).__name__}: {e}")

with open('lime_selection_diag.json', 'w') as fh:
    json.dump(results, fh, indent=2, default=str)

print("\n" + "=" * 96)
print("SUMMARY  (hit = top-k selection agreement with true local influence; captured = influence share)")
print("=" * 96)
hdr = (f"{'combo':<14}{'type':<12}{'hit_L':>7}{'hit_S':>7}{'hit_R':>7}"
       f"{'capt_L':>8}{'capt_S':>8}{'L-S':>7}{'abl':>9}")
print(hdr)
print("-" * len(hdr))
for r in results:
    print(f"{r['dataset']+'/'+r['model']:<14}{r['feat_type']:<12}"
          f"{r['hit_lime'][0]:>7.2f}{r['hit_shap'][0]:>7.2f}{r['hit_random_expected']:>7.2f}"
          f"{r['capt_lime'][0]:>8.2f}{r['capt_shap'][0]:>8.2f}"
          f"{r['hit_lime'][0]-r['hit_shap'][0]:>7.2f}{r['ablation_lime_minus_random']:>+9.4f}")
print("\nhit_L/S/R = LIME/SHAP/random selection agreement; L-S = LIME-minus-SHAP hit gap; "
      "abl = LIME-random attack-accuracy diff (from ablation).")
print("Wrote lime_selection_diag.json")
