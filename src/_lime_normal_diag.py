"""LIME-vs-SHAP boundary-normal cosine check (extends the SHAP-normal diagnostic).

Mechanism hypothesis (from the LIME-beats-random ablation): explanation-guided extraction
works with LIME because LIME's local linear coefficient recovers the boundary NORMAL, while
default SHAP (global-baseline attribution) does not. This driver tests it directly, PAIRED on
the same eval points, in PROB space (the attack's actual setup).

Headline columns (g = true finite-diff normal):
  cos(w_single, g)  — does the reconstructed SHAP normal recover g?          (expect low)
  cos(w_lime,   g)  — does the LIME normal recover g?                        (expect higher)
  cos(w_lime, w_s)  — do the LIME and SHAP normals even point the same way?
  topk_hit(g)       — fraction of top-k features that match the true-normal top-k, lime vs shap
                      (this is the feature-SELECTION quantity the +/-eps attack actually uses)

g is only defined on continuous axes, so the g-comparisons are meaningful on adult/breast;
mushroom is all-categorical (g = NaN) and is reported for completeness only.
Writes lime_normal_diag.json.
"""
import json
from attack_utils import run_shap_normal_diagnostic

# binary targets with continuous features first (g defined), then all-categorical for reference
DATASETS = {2: 'adult', 3: 'breast', 5: 'mushroom'}
MODELS = {1: 'lr', 2: 'nb', 3: 'knn', 5: 'mlp'}

N_EVAL = 80
LIME_NS = 1000

results = {'lime_vs_shap': []}

print("=" * 92)
print("LIME vs SHAP boundary-normal cosine check (prob space, paired eval points)")
print("=" * 92)
for wd, dname in DATASETS.items():
    for wm, mname in MODELS.items():
        try:
            r = run_shap_normal_diagnostic(
                wd, wm, output_space='prob', k_sweep=(30, 50, 100, 200),
                n_eval=N_EVAL, aux_size=350, shap_nsamples=128,
                include_lime=True, lime_num_samples=LIME_NS)
            results['lime_vs_shap'].append(r)
        except Exception as e:
            print(f"  [skip {dname}/{mname}] {type(e).__name__}: {e}")

with open('lime_normal_diag.json', 'w') as fh:
    json.dump(results, fh, indent=2, default=str)


def _v(m):
    return float('nan') if m is None else m[0]


print("\n" + "=" * 92)
print("SUMMARY  (mean cosine; higher = better recovery of the true boundary normal g)")
print("=" * 92)
hdr = (f"{'combo':<14}{'cos(ws,g)':>10}{'cos(wl,g)':>10}"
       f"{'cos(wl,ws)':>11}{'hit_s':>7}{'hit_l':>7}")
print(hdr)
print("-" * len(hdr))
for r in results['lime_vs_shap']:
    print(f"{r['dataset']+'/'+r['model']:<14}"
          f"{_v(r['cos_single_g']):>10.3f}{_v(r['cos_lime_g']):>10.3f}"
          f"{_v(r['cos_lime_single']):>11.3f}"
          f"{_v(r['topk_hit_shap_g']):>7.2f}{_v(r['topk_hit_lime_g']):>7.2f}")
print("\nws=SHAP single normal, wl=LIME normal, "
      "hit=top-k feature-selection agreement with g (continuous axes).")
print("Wrote lime_normal_diag.json")
