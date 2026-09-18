"""Magnitude-level explanation of the LIME-vs-random ablation, via a boundary-CROSSING
decomposition of the LIME attack move into a feature channel and a threshold (bin-edge) channel.
See run_lime_threshold_diagnostic. Correlates each channel with the ablation across the six
combos to see WHICH channel (and confound) drives the size of the LIME advantage.
Writes lime_threshold_diag.json.
"""
import json
import numpy as np
from attack_utils import run_lime_threshold_diagnostic

# (which_dataset, which_model, name, LIME-minus-random attack gain from the ablation image)
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
for wd, wm, name, abl in COMBOS:
    try:
        r = run_lime_threshold_diagnostic(wd, wm, n_eval=N_EVAL, aux_size=350, lime_num_samples=1000)
        r['ablation'] = abl
        results.append(r)
    except Exception as e:
        print(f"  [skip {name}] {type(e).__name__}: {e}")

with open('lime_threshold_diag.json', 'w') as fh:
    json.dump(results, fh, indent=2, default=str)


def corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.size < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float('nan')
    return float(np.corrcoef(a, b)[0, 1])


def spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return corr(ra, rb)


print("\n" + "=" * 96)
print("SUMMARY  (crossing rates + channel decomposition vs the ablation)")
print("=" * 96)
hdr = f"{'combo':<14}{'type':<12}{'bin':>7}{'limeN':>7}{'rand':>7}{'total':>8}{'feat_ch':>9}{'thr_ch':>8}{'abl':>9}"
print(hdr); print("-" * len(hdr))
for r in results:
    print(f"{r['dataset']+'/'+r['model']:<14}{r['feat_type']:<12}"
          f"{r['cr_lime_bin'][0]:>7.3f}{r['cr_lime_naive'][0]:>7.3f}{r['cr_rand_naive'][0]:>7.3f}"
          f"{r['total']:>+8.3f}{r['feat_ch']:>+9.3f}{r['thr_ch']:>+8.3f}{r['ablation']:>+9.4f}")

abl = [r['ablation'] for r in results]
cat = [i for i, r in enumerate(results) if r['feat_type'] == 'categorical']
print("\ncorrelation with ablation (all 6):")
for ch in ('total', 'feat_ch', 'thr_ch'):
    v = [r[ch] for r in results]
    print(f"  {ch:<8} pearson={corr(v, abl):+.2f}  spearman={spearman(v, abl):+.2f}")
print("correlation with ablation (categorical only, n=%d):" % len(cat))
for ch in ('total', 'feat_ch', 'thr_ch'):
    v = [results[i][ch] for i in cat]
    print(f"  {ch:<8} pearson={corr(v, [abl[i] for i in cat]):+.2f}  "
          f"spearman={spearman(v, [abl[i] for i in cat]):+.2f}")
print("\nWrote lime_threshold_diag.json")
