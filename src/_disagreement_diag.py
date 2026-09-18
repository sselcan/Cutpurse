"""Are the surrogate/target DISAGREEMENTS coverage errors or boundary errors?

Observation being tested: points the surrogate labels differently from the target often carry
HIGH target confidence -> they sit deep in a target class region, far from the target boundary,
so boundary search cannot fix them; only better COVERAGE (diverse generation) can. This driver
runs the real SHAP3 attack per combo, fits one surrogate, and classifies each test-set
disagreement as interior(coverage) vs boundary, then reports which failure mode dominates.

Reads nothing; writes disagreement_diag.json + one scatter PNG per combo.
Verdict per combo: coverage-dominant / mixed / boundary-dominant.
"""
import json
from attack_utils import run_disagreement_diagnostic

# (which_dataset, which_model, name) -- a spread over feat_type x model family.
#  datasets: 1 crop, 2 adult, 4 nursery, 5 mushroom   models: 0 dt, 1 lr, 2 nb, 3 knn, 4 rdf
COMBOS = [
    (2, 1, 'adult/lr'),      # mixed,       smooth
    (2, 4, 'adult/rdf'),     # mixed,       tree (the one robust densification win)
    (4, 3, 'nursery/knn'),   # categorical, piecewise (biggest LIME attack effect)
    (5, 1, 'mushroom/lr'),   # categorical, smooth
    (1, 1, 'crop/lr'),       # continuous  (where explanation gain ~0)
]

QUERY_LIMIT = 500
NFE = 3
SET_SIZE = 5

results = []
print("=" * 96)
print("Disagreement geometry: interior(coverage) vs boundary  (SHAP3 surrogate, target test set)")
print("=" * 96)
for wd, wm, name in COMBOS:
    try:
        r = run_disagreement_diagnostic(wd, wm, query_limit=QUERY_LIMIT, nfe=NFE,
                                        set_size=SET_SIZE)
        r['combo'] = name
        results.append(r)
    except Exception as e:
        print(f"  [skip {name}] {type(e).__name__}: {e}")

with open('disagreement_diag.json', 'w') as fh:
    json.dump(results, fh, indent=2, default=str)

print("\n" + "=" * 96)
print("SUMMARY  (d = normalized distance to nearest opposite-class point; higher = more interior)")
print("=" * 96)
hdr = (f"{'combo':<16}{'type':<12}{'fid':>6}{'#dis':>6}"
       f"{'ag_conf':>8}{'dg_conf':>8}{'ag_d':>7}{'dg_d':>7}"
       f"{'%hiconf':>8}{'intr':>6}{'bnd':>6}{'verdict':>18}")
print(hdr)
print("-" * len(hdr))
for r in results:
    dg, ag = r['disagree'], r['agree']
    print(f"{r['combo']:<16}{r['feat_type']:<12}{r['fidelity']:>6.3f}{dg['n']:>6}"
          f"{(ag['conf_med'] or 0):>8.3f}{(dg['conf_med'] or 0):>8.3f}"
          f"{(ag['d_med'] or 0):>7.3f}{(dg['d_med'] or 0):>7.3f}"
          f"{(dg['frac_highconf'] or 0):>8.2f}{r['interior']['n']:>6}{r['boundary']['n']:>6}"
          f"{str(r['verdict']):>18}")
print("\nintr/bnd = disagreements classified interior(coverage) / boundary; "
      "verdict coverage-dominant if >=50% interior, boundary-dominant if <=25%.")
print("Wrote disagreement_diag.json")
