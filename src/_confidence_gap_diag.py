"""Reproduce the 'disagreed vs agreed avg TARGET confidence' table on BASE Autolycus (no diverse,
no boundary search) and compare against the two-part method SHAP3, on the SAME sample sets.

Motivation: the original observation (measured on base Autolycus) was that on some combos the
surrogate/target DISAGREEMENTS carry HIGHER target confidence than the AGREEMENTS -- a negative
(agreed - disagreed) gap = confidently-wrong interior errors that boundary search cannot fix, only
coverage can. adult/lr showed a small positive gap; nursery/lr showed a NEGATIVE gap (the coverage
signature). Question: does SHAP3 (diverse + boundary) shrink disagreed confidence / turn the gap
positive? Writes confidence_gap_diag.json.
"""
import json
from attack_utils import run_confidence_gap_diagnostic

# combos from the reported tables (both on LR): adult (small +gap) and nursery (negative gap).
COMBOS = [
    (2, 1, 'adult/lr'),
    (4, 1, 'nursery/lr'),
]

HOW_MANY_SETS = 10
QUERY_LIMIT = 500
NFE = 3
SET_SIZE = 5

results = []
print("=" * 90)
print("Disagreed vs agreed TARGET confidence: BASE Autolycus vs SHAP3 (same sample sets)")
print("=" * 90)
for wd, wm, name in COMBOS:
    try:
        r = run_confidence_gap_diagnostic(wd, wm, how_many_sets=HOW_MANY_SETS,
                                          query_limit=QUERY_LIMIT, nfe=NFE, set_size=SET_SIZE)
        r['combo'] = name
        results.append(r)
    except Exception as e:
        print(f"  [skip {name}] {type(e).__name__}: {e}")

with open('confidence_gap_diag.json', 'w') as fh:
    json.dump(results, fh, indent=2, default=str)

print("\n" + "=" * 90)
print("SUMMARY  (avg over sets; gap = agreed - disagreed; negative = coverage signature)")
print("=" * 90)
hdr = (f"{'combo':<14}{'method':<7}{'disagreed':>11}{'agreed':>9}{'gap':>9}"
       f"{'sim':>8}{'neg-gap':>9}")
print(hdr)
print("-" * len(hdr))
for r in results:
    for m in ('base', 'shap3'):
        a = r['methods'][m]['avg']
        print(f"{r['combo']:<14}{m:<7}{a['disagreed_conf']:>11.4f}{a['agreed_conf']:>9.4f}"
              f"{a['gap']:>+9.4f}{a['similarity']:>8.4f}{a['n_neg_gap']:>6}/{HOW_MANY_SETS}")
    d = r.get('shap3_minus_base', {})
    if d:
        print(f"{'  -> SHAP3-base':<21}{d['disagreed_conf']:>+11.4f}{'':>9}{d['gap']:>+9.4f}"
              f"{d['similarity']:>+8.4f}")
print("\nIf SHAP3 raises the gap (less negative / more positive) and/or lifts similarity, the")
print("two-part method is converting confident interior errors into boundary errors it can fix.")
print("Wrote confidence_gap_diag.json")
