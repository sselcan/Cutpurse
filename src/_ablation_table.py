"""Aggregate the base-Autolycus 2x2 LIME ablation (+ SHAP 1x2) into one table.
Reads paper_results/autolycus_ablation_ds*.json and autolycus_ablation_shap_ds*.json.

LIME 2x2 deltas (top budget, default arm minus the ablated arm):
  attr_delta_top   = default - randfeat   (LIME feature selection value)
  thr_delta_top    = default - nothresh   (BIN-EDGE / threshold value)  <-- the thesis
  total_delta_top  = default - randnobin  (whole explanation vs no-explanation)
SHAP: attr_delta_top = shap - random (SHAP feature selection value; no threshold channel).

    python _ablation_table.py
"""
import os, json, glob

RES = os.path.join(os.path.dirname(__file__), 'paper_results')
# continuous = has real-valued features -> LIME bins are meaningful
CONTINUOUS = {'iris', 'crop', 'breast', 'digits', 'wine', 'pendigits', 'letter'}
CATEG = {'nursery', 'mushroom'}   # one-hot/label-coded, no bins to snap to
# adult = mixed


def star(p):
    if p is None:
        return '   '
    return '***' if p < .001 else '** ' if p < .01 else '*  ' if p < .05 else '   '


def load(pattern):
    out = {}
    for f in sorted(glob.glob(os.path.join(RES, pattern))):
        for r in json.load(open(f)):
            if 'error' in r:
                continue
            out[(r['dataset'], r['model'])] = r
    return out


def kind(ds):
    return 'cont' if ds in CONTINUOUS else 'categ' if ds in CATEG else 'mixed'


def main():
    lime = load('autolycus_ablation_ds*.json')
    shap = load('autolycus_ablation_shap_ds*.json')
    keys = sorted(set(lime) | set(shap), key=lambda k: (kind(k[0]) != 'cont', k[0], k[1]))
    print(f"{'dataset':10s} {'mdl':4s} {'type':5s} | {'LIME attr':>10s} {'LIME THR':>11s} {'LIME tot':>10s} | {'SHAP attr':>10s}")
    print('-' * 78)
    thr_cont = []
    for k in keys:
        ds, m = k
        L = lime.get(k, {})
        S = shap.get(k, {})
        at = L.get('attr_delta_top'); ap = L.get('attr_p_top')
        th = L.get('thr_delta_top');  tp = L.get('thr_p_top')
        to = L.get('total_delta_top')
        sa = S.get('attr_delta_top'); sp = S.get('attr_p_top')
        f = lambda v: '   -   ' if v is None else f'{v:+.3f}'
        print(f"{ds:10s} {m:4s} {kind(ds):5s} | {f(at):>7s}{star(ap)} {f(th):>7s}{star(tp)} {f(to):>7s}    | {f(sa):>7s}{star(sp)}")
        if kind(ds) == 'cont' and th is not None:
            thr_cont.append(th)
    print('-' * 78)
    if thr_cont:
        import statistics as st
        pos = sum(1 for x in thr_cont if x > 0.01)
        print(f"CONTINUOUS threshold effect: n={len(thr_cont)}  mean={st.mean(thr_cont):+.3f}  "
              f"median={st.median(thr_cont):+.3f}  positive(>0.01)={pos}/{len(thr_cont)}")


if __name__ == '__main__':
    main()
