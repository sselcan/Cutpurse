"""Run SHAP budget sweeps for Cutpurse and the base SHAP traversal.

The driver writes paired per-split fidelity results across the requested query
budgets. Command-line arguments select the evaluated configuration.
"""
import os, json, time, argparse, traceback
import numpy as np
from scipy.stats import wilcoxon
from attack_utils import run_attack_auto_compare

DATASETS = {0: 'iris', 1: 'crop', 2: 'adult', 3: 'breast', 4: 'nursery', 5: 'mushroom'}
MODELS   = {0: 'DT', 1: 'LR', 2: 'NB', 3: 'KNN', 4: 'RF', 5: 'P'}
DEFAULT_MODELS = [1, 2, 3, 0, 4]   # LR, NB, KNN, DT, RF


def _per_set(rtest):
    """Flatten per-set fidelity values returned by the comparison helper."""
    return [float(s) for batch in (rtest or []) for s in batch]


def run_dataset(ds, models=DEFAULT_MODELS, q_list=(100, 250, 500, 1000), hms=10, seed=0,
                nfe=3, size=5, main_variant='shap3', out_dir='.'):
    """Run the requested SHAP comparison for every model on one dataset."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f'shap_sweep_n{size}_k{nfe}_ds{ds}.json')
    q_list = list(q_list)
    rows = []
    for m in models:
        mname = MODELS.get(m, str(m)); t0 = time.time()
        try:
            main_all = {Q: [] for Q in q_list}
            base_all = {Q: [] for Q in q_list}
            for Q in q_list:
                # one budget per call so argmaxing() is a no-op; paired main/base on the same sets
                out = run_attack_auto_compare(ds, m, 1, hms, [size], [nfe], [Q], False,
                                              seed=seed, main_variant=main_variant)
                main_all[Q] = _per_set(out[1])   # ours (SHAP3, or SHAP4b for trees if main='best')
                base_all[Q] = _per_set(out[3])   # Autolycus base-SHAP

            def _pack(d):
                return {'fid_mean': [round(float(np.mean(d[Q])), 4) for Q in q_list],
                        'fid_std':  [round(float(np.std(d[Q])), 4) for Q in q_list],
                        'fid_all':  [[round(x, 4) for x in d[Q]] for Q in q_list]}

            row = {'dataset': DATASETS[ds], 'model': mname, 'ds': ds, 'm': m, 'q_cap': q_list,
                   'hms': hms, 'size': size, 'nfe': nfe, 'seed': seed, 'main_variant': main_variant,
                   'main': _pack(main_all), 'base': _pack(base_all), 'secs': round(time.time() - t0, 1)}
            L = np.array(main_all[q_list[-1]], float); B = np.array(base_all[q_list[-1]], float)
            n = min(len(L), len(B)); L, B = L[:n], B[:n]
            row['delta_top'] = round(float(L.mean() - B.mean()), 4) if n else None
            row['wilcoxon_p_top'] = (float(wilcoxon(L, B).pvalue) if (n and not np.allclose(L, B)) else 1.0)
        except Exception as e:
            traceback.print_exc()
            row = {'dataset': DATASETS[ds], 'model': mname, 'ds': ds, 'm': m,
                   'error': repr(e), 'secs': round(time.time() - t0, 1)}
        rows.append(row)
        print('ROWDONE', row['dataset'], row['model'], 'delta_top', row.get('delta_top'),
              'p', row.get('wilcoxon_p_top'), 'secs', row['secs'], flush=True)
        with open(path, 'w') as f:
            json.dump(rows, f, indent=2)
    return rows


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--ds', type=int, required=True)
    ap.add_argument('--qs', type=str, default='100,250,500,1000')
    ap.add_argument('--hms', type=int, default=10)
    ap.add_argument('--size', type=int, default=5)
    ap.add_argument('--nfe', type=int, default=3)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--main', type=str, default='shap3', choices=['shap3', 'best'])
    ap.add_argument('--models', type=str, default=','.join(map(str, DEFAULT_MODELS)))
    ap.add_argument('--out', type=str, default='paper_results')
    a = ap.parse_args()
    qs = [int(x) for x in a.qs.split(',') if x != '']
    mods = [int(x) for x in a.models.split(',') if x != '']
    run_dataset(a.ds, models=mods, q_list=qs, hms=a.hms, seed=a.seed, nfe=a.nfe, size=a.size,
                main_variant=a.main, out_dir=a.out)
