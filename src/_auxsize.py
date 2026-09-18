"""How much auxiliary data does the adversary need to reconstruct LIME's grid? (reviewer M2/M3/RE1)

The disclosure claim of the paper rests on the adversary estimating LIME's quartile bin edges from
its own auxiliary sample. That claim is conditional on the sample being large enough, and the review
correctly notes we never bound how large. This measures it directly, and costs no target queries at
all: the discretizer is fit on data, not on the model.

Two metrics, per dataset, over NSPLIT independent splits:
  edge_err  mean |q_aux - q_tgt| over the 3 cut points and all features, normalised by each
            feature's target-side interquartile range, so it is comparable across features.
  bin_agree fraction of evaluation points assigned to the SAME bin by the adversary's discretizer as
            by the target's. This is the operationally relevant quantity: the attack consumes bin
            membership, not the edge coordinate.

    python _auxsize.py --out paper_results
"""
import os, json, argparse, warnings
import numpy as np, random
import lime, lime.lime_tabular
from attack_utils import load_dataset

DATASETS = {1: 'crop', 2: 'adult', 3: 'breast', 4: 'nursery', 5: 'mushroom', 9: 'pendigits'}
N_AUX = [10, 25, 50, 100, 250, 500, None]      # None = the full auxiliary partition
NSPLIT = 10


def _disc(X):
    return lime.lime_tabular.LimeTabularExplainer(np.asarray(X, float),
                                                  discretize_continuous=True).discretizer


def _edges(d, nf):
    """(nf, 3) array of the 25/50/75 cut points, or nan where the feature was not discretized."""
    out = np.full((nf, 3), np.nan)
    for f in range(nf):
        try:
            out[f] = np.asarray(d.maxs[f][:3], float)     # upper edge of bins 0,1,2 == the 3 cuts
        except Exception:
            pass
    return out


def run(out_dir='paper_results'):
    warnings.filterwarnings('ignore')
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for ds, name in DATASETS.items():
        acc = {n: {'err': [], 'agree': []} for n in N_AUX}
        for s in range(NSPLIT):
            a1, _ = load_dataset(ds, seed=s)
            X_train, _, _, _, X_test_t, X_test_s, _, _ = a1
            Xtr = np.asarray(X_train.values, float)
            Xa = np.asarray(X_test_s.values, float)
            Xe = np.asarray(X_test_t.values, float)
            nf = Xtr.shape[1]
            d_t = _disc(Xtr); e_t = _edges(d_t, nf)
            iqr = np.abs(e_t[:, 2] - e_t[:, 0])
            iqr[~np.isfinite(iqr) | (iqr == 0)] = np.nan
            bin_t = np.column_stack([d_t.lambdas[f](Xe[:, f]) if f in d_t.lambdas
                                     else np.zeros(len(Xe)) for f in range(nf)])
            rng = np.random.RandomState(1000 + s)
            for n in N_AUX:
                sub = Xa if n is None or n >= len(Xa) else Xa[rng.choice(len(Xa), n, replace=False)]
                try:
                    d_a = _disc(sub); e_a = _edges(d_a, nf)
                except Exception:
                    continue
                acc[n]['err'].append(float(np.nanmean(np.abs(e_a - e_t) / iqr[:, None])))
                bin_a = np.column_stack([d_a.lambdas[f](Xe[:, f]) if f in d_a.lambdas
                                         else np.zeros(len(Xe)) for f in range(nf)])
                acc[n]['agree'].append(float(np.mean(bin_a == bin_t)))
        row = {'dataset': name, 'n_aux_pool': int(len(Xa)), 'nsplit': NSPLIT,
               'n_aux': ['all' if n is None else n for n in N_AUX],
               'edge_err': [round(float(np.nanmean(acc[n]['err'])), 4) for n in N_AUX],
               'bin_agree': [round(float(np.nanmean(acc[n]['agree'])), 4) for n in N_AUX]}
        rows.append(row)
        print(f"{name:10s} pool={row['n_aux_pool']:5d} "
              f"agree={[f'{x:.3f}' for x in row['bin_agree']]} "
              f"err={[f'{x:.3f}' for x in row['edge_err']]}", flush=True)
        with open(os.path.join(out_dir, 'auxsize.json'), 'w') as f:
            json.dump(rows, f, indent=2)
    return rows


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='paper_results')
    run(ap.parse_args().out)
