from importlib import metadata
import itertools
import lime
import lime.lime_tabular
import numpy as np
import pandas as pd
import random
# import seaborn as sns
import shap
import sklearn
import warnings
import pickle

from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
import sklearn.tree as tree
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier as dt
from sklearn.linear_model import LogisticRegression as lr
from sklearn.naive_bayes import MultinomialNB as mnb
from sklearn.neighbors import KNeighborsClassifier as knn, NearestNeighbors
from sklearn.ensemble import RandomForestClassifier as rf
from sklearn.neural_network import MLPClassifier as mlp
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import os, sys
from matplotlib import cm

#for coupla-based sampling
# from sdv.metadata import SingleTableMetadata
# from sdv.single_table import GaussianCopulaSynthesizer


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


np.set_printoptions(suppress=True)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)


def takeFourth(elem):
    return abs(elem[3])


def explanation_parser(expMap, expList, key, features):
    result = []
    for i in features:
        result = result + [[i, -1, -1, 0]]
    #result = [[0,-1,-1,0],[1,-1,-1,0],[2,-1,-1,0],[3,-1,-1,0]] 
    # Use an impossible value like -1 for no-information parts. 
    indices = []
    tmp = 0
    for i in expMap[key]:
        indices = indices + [i[0]]
    for i in expList:
        txt = i[0].split(' ')
        #print(txt)
        if len(txt) < 4:  #length of lime explanation is 7 for this particular dataset's normal samples
            if (txt[-2] == '<=') or (txt[-2] == '<'):
                result[indices[tmp]][2] = float(txt[-1])
            else:
                result[indices[tmp]][1] = float(txt[-1])
        else:
            result[indices[tmp]][1] = float(txt[0])
            result[indices[tmp]][2] = float(txt[-1])
        result[indices[tmp]][3] = round(float(i[1]), 2)
        tmp = tmp + 1
    result.sort(key=takeFourth, reverse=True)
    return result


def grid_rules(explainer, x, features, isCat=None):
    """Bin rules for `x` read straight off the explainer's discretizer -- zero model queries.

    LIME's QuartileDiscretizer is fit on the explainer's training data when the explainer is
    constructed (per-feature percentiles 25/50/75); it is a property of that data alone, not of the
    model. An adversary that builds a LimeTabularExplainer over its own auxiliary pool therefore
    already holds this grid without ever calling explain_instance -- which is what E5 measures.

    Returns the same [name, low_edge, high_edge, weight] rows as `explanation_parser`, with weight 0
    (no attribution is available, and none is used), so callers consume either source unchanged.
    """
    d = getattr(explainer, 'discretizer', None)
    rows = []
    for fi, name in enumerate(features):
        lo = hi = -1
        if d is not None and not (isCat is not None and isCat[fi]):
            try:
                b = int(d.lambdas[fi](np.array([float(x[fi])]))[0])
                lo, hi = float(d.mins[fi][b]), float(d.maxs[fi][b])
            except Exception:
                lo = hi = -1
        rows.append([name, lo, hi, 0])
    return rows


def extract_explanation_boundaries(model, explainer, n_ft):
    boundaries = np.zeros((n_ft, 3))
    sample = np.zeros(n_ft)
    for i in range(3):
        exp = explainer.explain_instance(sample, model.predict_proba, top_labels=1)
        exp_map = exp.as_map()
        key = list(exp_map.keys())[0]
        exp_list = exp.as_list(key)
        exp_parsed = explanation_parser(exp_map, exp_list, key, features)
        for j in range(n_ft):
            tmp_exp = exp_parsed[j]
            feature_index = [index for index, content in enumerate(features) if tmp_exp[0] in content][0]
            boundaries[feature_index, i] = tmp_exp[2]
            sample[feature_index] = tmp_exp[2] + 1  #0.01
    return boundaries

''' Generate sample set
@param dataset: The dataset from which the sample set is generated  
@param n_classes: Number of classes in the dataset
@param n_samples_per_class: Number of samples per class in the sample set'''
def sample_set_generation(dataset, n_classes, n_samples_per_class):  # Make sure that the dataset is sorted&balanced
    sample_set = []
    if isinstance(n_samples_per_class, list):
        lister = n_samples_per_class
    else:
        lister = np.ones((n_classes,), dtype=int) * n_samples_per_class
    for i in range(len(lister)):
        tmp = np.where(dataset[:, -1] == i)
        indices = random.sample(tmp[0].tolist(), lister[i])
        for j in indices:
            sample_set += [dataset[j][:-1]]
    return sample_set


def traverse_explanations_LIME(sample_set, explainer, model, n_visits_lb, n_visits_ub, upper_limit, n_f_e, args2,
                               feature_select='explanation', use_threshold=True, online_disc_every=None,
                               use_explanation=True, eps_per_feature=False):
    """eps_per_feature=False (default) keeps Autolycus's published behaviour: a fixed step of 1 for
    every feature, ignoring epsilon_set (see the original, utils.py:113). Set True to step by
    epsilon_set[i] instead, as the SHAP traversals and traverse_explanations_LIME3 do. That makes a
    step-matched baseline possible: without it, any LIME3-vs-LIME contrast confounds the phases with
    the step rule (30x on pendigits, 8x on crop, 1x on nursery/mushroom)."""
    if len(args2) == 10:
        classes, features, n_classes, n_features, isCat, epsilon_set, canNegative, classPossibilities, dataset_name, feature_ranges = args2
    else:
        classes, features, n_classes, n_features, isCat, epsilon_set, canNegative, classPossibilities, dataset_name = args2
    if isinstance(n_visits_lb, int):
        n_visits_lb = np.ones(len(classes)) * n_visits_lb
        n_visits_ub = np.ones(len(classes)) * n_visits_ub
    n_visits = np.zeros(len(classes))
    samples = sample_set.copy()
    init_preds = model.predict_proba(samples)
    preds = []
    visited_samples = []
    k = n_f_e  # number of features to explore
    for i in init_preds:
        preds.append(np.argmax(i))
    for i in samples:
        visited_samples += [i]  #.tolist()]
    query = 1
    epsilon = 1
    isPassed = [n_visits[i] >= n_visits_lb[i] for i in range(len(n_visits_lb))]
    while len(samples) != 0 and not all(isPassed) and not query > upper_limit:
        # 1. Print the information about the current sample
        curr = samples.pop(0)
        pred = model.predict_proba([curr])  #.astype(int)[0]
        class_index = np.argmax(pred)
        #query += 1
        # No need to further visit overly explored sample classes
        if n_visits[class_index] < n_visits_ub[class_index]:
            query += 1
            n_visits[class_index] += 1
            preds += [classes[class_index]]

            visited_samples += [curr]
            if online_disc_every and query % online_disc_every == 0 and len(visited_samples) >= 20:
                # online update: refit the discretizer on the attacker's growing query set (seeds +
                # everything queried so far) -- quartiles sharpen as the attack proceeds
                try:
                    explainer = lime.lime_tabular.LimeTabularExplainer(
                        np.array(visited_samples, float), discretize_continuous=True)
                except Exception:
                    pass
            # 2. Get the explanation about the current sample
            if use_explanation:
                exp = explainer.explain_instance(curr, model.predict_proba)  #, top_labels=1)
                exp_map = exp.as_map()
                key = list(exp_map.keys())[0]
                exp_list = exp.as_list(key)
                exp_parsed = explanation_parser(exp_map, exp_list, key, features)
            else:
                # explanation-free adversary: bin edges from its OWN discretizer, no explain_instance
                # (and hence none of LIME's ~5000 internal predict_proba calls on the target)
                exp_parsed = grid_rules(explainer, curr, features, isCat)

            # 3. Generate new samples and check if they were visited before
            tmp_exps, indices, cpys = [], [], []
            _sel = (exp_parsed[:k] if feature_select == 'explanation'      # E2/H1: top-k vs random features
                    else random.sample(exp_parsed, min(k, len(exp_parsed))))
            for row in _sel:
                tmp_exps += [row]
            for i in range(k):
                #print(tmp_exps[i][0])
                indices += [index for index, content in enumerate(features) if tmp_exps[i][0] in content]  #[0]
            for i in range(2 * k):
                cpys += [np.copy(curr)]
            for i in range(k):
                lo_e, hi_e = tmp_exps[i][1], tmp_exps[i][2]
                if use_threshold and lo_e == -1 and hi_e == -1:
                    # LIME fits a SPARSE local model (num_features=10 by default), so a feature it
                    # never scored has both edges at -1. Snapping to -1 would write an out-of-range
                    # value. This is unreachable on the explanation arm (top-k always come from the
                    # scored set) and arises only in the random-feature ablation on datasets with
                    # more features than LIME returns. We must not simply skip the perturbation:
                    # that would strip the THRESHOLD channel from the arm meant to ablate only the
                    # ATTRIBUTION channel. Instead we recover the bin the current value falls in
                    # from the discretizer itself -- the same global quantile grid LIME would have
                    # used had it reported this feature -- so the arm differs from `default` only in
                    # WHICH features are chosen.
                    try:
                        d = explainer.discretizer
                        b = int(d.lambdas[indices[i]](np.array([curr[indices[i]]]))[0])
                        lo_e, hi_e = d.mins[indices[i]][b], d.maxs[indices[i]][b]
                    except Exception:
                        lo_e = hi_e = -1          # no discretizer (categorical): fall through below
                # NB: a feature LIME *did* score may still have one edge at -1 (a one-sided rule);
                # that is Autolycus's own behaviour and is deliberately preserved. We test the EDGES,
                # not the weight: explanation_parser rounds weights to 2dp, so a genuinely scored
                # feature with |w| < 0.005 also reads as weight 0 while carrying valid edges.
                if use_threshold and not (lo_e == -1 and hi_e == -1):
                    cpys[2 * i][indices[i]] = hi_e
                    cpys[2 * i + 1][indices[i]] = lo_e
                else:                                          # E3a / no edge: step from current value
                    cpys[2 * i][indices[i]] = curr[indices[i]]
                    cpys[2 * i + 1][indices[i]] = curr[indices[i]]
            for i in range(2 * k):
                ind_i = int(i / 2)
                #tmp = (any((cpys[i]==x).all() for x in visited_samples) or any((cpys[i]==x).all() for x in samples))
                #if (tmp and (cpys[i][indices[ind_i]] >= 0)):
                if (cpys[i][indices[ind_i]] >= 0):
                    eps_i = epsilon_set[indices[ind_i]] if eps_per_feature else epsilon
                    if i % 2 == 0:
                        cpys[i][indices[ind_i]] += eps_i
                    else:
                        cpys[i][indices[ind_i]] -= eps_i
                tmp = (any((cpys[i] == x).all() for x in visited_samples) or
                       (any((cpys[i] == x).all() for x in samples)) or
                       (cpys[i][indices[ind_i]] < 0) or  #and isCat[indices[ind_i]] or
                       (cpys[i][indices[ind_i]] >= classPossibilities[indices[ind_i]]))
                if not tmp:
                    samples += [cpys[i]]  #.tolist()]
    return visited_samples, preds, query

def takeTopnSHAP(exp, top_exp):
    # print(exp)
    # print("picking top features...")
    # take only the k most important features in the explanation and make other entries zero
    k = min(np.count_nonzero(exp), top_exp) 
    sort_index = np.flip(np.argsort(abs(exp)))[:k]
    for i in range(len(exp)):
        if i not in sort_index:
                exp[i] = 0
    # print(exp)
    return exp

def takeRandomnSHAP(exp, n_exp):
    # print(exp)
    # print("picking random features...")
    # take only the k random features in the explanation and make other entries zero
    k = min(np.count_nonzero(exp), n_exp) 
    sort_index = random.sample(range(len(exp)), k)
    for i in range(len(exp)):
        if i not in sort_index:
            exp[i] = 0
    # print(exp)
    return exp

def returnAllZero(exp):
    for i in range(len(exp)):
        # print("zeroing all feature explanations...")
        exp[i] = 0
    return exp

# ...existing code...
def compute_hybrid_shap_features(explainer, model, X_train, x_q,
                                 y_train=None, feature_names=None,
                                 top_k_local=1, top_k_coverage=5, num_cover_features=4):
    """
    Produce a hybrid feature set: local top-k (by SHAP) + top features that maximize
    coverage over contrast-class instances. Also return a modified SHAP vector for the
    query with non-hybrid feature contributions zeroed out.

    Parameters
    - explainer: shap.Explainer already fitted (TreeExplainer, etc.)
    - model: trained classifier (used to get query_pred_class)
    - X_train: training set (DataFrame or ndarray) used by explainer
    - X_contrast: subset of X_train (DataFrame/ndarray) containing only contrast-class samples
    - x_q: single-row DataFrame/ndarray for query instance (shape (1, n_features))
    - y_train: optional, used to infer number of classes; can be None
    - feature_names: list/Index of feature names; if None, taken from X_train.columns when possible
    - top_k_local: number of top local features to keep
    - top_k_coverage: per-contrast-instance: number of top features that count as "covering" that instance
    - num_cover_features: how many global coverage-maximizing features to add to hybrid set

    Returns (dict):
    {
      "hybrid_indices": list of int feature indices,
      "hybrid_names": list of feature names,
      "phi_q_orig": 1D numpy array original SHAP for query (predicted class),
      "phi_q_mod": 1D numpy array modified SHAP (non-hybrid entries zeroed),
      "coverage_df": pandas DataFrame with columns ['idx','feature','count','percent'] sorted desc,
      "coverage_sets": list of sets (index -> set of covered contrast-instance indices)
    }
    """
    
    # Feature names
    if feature_names is None:
        try:
            feature_names = list(X_train.columns)
        except Exception:
            n_features_guess = np.asarray(X_train).shape[1]
            feature_names = [f"f{i}" for i in range(n_features_guess)]

    # ---- local SHAP for query ----
    # predict query class
    query_pred_class = int(np.asarray(model.predict(x_q)).ravel()[0])
    # get raw shap values for the query (explainer supports lists/arrays)
    sv_q_raw = explainer.shap_values(x_q)
    # robust extraction of 1D SHAP vector for the predicted class
    svq_arr = np.asarray(sv_q_raw)
    if svq_arr.ndim == 3:
        # common shapes: (n_samples, n_features, n_classes) or (n_classes, n_samples, n_features)
        if svq_arr.shape[0] == 1:
            # (1, n_features, n_classes)
            if svq_arr.shape[2] > query_pred_class:
                phi_q = svq_arr[0, :, query_pred_class].astype(float).flatten()
            else:
                phi_q = svq_arr[0, :, 0].astype(float).flatten()
        elif svq_arr.shape[2] == svq_arr.shape[2]:  # fallthrough - treat as (1,n_features,n_classes)
            phi_q = svq_arr.reshape(svq_arr.shape[0], -1).flatten()
        else:
            # try common alternative: (n_classes, n_samples, n_features)
            if svq_arr.shape[0] > query_pred_class and svq_arr.shape[1] == 1:
                phi_q = svq_arr[query_pred_class, 0, :].astype(float).flatten()
            else:
                phi_q = svq_arr.flatten()
    elif svq_arr.ndim == 2:
        # shapes like (1, n_features) or (n_features, n_classes) - attempt safe pick
        if svq_arr.shape[0] == 1:
            phi_q = svq_arr[0].astype(float).flatten()
        elif svq_arr.shape[1] > query_pred_class:
            # (n_features, n_classes) -> pick column
            phi_q = svq_arr[:, query_pred_class].astype(float).flatten()
        else:
            phi_q = svq_arr.flatten()
    else:
        phi_q = svq_arr.flatten()

    phi_q = np.asarray(phi_q, dtype=float).flatten()
    n_features = phi_q.shape[0]

    # Local ranking
    abs_phi_q = np.abs(phi_q)
    shap_ranking_q_idx = np.argsort(abs_phi_q)[::-1]
    local_best_feature_idx = shap_ranking_q_idx[:top_k_local].tolist()

    # Prepare contrast class set
    X_contrast = X_train[y_train != query_pred_class]

    # ---- contrast-class SHAP for coverage ----
    sv_contrast_all = explainer.shap_values(X_contrast)
    arr = np.asarray(sv_contrast_all)
    if arr.ndim != 3:
        raise ValueError(f"Unhandled SHAP array ndim for contrast set: {arr.ndim}, shape={arr.shape}")

    # infer number of classes
    n_classes = None
    if y_train is not None:
        n_classes = len(np.unique(y_train))
    else:
        # try to infer from arr
        if arr.shape[2] <= 10:  # small heuristic
            n_classes = arr.shape[2]
    contrast_class_label = 1 - int(query_pred_class) if n_classes == 2 else int(contrast_class_label) if 'contrast_class_label' in locals() else None

    # Common layouts:
    # (n_samples, n_features, n_classes)
    # (n_classes, n_samples, n_features)
    if arr.shape[1] == n_features and arr.shape[0] == X_contrast.shape[0]:
        # (n_samples, n_features, n_classes)
        shap_values_contrast = arr[:, :, query_pred_class] if arr.shape[2] > query_pred_class else arr[:, :, 0]
    elif arr.shape[0] == n_classes and arr.shape[1] == X_contrast.shape[0]:
        # (n_classes, n_samples, n_features)
        shap_values_contrast = arr[query_pred_class, :, :]
    else:
        # fallback: try to locate axis equal to n_features
        if arr.shape[2] == n_features and arr.shape[0] == X_contrast.shape[0]:
            shap_values_contrast = arr[:, :, query_pred_class]
        else:
            # final fallback: transpose if necessary
            shap_values_contrast = arr.reshape(X_contrast.shape[0], n_features, -1)[:, :, 0]

    shap_values_contrast = np.asarray(shap_values_contrast, dtype=float)
    n_instances = shap_values_contrast.shape[0]

    # Build coverage sets: feature -> set of contrast-instance indices it covers
    coverage_sets = [set() for _ in range(n_features)]
    for j in range(n_instances):
        top_feats = np.argsort(-np.abs(shap_values_contrast[j]))[:top_k_coverage]
        for i in top_feats:
            coverage_sets[i].add(j)

    coverage_counts = [len(s) for s in coverage_sets]
    coverage_percent = [c / float(max(1, n_instances)) for c in coverage_counts]

    coverage_df = pd.DataFrame({
        "idx": np.arange(n_features),
        "feature": list(feature_names),
        "count": coverage_counts,
        "percent": coverage_percent
    }).sort_values("count", ascending=False).reset_index(drop=True)

    coverage_topk_indices = coverage_df.iloc[:num_cover_features]["idx"].tolist()

    # Combine local + coverage features
    hybrid_indices = list(dict.fromkeys(list(local_best_feature_idx) + coverage_topk_indices))
    hybrid_names = [feature_names[i] for i in hybrid_indices]

    # Modify phi_q: zero out non-hybrid features
    phi_q_mod = phi_q.copy()
    hybrid_set = set(hybrid_indices)
    for i in range(len(phi_q_mod)):
        if i not in hybrid_set:
            phi_q_mod[i] = 0.0

    # return {
    #     "hybrid_indices": hybrid_indices,
    #     "hybrid_names": hybrid_names,
    #     "phi_q_orig": phi_q,
    #     "phi_q_mod": phi_q_mod,
    #     "coverage_df": coverage_df,
    #     "coverage_sets": coverage_sets,
    #     "local_best_feature_idx": local_best_feature_idx,
    #     "shap_ranking_q_idx": shap_ranking_q_idx
    # }
    return phi_q_mod

def traverse_explanations_SHAP(sample_set, explainer, model, n_visits_lb, n_visits_ub, upper_limit, n_f_e, args2,
                               model_name, X_train=None, y_train=None, explanation_type='vanilla', num_exp = 5,
                               quantile_grid=None):
    if len(args2) == 10:
        classes, features, n_classes, n_features, isCat, epsilon_set, canNegative, classPossibilities, dataset_name, feature_ranges = args2
    else:
        classes, features, n_classes, n_features, isCat, epsilon_set, canNegative, classPossibilities, dataset_name = args2
    if isinstance(n_visits_lb, int):
        n_v_lb = np.ones(len(classes)) * n_visits_lb
        n_v_ub = np.ones(len(classes)) * n_visits_ub
    n_visits = np.zeros(len(classes))
    samples = sample_set.copy()
    init_preds = model.predict_proba(samples)
    # print("Local model created. predictions are as follows:", init_preds)
    preds = []
    visited_samples = []
    for i in init_preds:
        preds.append(np.argmax(i))
    for i in samples:
        visited_samples += [i]
    query = 1
    isPassed = [n_visits[i] >= n_v_lb[i] for i in range(len(n_v_lb))]
    while len(samples) != 0 and not all(isPassed) and not query > upper_limit:
        # 1. Print the information about the current sample
        query += 1
        curr = samples.pop(0)
        pred = model.predict_proba([curr])  #.astype(int)[0]
        class_index = np.argmax(pred)
        if query % 100 == 0:
            print(int(query / 100), end=" ")
        # No need to further visit overly explored sample classes
        if n_visits[class_index] < n_v_ub[class_index]:
            #query += 1
            n_visits[class_index] += 1
            preds += [classes[class_index]]
            visited_samples += [curr]
            # 2. Get the SHAP explanation
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                if model_name == 'dt' or model_name == 'rdf':
                    exp = explainer.shap_values(curr)[:, class_index]
                else:
                    exp = explainer.shap_values(curr)
            if explanation_type == 'hybrid': #my method
                exp = compute_hybrid_shap_features(explainer, model,
                                                         X_train=X_train,
                                                         x_q=curr.reshape(1, -1),
                                                         y_train=y_train,
                                                         feature_names=features,
                                                         top_k_local=0,
                                                         top_k_coverage=5,
                                                         num_cover_features=5)
            elif explanation_type == 'topk':
                        exp = takeTopnSHAP(exp, num_exp)    #return shap values of top-k important features
            elif explanation_type == 'random':
                        exp = takeRandomnSHAP(exp, num_exp)   #return shap values of k random features
            elif explanation_type == 'zero':
                        exp = returnAllZero(exp)    #return all-zero explanation
            elif explanation_type == 'vanilla' or "test":
                pass  # use original explanation
            else:
                raise ValueError(f"Unknown explanation_type: {explanation_type}")

            k = min(np.count_nonzero(exp), n_f_e)  #k = n_f_e
            sort_index = np.flip(np.argsort(abs(exp)))[:k]
            cpys = []
            oldOption = False
            if quantile_grid is not None:
                # SHAP + quantile grid (diagnostic): mirror base-LIME's mechanism EXACTLY -- snap each
                # top-k SHAP feature to its bracketing quartile edge, nudge +/-1 across it, same bounds
                # -- but the grid is built from the ATTACKER's data (aux) or the TARGET's (tgt), not from
                # the target's LIME discretization. Tests whether LIME's threshold gain is a self-
                # computable data-quantile grid. Feature choice is SHAP's top-k; categoricals aren't
                # discretized (LIME doesn't either).
                for _ in range(2 * k):
                    cpys += [np.copy(curr)]
                for i in range(k):
                    f = sort_index[i]; v = curr[f]
                    if isCat[f]:
                        cpys[2 * i][f] = v; cpys[2 * i + 1][f] = v
                    else:
                        edges = quantile_grid[f]
                        hi = edges[edges > v]; lo = edges[edges < v]
                        cpys[2 * i][f]     = hi.min() if len(hi) else v   # snap to high bin edge
                        cpys[2 * i + 1][f] = lo.max() if len(lo) else v   # snap to low  bin edge
                for i in range(2 * k):
                    ind_i = sort_index[int(i / 2)]
                    if cpys[i][ind_i] >= 0:                               # nudge +/-1 across the edge (LIME epsilon=1)
                        cpys[i][ind_i] += 1 if i % 2 == 0 else -1
                    bad = (any((cpys[i] == x).all() for x in visited_samples) or
                           any((cpys[i] == x).all() for x in samples) or
                           (cpys[i][ind_i] < 0) or (cpys[i][ind_i] >= classPossibilities[ind_i]))
                    if not bad:
                        samples += [cpys[i]]
            elif oldOption:
                # 2.1. Old version with single feature changing
                for i in range(2 * k):
                    cpys += [np.copy(curr)]
                for i in range(k):
                    cpys[2 * i][sort_index[i]] += epsilon_set[sort_index[i]]
                    cpys[2 * i + 1][sort_index[i]] -= epsilon_set[sort_index[i]]
                for i in range(2 * k):
                    ind_i = int(i / 2)
                    tmp = (any((cpys[i] == x).all() for x in visited_samples) or
                           (any((cpys[i] == x).all() for x in samples)) or
                           (cpys[i][sort_index[ind_i]] < 0) or
                           (cpys[i][sort_index[ind_i]] >= classPossibilities[sort_index[ind_i]]))
                    if not tmp:
                        samples += [cpys[i]]
            else:
                # 2.2 New version with k features changing at the same time
                for i in range(2):
                    cpys += [np.copy(curr)]
                for i in range(k):
                    num = random.random()
                    mult = random.randint(1, 1)  # apply 1, 2 or 3 epsilons randomly
                    tmp0 = cpys[0][sort_index[i]] + epsilon_set[sort_index[i]] * mult
                    tmp1 = cpys[0][sort_index[i]] - epsilon_set[sort_index[i]] * mult
                    if not isCat[sort_index[i]]:
                        cond1 = True
                    else:
                        cond1 = (tmp0 < classPossibilities[sort_index[i]])
                    cond2 = (tmp1 >= 0)
                    if num < 0.8:
                        if cond1:
                            cpys[0][sort_index[i]] += epsilon_set[sort_index[i]] * mult
                            if cond2:
                                cpys[1][sort_index[i]] -= epsilon_set[sort_index[i]] * mult
                        else:
                            cpys[0][sort_index[i]] -= epsilon_set[sort_index[i]] * mult
                            if cond1:
                                cpys[1][sort_index[i]] += epsilon_set[sort_index[i]] * mult
                    else:
                        if cond1:
                            cpys[1][sort_index[i]] += epsilon_set[sort_index[i]] * mult
                            if cond2:
                                cpys[0][sort_index[i]] -= epsilon_set[sort_index[i]] * mult
                        else:
                            cpys[1][sort_index[i]] -= epsilon_set[sort_index[i]] * mult
                            if cond1:
                                cpys[0][sort_index[i]] += epsilon_set[sort_index[i]] * mult
                for i in range(2):
                    tmp = (any((cpys[i] == x).all() for x in visited_samples) or
                           (any((cpys[i] == x).all() for x in samples)))
                    if not tmp:
                        samples += [cpys[i]]
    # for i in range(len(visited_samples)):
    #     print("Visited sample ", i, ": ", visited_samples[i], " Predicted as class ", preds[i])
    return visited_samples, preds, query

def traverse_explanations_SHAP2(sample_set, explainer, model, n_visits_lb, n_visits_ub, upper_limit, n_f_e, args2,
                               model_name, X_train=None, y_train=None, explanation_type='vanilla', num_exp = 5):
    if len(args2) == 10:
        classes, features, n_classes, n_features, isCat, epsilon_set, canNegative, classPossibilities, dataset_name, feature_ranges = args2
    else:
        classes, features, n_classes, n_features, isCat, epsilon_set, canNegative, classPossibilities, dataset_name = args2
    if isinstance(n_visits_lb, int):
        n_v_lb = np.ones(len(classes)) * n_visits_lb
        n_v_ub = np.ones(len(classes)) * n_visits_ub
    n_visits = np.zeros(len(classes))
    samples = sample_set.copy()
    init_preds = model.predict_proba(samples)
    #create a local model using the sample set    
    print(model_name)
    preds = []
    visited_samples = []
    for i in init_preds:
        preds.append(np.argmax(i))
    for i in samples:
        visited_samples += [i]
    local_model, _ = load_model(1, sample_set, preds)
    query = 1
    isPassed = [n_visits[i] >= n_v_lb[i] for i in range(len(n_v_lb))]
    iter = 0
    # curr = samples[0]
    # samples = samples[1:]
    # while len(samples) != 0 and not all(isPassed) and not query > upper_limit and iter < len(sample_set):
    #     iter = +1
    #     query += 1
    #     pred = model.predict_proba([curr])  #.astype(int)[0]
    #     class_index = np.argmax(pred)
    #     if query % 100 == 0:
    #         print(int(query / 100), end=" ")
    #     # No need to further visit overly explored sample classes
    #     if n_visits[class_index] < n_v_ub[class_index]:
    #         #query += 1
    #         n_visits[class_index] += 1
    #         preds += [classes[class_index]]
    #         visited_samples += [curr]
    #         # 2. Get the SHAP explanation
    #         with warnings.catch_warnings():
    #             warnings.filterwarnings("ignore")
    #             if model_name == 'dt' or model_name == 'rdf':
    #                 exp = explainer.shap_values(curr)[:, class_index]
    #             else:
    #                 exp = explainer.shap_values(curr)
    #         k = min(np.count_nonzero(exp), n_f_e)  #k = n_f_e
    #         sort_index = np.flip(np.argsort(abs(exp)))[:k]
    #         cpys = []
    #         for i in range(2):
    #             cpys += [np.copy(curr)]
    #         for i in sort_index:
    #             # Determine direction: move against the SHAP value to approach the boundary
    #             # If SHAP > 0, the feature helps the current class, so we decrease/change it.
    #             direction = -1 if exp[i] > 0 else 1
                
    #             for variant_idx in range(len(cpys)):
    #                 multiplier = variant_idx*5 + 1  # Increase perturbation for the second copy
                    
    #             if isCat[i]:
    #             # Get the possible values for this categorical feature
    #                 p_values = classPossibilities[i]
                    
    #                 # If it's an integer (e.g., 3), convert it to a range (e.g., [0, 1, 2])
    #                 if isinstance(p_values, (int, np.integer)):
    #                     iterable_possibilities = range(p_values)
    #                 else:
    #                     iterable_possibilities = p_values

    #                 # Filter out the current value to find alternatives
    #                 options = [v for v in iterable_possibilities if v != curr[i]]
                    
    #                 if options:
    #                     # Choose the first alternative category
    #                     cpys[variant_idx][i] = options[0]   

    #         # Add these new samples to the set to be explored in future iterations
    #         for new_sample in cpys:
    #             samples = np.vstack([samples, new_sample])
        
    #     # Move to the next sample in the stack
    #     if len(samples) > 0:
    #         curr = samples[0]
    #         samples = samples[1:]
        
    #     iter += 1 # Fixed the "iter = +1" bug from the original code

    while len(samples) != 0 and not all(isPassed) and not query > upper_limit:
        local_model, _ = load_model(1, visited_samples, preds)
        curr = samples.pop(0) 
        local_pred = local_model.predict_proba([curr])
        if((len(visited_samples) > 100) and local_pred.any() > 0.9):
            continue
        else:
            query += 1
            pred = model.predict_proba([curr])  #.astype(int)[0]
            print("pred:", pred)
            class_index = np.argmax(pred)
            if query % 100 == 0:
                print(int(query / 100), end=" ")
            # No need to further visit overly explored sample classes
            if n_visits[class_index] < n_v_ub[class_index]:
                #query += 1
                n_visits[class_index] += 1
                preds += [classes[class_index]]
                visited_samples += [curr]
                # 2. Get the SHAP explanation
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")
                    if model_name == 'dt' or model_name == 'rdf':
                        exp = explainer.shap_values(curr)[:, class_index]
                    else:
                        exp = explainer.shap_values(curr)
                k = min(np.count_nonzero(exp), n_f_e)  #k = n_f_e
                sort_index = np.flip(np.argsort(abs(exp)))[:k]
                cpys = []
                for i in range(2):
                    cpys += [np.copy(curr)]
                for i in range(k):
                    num = random.random()
                    mult = random.randint(1, 1)  # apply 1, 2 or 3 epsilons randomly
                    tmp0 = cpys[0][sort_index[i]] + epsilon_set[sort_index[i]] * mult
                    tmp1 = cpys[0][sort_index[i]] - epsilon_set[sort_index[i]] * mult
                    if not isCat[sort_index[i]]:
                        cond1 = True
                    else:
                        cond1 = (tmp0 < classPossibilities[sort_index[i]])
                    cond2 = (tmp1 >= 0)
                    if num < 0.8:
                        if cond1:
                            cpys[0][sort_index[i]] += epsilon_set[sort_index[i]] * mult
                            if cond2:
                                cpys[1][sort_index[i]] -= epsilon_set[sort_index[i]] * mult
                        else:
                            cpys[0][sort_index[i]] -= epsilon_set[sort_index[i]] * mult
                            if cond1:
                                cpys[1][sort_index[i]] += epsilon_set[sort_index[i]] * mult
                    else:
                        if cond1:
                            cpys[1][sort_index[i]] += epsilon_set[sort_index[i]] * mult
                            if cond2:
                                cpys[0][sort_index[i]] -= epsilon_set[sort_index[i]] * mult
                        else:
                            cpys[1][sort_index[i]] -= epsilon_set[sort_index[i]] * mult
                            if cond1:
                                cpys[0][sort_index[i]] += epsilon_set[sort_index[i]] * mult
                for i in range(2):
                    tmp = (any((cpys[i] == x).all() for x in visited_samples) or
                        (any((cpys[i] == x).all() for x in samples)))
                    if not tmp:
                        samples += [cpys[i]]
    # for i in range(len(visited_samples)):
    #     print("Visited sample ", i, ": ", visited_samples[i], " Predicted as class ", preds[i])
    return visited_samples, preds, query


def create_diverse_samples_hybrid(samples, classPossibilities, isCat, feature_ranges, num_desired_samples=10, pool_size=1000, refinement_steps=2):
    current_pool = np.array(samples, dtype=float)
    n_features = len(classPossibilities)
    new_diverse_samples = []

    for _ in range(num_desired_samples):
        # PHASE 1: RANDOM SEARCH
        # Initialize as float to support continuous values
        candidates = np.zeros((pool_size, n_features), dtype=float)
        
        for i in range(n_features):
            if isCat[i]:
                candidates[:, i] = np.random.randint(0, classPossibilities[i], size=pool_size)
            else:
                low, high = feature_ranges[i]
                candidates[:, i] = np.random.uniform(low, high, size=pool_size)

        # Calculate Distance Matrix for Mixed Data
        # We calculate distances for all candidates against the current_pool
        # (Pool_Size, Current_Pool_Size)
        dist_matrix = np.zeros((pool_size, len(current_pool)))
        
        for i in range(n_features):
            col_candidates = candidates[:, i][:, np.newaxis]
            col_pool = current_pool[:, i][np.newaxis, :]
            
            if isCat[i]:
                # Hamming component for this feature
                dist_matrix += (col_candidates != col_pool).astype(float)
            else:
                # Squared Euclidean component for this feature
                dist_matrix += (col_candidates - col_pool)**2

        # Pick the candidate whose MINIMUM distance to any point in the pool is LARGEST
        min_dists = dist_matrix.min(axis=1)
        best_idx = np.argmax(min_dists)
        best_candidate = candidates[best_idx].copy()
        current_max_min_dist = min_dists[best_idx]

        # PHASE 2: LOCAL REFINEMENT (HILL CLIMBING)
        for _ in range(refinement_steps):
            improved = False
            for f_idx in range(n_features):
                original_val = best_candidate[f_idx]
                
                # Define values to test for this specific feature
                if isCat[f_idx]:
                    test_values = range(int(classPossibilities[f_idx]))
                else:
                    # For continuous, try 10 random values within its range
                    low, high = feature_ranges[f_idx]
                    test_values = np.random.uniform(low, high, size=10)

                for val in test_values:
                    if val == original_val: continue
                    
                    best_candidate[f_idx] = val
                    
                    # Calculate new distance to pool
                    if isCat[f_idx]:
                        # Quick update logic: calculate just this feature's contribution change
                        # But for simplicity, we recalculate full distance:
                        dists = np.zeros(len(current_pool))
                        for j in range(n_features):
                            if isCat[j]:
                                dists += (best_candidate[j] != current_pool[:, j])
                            else:
                                dists += (best_candidate[j] - current_pool[:, j])**2
                        new_dist = dists.min()

                    if new_dist > current_max_min_dist:
                        current_max_min_dist = new_dist
                        improved = True
                        break 
                    else:
                        best_candidate[f_idx] = original_val
            
            if not improved:
                break

        new_diverse_samples.append(best_candidate)
        current_pool = np.vstack([current_pool, best_candidate])

    return new_diverse_samples

# optimized version of create_diverse_samples using random sampling of candidates
def create_diverse_samples_optimized(samples, classPossibilities, isCat, num_desired_samples=10, candidate_pool_size=5000):
    current_pool = np.array(samples)
    new_diverse_samples = []
    
    # feature_ranges tells us the max value for each column
    n_features = len(classPossibilities)

    for _ in range(1000):
        # 1. Generate a random pool of candidates instead of ALL combinations
        candidates = np.zeros((candidate_pool_size, n_features), dtype=int)
        for i in range(n_features):
            candidates[:, i] = np.random.randint(0, classPossibilities[i], size=candidate_pool_size)

        if isCat:
            # Hamming Distance: Using broadcasting for speed
            # (candidates: P x F) != (current_pool: N x F)
            # Result is (P x N x F), sum over F to get (P x N) distance matrix
            diff = candidates[:, np.newaxis, :] != current_pool[np.newaxis, :, :]
            dist_matrix = diff.sum(axis=2)
        else:
            # Euclidean Distance for continuous
            diff = candidates[:, np.newaxis, :] - current_pool[np.newaxis, :, :]
            dist_matrix = np.linalg.norm(diff, axis=2)

        # 2. Max-Min Strategy
        # Find the distance to the NEAREST existing neighbor for each candidate
        min_dists = dist_matrix.min(axis=1)
        
        # Pick the candidate that is FARTHEST from its nearest neighbor
        best_idx = np.argmax(min_dists)
        best_candidate = candidates[best_idx]

        new_diverse_samples.append(best_candidate)
        
        # 3. Update current_pool so the next iteration accounts for this new sample
        current_pool = np.vstack([current_pool, best_candidate])

    return new_diverse_samples

# generate diverse samples for categorical datasets like nursery and mushroom
# genarate the samples using the hamming distance: furthest samples from the existing ones
def create_diverse_samples(samples, classPossibilities, isCat):
    feature_ranges = [range(count) for count in classPossibilities]
    all_combinations = np.array(list(itertools.product(*feature_ranges)))
    current_pool = np.array(samples)
    new_diverse_samples = []
    num_desired_samples = 10  # number of diverse samples to generate
    while len(new_diverse_samples) < num_desired_samples:
        if isCat:
            # Hamming Distance: Count how many encoded integers differ
            diff = all_combinations[:, np.newaxis] != current_pool
            dist_matrix = diff.sum(axis=2)
        else:
            # Euclidean Distance: Standard for continuous encoded data
            diff = all_combinations[:, np.newaxis] - current_pool
            dist_matrix = np.linalg.norm(diff, axis=2)

        # Max-Min: Find the candidate furthest from its nearest neighbor
        min_dists = dist_matrix.min(axis=1)
        best_idx = np.argmax(min_dists)
        best_candidate = all_combinations[best_idx]

        new_diverse_samples.append(best_candidate)
        
        # Update pool so the next 'diverse' sample is also far from this one
        current_pool = np.vstack([current_pool, best_candidate])

    return new_diverse_samples

# For continuous datasets, we can use a Gaussian copula to model the joint distribution and sample from it to get diverse samples that respect feature correlations.
# uses knn to select the most distinct samples from the copula-generated pool
def create_copula_diverse_samples(samples, classPossibilities, isCat, feature_ranges, num_desired_samples=10, pool_size=1000):
    metadata = SingleTableMetadata()
    column_names = [f"feat_{i}" for i in range(len(samples[0]))]
    df_real = pd.DataFrame(samples, columns=column_names)
    metadata.detect_from_dataframe(df_real)

    copula_synth = GaussianCopulaSynthesizer(metadata)
    copula_synth.fit(df_real)
    # synthetic_data_copula = copula_synth.sample(num_rows=10)
    # return [row for row in synthetic_data_copula.values]
    
    #generate 100 samples and pick the outliers from the synthetic dataset
    synthetic_data_copula = copula_synth.sample(num_rows=pool_size)
    scaler = StandardScaler()
    real_scaled = scaler.fit_transform(samples)
    synth_scaled = scaler.transform(synthetic_data_copula)

    # 2. Fit Nearest Neighbors on the REAL data
    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(real_scaled)
    # nn.fit(samples)

    # 3. Find the distance from each SYNTHETIC row to its closest REAL row
    distances, indices = nn.kneighbors(synth_scaled)
    # distances, indices = nn.kneighbors(synthetic_data_copula.values)


    # 4. Add distances to your synthetic dataframe
    synthetic_data_copula['dist_from_real'] = distances

    # 5. Pick the top 10 most "distinct" (largest distance)
    most_distinct = synthetic_data_copula.nlargest(10, 'dist_from_real')
    most_distinct = most_distinct.drop(columns=['dist_from_real'])
    # print(most_distinct.values)
    return [row for row in most_distinct.values]


# Distance-based selection method for copula-generated samples
# Uses manual distance calculation (compatible with mixed categorical/continuous data) instead of KNN
def create_copula_diverse_samples_distance_based(samples, classPossibilities, isCat, feature_ranges, num_desired_samples=10, pool_size=100):
    """
    Generate diverse samples using Gaussian Copula with distance-based selection.
    
    Uses manual distance matrix calculation (Hamming for categorical, normalized Euclidean for continuous)
    instead of KNN, allowing mixed feature type handling.
    
    @param samples: Real samples to base copula on
    @param classPossibilities: Max values/cardinalities for each feature
    @param isCat: Boolean array indicating categorical features
    @param feature_ranges: (min, max) tuples for continuous features
    @param num_desired_samples: Number of diverse samples to return
    @param pool_size: Number of synthetic samples to generate before selection
    @return: List of selected diverse samples
    """
    metadata = SingleTableMetadata()
    column_names = [f"feat_{i}" for i in range(len(samples[0]))]
    df_real = pd.DataFrame(samples, columns=column_names)
    metadata.detect_from_dataframe(df_real)

    copula_synth = GaussianCopulaSynthesizer(metadata)
    copula_synth.fit(df_real)
    
    # Generate synthetic samples from copula model
    synthetic_data = copula_synth.sample(num_rows=pool_size)
    synthetic_array = np.array(synthetic_data.values, dtype=float)
    real_array = np.array(samples, dtype=float)
    
    n_features = len(samples[0])
    selected_samples = []
    current_pool = real_array.copy()
    
    # Iteratively select most diverse samples using distance matrix approach
    for _ in range(num_desired_samples):
        # Calculate distance matrix between all synthetic samples and current pool
        dist_matrix = np.zeros((len(synthetic_array), len(current_pool)))
        
        for feat_idx in range(n_features):
            col_synthetic = synthetic_array[:, feat_idx][:, np.newaxis]  # (pool_size, 1)
            col_pool = current_pool[:, feat_idx][np.newaxis, :]          # (1, current_pool_size)
            
            if isCat[feat_idx]:
                # Hamming distance for categorical features
                dist_matrix += (col_synthetic != col_pool).astype(float)
            else:
                # Normalized squared Euclidean distance for continuous features
                low, high = feature_ranges[feat_idx]
                range_val = (high - low) if (high - low) > 0 else 1
                dist_matrix += ((col_synthetic - col_pool) / range_val)**2
        
        # Max-Min strategy: select synthetic sample farthest from its nearest neighbor
        min_dists = dist_matrix.min(axis=1)
        best_idx = np.argmax(min_dists)
        best_sample = synthetic_array[best_idx]
        
        selected_samples.append(best_sample)
        # Add to pool so next iteration considers this new sample
        current_pool = np.vstack([current_pool, best_sample])
        # Remove selected sample from synthetic pool to avoid reselection
        synthetic_array = np.delete(synthetic_array, best_idx, axis=0)
        
        if len(synthetic_array) == 0:
            break
    
    return selected_samples


# For a hybrid approach, we can start with random sampling 
# and then apply a local search (like a small mutation) 
# to further increase diversity while ensuring samples remain realistic.
def create_manifold_aware_diverse_samples_corrected(samples, classPossibilities, feature_ranges, isCat,
                                         num_desired_samples=10, pool_size=1000, mutation_rate=0.3,
                                         crossover_ratio=0.5, plausibility_percentile=90):
    # SANITIZE INPUTS: Ensure no NaNs exist in the input samples or ranges
    original_pool = np.array(samples, dtype=float)
    if np.any(np.isnan(original_pool)):
        original_pool = np.nan_to_num(original_pool)

    current_pool = original_pool.copy()
    n_features = len(classPossibilities)
    n_crossover = int(pool_size * crossover_ratio)
    n_mutation = pool_size - n_crossover
    new_diverse_samples = []

    # Precompute plausibility threshold from pairwise distances in original data
    # Candidates farther than this from all original samples are considered off-manifold
    if len(original_pool) > 1:
        orig_dists = _compute_dist_matrix(original_pool, original_pool, n_features, isCat, feature_ranges)
        np.fill_diagonal(orig_dists, np.inf)
        nn_dists = orig_dists.min(axis=1)  # nearest-neighbor distances within original data
        plausibility_threshold = np.percentile(nn_dists, plausibility_percentile)
    else:
        plausibility_threshold = np.inf

    for _ in range(num_desired_samples):
        candidates = []

        # --- Crossover candidates: combine two parents to preserve feature correlations ---
        for _ in range(n_crossover):
            idx_a, idx_b = np.random.choice(len(current_pool), size=2, replace=False) if len(current_pool) > 1 \
                else (0, 0)
            parent_a, parent_b = current_pool[idx_a], current_pool[idx_b]
            candidate = parent_a.copy()
            # For each feature, randomly pick from one parent
            mask = np.random.random(n_features) < 0.5
            for i in range(n_features):
                if mask[i]:
                    candidate[i] = parent_b[i]
            # Light mutation on a few features to add exploration
            for i in range(n_features):
                if np.random.random() < mutation_rate * 0.3:
                    candidate[i] = _mutate_feature(candidate[i], i, isCat, classPossibilities, feature_ranges)
            if not np.any(np.isnan(candidate)):
                candidates.append(candidate)

        # --- Mutation candidates: mutate a single parent ---
        for _ in range(n_mutation):
            parent = current_pool[np.random.randint(len(current_pool))]
            candidate = parent.copy()
            for i in range(n_features):
                if np.random.random() < mutation_rate:
                    candidate[i] = _mutate_feature(candidate[i], i, isCat, classPossibilities, feature_ranges)
            if not np.any(np.isnan(candidate)):
                candidates.append(candidate)

        if not candidates:
            continue

        candidates = np.array(candidates)

        # --- Plausibility filter: reject candidates too far from all original samples ---
        dist_to_original = _compute_dist_matrix(candidates, original_pool, n_features, isCat, feature_ranges)
        min_dist_to_original = dist_to_original.min(axis=1)
        plausible_mask = min_dist_to_original <= plausibility_threshold
        if not np.any(plausible_mask):
            # Fallback: keep the closest half if none pass the threshold
            cutoff = np.median(min_dist_to_original)
            plausible_mask = min_dist_to_original <= cutoff
        candidates = candidates[plausible_mask]
        min_dist_to_original = min_dist_to_original[plausible_mask]

        if len(candidates) == 0:
            continue

        # --- Diversity selection among plausible candidates ---
        dist_to_pool = _compute_dist_matrix(candidates, current_pool, n_features, isCat, feature_ranges)
        min_dist_to_pool = dist_to_pool.min(axis=1)

        # Score = diversity (distance from pool) — candidates that are plausible but cover new ground
        best_idx = np.argmax(min_dist_to_pool)
        best_candidate = candidates[best_idx]
        new_diverse_samples.append(best_candidate)
        current_pool = np.vstack([current_pool, best_candidate])

    return new_diverse_samples


def _mutate_feature(value, idx, isCat, classPossibilities, feature_ranges):
    """Mutate a single feature value."""
    if isCat[idx]:
        return np.random.randint(0, classPossibilities[idx])
    else:
        low, high = feature_ranges[idx]
        if np.isnan(low) or np.isnan(high) or low == high:
            return value
        std_dev = (high - low) * 0.1
        noise = np.random.normal(0, std_dev)
        return np.clip(value + noise, low, high)


def _compute_dist_matrix(A, B, n_features, isCat, feature_ranges):
    """Compute pairwise distance matrix between rows of A and rows of B."""
    dist_matrix = np.zeros((len(A), len(B)))
    for i in range(n_features):
        col_a = A[:, i][:, np.newaxis]
        col_b = B[:, i][np.newaxis, :]
        if isCat[i]:
            dist_matrix += (col_a != col_b).astype(float)
        else:
            low, high = feature_ranges[i]
            range_val = (high - low) if (high - low) > 0 else 1
            dist_matrix += ((col_a - col_b) / range_val) ** 2
    return dist_matrix


# For a hybrid approach, we can start with random sampling 
# and then apply a local search (like a small mutation) 
# to further increase diversity while ensuring samples remain realistic.
def create_manifold_aware_diverse_samples(samples, classPossibilities, feature_ranges, isCat, 
                                         num_desired_samples=10, pool_size=1000, mutation_rate=0.7):
    # SANITIZE INPUTS: Ensure no NaNs exist in the input samples or ranges
    current_pool = np.array(samples, dtype=float)
    if np.any(np.isnan(current_pool)):
        current_pool = np.nan_to_num(current_pool)  # Fill NaNs with 0 or mean as a fallback
        
    n_features = len(classPossibilities)
    new_diverse_samples = []

    for _ in range(num_desired_samples):
        candidates = []
        for _ in range(pool_size):
            parent = current_pool[np.random.randint(len(current_pool))] # Randomly pick a parent from the current pool
            candidate = parent.copy()
            
            for i in range(n_features):
                if np.random.random() < mutation_rate: # Mutate this feature with some probability
                    if isCat[i]:
                        candidate[i] = np.random.randint(0, classPossibilities[i]) # Random category
                    else:
                        low, high = feature_ranges[i]
                        # GUARD: If range is 0 or NaN, use a small default epsilon or skip
                        if np.isnan(low) or np.isnan(high) or low == high:
                            # If feature is constant, don't mutate it
                            continue 
                            
                        std_dev = (high - low) * 0.1 
                        noise = np.random.normal(0, std_dev)
                        candidate[i] = np.clip(candidate[i] + noise, low, high) # Ensure we stay within valid range
            
            # Final safety check for the candidate
            if not np.any(np.isnan(candidate)):
                candidates.append(candidate)
        
        if not candidates: # If all candidates were invalid, skip this iteration
            continue
            
        candidates = np.array(candidates)
        # PHASE 2: MAX-MIN DISTANCE SELECTION (Diversity Check)
        # We still want the most "diverse" of the realistic candidates
        dist_matrix = np.zeros((pool_size, len(current_pool)))
        for i in range(n_features):
            col_candidates = candidates[:, i][:, np.newaxis]
            col_pool = current_pool[:, i][np.newaxis, :]
            if isCat[i]:
                dist_matrix += (col_candidates != col_pool).astype(float)
            else:
                # Normalize continuous distance so one feature doesn't dominate
                low, high = feature_ranges[i]
                range_val = (high - low) if (high - low) > 0 else 1
                dist_matrix += ((col_candidates - col_pool) / range_val)**2
        min_dists = dist_matrix.min(axis=1)
        best_idx = np.argmax(min_dists)
        best_candidate = candidates[best_idx]
        new_diverse_samples.append(best_candidate)
        current_pool = np.vstack([current_pool, best_candidate])
    return new_diverse_samples


# SHAP-directed exploration with distance-band filtering.
# Generates candidates by walking along SHAP directions from existing samples and
# sweeping top-k important features. Filters by distance band (not too close = redundant,
# not too far = off-manifold gibberish). Selects via greedy farthest-point among survivors.
def create_shap_directed_diverse_samples(samples, classPossibilities, feature_ranges, isCat,
                                         model, explainer, num_desired_samples=10):
    samples_array = np.array(samples, dtype=float)
    if np.any(np.isnan(samples_array)):
        samples_array = np.nan_to_num(samples_array)

    n_features = len(classPossibilities)
    n_samples = len(samples_array)

    # Compute SHAP values for exploration directions
    shap_results = explainer.shap_values(samples_array)
    if not isinstance(shap_results, list):
        shap_results = [shap_results]
    shap_vals = shap_results[0]  # shape: (n_samples, n_features)

    # Feature range scales for normalization
    range_scales = np.array([
        (feature_ranges[i][1] - feature_ranges[i][0])
        if not isCat[i] and (feature_ranges[i][1] - feature_ranges[i][0]) > 0 else 1.0
        for i in range(n_features)
    ])

    # Distance band thresholds from pairwise distances in original data
    if n_samples > 1:
        orig_dists = _compute_dist_matrix(samples_array, samples_array, n_features, isCat, feature_ranges)
        np.fill_diagonal(orig_dists, np.inf)
        nn_dists = orig_dists.min(axis=1)
        d_lower = np.percentile(nn_dists, 10)    # closer than this = redundant
        d_upper = np.percentile(nn_dists, 95) * 2  # farther than this = off-manifold
    else:
        d_lower, d_upper = 0, np.inf

    candidates = []

    # --- Strategy A: SHAP-direction walk ---
    # Step from each sample along its signed SHAP vector (and its negation) at varying magnitudes.
    # Explores along directions the model cares about, staying near the manifold by construction.
    step_sizes = [0.1, 0.2, 0.4, 0.7, 1.0]
    for s_idx in range(n_samples):
        direction = shap_vals[s_idx].copy()
        # Normalize in feature-range-scaled space so no single feature dominates the step
        scaled_dir = direction / range_scales
        norm = np.linalg.norm(scaled_dir)
        if norm < 1e-8:
            continue
        scaled_dir = scaled_dir / norm
        direction_normalized = scaled_dir * range_scales  # back to feature space

        for step in step_sizes:
            for sign in [1, -1]:
                candidate = samples_array[s_idx].copy()
                for i in range(n_features):
                    if isCat[i]:
                        # Flip important categorical features at larger steps
                        feat_importance_pct = np.percentile(np.abs(shap_vals[s_idx]), 70)
                        if abs(shap_vals[s_idx][i]) > feat_importance_pct and step > 0.3:
                            candidate[i] = np.random.randint(0, classPossibilities[i])
                    else:
                        low, high = feature_ranges[i]
                        if np.isnan(low) or np.isnan(high) or low == high:
                            continue
                        candidate[i] += sign * step * direction_normalized[i]
                        candidate[i] = np.clip(candidate[i], low, high)
                if not np.any(np.isnan(candidate)):
                    candidates.append(candidate)

    # --- Strategy B: Single-feature sweep on top-k important features ---
    # For each sample, vary its most influential features individually across their range.
    # Discovers how far each important feature can stretch before leaving the manifold.
    top_k = min(5, n_features)
    alphas = [0.1, 0.3, 0.5, 0.7, 0.9]
    for s_idx in range(n_samples):
        importance = np.abs(shap_vals[s_idx])
        top_feats = np.argsort(importance)[-top_k:]
        for feat in top_feats:
            if isCat[feat]:
                for cat_val in range(classPossibilities[feat]):
                    if cat_val != samples_array[s_idx][feat]:
                        candidate = samples_array[s_idx].copy()
                        candidate[feat] = cat_val
                        candidates.append(candidate)
            else:
                low, high = feature_ranges[feat]
                if np.isnan(low) or np.isnan(high) or low == high:
                    continue
                for alpha in alphas:
                    candidate = samples_array[s_idx].copy()
                    candidate[feat] = low + alpha * (high - low)
                    candidates.append(candidate)

    if not candidates:
        return []

    candidates = np.array(candidates)

    # --- Distance-band filter ---
    # Keep candidates in [d_lower, d_upper] from original samples.
    # Too close = redundant, too far = off-manifold gibberish.
    dist_to_original = _compute_dist_matrix(candidates, samples_array, n_features, isCat, feature_ranges)
    min_dist = dist_to_original.min(axis=1)
    band_mask = (min_dist >= d_lower) & (min_dist <= d_upper)

    if np.sum(band_mask) < num_desired_samples:
        # Relax: take candidates closest to the band center
        band_center = (d_lower + d_upper) / 2
        dist_to_center = np.abs(min_dist - band_center)
        relaxed_indices = np.argsort(dist_to_center)[:max(num_desired_samples * 3, 100)]
        candidates = candidates[relaxed_indices]
    else:
        candidates = candidates[band_mask]

    if len(candidates) == 0:
        return []

    # --- Greedy farthest-point selection for diversity ---
    current_pool = samples_array.copy()
    selected = []
    remaining_mask = np.ones(len(candidates), dtype=bool)

    for _ in range(min(num_desired_samples, len(candidates))):
        remaining_indices = np.where(remaining_mask)[0]
        if len(remaining_indices) == 0:
            break
        dist_to_pool = _compute_dist_matrix(candidates[remaining_indices], current_pool,
                                            n_features, isCat, feature_ranges)
        min_dists_to_pool = dist_to_pool.min(axis=1)
        local_best = np.argmax(min_dists_to_pool)
        best_idx = remaining_indices[local_best]

        selected.append(candidates[best_idx])
        current_pool = np.vstack([current_pool, candidates[best_idx]])
        remaining_mask[best_idx] = False

    return selected


# KNN-based version of create_manifold_aware_diverse_samples
# Uses StandardScaler + NearestNeighbors for diversity selection (like create_copula_diverse_samples)
def create_manifold_aware_diverse_samples_knn(samples, classPossibilities, feature_ranges, isCat,
                                              num_desired_samples=10, pool_size=1000, mutation_rate=0.7):
    """
    Generate diverse samples using mutation-based candidate generation with KNN-based diversity selection.
    
    Similar to create_manifold_aware_diverse_samples but uses StandardScaler + NearestNeighbors
    for diversity checking (like create_copula_diverse_samples) instead of manual distance matrix.
    
    @param samples: Real samples to base mutations on
    @param classPossibilities: Max values/cardinalities for each feature
    @param feature_ranges: (min, max) tuples for continuous features
    @param isCat: Boolean array indicating categorical features
    @param num_desired_samples: Number of diverse samples to return
    @param pool_size: Number of candidate samples to generate before selection
    @param mutation_rate: Probability of mutating each feature
    @return: List of selected diverse samples
    """
    # SANITIZE INPUTS: Ensure no NaNs exist in the input samples or ranges
    current_pool = np.array(samples, dtype=float)
    if np.any(np.isnan(current_pool)):
        current_pool = np.nan_to_num(current_pool)
        
    n_features = len(classPossibilities)
    new_diverse_samples = []

    for _ in range(num_desired_samples):
        # PHASE 1: Generate candidate pool via mutations
        candidates = []
        for _ in range(pool_size):
            parent = current_pool[np.random.randint(len(current_pool))]
            candidate = parent.copy()
            
            for i in range(n_features):
                if np.random.random() < mutation_rate:
                    if isCat[i]:
                        candidate[i] = np.random.randint(0, classPossibilities[i])
                    else:
                        low, high = feature_ranges[i]
                        if np.isnan(low) or np.isnan(high) or low == high:
                            continue
                        std_dev = (high - low) * 0.1
                        noise = np.random.normal(0, std_dev)
                        candidate[i] = np.clip(candidate[i] + noise, low, high)
            
            if not np.any(np.isnan(candidate)):
                candidates.append(candidate)
        
        if not candidates:
            continue
            
        candidates = np.array(candidates)
        
        # PHASE 2: KNN-based diversity selection
        # Scale the data for fair distance computation
        scaler = StandardScaler()
        pool_scaled = scaler.fit_transform(current_pool)
        candidates_scaled = scaler.transform(candidates)
        
        # Fit Nearest Neighbors on the current pool
        nn = NearestNeighbors(n_neighbors=1)
        nn.fit(pool_scaled)
        
        # Find the distance from each candidate to its closest point in the pool
        distances, _ = nn.kneighbors(candidates_scaled)
        distances = distances.flatten()
        
        # Pick the candidate that is farthest from its nearest neighbor
        best_idx = np.argmax(distances)
        best_candidate = candidates[best_idx]
        
        new_diverse_samples.append(best_candidate)
        current_pool = np.vstack([current_pool, best_candidate])
        
    return new_diverse_samples


def find_boundary_point(target_model, x_a, x_b, iterations=10):
    """
    Finds a point close to the decision boundary between x_a and x_b
    using binary search (bisection) along the interpolation line.
    Note: uses full-vector interpolation; best for continuous features.
    """
    label_a = int(target_model.predict(x_a.reshape(1, -1))[0])
    label_b = int(target_model.predict(x_b.reshape(1, -1))[0])

    if label_a == label_b:
        raise ValueError("Points x_a and x_b must have different predicted classes.")

    low = 0.0
    high = 1.0
    boundary_sample = x_a.copy()
    query_count = 0

    for _ in range(iterations):
        mid = (low + high) / 2
        x_mid = x_a + mid * (x_b - x_a)
        current_label = int(target_model.predict(x_mid.reshape(1, -1))[0])
        query_count += 1

        if current_label == label_a:
            low = mid
            boundary_sample = x_mid
        else:
            high = mid

    return boundary_sample, query_count


def generate_tree_boundary_samples(model, seed_samples, seed_preds,
                                   max_queries=120, bisect_iterations=8, max_pairs=20):
    """
    TRA-inspired: bisects between cross-class seed pairs to find decision boundary points.
    For tree-based models (DT, RF), boundary points directly reveal axis-parallel split
    thresholds, which are far more informative for surrogate training than interior samples.

    Args:
        model: target black-box model
        seed_samples: list of seed samples (already queried)
        seed_preds: list of predicted class labels corresponding to seed_samples
        max_queries: query budget for this phase
        bisect_iterations: bisection depth per pair (precision ≈ range / 2^iterations)
        max_pairs: max cross-class pairs to process
    Returns:
        (boundary_samples, query_count)
    """
    boundary_samples = []
    query_count = 0

    cross_class_pairs = [
        (i, j)
        for i in range(len(seed_samples))
        for j in range(i + 1, len(seed_samples))
        if seed_preds[i] != seed_preds[j]
    ]

    random.shuffle(cross_class_pairs)
    for i, j in cross_class_pairs[:max_pairs]:
        if query_count + bisect_iterations > max_queries:
            break
        x_a = np.array(seed_samples[i], dtype=float)
        x_b = np.array(seed_samples[j], dtype=float)
        try:
            boundary_pt, q = find_boundary_point(model, x_a, x_b, iterations=bisect_iterations)
            query_count += q
            boundary_samples.append(boundary_pt)
        except ValueError:
            continue

    return boundary_samples, query_count

def generate_shap_informed_samples(model, samples, explainer, isCat, feature_ranges, num_new_samples=10, top_k=5):
    samples_array = np.array(samples, dtype=float)
    preds = model.predict(samples_array)
    
    shap_results = explainer.shap_values(samples_array)
    
    # Standardize to a list of arrays
    if not isinstance(shap_results, list):
        shap_results = [shap_results]

    new_samples = []
    for _ in range(num_new_samples):
        # 2. Pick two samples from different classes
        idx_a = random.choice(range(len(samples_array)))
        class_a = int(preds[idx_a])
        
        diff_indices = [i for i, p in enumerate(preds) if p != class_a]
        if not diff_indices: continue
        idx_b = random.choice(diff_indices)
        class_b = int(preds[idx_b])

        s_a, s_b = samples_array[idx_a], samples_array[idx_b]
        
        # FIX: Check if we have one array or one per class
        if len(shap_results) == 1:
            # For binary models with 1 output, class index doesn't exist in shap_results
            importance_a = np.abs(shap_results[0][idx_a])
            importance_b = np.abs(shap_results[0][idx_b])
        else:
            # For multi-output models (or predict_proba)
            importance_a = np.abs(shap_results[class_a][idx_a])
            importance_b = np.abs(shap_results[class_b][idx_b])
            
        combined_importance = importance_a + importance_b
        top_features = np.argsort(combined_importance)[-top_k:]
        
        child = s_a.copy()
        for i in range(len(s_a)):
            if i in top_features:
                if isCat[i]:
                    child[i] = random.choice([s_a[i], s_b[i]]) # Pick one of the two categories
                else:
                    child[i] = np.random.uniform(min(s_a[i], s_b[i]), max(s_a[i], s_b[i])) # Pick a random float in between
            else:
                child[i] = s_a[i] if random.random() > 0.5 else s_b[i]
        
        new_samples.append(child)
        
    return new_samples

# Use SHAP values to identify the most influential features for two samples from different classes and then create new samples by mixing those features.
# Decision boundary samples can be generated by interpolating between two samples from different classes and using SHAP values to identify which features to perturb for creating new samples that are likely near the boundary.
def generate_shap_informed_samples_new(model, samples, explainer, isCat, feature_ranges, num_new_samples=10, top_k=5, mid_log=None, use_shap=True):
    samples_array = np.array(samples, dtype=float)
    preds = model.predict_proba(samples_array)
    if use_shap:
        shap_results = explainer.shap_values(samples_array)
        # Standardize to a list of arrays
        if not isinstance(shap_results, list):
            shap_results = [shap_results]
    else:
        shap_results = None  # ablation: no SHAP -> random top-k features (below)

    top_k = min(top_k, samples_array.shape[1])

    # Pre-enumerate all cross-class pairs for diversity guarantee
    all_pairs = [(i, j) for i in range(len(samples_array))
                 for j in range(len(samples_array))
                 if preds[i].argmax() != preds[j].argmax()]
    if not all_pairs:
        return [], 0
    random.shuffle(all_pairs)
    pair_pool = list(all_pairs)

    new_samples = []
    query_count = 0
    for _ in range(num_new_samples):
        # Pick diverse cross-class pairs by exhausting the pre-shuffled pool before repeating
        if not pair_pool:
            pair_pool = list(all_pairs)
            random.shuffle(pair_pool)           ## TODO instead of all pairs create pairs ffrom the closest classes
        idx_a, idx_b = pair_pool.pop()
        class_a = int(preds[idx_a].argmax())
        class_b = int(preds[idx_b].argmax())

        # Ensure 1D float vectors regardless of how samples_array was constructed
        s_a = np.array(samples_array[idx_a], dtype=float).flatten()
        s_b = np.array(samples_array[idx_b], dtype=float).flatten()

        # Select top-k features to blend at the midpoint. SHAP-guided (default): most SHAP-important
        # features of the two parents. Ablation (use_shap=False): random top_k -- isolates whether
        # SHAP's feature choice in boundary generation actually helps.
        if use_shap:
            if len(shap_results) == 1:
                importance_a = np.abs(shap_results[0][idx_a])
                importance_b = np.abs(shap_results[0][idx_b])
            else:
                importance_a = np.abs(shap_results[class_a][idx_a])
                importance_b = np.abs(shap_results[class_b][idx_b])
            combined_importance = np.array(importance_a + importance_b, dtype=float)
            if combined_importance.ndim > 1:
                combined_importance = combined_importance.sum(axis=tuple(range(1, combined_importance.ndim)))
            combined_importance = combined_importance.flatten()
            top_features = set(int(x) for x in np.argsort(combined_importance)[-top_k:])
        else:
            top_features = set(random.sample(range(samples_array.shape[1]), min(top_k, samples_array.shape[1])))

        # Precompute midpoint for nudging and non-top interpolation
        midpoint = np.array([
            random.choice([s_a[i], s_b[i]]) if isCat[i] else 0.5 * s_a[i] + 0.5 * s_b[i]
            for i in range(len(s_a))
        ])

        child = s_a.copy()
        for i in range(len(s_a)):
            if i in top_features:
                if isCat[i]:
                    child[i] = random.choice([s_a[i], s_b[i]])
                else:
                    child[i] = midpoint[i]  # midpoint on influential features to push toward boundary
            else:
                if isCat[i]:
                    child[i] = random.choice([s_a[i], s_b[i]])
                else:
                    child[i] = np.random.uniform(min(s_a[i], s_b[i]), max(s_a[i], s_b[i]))  # random for low-impact features

        # Adaptive boundary search: bisect between anchor_a (class_a side) and
        # anchor_b (class_b side). Each iteration narrows toward the decision boundary.
        anchor_a = child.copy()
        anchor_b = s_b.copy()

        child_label = int(model.predict_proba([child])[0].argmax())
        # Note: child_label is NOT counted here. The final child of this iteration
        # will be queried again in Phase 3. Not counting child_label offsets that
        # double-count: either child_label IS the final sample (Phase 3 counts it),
        # or it's intermediate but a bisection mid that became the final child is
        # double-counted instead — the two errors cancel, keeping the total correct.

        for _ in range(8):
            mid = np.array([
                random.choice([anchor_a[i], anchor_b[i]]) if isCat[i]
                else 0.5 * anchor_a[i] + 0.5 * anchor_b[i]
                for i in range(len(anchor_a))
            ])
            mid_label = int(model.predict_proba([mid])[0].argmax())
            query_count += 1  # count all bisection intermediates
            if mid_log is not None:
                mid_log.append((mid.copy(), mid_label))  # recycle: every queried mid is free labelled data
            if mid_label != child_label:
                # label flipped — boundary is between anchor_a and mid
                anchor_b = mid
                child = mid
            else:
                # same label — boundary is between mid and anchor_b
                anchor_a = mid

        new_samples.append(child)

    return new_samples, query_count


def _feature_index(feat_name, features):
    """Map a LIME rule's feature name back to its column index, mirroring the substring match
    used in `traverse_explanations_LIME`. Returns None if unmatched."""
    for idx, content in enumerate(features):
        if feat_name in content:
            return idx
    return None


def generate_lime_informed_samples(model, samples, lime_explainer, isCat, feature_ranges,
                                   features, num_new_samples=10, top_k=5, mid_log=None,
                                   bisect_refine=True, lime_num_samples=1000,
                                   densify=0, densify_step_frac=0.03, feature_select='explanation',
                                   use_threshold=True, use_explanation=True):
    """LIME counterpart to `generate_shap_informed_samples_new` (the Phase-2 boundary search).

    Same skeleton -- cross-class pairs, pick top-k features, build a child toward the opposite
    anchor, then bisect -- but the top-k features are SNAPPED TO LIME'S BIN EDGE (the local
    discretization threshold LIME hands over for free) instead of the blind midpoint. Per the
    mechanism study, that bin edge is LIME's unique signal over SHAP: it tells the attacker WHERE
    the boundary along a feature is, which SHAP (magnitude-only) must reconstruct by bisection.

    bisect_refine=True keeps an 8-step bisection AFTER the edge snap (free-threshold init + search);
    bisect_refine=False returns the point sitting on LIME's edge (pure free threshold) so the two
    can be A/B'd. LIME's internal queries are NOT charged (explanation is bundled with the label,
    as in `traverse_explanations_LIME`); only bisection intermediates are counted in query_count.
    """
    samples_array = np.array(samples, dtype=float)
    preds = model.predict_proba(samples_array)
    n_features = samples_array.shape[1]
    top_k = min(top_k, n_features)

    all_pairs = [(i, j) for i in range(len(samples_array))
                 for j in range(len(samples_array))
                 if preds[i].argmax() != preds[j].argmax()]
    if not all_pairs:
        return [], 0
    random.shuffle(all_pairs)
    pair_pool = list(all_pairs)

    lime_cache = {}   # anchor index -> parsed LIME rules (explain_instance is the expensive call)
    new_samples = []
    query_count = 0
    for _ in range(num_new_samples):
        if not pair_pool:
            pair_pool = list(all_pairs)
            random.shuffle(pair_pool)
        idx_a, idx_b = pair_pool.pop()
        s_a = np.array(samples_array[idx_a], dtype=float).flatten()
        s_b = np.array(samples_array[idx_b], dtype=float).flatten()

        if idx_a not in lime_cache:
            if use_explanation:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    exp = lime_explainer.explain_instance(s_a, model.predict_proba,
                                                          num_features=n_features,
                                                          num_samples=lime_num_samples)
                key = list(exp.as_map().keys())[0]
                lime_cache[idx_a] = explanation_parser(exp.as_map(), exp.as_list(key), key, features)
            else:
                # explanation-free: the bin edges this phase actually consumes come from the
                # discretizer, so no explain_instance (no lime_num_samples target queries) is needed
                lime_cache[idx_a] = grid_rules(lime_explainer, s_a, features, isCat)
        parsed = lime_cache[idx_a]   # [feat_name, low_edge, high_edge, weight], sorted by |weight|

        top_feats = {}
        sel = (parsed[:top_k] if feature_select == 'explanation'
               else random.sample(parsed, min(top_k, len(parsed))))   # E2/H1: random-feature ablation
        for row in sel:
            fi = _feature_index(row[0], features)
            if fi is not None:
                top_feats[fi] = (row[1], row[2])   # (low_edge, high_edge); -1 = no info

        child = s_a.copy()
        for i in range(n_features):
            if i in top_feats:
                if isCat[i]:
                    child[i] = random.choice([s_a[i], s_b[i]])
                    continue
                low, high = top_feats[i]
                # snap to the bin edge on the side toward the opposite-class anchor b
                edge = high if s_b[i] >= s_a[i] else low
                if (not use_threshold) or edge is None or edge == -1:
                    child[i] = 0.5 * s_a[i] + 0.5 * s_b[i]   # E3a/no-threshold or no edge -> midpoint
                else:
                    child[i] = float(edge)
            else:
                if isCat[i]:
                    child[i] = random.choice([s_a[i], s_b[i]])
                else:
                    child[i] = np.random.uniform(min(s_a[i], s_b[i]), max(s_a[i], s_b[i]))

        if not bisect_refine:
            new_samples.append(child)
            continue

        # Adaptive boundary search: bisect the edge-snapped child against s_b (mirrors the SHAP
        # version). Every queried mid is recycled via mid_log as free boundary-labelled data.
        anchor_a = child.copy()
        anchor_b = s_b.copy()
        child_label = int(model.predict_proba([child])[0].argmax())
        for _ in range(8):
            mid = np.array([
                random.choice([anchor_a[i], anchor_b[i]]) if isCat[i]
                else 0.5 * anchor_a[i] + 0.5 * anchor_b[i]
                for i in range(len(anchor_a))
            ])
            mid_label = int(model.predict_proba([mid])[0].argmax())
            query_count += 1
            if mid_log is not None:
                mid_log.append((mid.copy(), mid_label))
            if mid_label != child_label:
                anchor_b = mid
                child = mid
            else:
                anchor_a = mid
        new_samples.append(child)

        # Threshold DENSIFICATION: pile target-labelled points bracketing the refined threshold on
        # the primary continuous boundary feature, so the surrogate pins the split. Uses LIME's free
        # bin edge (already localised by the bisection above); each point is a real query (counted)
        # recycled as boundary-labelled training data via mid_log.
        if densify > 0 and mid_log is not None and bisect_refine:
            fdi = next((fi for fi in top_feats if not isCat[fi]), None)
            if fdi is not None:
                lo, hi = feature_ranges[fdi]
                rng = (hi - lo) if (np.isfinite(lo) and np.isfinite(hi) and hi > lo) else (abs(child[fdi]) or 1.0)
                step = densify_step_frac * rng
                for m in range(1, densify + 1):
                    for sgn in (-1.0, 1.0):
                        pt = child.copy()
                        pt[fdi] = child[fdi] + sgn * m * step
                        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                            pt[fdi] = float(np.clip(pt[fdi], lo, hi))
                        lbl = int(model.predict_proba([pt])[0].argmax())
                        query_count += 1
                        mid_log.append((pt.copy(), lbl))

    return new_samples, query_count


def shap_guided_counterfactual_flip(model, explainer, s_a, s_b, model_name, max_flips=None, query_log=None, use_shap=True):
    """
    SHAP-guided greedy categorical counterfactual walk from s_a toward s_b.

    At each step, flips the single feature with the highest positive SHAP value
    for the current predicted class (i.e. the feature most strongly holding the
    prediction where it is) to the corresponding value from s_b. This monotonically
    weakens the current class's SHAP-attributed margin, so the target model will
    eventually flip its prediction in at most `num_differing_features` steps.

    Returns (pre_boundary, post_boundary, queries):
        pre_boundary : last sample BEFORE the prediction flipped (confident orig class)
        post_boundary: first sample AFTER the prediction flipped (confident target class)
        queries      : number of target-model predict_proba calls made
    If no flip occurs within max_flips, returns (last_sample, last_sample, queries).
    """
    current = np.asarray(s_a, dtype=float).copy()
    target  = np.asarray(s_b, dtype=float).copy()
    n_feat  = len(current)
    if max_flips is None:
        max_flips = n_feat

    orig_class = int(np.asarray(model.predict_proba([current])).ravel().argmax()) \
                 if model.predict_proba([current]).shape[-1] > 1 \
                 else int(model.predict_proba([current])[0].argmax())
    queries = 1
    flipped = set()
    pre_boundary = current.copy()
    if query_log is not None:
        query_log.append((current.copy(), orig_class))

    for _ in range(max_flips):
        # 1. Eligible features: still differ from target, not already flipped.
        candidates = [i for i in range(n_feat)
                      if i not in flipped and current[i] != target[i]]
        if not candidates:
            break

        # 2. Pick the feature to flip. SHAP-guided: feature with the strongest pull toward the current
        #    class (highest positive SHAP). Ablation (use_shap=False): random eligible feature, with NO
        #    SHAP query -- isolates whether SHAP guidance helps vs. the flip/boundary machinery itself.
        if use_shap:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                sv = explainer.shap_values(current)
            sv_arr = np.asarray(sv)
            if model_name in ('dt', 'rdf'):
                if sv_arr.ndim == 2 and sv_arr.shape[1] > orig_class:
                    sv_orig = sv_arr[:, orig_class]
                else:
                    sv_orig = sv_arr.ravel()
            else:
                if isinstance(sv, list):
                    sv_orig = np.asarray(sv[orig_class] if len(sv) > orig_class else sv[0]).ravel()
                else:
                    sv_orig = sv_arr.ravel()
            sv_orig = sv_orig.astype(float).flatten()
            if sv_orig.shape[0] != n_feat:
                sv_orig = sv_orig[:n_feat] if sv_orig.shape[0] > n_feat else np.pad(sv_orig, (0, n_feat - sv_orig.shape[0]))
            best = max(candidates, key=lambda i: sv_orig[i])
        else:
            best = random.choice(candidates)

        # 4. Execute the flip.
        pre_boundary = current.copy()
        current = current.copy()
        current[best] = target[best]
        flipped.add(best)

        # 5. Query the target model.
        new_class = int(model.predict_proba([current])[0].argmax())
        queries += 1
        if query_log is not None:
            query_log.append((current.copy(), new_class))

        if new_class != orig_class:
            # Boundary crossed. pre_boundary is confidently orig_class,
            # current is confidently new_class, and they differ in exactly
            # one feature — the feature that controls the local boundary.
            return pre_boundary, current.copy(), queries

    # No flip occurred within the budget — return the final state twice.
    return pre_boundary, current.copy(), queries


def expand_confident_descendants(model, explainer, pre_boundary, post_boundary,
                                  isCat, classPossibilities, epsilon_set,
                                  model_name, num_per_side=3, confidence_threshold=0.8,
                                  max_attempts_per_side=10, canNegative=None, query_log=None):
    """
    Given a boundary straddle pair (pre, post), walk ε-perturbations AWAY from the
    boundary on each side to produce confidently-labeled training samples.

    For each side, starting from the straddle sample, repeatedly applies a single
    ε-perturbation in the SHAP-indicated "into the class" direction (opposite the
    feature that controls the boundary), keeping each candidate only if the target
    model still predicts the expected class with probability >= confidence_threshold.

    Returns (confident_samples, confident_labels, queries).
    """
    confident_samples = []
    confident_labels  = []
    queries = 0

    for anchor, expected_class in [(pre_boundary, None), (post_boundary, None)]:
        anchor = np.asarray(anchor, dtype=float).copy()
        expected_class = int(model.predict_proba([anchor])[0].argmax())
        queries += 1
        if query_log is not None:
            query_log.append((anchor.copy(), expected_class))

        # SHAP on the anchor w.r.t. its own class tells us which features pull
        # the prediction deeper into that class. We ε-perturb those features in
        # the direction that INCREASES the SHAP-attributed margin.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            sv = explainer.shap_values(anchor)
        sv_arr = np.asarray(sv)
        if model_name in ('dt', 'rdf'):
            if sv_arr.ndim == 2 and sv_arr.shape[1] > expected_class:
                sv_own = sv_arr[:, expected_class]
            else:
                sv_own = sv_arr.ravel()
        else:
            if isinstance(sv, list):
                sv_own = np.asarray(sv[expected_class] if len(sv) > expected_class else sv[0]).ravel()
            else:
                sv_own = sv_arr.ravel()
        sv_own = sv_own.astype(float).flatten()
        if sv_own.shape[0] != len(anchor):
            sv_own = sv_own[:len(anchor)] if sv_own.shape[0] > len(anchor) \
                else np.pad(sv_own, (0, len(anchor) - sv_own.shape[0]))

        # Rank features by how strongly they already support expected_class.
        # Positive SHAP -> pushing toward expected_class -> perturbing in its
        # natural direction should deepen confidence.
        feat_order = list(np.argsort(-sv_own))  # descending

        kept = 0
        attempts = 0
        seed = anchor.copy()
        while kept < num_per_side and attempts < max_attempts_per_side:
            attempts += 1
            candidate = seed.copy()
            # Perturb the top-k most supportive features by one epsilon step each,
            # choosing the sign that moves AWAY from the boundary (positive SHAP
            # contribution increased). For categorical features we re-randomize
            # within valid categories, biased to stay away from post_boundary's value.
            k = min(3, len(feat_order))
            for i in range(k):
                fi = feat_order[i]
                if isCat[fi]:
                    # Stay at anchor's value (which is the "confident" side)
                    # but occasionally jitter to another category != the opposite side.
                    if random.random() < 0.5:
                        max_cat = int(classPossibilities[fi])
                        choices = [c for c in range(max_cat) if c != int(anchor[fi])]
                        if choices:
                            candidate[fi] = float(random.choice(choices))
                else:
                    step = epsilon_set[fi] * random.randint(1, 2)
                    # Sign: if sv_own[fi] > 0, feature value pushes toward expected_class;
                    # nudge further in whichever direction increases that contribution.
                    # Without access to gradient, use a +/- trial biased by SHAP sign.
                    direction = 1.0 if sv_own[fi] >= 0 else -1.0
                    new_val = candidate[fi] + direction * step
                    # Respect the dataset's non-negative feature constraint the
                    # same way the main SHAP3 loop does (cond2 = tmp1 >= 0).
                    # Without this guard MultinomialNB fit() crashes on datasets
                    # like mushroom / nursery that disallow negatives.
                    feature_can_neg = True if canNegative is None else bool(canNegative[fi])
                    if (not feature_can_neg) and new_val < 0:
                        # Try the opposite direction; if that still violates,
                        # leave the feature unchanged for this candidate.
                        alt_val = candidate[fi] - direction * step
                        if alt_val >= 0:
                            new_val = alt_val
                        else:
                            new_val = candidate[fi]
                    candidate[fi] = new_val

            # Confidence check against the target model.
            proba = model.predict_proba([candidate])[0]
            queries += 1
            pred = int(np.argmax(proba))
            if query_log is not None:
                query_log.append((candidate.copy(), pred))
            if pred == expected_class and float(np.max(proba)) >= confidence_threshold:
                confident_samples.append(candidate)
                confident_labels.append(expected_class)
                kept += 1
                seed = candidate  # chain: walk further away from the boundary

    return confident_samples, confident_labels, queries


def generate_counterfactual_confident_samples(model, explainer, sample_set, preds,
                                               isCat, classPossibilities, epsilon_set, model_name,
                                               num_cross_pairs=5, num_per_side=3,
                                               confidence_threshold=0.8, canNegative=None, query_log=None):
    """
    Phase 2 orchestration helper (mirroring the role that
    `generate_shap_informed_samples_new` plays for SHAP3).

    For each cross-class seed pair (capped at `num_cross_pairs`):
      1. shap_guided_counterfactual_flip  -> straddle pair (pre, post)
      2. confirm the flip actually crossed the boundary
      3. expand_confident_descendants      -> clean-label samples each side
      4. confident descendants -> TRAVERSAL QUEUE (clean-label parents, will
         spawn useful ε-children and also enter training data via the main loop)
      5. straddle pair (pre, post) -> DIRECT TRAINING DATA for trees only:
         axis-aligned split revelation; goes straight into visited_samples/preds
         with labels already known from the flip, so the main loop does NOT
         re-query them and does NOT ε-perturb them (avoids boundary noise
         cascading into ε-children).

    Returns:
        queue_samples     : list[np.ndarray]  — append to traversal queue (samples)
        direct_samples    : list[np.ndarray]  — append directly to visited_samples
        direct_labels     : list[int]         — append directly to preds
        query_count       : int               — total overhead target-model queries
    """
    overhead = 0
    queue_samples = []
    direct_samples = []
    direct_labels = []

    cross_pairs = []
    for i in range(len(sample_set)):
        for j in range(len(sample_set)):
            if preds[i] != preds[j]:
                cross_pairs.append((sample_set[i], sample_set[j]))
    random.shuffle(cross_pairs)
    cross_pairs = cross_pairs[:num_cross_pairs]

    successful_flips = 0
    for s_a, s_b in cross_pairs:
        pre, post, q1 = shap_guided_counterfactual_flip(
            model, explainer, s_a, s_b, model_name, query_log=query_log
        )
        overhead += q1

        # Verify the walk actually crossed the target's decision boundary.
        pred_pre  = int(model.predict_proba([pre])[0].argmax())
        pred_post = int(model.predict_proba([post])[0].argmax())
        overhead += 2
        if query_log is not None:
            query_log.append((np.asarray(pre, dtype=float).copy(), pred_pre))
            query_log.append((np.asarray(post, dtype=float).copy(), pred_post))
        if pred_pre == pred_post:
            continue
        successful_flips += 1

        conf, conf_labels, q2 = expand_confident_descendants(
            model, explainer, pre, post,
            isCat, classPossibilities, epsilon_set, model_name,
            num_per_side=num_per_side, confidence_threshold=confidence_threshold,
            canNegative=canNegative, query_log=query_log,
        )
        overhead += q2
        # Confident descendants -> traversal queue (clean-label, safe to ε-perturb).
        queue_samples.extend(conf)

        # Straddle pair: trees only. Direct-add to training data with the
        # labels we already know from pred_pre / pred_post. Do NOT put them
        # on the traversal queue — ε-perturbing boundary points cascades
        # label noise through their ε-children.
        if model_name in ('dt', 'rdf'):
            direct_samples.extend([pre, post])
            direct_labels.extend([pred_pre, pred_post])

    print(f"[cf+desc] {successful_flips}/{len(cross_pairs)} flips succeeded,"
          f" queue={len(queue_samples)}, direct={len(direct_samples)},"
          f" {overhead} overhead queries")
    return queue_samples, direct_samples, direct_labels, overhead


#   version 3: adding diverse and target-model-confident samples for categorical datasets (nursey and mushroom) in the beginning
def traverse_explanations_SHAP3(sample_set, explainer, model, n_visits_lb, n_visits_ub, upper_limit, n_f_e, args2,
                               model_name, X_train=None, y_train=None, explanation_type='vanilla', num_exp = 5,
                               recycle_bisection=True, use_shap_gen=True, use_shap_traverse=True,
                               use_diverse=True, n_middle=10):
    classes, features, n_classes, n_features, isCat, epsilon_set, canNegative, classPossibilities, dataset_name, feature_ranges = args2
    print("Dataset name in traverse_explanations_SHAP3:", dataset_name)
    if isinstance(n_visits_lb, int):
        n_v_lb = np.ones(len(classes)) * n_visits_lb
        n_v_ub = np.ones(len(classes)) * n_visits_ub
    n_visits = np.zeros(len(classes))
    samples = sample_set.copy()
    seed_samples = list(sample_set)  # preserve original seed for tree boundary bisection
    init_preds = model.predict_proba(samples)
    preds = []
    visited_samples = []
    for i in init_preds:
        preds.append(np.argmax(i))
    for i in samples:
        visited_samples += [i]
    query = 1
    isPassed = [n_visits[i] >= n_v_lb[i] for i in range(len(n_v_lb))]

    # print("Generating diverse samples for categorical dataset:", dataset_name)
    # for categorical datasets, create more diverse initial samples
    
    # diverse_samples = create_copula_diverse_samples(samples, classPossibilities, feature_ranges, isCat)
    # diverse_samples = create_manifold_aware_diverse_samples(samples, classPossibilities, feature_ranges, isCat)
    # diverse_samples = create_manifold_aware_diverse_samples_knn(samples, classPossibilities, feature_ranges, isCat)
    # diverse_samples = create_copula_diverse_samples_distance_based(samples, classPossibilities, isCat, feature_ranges)


    selected_diverse_samples = []
    if use_diverse:  # Phase 1 diverse generation (ablatable)
        diverse_samples = create_manifold_aware_diverse_samples_corrected(samples, classPossibilities, feature_ranges, isCat)
        while len(diverse_samples) > 0:
            current_diverse  = diverse_samples.pop(0)
            query += 1
            if(model.predict_proba([current_diverse]).max() > 0.8):  # only add confidently-classified samples
                selected_diverse_samples += [current_diverse]

    bisection_mids = [] if recycle_bisection else None
    middle_samples, shap_query_count = generate_shap_informed_samples_new(model, samples, explainer, isCat, feature_ranges, num_new_samples=n_middle, top_k=5, mid_log=bisection_mids, use_shap=use_shap_gen)
    # middle_samples = generate_shap_informed_samples(model, samples, explainer, isCat, feature_ranges, num_new_samples=10, top_k=5)

    # Track bisection overhead separately: these are real model queries but they are
    # intermediate bisection steps that don't produce training samples. Adding them to
    # `query` would starve Phase 3 of its budget. They are included in the returned
    # total so the reported query count remains accurate.
    overhead_queries = 0
    overhead_queries = shap_query_count
    samples += middle_samples
    samples += selected_diverse_samples

    # Recycle the bisection intermediates: every mid was queried (already counted in overhead) and is
    # target-labelled, so add it directly to training instead of discarding it -- boundary-concentrated,
    # clean-label, zero extra budget. Mirrors SHAP4b's train-on-all-queried. Added to visited_samples
    # (not the queue) so they are NOT ε-perturbed.
    if recycle_bisection and bisection_mids:
        for ms, mlab in bisection_mids:
            if n_visits[int(mlab)] < n_v_ub[int(mlab)]:
                n_visits[int(mlab)] += 1
                preds += [classes[int(mlab)]]
                visited_samples += [np.asarray(ms, dtype=float)]

    # Phase 2 (NEW): SHAP-guided counterfactual flip + confident descendant expansion.
    # Replaces the midpoint-bisection middle_samples block above with a principled
    # boundary search. Toggle by commenting this block in/out, same as the
    # diverse-sample and generate_shap_informed_samples_new alternatives.
    #   - shap_guided_counterfactual_flip: greedy SHAP-guided walk to the boundary
    #   - expand_confident_descendants:    ε-walk AWAY from the boundary, rejection-
    #                                       sampled on predict_proba.max() >= threshold
    #   - Confident descendants -> traversal queue (spawn ε-children).
    #   - Straddle pair (pre/post) -> DIRECT training data for trees only
    #     (dt/rdf): bypasses the ε-perturbation loop entirely so boundary noise
    #     doesn't cascade into their children.
    # queue_samples, direct_samples, direct_labels, new_overhead = generate_counterfactual_confident_samples(
    #     model, explainer, samples, preds,
    #     isCat, classPossibilities, epsilon_set, model_name,
    #     num_cross_pairs=5, num_per_side=3, confidence_threshold=0.8,
    #     canNegative=canNegative,
    # )
    # overhead_queries += new_overhead
    # samples += queue_samples
    # # Direct training-data injection (tree-only payload from the helper).
    # for ds, dl in zip(direct_samples, direct_labels):
    #     visited_samples += [ds]
    #     preds += [dl]
    #     # Respect per-class visit accounting so the main loop's n_v_ub caps
    #     # stay consistent with what actually ended up in the training set.
    #     if 0 <= int(dl) < len(n_visits):
    #         n_visits[int(dl)] += 1

    # for ds in middle_samples:
    #     dl = model.predict_proba([ds])
    #     class_index = np.argmax(dl)
    #     # Respect per-class visit accounting so the main loop's n_v_ub caps
    #     # stay consistent with what actually ended up in the training set.
    #     if n_visits[class_index] < n_v_ub[class_index]:
    #         n_visits[class_index] += 1
    #         preds += [classes[class_index]]
    #         visited_samples += [ds]

    # Phase 2.5 (TRA-inspired): for tree-based models, bisect cross-class seed pairs to
    # find exact decision boundary points. Each boundary point directly reveals an
    # axis-parallel split threshold, which is far more informative for surrogate DT/RF
    # training than interior samples found by ε-perturbation alone.
    # if model_name in ('dt', 'rdf'):
    #     seed_preds = [int(np.argmax(p)) for p in init_preds]
    #     if upper_limit > query:
    #         tree_bnd_samples, tree_bnd_queries = generate_tree_boundary_samples(
    #             model, seed_samples, seed_preds,
    #             max_queries=120, bisect_iterations=8, max_pairs=20
    #         )
    #         overhead_queries += tree_bnd_queries
    #         samples += tree_bnd_samples
    #         print(f"[Tree boundary] Added {len(tree_bnd_samples)} boundary samples ({tree_bnd_queries} overhead queries)")

   # for i in range(len(classes)):
    #     # generate target-model-confident samples near the decision boundary between class i and other classes
    #     print("Generating boundary samples for class:", classes[i])
    #     class_i_samples = [s for s, p in zip(samples, preds) if p == classes[i]]
    #     other_class_samples = [s for s, p in zip(samples, preds) if p != classes[i]]
    #     for s_a in class_i_samples:
    #         for s_b in other_class_samples:
    #             try:
    #                 # todo: increase query also in find_boundary_point
    #                 boundary_sample, query_count = find_boundary_point(model, s_a, s_b, iterations=3)
    #                 query += query_count
    #                 #if(model.predict_proba([boundary_sample]).max() > 0.8): # only add samples that are confidently classified
    #                 samples += [boundary_sample]
    #                 print("Added boundary sample between classes", classes[i], "and", "other class")
    #             except ValueError:
    #                 # Points have the same predicted class, skip
    #                 continue

    upper_limit = upper_limit - overhead_queries  # Adjust upper limit for Phase 3 to account for overhead queries in Phase 2.5
    while len(samples) != 0 and not all(isPassed) and not query > upper_limit:
        # 1. Print the information about the current sample
        query += 1
        curr = samples.pop(0)
        pred = model.predict_proba([curr])  #.astype(int)[0]
        class_index = np.argmax(pred)
        if query % 100 == 0:
            print(int(query / 100), end=" ")
        # No need to further visit overly explored sample classes
        if n_visits[class_index] < n_v_ub[class_index]:
            #query += 1
            n_visits[class_index] += 1
            preds += [classes[class_index]]
            visited_samples += [curr]
            # 2. Choose features to perturb. SHAP-guided (default): top-k by |SHAP|. Ablation
            #    (use_shap_traverse=False): random k features, no SHAP query.
            if use_shap_traverse:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")
                    if model_name == 'dt' or model_name == 'rdf':
                        exp = explainer.shap_values(curr)[:, class_index]
                    else:
                        exp = explainer.shap_values(curr)
                k = min(np.count_nonzero(exp), n_f_e)  #k = n_f_e
                sort_index = np.flip(np.argsort(abs(exp)))[:k]
            else:
                k = min(n_f_e, n_features)
                sort_index = np.array(random.sample(range(n_features), k))
            cpys = []
            oldOption = False
            if oldOption:
                # 2.1. Old version with single feature changing
                for i in range(2 * k):
                    cpys += [np.copy(curr)]
                for i in range(k):
                    cpys[2 * i][sort_index[i]] += epsilon_set[sort_index[i]]
                    cpys[2 * i + 1][sort_index[i]] -= epsilon_set[sort_index[i]]
                for i in range(2 * k):
                    ind_i = int(i / 2)
                    tmp = (any((cpys[i] == x).all() for x in visited_samples) or
                           (any((cpys[i] == x).all() for x in samples)) or
                           (cpys[i][sort_index[ind_i]] < 0) or
                           (cpys[i][sort_index[ind_i]] >= classPossibilities[sort_index[ind_i]]))
                    if not tmp:
                        samples += [cpys[i]]
            else:
                # 2.2 New version with k features changing at the same time
                for i in range(2):
                    cpys += [np.copy(curr)]
                for i in range(k):
                    num = random.random()
                    mult = random.randint(1, 1)  # apply 1, 2 or 3 epsilons randomly
                    tmp0 = cpys[0][sort_index[i]] + epsilon_set[sort_index[i]] * mult
                    tmp1 = cpys[0][sort_index[i]] - epsilon_set[sort_index[i]] * mult
                    if not isCat[sort_index[i]]:
                        cond1 = True
                    else:
                        cond1 = (tmp0 < classPossibilities[sort_index[i]])
                    cond2 = (tmp1 >= 0)
                    if num < 0.8:
                        if cond1:
                            cpys[0][sort_index[i]] += epsilon_set[sort_index[i]] * mult
                            if cond2:
                                cpys[1][sort_index[i]] -= epsilon_set[sort_index[i]] * mult
                        else:
                            cpys[0][sort_index[i]] -= epsilon_set[sort_index[i]] * mult
                            if cond1:
                                cpys[1][sort_index[i]] += epsilon_set[sort_index[i]] * mult
                    else:
                        if cond1:
                            cpys[1][sort_index[i]] += epsilon_set[sort_index[i]] * mult
                            if cond2:
                                cpys[0][sort_index[i]] -= epsilon_set[sort_index[i]] * mult
                        else:
                            cpys[1][sort_index[i]] -= epsilon_set[sort_index[i]] * mult
                            if cond1:
                                cpys[0][sort_index[i]] += epsilon_set[sort_index[i]] * mult
                for i in range(2):
                    tmp = (any((cpys[i] == x).all() for x in visited_samples) or
                           (any((cpys[i] == x).all() for x in samples)))
                    if not tmp:
                        samples += [cpys[i]]
    # for i in range(len(visited_samples)):
    #     print("Visited sample ", i, ": ", visited_samples[i], " Predicted as class ", preds[i])
    return visited_samples, preds, query + overhead_queries


def create_lime_threshold_diverse_samples(samples, explainer, model, features, isCat,
                                          classPossibilities, feature_ranges,
                                          num_desired_samples=10, top_k=5, pool_size=400,
                                          plausibility_percentile=90, lime_num_samples=1000,
                                          max_cross_feats=3, max_seeds_explained=20):
    """Decision-cell-covering diverse generation using LIME's bin-edge thresholds.

    Alternative to the geometric create_manifold_aware_diverse_samples_corrected. Instead of max-min
    Euclidean spread (model-blind), it (1) harvests the target's LIME thresholds for FREE from the
    already-queried seeds, (2) generates candidates that CROSS those thresholds into different
    decision cells (anchored on real seeds, so on-manifold), (3) plausibility-filters them, and
    (4) greedily selects the candidates that cover the most distinct (feature, threshold, side)
    configurations -- i.e. the most distinct behavioural regions of the target. Returns UNQUERIED
    candidates; the caller queries + confidence-filters them (as with the geometric generator).
    Falls back to the geometric generator if LIME surfaces no thresholds (e.g. a linear target)."""
    original_pool = np.array(samples, dtype=float)
    if np.any(np.isnan(original_pool)):
        original_pool = np.nan_to_num(original_pool)
    n_features = original_pool.shape[1]

    # 1. Harvest LIME bin-edge thresholds + per-feature importance from the seeds (free explanations)
    thresholds, importance = {}, np.zeros(n_features)
    seed_idx = list(range(len(original_pool)))
    if len(seed_idx) > max_seeds_explained:
        seed_idx = list(np.random.choice(len(original_pool), max_seeds_explained, replace=False))
    for si in seed_idx:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            exp = explainer.explain_instance(original_pool[si], model.predict_proba,
                                             num_features=n_features, num_samples=lime_num_samples)
        key = list(exp.as_map().keys())[0]
        parsed = explanation_parser(exp.as_map(), exp.as_list(key), key, features)  # [name, low, high, w]
        for row in parsed[:top_k]:
            fi = _feature_index(row[0], features)
            if fi is None:
                continue
            importance[fi] += abs(row[3])
            for edge in (row[1], row[2]):
                if edge is not None and edge != -1:
                    thresholds.setdefault(fi, set()).add(float(edge))
    thresholds = {f: sorted(v) for f, v in thresholds.items() if v}
    if not thresholds:   # no cells discovered (e.g. linear target) -> geometric fallback
        return create_manifold_aware_diverse_samples_corrected(
            samples, classPossibilities, feature_ranges, isCat, num_desired_samples=num_desired_samples)
    cross_feats = sorted(thresholds.keys(), key=lambda f: -importance[f])

    def point_cover(x):   # (feature, threshold, side) triples this point realises
        return {(f, t, int(x[f] >= t)) for f, edges in thresholds.items() for t in edges}

    def cross_value(anchor_val, f):   # a value for f in a DIFFERENT threshold interval than the anchor
        edges = thresholds[f]
        lo, hi = feature_ranges[f]
        bounds = [lo] + list(edges) + [hi]
        ai = next((bi for bi in range(len(bounds) - 1) if bounds[bi] <= anchor_val <= bounds[bi + 1]), 0)
        choices = [bi for bi in range(len(bounds) - 1) if bi != ai]
        if not choices:
            return anchor_val
        a, b = (lambda ci: (bounds[ci], bounds[ci + 1]))(random.choice(choices))
        if not (np.isfinite(a) and np.isfinite(b)) or b <= a:
            return anchor_val
        if isCat[f]:
            valid = [v for v in range(max(0, int(np.ceil(a))), min(classPossibilities[f], int(np.floor(b)) + 1))
                     if v != anchor_val]
            return random.choice(valid) if valid else anchor_val
        mid, half = 0.5 * (a + b), 0.5 * (b - a)     # aim for the interval interior (stay confident)
        return float(np.clip(mid + np.random.uniform(-0.6, 0.6) * half, a, b))

    # 2. plausibility threshold from the real data (same as the geometric generator)
    if len(original_pool) > 1:
        od = _compute_dist_matrix(original_pool, original_pool, n_features, isCat, feature_ranges)
        np.fill_diagonal(od, np.inf)
        plausibility_threshold = np.percentile(od.min(axis=1), plausibility_percentile)
    else:
        plausibility_threshold = np.inf

    # 3. generate cell-crossing candidates anchored on real seeds
    candidates = []
    for _ in range(pool_size):
        anchor = original_pool[np.random.randint(len(original_pool))].copy()
        chosen = random.sample(cross_feats, random.randint(1, min(max_cross_feats, len(cross_feats))))
        for f in chosen:
            anchor[f] = cross_value(anchor[f], f)
        if not np.any(np.isnan(anchor)):
            candidates.append(anchor)
    if not candidates:
        return []
    candidates = np.array(candidates)

    dmin = _compute_dist_matrix(candidates, original_pool, n_features, isCat, feature_ranges).min(axis=1)
    keep = dmin <= plausibility_threshold
    if not np.any(keep):
        keep = dmin <= np.median(dmin)
    candidates = candidates[keep]
    if len(candidates) == 0:
        return []

    # 4. greedy (feature, threshold, side) coverage selection
    covered = set().union(*(point_cover(s) for s in original_pool)) if len(original_pool) else set()
    cand_cover = [point_cover(c) for c in candidates]
    selected, used = [], np.zeros(len(candidates), dtype=bool)
    for _ in range(min(num_desired_samples, len(candidates))):
        gains = [(-1 if used[i] else len(cand_cover[i] - covered)) for i in range(len(candidates))]
        bi = int(np.argmax(gains))
        if gains[bi] <= 0:                     # coverage saturated -> fill with any unused candidate
            rem = np.where(~used)[0]
            if len(rem) == 0:
                break
            bi = int(rem[0])
        used[bi] = True
        selected.append(candidates[bi])
        covered |= cand_cover[bi]
    return selected


def traverse_explanations_LIME3(sample_set, explainer, model, n_visits_lb, n_visits_ub, upper_limit,
                                n_f_e, args2, model_name, X_train=None, y_train=None,
                                recycle_bisection=True, use_diverse=True, n_middle=10,
                                bisect_refine=True, lime_num_samples=1000, diverse_method='manifold',
                                budget_scale=True, overhead_frac=0.4, div_frac=0.15, div_cap=10,
                                densify=0, feature_select='explanation', use_threshold=True,
                                use_explanation=True, eps_override=None):
    """LIME analog of traverse_explanations_SHAP3. Same three-phase scaffold, LIME throughout the
    explanation-driven parts:
      Phase 1  diverse generation (explanation-free, identical to SHAP3; ablatable via use_diverse)
      Phase 2  LIME boundary search  (generate_lime_informed_samples: snap top-k to bin edge + bisect)
      Phase 3  LIME-guided traversal (explain each popped sample, snap its top-k features to their
               bin edges, then step +/- epsilon)  -- mirrors traverse_explanations_LIME's mechanism.

    `explainer` must be a lime.lime_tabular.LimeTabularExplainer. LIME's internal queries are NOT
    charged (the explanation is bundled with the label, as in traverse_explanations_LIME); only
    Phase-2 bisection intermediates count, matching SHAP3's overhead accounting.
    """
    classes, features, n_classes, n_features, isCat, epsilon_set, canNegative, classPossibilities, dataset_name, feature_ranges = args2
    print("Dataset name in traverse_explanations_LIME3:", dataset_name)
    if isinstance(n_visits_lb, int):
        n_v_lb = np.ones(len(classes)) * n_visits_lb
        n_v_ub = np.ones(len(classes)) * n_visits_ub
    n_visits = np.zeros(len(classes))
    samples = sample_set.copy()
    init_preds = model.predict_proba(samples)
    preds = []
    visited_samples = []
    for i in init_preds:
        preds.append(np.argmax(i))
    for i in samples:
        visited_samples += [i]
    query = 1
    isPassed = [n_visits[i] >= n_v_lb[i] for i in range(len(n_v_lb))]

    # Budget-aware sizing: at small Q the Phase-1 diverse + Phase-2 boundary overhead (~n_middle*8
    # bisection queries) can eat the whole budget and starve Phase-3 -> a fidelity dip at small Q
    # (the boundary bisection is the dominant cost, ~80 queries; diverse is ~n_div). Cap that
    # overhead to ~overhead_frac of the budget; both clamp back to full size at large Q.
    if budget_scale:
        # div_frac controls the Phase-1 diverse budget (fraction of Q, capped at div_cap);
        # div_frac=0 -> no diverse. (Was hardcoded 0.15 / cap 10; now tunable.)
        n_div_eff = int(np.clip(upper_limit * div_frac, 0, div_cap))
        n_mid_eff = int(np.clip(upper_limit * overhead_frac / 8.0, 2, n_middle))
    else:
        n_div_eff, n_mid_eff = div_cap, n_middle

    # Phase 1: diverse generation. 'manifold' = geometric (explanation-free, same as SHAP3);
    # 'lime_threshold' = LIME decision-cell coverage (uses the free seed explanations).
    selected_diverse_samples = []
    if use_diverse:
        if diverse_method == 'lime_threshold':
            diverse_samples = create_lime_threshold_diverse_samples(
                samples, explainer, model, features, isCat, classPossibilities, feature_ranges,
                num_desired_samples=n_div_eff, lime_num_samples=lime_num_samples)
        else:
            diverse_samples = create_manifold_aware_diverse_samples_corrected(
                samples, classPossibilities, feature_ranges, isCat, num_desired_samples=n_div_eff)
        while len(diverse_samples) > 0:
            current_diverse = diverse_samples.pop(0)
            query += 1
            if model.predict_proba([current_diverse]).max() > 0.8:
                selected_diverse_samples += [current_diverse]

    # Phase 2: LIME boundary search (bin-edge threshold, then optional bisection refine)
    bisection_mids = [] if recycle_bisection else None
    middle_samples, overhead_queries = generate_lime_informed_samples(
        model, samples, explainer, isCat, feature_ranges, features,
        num_new_samples=n_mid_eff, top_k=5, mid_log=bisection_mids,
        bisect_refine=bisect_refine, lime_num_samples=lime_num_samples, densify=densify,
        feature_select=feature_select, use_threshold=use_threshold,
        use_explanation=use_explanation)
    samples += middle_samples
    samples += selected_diverse_samples

    # Recycle bisection intermediates as free boundary-labelled training data (not eps-perturbed)
    if recycle_bisection and bisection_mids:
        for ms, mlab in bisection_mids:
            if n_visits[int(mlab)] < n_v_ub[int(mlab)]:
                n_visits[int(mlab)] += 1
                preds += [classes[int(mlab)]]
                visited_samples += [np.asarray(ms, dtype=float)]

    # Phase 3: LIME-guided traversal -- snap each top-k feature to its bin edge, then +/- epsilon
    upper_limit = upper_limit - overhead_queries
    while len(samples) != 0 and not all(isPassed) and not query > upper_limit:
        query += 1
        curr = samples.pop(0)
        class_index = int(np.argmax(model.predict_proba([curr])))
        if query % 100 == 0:
            print(int(query / 100), end=" ")
        if n_visits[class_index] < n_v_ub[class_index]:
            n_visits[class_index] += 1
            preds += [classes[class_index]]
            visited_samples += [curr]
            if use_explanation:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    exp = explainer.explain_instance(curr, model.predict_proba,
                                                     num_features=n_features, num_samples=lime_num_samples)
                key = list(exp.as_map().keys())[0]
                exp_parsed = explanation_parser(exp.as_map(), exp.as_list(key), key, features)  # sorted by |w|
            else:
                exp_parsed = grid_rules(explainer, curr, features, isCat)   # self-computed grid only
            k = min(n_f_e, n_features)
            # E2/H1: 'explanation' = top-k by |weight|; 'random' = random k features (still snapped
            # to THEIR bin edges below), isolating feature choice from the threshold channel.
            top = (exp_parsed[:k] if feature_select == 'explanation'
                   else random.sample(exp_parsed, min(k, len(exp_parsed))))
            idxs = [_feature_index(row[0], features) for row in top]
            # two single-feature children per top feature: high-edge (+eps) and low-edge (-eps)
            cpys = [np.copy(curr) for _ in range(2 * k)]
            for i in range(k):
                fi = idxs[i]
                if fi is None:
                    continue
                low, high = top[i][1], top[i][2]
                base_hi = float(high) if high != -1 else float(curr[fi])
                base_lo = float(low) if low != -1 else float(curr[fi])
                if not use_threshold:                      # E3a: ignore the bin edge, step from current value
                    base_hi = base_lo = float(curr[fi])
                # STEP SIZE. Default is the per-feature epsilon_set, which is NOT what the base
                # Autolycus LIME traversal does: traverse_explanations_LIME hardcodes epsilon = 1
                # and ignores epsilon_set entirely (so does the original, utils.py:113). Only the
                # SHAP traversals use epsilon_set. That makes every ladder/defense-sweep contrast
                # between this function and the base one confounded by step size, by 8x on crop and
                # 30x on pendigits, and not at all on nursery/mushroom where epsilon_set is all 1s.
                # eps_override=1.0 matches the baseline so the phases can be isolated from the step.
                eps_i = epsilon_set[fi] if eps_override is None else eps_override
                cpys[2 * i][fi] = base_hi + eps_i
                cpys[2 * i + 1][fi] = base_lo - eps_i
            for i in range(2 * k):
                fi = idxs[i // 2]
                if fi is None:
                    continue
                if isCat[fi]:      # keep categoricals in [0, classPossibilities)
                    if cpys[i][fi] < 0 or cpys[i][fi] >= classPossibilities[fi]:
                        continue
                else:              # clip continuous to the observed range
                    lo, hi = feature_ranges[fi]
                    if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                        cpys[i][fi] = float(np.clip(cpys[i][fi], lo, hi))
                dup = (any((cpys[i] == x).all() for x in visited_samples) or
                       any((cpys[i] == x).all() for x in samples))
                if not dup:
                    samples += [cpys[i]]
    return visited_samples, preds, query + overhead_queries


#   version 4: SHAP-guided counterfactual flip + confident-descendant expansion for boundary search.
#   Replaces SHAP3's midpoint-bisection Phase 2 (generate_shap_informed_samples_new) with a
#   principled, axis-aligned boundary search (see DEVELOPMENT_NOTES.md #12). Phase 1 (diverse
#   seeds) and Phase 3 (ε-perturbation traversal) are behaviourally identical to SHAP3, so an
#   A/B against SHAP3 isolates the effect of the counterfactual Phase 2.
def traverse_explanations_SHAP4(sample_set, explainer, model, n_visits_lb, n_visits_ub, upper_limit, n_f_e, args2,
                               model_name, X_train=None, y_train=None, explanation_type='vanilla', num_exp=5):
    classes, features, n_classes, n_features, isCat, epsilon_set, canNegative, classPossibilities, dataset_name, feature_ranges = args2
    print("Dataset name in traverse_explanations_SHAP4:", dataset_name)
    if isinstance(n_visits_lb, int):
        n_v_lb = np.ones(len(classes)) * n_visits_lb
        n_v_ub = np.ones(len(classes)) * n_visits_ub
    n_visits = np.zeros(len(classes))
    samples = sample_set.copy()
    seed_samples = list(sample_set)  # preserve original seed for cross-class pairing
    init_preds = model.predict_proba(samples)
    preds = []
    visited_samples = []
    for i in init_preds:
        preds.append(np.argmax(i))
    for i in samples:
        visited_samples += [i]
    query = 1
    isPassed = [n_visits[i] >= n_v_lb[i] for i in range(len(n_v_lb))]

    # --- Phase 1: diverse, target-model-confident seed samples (identical to SHAP3) ---
    diverse_samples = create_manifold_aware_diverse_samples_corrected(samples, classPossibilities, feature_ranges, isCat)
    selected_diverse_samples = []
    while len(diverse_samples) > 0:
        current_diverse = diverse_samples.pop(0)
        query += 1
        if model.predict_proba([current_diverse]).max() > 0.8:  # only add confidently-classified samples
            selected_diverse_samples += [current_diverse]

    # --- Phase 2 (SHAP4): SHAP-guided counterfactual flip + confident descendants ---
    # Forms cross-class pairs from the seed set, walks each to the decision boundary by flipping the
    # single highest-SHAP feature at a time, then ε-walks confidently-labelled descendants away from
    # the boundary. For tree targets (dt/rdf) the straddle pair itself is injected directly as
    # training data (axis-aligned split revelation) and is NOT ε-perturbed (avoids boundary noise).
    # Phase 2 queries every flip/rejection sample against the target, so all are
    # target-labelled. Recycling them into the training set is what makes the
    # counterfactual phase pay for itself instead of just burning query budget
    # (without this, SHAP4 lost to SHAP3; with it, it is on par — see DEV NOTES #12).
    cf_query_log = []
    queue_samples, direct_samples, direct_labels, cf_overhead = generate_counterfactual_confident_samples(
        model, explainer, seed_samples, preds,
        isCat, classPossibilities, epsilon_set, model_name,
        num_cross_pairs=5, num_per_side=3, confidence_threshold=0.8,
        canNegative=canNegative, query_log=cf_query_log,
    )
    overhead_queries = cf_overhead
    samples += queue_samples
    samples += selected_diverse_samples

    # Recycle the Phase-2 queried samples as clean (target-labelled) training data.
    for qs, qlab in cf_query_log:
        if n_visits[int(qlab)] < n_v_ub[int(qlab)]:
            n_visits[int(qlab)] += 1
            preds += [classes[int(qlab)]]
            visited_samples += [np.asarray(qs, dtype=float)]

    # Straddle pairs (tree targets only) go straight into training with their known
    # labels from the flip, bypassing the ε-perturbation loop.
    for ds, dl in zip(direct_samples, direct_labels):
        if n_visits[int(dl)] < n_v_ub[int(dl)]:
            n_visits[int(dl)] += 1
            preds += [classes[int(dl)]]
            visited_samples += [ds]

    upper_limit = upper_limit - overhead_queries  # reserve Phase-2 overhead out of the Phase-3 budget
    # --- Phase 3: ε-perturbation traversal (identical to SHAP3's active branch) ---
    while len(samples) != 0 and not all(isPassed) and not query > upper_limit:
        query += 1
        curr = samples.pop(0)
        pred = model.predict_proba([curr])
        class_index = np.argmax(pred)
        if query % 100 == 0:
            print(int(query / 100), end=" ")
        if n_visits[class_index] < n_v_ub[class_index]:
            n_visits[class_index] += 1
            preds += [classes[class_index]]
            visited_samples += [curr]
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                if model_name == 'dt' or model_name == 'rdf':
                    exp = explainer.shap_values(curr)[:, class_index]
                else:
                    exp = explainer.shap_values(curr)
            k = min(np.count_nonzero(exp), n_f_e)
            sort_index = np.flip(np.argsort(abs(exp)))[:k]
            cpys = []
            for i in range(2):
                cpys += [np.copy(curr)]
            for i in range(k):
                num = random.random()
                mult = random.randint(1, 1)
                tmp0 = cpys[0][sort_index[i]] + epsilon_set[sort_index[i]] * mult
                tmp1 = cpys[0][sort_index[i]] - epsilon_set[sort_index[i]] * mult
                if not isCat[sort_index[i]]:
                    cond1 = True
                else:
                    cond1 = (tmp0 < classPossibilities[sort_index[i]])
                cond2 = (tmp1 >= 0)
                if num < 0.8:
                    if cond1:
                        cpys[0][sort_index[i]] += epsilon_set[sort_index[i]] * mult
                        if cond2:
                            cpys[1][sort_index[i]] -= epsilon_set[sort_index[i]] * mult
                    else:
                        cpys[0][sort_index[i]] -= epsilon_set[sort_index[i]] * mult
                        if cond1:
                            cpys[1][sort_index[i]] += epsilon_set[sort_index[i]] * mult
                else:
                    if cond1:
                        cpys[1][sort_index[i]] += epsilon_set[sort_index[i]] * mult
                        if cond2:
                            cpys[0][sort_index[i]] -= epsilon_set[sort_index[i]] * mult
                    else:
                        cpys[1][sort_index[i]] -= epsilon_set[sort_index[i]] * mult
                        if cond1:
                            cpys[0][sort_index[i]] += epsilon_set[sort_index[i]] * mult
            for i in range(2):
                tmp = (any((cpys[i] == x).all() for x in visited_samples) or
                       (any((cpys[i] == x).all() for x in samples)))
                if not tmp:
                    samples += [cpys[i]]
    return visited_samples, preds, query + overhead_queries


def densify_boundary_pairs(model, explainer, seeds, preds, args2, model_name,
                           max_queries=300, n_pairs=12, n_dense=8, query_log=None, use_shap=True):
    """
    DualCF-style boundary densification (Phase 2 for SHAP4b).

    Locates single-axis decision boundaries with the SHAP-guided flip, then densely samples BOTH
    sides of each boundary — varying the NON-boundary features by small perturbations while pinning
    the boundary feature to each side's value — so the *bulk* of the training set concentrates right
    at the target's decision boundaries. For a tree surrogate this pulls the impurity-optimal split
    onto the true threshold (rather than somewhere in the diffuse margin Autolycus leaves).

    Unlike `expand_confident_descendants` (which walks AWAY from the boundary), this stays AT it and
    emits paired samples on both sides. Every sample is target-labelled. Continuous boundary features
    are localised by deep bisection first; categorical features use the flip's exact category boundary.

    Returns (samples, labels, queries). The SHAP-flip walk samples are logged to `query_log`
    (recycled by the caller); the densified bracket/bisection samples are returned directly.
    """
    classes, features, n_classes, n_features, isCat, epsilon_set, canNegative, classPossibilities, dataset_name, feature_ranges = args2
    out_samples, out_labels = [], []
    recovered = {}  # feature_idx -> list of bisection-localised continuous split thresholds (for snapping)
    queries = 0

    cross_pairs = [(seeds[i], seeds[j]) for i in range(len(seeds)) for j in range(len(seeds)) if preds[i] != preds[j]]
    random.shuffle(cross_pairs)
    cross_pairs = cross_pairs[:n_pairs]

    def q_label(x):
        # query + label a densified candidate (counted here, NOT in query_log, to avoid double-add)
        nonlocal queries
        queries += 1
        return int(np.argmax(model.predict_proba([np.asarray(x, dtype=float)])[0]))

    for s_a, s_b in cross_pairs:
        if queries >= max_queries:
            break
        pre, post, q1 = shap_guided_counterfactual_flip(model, explainer, s_a, s_b, model_name, query_log=query_log, use_shap=use_shap)
        queries += q1
        diff = [i for i in range(n_features) if pre[i] != post[i]]
        if len(diff) != 1:
            continue  # only use clean single-axis straddle pairs
        f = diff[0]
        lo_val, hi_val = float(pre[f]), float(post[f])

        # Continuous boundary: bisect along f (other features held at pre) to pin the threshold.
        if not isCat[f]:
            cls_pre = q_label(pre)
            a, b = lo_val, hi_val
            for _ in range(6):
                if queries >= max_queries:
                    break
                mid = pre.copy(); mid[f] = (a + b) / 2.0
                cls_mid = q_label(mid)
                out_samples.append(mid.copy()); out_labels.append(cls_mid)  # mids sit near the boundary
                if cls_mid == cls_pre:
                    a = (a + b) / 2.0
                else:
                    b = (a + b) / 2.0
            lo_val, hi_val = a, b  # tight bracket around the localised threshold
            recovered.setdefault(f, []).append((a + b) / 2.0)  # the recovered true split threshold f*

        # Densify: siblings differing only in feature f (lo vs hi side), with small perturbations on
        # a couple of OTHER features so many points pile up on both sides of this one boundary.
        others = [i for i in range(n_features) if i != f]
        for _ in range(n_dense):
            if queries >= max_queries:
                break
            base = pre.copy()
            for i in random.sample(others, min(2, len(others))):
                if isCat[i]:
                    maxc = int(classPossibilities[i])
                    if maxc > 1:
                        base[i] = float(random.randrange(maxc))
                else:
                    lo, hi = feature_ranges[i]
                    base[i] = float(np.clip(base[i] + np.random.uniform(-1, 1) * epsilon_set[i], lo, hi))
            ca = base.copy(); ca[f] = lo_val
            cb = base.copy(); cb[f] = hi_val
            out_samples.append(ca); out_labels.append(q_label(ca))
            out_samples.append(cb); out_labels.append(q_label(cb))

    print(f"[densify] {len(cross_pairs)} pairs -> {len(out_samples)} boundary samples, {queries} queries")
    return out_samples, out_labels, queries, recovered


#   version 4b: boundary DENSIFICATION. Same Phase 1 (diverse) and Phase 3 (ε-perturbation) as
#   SHAP4, but Phase 2 concentrates the training distribution AT the decision boundaries
#   (DualCF-style dense straddling pairs) instead of sprinkling a few boundary points. The bet:
#   a tree surrogate's splits snap to the true thresholds when most training data brackets them.
def traverse_explanations_SHAP4b(sample_set, explainer, model, n_visits_lb, n_visits_ub, upper_limit, n_f_e, args2,
                                 model_name, X_train=None, y_train=None, explanation_type='vanilla', num_exp=5,
                                 use_diverse=True, densify_n_pairs=12, return_boundary_mask=False,
                                 return_thresholds=False, skip_traversal=False,
                                 densify_cap_frac=None, use_shap_flip=True, use_shap_traverse=True):
    classes, features, n_classes, n_features, isCat, epsilon_set, canNegative, classPossibilities, dataset_name, feature_ranges = args2
    print("Dataset name in traverse_explanations_SHAP4b:", dataset_name)
    if isinstance(n_visits_lb, int):
        n_v_lb = np.ones(len(classes)) * n_visits_lb
        n_v_ub = np.ones(len(classes)) * n_visits_ub
    n_visits = np.zeros(len(classes))
    samples = sample_set.copy()
    seed_samples = list(sample_set)
    init_preds = model.predict_proba(samples)
    preds = []
    visited_samples = []
    for i in init_preds:
        preds.append(np.argmax(i))
    for i in samples:
        visited_samples += [i]
    is_boundary = [False] * len(visited_samples)  # True for densified boundary samples (for up-weighting)
    query = 1
    isPassed = [n_visits[i] >= n_v_lb[i] for i in range(len(n_v_lb))]

    # --- Phase 1: diverse, target-model-confident seed samples. Skipped when use_diverse=False
    #     so the query budget can be reallocated to boundary search (tree-model experiment). ---
    selected_diverse_samples = []
    if use_diverse:
        diverse_samples = create_manifold_aware_diverse_samples_corrected(samples, classPossibilities, feature_ranges, isCat)
        while len(diverse_samples) > 0:
            current_diverse = diverse_samples.pop(0)
            query += 1
            if model.predict_proba([current_diverse]).max() > 0.8:
                selected_diverse_samples += [current_diverse]

    # --- Phase 2: boundary densification (replaces SHAP4's flip+descendants).
    #     densify_n_pairs=0 disables it entirely -> scaffold control (Phase 1 + Phase 3 only).
    #     When Phase 1 is skipped, reallocate that budget to the boundary search (larger cap). ---
    cf_query_log = []
    _frac = densify_cap_frac if densify_cap_frac is not None else (0.85 if not use_diverse else 0.7)
    densify_cap = min(400, int(_frac * upper_limit))
    dense_samples, dense_labels, dense_q, recovered_thresholds = densify_boundary_pairs(
        model, explainer, seed_samples, preds, args2, model_name,
        max_queries=densify_cap, n_pairs=densify_n_pairs, n_dense=8, query_log=cf_query_log, use_shap=use_shap_flip,
    )
    overhead_queries = dense_q

    # Densified boundary samples -> training data directly (target-labelled).
    for s, lab in zip(dense_samples, dense_labels):
        if n_visits[int(lab)] < n_v_ub[int(lab)]:
            n_visits[int(lab)] += 1
            preds += [classes[int(lab)]]
            visited_samples += [np.asarray(s, dtype=float)]
            is_boundary.append(True)
    # Recycle the SHAP-flip walk samples too (also target-labelled, also near boundaries).
    for qs, qlab in cf_query_log:
        if n_visits[int(qlab)] < n_v_ub[int(qlab)]:
            n_visits[int(qlab)] += 1
            preds += [classes[int(qlab)]]
            visited_samples += [np.asarray(qs, dtype=float)]
            is_boundary.append(True)
    samples += selected_diverse_samples

    upper_limit = upper_limit - overhead_queries
    # --- Phase 3: ε-perturbation traversal (identical to SHAP4). Skipped when skip_traversal=True
    #     (densification-only variant: tests whether SHAP-guided densification can replace the
    #     Autolycus-style traversal). use_shap=False -> random feature selection (no SHAP query). ---
    while not skip_traversal and len(samples) != 0 and not all(isPassed) and not query > upper_limit:
        query += 1
        curr = samples.pop(0)
        pred = model.predict_proba([curr])
        class_index = np.argmax(pred)
        if query % 100 == 0:
            print(int(query / 100), end=" ")
        if n_visits[class_index] < n_v_ub[class_index]:
            n_visits[class_index] += 1
            preds += [classes[class_index]]
            visited_samples += [curr]
            is_boundary.append(False)
            if use_shap_traverse:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")
                    if model_name == 'dt' or model_name == 'rdf':
                        exp = explainer.shap_values(curr)[:, class_index]
                    else:
                        exp = explainer.shap_values(curr)
                k = min(np.count_nonzero(exp), n_f_e)
                sort_index = np.flip(np.argsort(abs(exp)))[:k]
            else:
                k = min(n_f_e, n_features)
                sort_index = np.array(random.sample(range(n_features), k))
            cpys = []
            for i in range(2):
                cpys += [np.copy(curr)]
            for i in range(k):
                num = random.random()
                mult = random.randint(1, 1)
                tmp0 = cpys[0][sort_index[i]] + epsilon_set[sort_index[i]] * mult
                tmp1 = cpys[0][sort_index[i]] - epsilon_set[sort_index[i]] * mult
                if not isCat[sort_index[i]]:
                    cond1 = True
                else:
                    cond1 = (tmp0 < classPossibilities[sort_index[i]])
                cond2 = (tmp1 >= 0)
                if num < 0.8:
                    if cond1:
                        cpys[0][sort_index[i]] += epsilon_set[sort_index[i]] * mult
                        if cond2:
                            cpys[1][sort_index[i]] -= epsilon_set[sort_index[i]] * mult
                    else:
                        cpys[0][sort_index[i]] -= epsilon_set[sort_index[i]] * mult
                        if cond1:
                            cpys[1][sort_index[i]] += epsilon_set[sort_index[i]] * mult
                else:
                    if cond1:
                        cpys[1][sort_index[i]] += epsilon_set[sort_index[i]] * mult
                        if cond2:
                            cpys[0][sort_index[i]] -= epsilon_set[sort_index[i]] * mult
                    else:
                        cpys[1][sort_index[i]] -= epsilon_set[sort_index[i]] * mult
                        if cond1:
                            cpys[0][sort_index[i]] += epsilon_set[sort_index[i]] * mult
            for i in range(2):
                tmp = (any((cpys[i] == x).all() for x in visited_samples) or
                       (any((cpys[i] == x).all() for x in samples)))
                if not tmp:
                    samples += [cpys[i]]
    if return_thresholds:
        return visited_samples, preds, query + overhead_queries, recovered_thresholds
    if return_boundary_mask:
        return visited_samples, preds, query + overhead_queries, is_boundary
    return visited_samples, preds, query + overhead_queries


def decode_pred(target, v_preds):
    v_pred_dec = np.zeros(len(v_preds))
    for i in range(len(v_preds)):
        for j in range(len(target)):
            if v_preds[i] == target[j]:  #+1]:
                v_pred_dec[i] = j  #+1
    return v_pred_dec

''' Generate mega samples
@param testx: Test data features
@param testy: Test data labels
@param n: Number of classes
@param sizes: List of sizes of auxiliary datasets per class
@param n_set: Number of auxiliary datasets to generate'''
def mega_sample_generation(testx, testy, n, sizes, n_set):
    test_cpy = []
    for i in range(len(testy)):
        test_cpy += [np.append(testx[i], testy[i])]
    test_cpy = np.array(test_cpy)
    samples_mega = []
    for i in range(n_set):
        sample_sets = []
        sample_set_sui = []
        for j in range(len(sizes)):
            sample_set_sui = sample_set_generation(test_cpy, n, sizes[j])
            sample_sets += [sample_set_sui.copy()]
        samples_mega += [sample_sets]
    return samples_mega


def rtest_sim(shadow, target, test_data):
    count = 0
    shadow_result = shadow.predict_proba(test_data)
    target_result = target.predict_proba(test_data)
    for i in range(len(shadow_result)):
        #if np.argmax(shadow.predict_proba([i])) == np.argmax(target.predict_proba([i])):
        if np.argmax(shadow_result[i]) == np.argmax(target_result[i]):
            count += 1
        # else: 
            # print the misclassified samples and their prediction probabilities
            # print("Misclassified sample:", test_data[i], "Shadow prediction probabilities:", shadow_result[i], "Target prediction probabilities:", target_result[i])

    # average confidence score for misclassified samples among all test samples
    # print("Confidence scores for misclassified samples:", confidence_scores)
    # #visualize the confidence scores
    # if confidence_scores:
    #     plt.figure(figsize=(8, 6))
    #     plt.hist(confidence_scores, bins=20, alpha=0.7, color='blue', edgecolor='black')
    #     plt.title('Confidence Scores for Misclassified Samples')
    #     plt.xlabel('Confidence Score')
    #     plt.ylabel('Frequency')
    #     plt.grid(True, alpha=0.05)
    #     plt.savefig('confidence_scores_histogram.png', dpi=150, bbox_inches='tight')
    #     plt.close()
    #     print("Confidence scores histogram saved as 'confidence_scores_histogram.png'")
    
    # confidence_scores = [target_result[i][np.argmax(target_result[i])] for i in range(len(target_result)) if np.argmax(shadow_result[i]) != np.argmax(target_result[i])]
    # avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
    # print("Average confidence score for samples classified differently by surrogate model:", avg_confidence)

    # correct_confidences = [target_result[i][np.argmax(target_result[i])] for i in range(len(target_result)) if np.argmax(shadow_result[i]) == np.argmax(target_result[i])]
    # avg_correct_confidence = sum(correct_confidences) / len(correct_confidences) if correct_confidences else 0
    # print("Average confidence score for samples classified same by surrogate model:", avg_correct_confidence)
    return count / len(test_data)


# Change the addresses to prevent overriding
def pickling(dataset, modelName, accuracies, rtest_sims, samples_mega):
    if explanation_tool == 1:
        tool = "SHAP"
    else:
        tool = "LIME"

    address_acc = tool + "/_models/" + modelName + "/" + modelName + "_accuracies_" + dataset
    address_sim = tool + "/_models/" + modelName + "/" + modelName + "_similarities_" + dataset
    address_smega = tool + "/_models/" + modelName + "/" + modelName + "_samples_mega_" + dataset

    acc_file = open(address_acc, 'wb')
    pickle.dump(accuracies, acc_file)
    acc_file.close()
    sim_file = open(address_sim, 'wb')
    pickle.dump(rtest_sims, sim_file)
    acc_file.close()
    smega_file = open(address_smega, 'wb')
    pickle.dump(samples_mega, smega_file)
    smega_file.close()


def unpickling(dataset, modelName):
    if explanation_tool == 1:
        tool = "SHAP"
    else:
        tool = "LIME"

    address_acc = tool + "/_models/" + modelName + "/" + modelName + "_accuracies_" + dataset
    address_sim = tool + "/_models/" + modelName + "/" + modelName + "_similarities_" + dataset
    address_smega = tool + "/_models/" + modelName + "/" + modelName + "_samples_mega_" + dataset

    acc_file = open(address_acc, 'rb')
    accs = pickle.load(acc_file)
    acc_file.close()
    sim_file = open(address_sim, 'rb')
    sims = pickle.load(sim_file)
    acc_file.close()
    smega_file = open(address_smega, 'rb')
    samples_mega = pickle.load(smega_file)
    smega_file.close()
    return accs, sims, samples_mega


def argmaxing(accs, rss, args4):  # Select the most similar model up until given query limit
    how_many_sets, sample_set_sizes, nfe, query_limit = args4
    ql = len(query_limit)
    argmax_acc = accs.copy()
    argmax_sim = rss.copy()
    for i in range(1, ql):
        for j in range(max(len(nfe), len(sample_set_sizes))):
            idx = (ql * j) + i
            for k in range(how_many_sets):
                if argmax_sim[idx][k] < argmax_sim[idx - 1][k]:
                    argmax_acc[idx][k] = argmax_acc[idx - 1][k]
                    argmax_sim[idx][k] = argmax_sim[idx - 1][k]
    return argmax_acc, argmax_sim

#todo add feature ranges for continuous features in the datasets, and use them to ensure the generated diverse samples are within valid ranges
def load_dataset(which_dataset, seed=None):
    _rs = 42 if seed is None else seed  # split random_state; override to sample different train/test splits
    if which_dataset == 0:
        #iris = sklearn.datasets.load_iris()
        #X = iris.data
        #y = iris.target
        X, y = shap.datasets.iris()
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y,
                                                                                    test_size=0.25,
                                                                                    random_state=_rs)
        X_test_t, X_test_s, y_test_t, y_test_s = sklearn.model_selection.train_test_split(X_test, y_test,
                                                                                          train_size=0.60,
                                                                                          random_state=21)
        #features = iris.feature_names
        features = list(X.columns)
        classes = [0, 1, 2]  #iris.target_names
        n_features = len(features)
        n_classes = len(classes)
        targets = dict({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
        dataset_name = 'iris'
        isCategorical = [False] * n_features
        canNegative = [False] * n_features
        epsilon_set = [0.828, 0.436, 1.765, 0.762]
        epsilon_set = [x // 4 for x in epsilon_set]
        feature_ranges = [(X[features[i]].min(), X[features[i]].max()) for i in range(n_features)]
        #epsilon_set = [1]*n_features   
    elif which_dataset == 1:
        n_crops = 17
        crop = pd.read_csv('/Users/sesame/AUTOLYCUS/data/crop/Crop_recommendation.csv')  # Dataset 2
        #crop = crop[0:(n_crops*100)]
        crop.drop(crop.index[1800:1900], inplace=True)
        crop.drop(crop.index[1400:1500], inplace=True)
        crop.drop(crop.index[1000:1100], inplace=True)
        crop.drop(crop.index[800:900], inplace=True)
        crop.drop(crop.index[200:300], inplace=True)
        features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 'label']
        label_encoder = LabelEncoder()
        for col in features:
            label_encoder.fit(crop[col])
            crop[col] = label_encoder.transform(crop[col])
        features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        X = crop[features]
        y = crop['label'].to_numpy()

        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.25, stratify=y,
                                                                                    random_state=_rs)
        X_test_t, X_test_s, y_test_t, y_test_s = sklearn.model_selection.train_test_split(X_test, y_test,
                                                                                          train_size=0.60,
                                                                                          stratify=y_test,
                                                                                          random_state=21)

        n_features = len(features)
        classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        n_classes = len(classes)
        dataset_name = 'crop'
        isCategorical = [False] * n_features
        canNegative = [False] * n_features
        epsilon_set = [36.26, 34.17, 56.48, 5.34, 19.98, 0.79, 54.04]
        epsilon_set = [x // 4 for x in epsilon_set]
        feature_ranges = [(X[features[i]].min(), X[features[i]].max()) for i in range(n_features)]
    elif which_dataset == 2:
        X, y = shap.datasets.adult()
        #Preprocessing
        X.iloc[:, 2] -= 1
        X['Country'] = np.where(X['Country'] == 39, 1, 0)
        X['Capital Gain'] = X['Capital Gain'] - X['Capital Loss'] + 4356
        X = X.drop(['Capital Loss'], axis=1)
        X.rename(columns={'Capital Gain': 'Net Capital'}, inplace=True)
        y = y.astype(int)

        X_display, y_display = shap.datasets.adult(display=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=_rs)
        X_test_t, X_test_s, y_test_t, y_test_s = train_test_split(X_test, y_test, train_size=0.60, random_state=21)

        classes = [0, 1]
        features = list(X.columns)
        n_features = len(features)
        n_classes = len(classes)
        epsilon_set = [20, 2, 4, 2, 4, 2, 2, 1, 3000, 5, 1]
        isCategorical = [False, True, True, True, True, True, True, True, False, False, True]
        canNegative = [False, False, False, False, False, False, False, False, True, False, False]
        dataset_name = 'adult'
        feature_ranges = [(X[features[i]].min(), X[features[i]].max()) for i in range(n_features)]
    elif which_dataset == 3:
        from sklearn.datasets import load_breast_cancer
        bc = load_breast_cancer()
        X = pd.DataFrame(bc.data)
        y = bc.target
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.25,
                                                                                    stratify=bc.target,
                                                                                    random_state=_rs)
        X_test_t, X_test_s, y_test_t, y_test_s = sklearn.model_selection.train_test_split(X_test, y_test,
                                                                                          train_size=0.60,
                                                                                          stratify=y_test,
                                                                                          random_state=21)
        features = bc.feature_names
        classes = [0, 1]
        n_features = len(features)
        n_classes = len(classes)
        targets = dict(enumerate(bc.target_names))
        isCategorical = [False] * n_features
        canNegative = [False] * n_features
        epsilon_set = list(X.std())
        #epsilon_set = [x//4 for x in epsilon_set]
        #epsilon_set = [1]*n_features
        dataset_name = 'breast'
        X = pd.DataFrame(X, columns=features)
        feature_ranges = [(X[features[i]].min(), X[features[i]].max()) for i in range(n_features)]
    elif which_dataset == 4:
        nursery = pd.read_csv('/Users/sesame/AUTOLYCUS/data/nursery/nursery.csv')
        nursery[nursery == '?'] = np.nan
        #nursery[nursery['final evaluation']>=2]

        features = list(
            nursery.columns)  #['parents','has_nurs','form','children','housing','finance','social','health','final evaluation']
        label_encoder = LabelEncoder()
        for col in features:
            label_encoder.fit(nursery[col])
            nursery[col] = label_encoder.transform(nursery[col])

        nursery.loc[nursery['final evaluation'] >= 2, 'final evaluation'] = 2
        features = nursery.columns[:-1]
        X = nursery[features]
        y = nursery['final evaluation'].to_numpy()

        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.25,
                                                                                    random_state=_rs)
        X_test_t, X_test_s, y_test_t, y_test_s = sklearn.model_selection.train_test_split(X_test, y_test,
                                                                                          train_size=0.60,
                                                                                          random_state=21)
        targets = dict({0: 'not_recom', 1: 'recommend', 2: 'very_recom'})
        classes = [0, 1, 2]
        n_features = len(features)
        n_classes = len(classes)
        isCategorical = [True] * n_features
        epsilon_set = [1] * n_features
        canNegative = [False] * n_features
        dataset_name = 'nursery'
        feature_ranges = [(X[features[i]].min(), X[features[i]].max()) for i in range(n_features)]
    elif which_dataset == 5:
        mushroom = pd.read_csv('/Users/sesame/AUTOLYCUS/data/mushroom/mushroom_data.csv')
        mushroom[mushroom == '?'] = np.nan
        mushroom = mushroom.drop(mushroom.columns[16], axis=1)

        features = list(mushroom.columns)
        label_encoder = LabelEncoder()
        for col in features:
            label_encoder.fit(mushroom[col])
            mushroom[col] = label_encoder.transform(mushroom[col])

        features = mushroom.columns[1::]
        X = mushroom[features]
        y = mushroom['p'].to_numpy()

        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.25,
                                                                                    random_state=_rs)
        X_test_t, X_test_s, y_test_t, y_test_s = sklearn.model_selection.train_test_split(X_test, y_test,
                                                                                          train_size=0.60,
                                                                                          random_state=21)
        targets = dict({0: 'not_poisonous', 1: 'poisonous'})
        classes = [0, 1]
        n_features = len(features)
        n_classes = len(classes)
        isCategorical = [True] * n_features
        epsilon_set = [1] * n_features
        canNegative = [False] * n_features
        dataset_name = 'mushroom'
        feature_ranges = [(X[features[i]].min(), X[features[i]].max()) for i in range(n_features)]
    elif which_dataset == 6:
        normal = pd.read_csv("/Users/sesame/AUTOLYCUS/data/bctcga/BC-TCGA-Normal.txt", sep="\t")
        normal = normal.drop(columns=["Hybridization REF"], errors="ignore").T

        tumor = pd.read_csv("/Users/sesame/AUTOLYCUS/data/bctcga/BC-TCGA-Tumor.txt", sep="\t")
        tumor = tumor.drop(columns=["Hybridization REF"], errors="ignore").T

        normal['label'] = 0
        tumor['label'] = 1

        data = pd.concat([normal, tumor], ignore_index=True)

        # Separate features and labels
        X = data.drop(columns=["label"], errors="ignore")
        y = data['label'].to_numpy()

        # Convert all values to numeric (handling missing values if necessary)
        X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

        scaler = MinMaxScaler(feature_range=(-1, 1))
        X = scaler.fit_transform(X)
        X = pd.DataFrame(X)

        # Train/test split
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.25, stratify=y,
                                                                                    random_state=_rs)
        X_test_t, X_test_s, y_test_t, y_test_s = sklearn.model_selection.train_test_split(X_test, y_test,
                                                                                          train_size=0.60,
                                                                                          stratify=y_test,
                                                                                          random_state=21)

    
        # Metadata
        features = list(X.columns)
        classes = sorted(list(set(y)))
        n_features = len(features)
        n_classes = len(classes)
        targets = {cls: 'normal' if cls == 0 else 'tumor' for cls in classes}
        isCategorical = [False] * n_features
        canNegative = [False] * n_features
        epsilon_set = list(X.std())
        dataset_name = 'bc_tcga'
        feature_ranges = [(X[features[i]].min(), X[features[i]].max()) for i in range(n_features)]  
    elif which_dataset == 7:
        digits = sklearn.datasets.load_digits()
        #convert to dataframe
        X = pd.DataFrame(digits.data)
        y = digits.target

        # Split into training and test set
    
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=_rs, stratify=y)
        X_test_t, X_test_s, y_test_t, y_test_s = sklearn.model_selection.train_test_split(X_test, y_test,
                                                                                            train_size=0.60,
                                                                                            stratify=y_test,
                                                                                            random_state=21)
        features = [f'pixel_{i}' for i in range(X.shape[1])]
        X.columns = features
        classes = sorted(list(set(y)))
        n_features = len(features)
        n_classes = len(classes)
        targets = {i: str(i) for i in classes}
        isCategorical = [False] * n_features
        canNegative = [False] * n_features
        epsilon_set = [16] * n_features  # assuming pixel values range from 0 to 16
        dataset_name = 'digits'
        feature_ranges = [(X[features[i]].min(), X[features[i]].max()) for i in range(n_features)]

    elif which_dataset == 8:
        w = sklearn.datasets.load_wine()
        X = pd.DataFrame(w.data, columns=[f'f{i}' for i in range(w.data.shape[1])])
        y = w.target
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=_rs, stratify=y)
        X_test_t, X_test_s, y_test_t, y_test_s = train_test_split(X_test, y_test, train_size=0.60,
                                                                  random_state=21, stratify=y_test)
        features = list(X.columns); classes = sorted(set(int(v) for v in y))
        n_features = len(features); n_classes = len(classes)
        isCategorical = [False] * n_features; canNegative = [False] * n_features
        epsilon_set = list(X.std()); dataset_name = 'wine'
        feature_ranges = [(X[features[i]].min(), X[features[i]].max()) for i in range(n_features)]

    elif which_dataset in (9, 10, 11, 12):
        _map = {9: ('pendigits', 'pendigits'), 10: ('letter', 'letter'),
                11: ('waveform-5000', 'waveform'), 12: ('segment', 'segment')}  # OpenML name, display name
        _fetch, _disp = _map[which_dataset]
        d = sklearn.datasets.fetch_openml(_fetch, version=1, as_frame=True)
        X = d.data.reset_index(drop=True).astype(float)
        X.columns = [f'f{i}' for i in range(X.shape[1])]
        y = LabelEncoder().fit_transform(d.target)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=_rs, stratify=y)
        X_test_t, X_test_s, y_test_t, y_test_s = train_test_split(X_test, y_test, train_size=0.60,
                                                                  random_state=21, stratify=y_test)
        features = list(X.columns); classes = sorted(set(int(v) for v in y))
        n_features = len(features); n_classes = len(classes)
        isCategorical = [False] * n_features; canNegative = [False] * n_features
        epsilon_set = list(X.std()); dataset_name = _disp
        feature_ranges = [(X[features[i]].min(), X[features[i]].max()) for i in range(n_features)]

    else:
        print('there is no such dataset')

    classPossibilities = []
    for i in range(n_features):
        uniques, counts = np.unique(X_train.iloc[:, i], return_counts=True)
        classPossibilities.append(len(uniques))

    args1 = [X_train, X_test, y_train, y_test, X_test_t, X_test_s, y_test_t, y_test_s]
    args2 = [classes, features, n_classes, n_features, isCategorical, epsilon_set, canNegative, classPossibilities,
             dataset_name, feature_ranges]
    return args1, args2


def load_model(which_model, X_train, y_train):
    if which_model == 0:
        depth = 15
        t_model = dt(max_depth=depth, random_state=101).fit(X_train.values, y_train)
        model_name = 'dt'
    elif which_model == 1:
        #t_model = lr(solver='sag', random_state=101, max_iter=10000).fit(X_train.values, y_train)
        t_model = lr(random_state=101, max_iter=10000).fit(X_train, y_train)
        model_name = 'lr'
    elif which_model == 2:
        t_model = mnb().fit(X_train.values, y_train)
        model_name = 'nb'
    elif which_model == 3:
        n_classes = len(np.unique(y_train, return_counts=True)[0])
        t_model = knn(n_neighbors=n_classes).fit(X_train.values, y_train)
        model_name = 'knn'
    elif which_model == 4:
        depth = 15
        t_model = rf(max_depth=depth, random_state=101).fit(X_train.values, y_train)
        model_name = 'rdf'
    elif which_model == 5:
        t_model = mlp(hidden_layer_sizes=(20,), activation='relu', solver='adam', max_iter=10000, random_state=101).fit(
            X_train.values, y_train)
        t_model = mlp(activation='relu', solver='adam', max_iter=10000, random_state=101).fit(X_train.values, y_train)
        model_name = 'mlp'
    else:
        print('No such model exists!')

    return t_model, model_name


def load_explainer(explanation_tool, t_model, model_name, X_train):
    if explanation_tool == 1:
        shap.initjs()
        if model_name == 'dt' or model_name == 'rdf':
            t_explainer = shap.Explainer(t_model)
        else:
            f = lambda x: t_model.predict_proba(x)[:, 1]
            med = X_train.median().values.reshape((1, X_train.shape[1]))
            t_explainer = shap.KernelExplainer(f, med, normalize=False)

            # -- Slower but more precise version --
            # f = lambda x: t_model.predict_proba(x)
            # t_explainer = shap.KernelExplainer(f, X_train)
    else:
        t_explainer = lime.lime_tabular.LimeTabularExplainer(X_train.values, discretize_continuous=True)
    return t_explainer


def getModelInfo(t_model, X_train, y_train, X_test_t, y_test_t):
    predict_train = t_model.predict(X_train.values)
    predict_test = t_model.predict(X_test_t.values)
    print('Train results')
    print(confusion_matrix(y_train, predict_train))
    print(classification_report(y_train, predict_train))
    print('Test results')
    print(confusion_matrix(y_test_t, predict_test))
    print(classification_report(y_test_t, predict_test))
    #if model_name == 'dt':
    #    print(features)
    #    text_representation_t = tree.export_text(t_model)
    #    print("Target\n",text_representation_t)
    predictions = t_model.predict(X_test_t.values)
    t_accuracy = round(accuracy_score(y_test_t, t_model.predict(X_test_t.values)), 4)
    print('Model test accuracy: ', t_accuracy, '\n')

    return t_accuracy


def load_experiment_dicts():
    dataset_dict = {0: 'Iris', 1: 'Crop', 2: 'Adult Income', 3: 'Breast Cancer', 4: 'Nursery', 5: 'Mushroom', 6: 'BC-TCGA', 7: 'Digits'}
    model_dict = {0: 'Decision Tree', 1: 'Logistic Regression', 2: 'Multinomial Naive Bayes', 3: 'K Nearest Neighbor',
                  4: 'Random Forest', 5: 'Multilayer Perceptron'}
    exp_dict = {0: 'LIME', 1: 'SHAP'}
    return dataset_dict, model_dict, exp_dict

def visualize_samples(X_train, y_train, X_test_s, y_test_s, X_generated, y_generated):
    """
    Performs t-SNE reduction and visualizes samples.
    Colors represent class labels, marker shapes represent source (Train=o, Test=s, Generated=^).
    Creates: combined figure with all sources, and separate figures for each source type.
    """
    import os
    point_size = 100
    alpha_level = 0.6
    jitter_strength = 0.0
    
    # --- 1. Combine Features (X), Class Labels (y), and Create Source Labels ---
    X_train = np.array(X_train)
    X_test_s = np.array(X_test_s)
    X_generated = np.array(X_generated)
    
    y_train = np.array(y_train).flatten()
    y_test_s = np.array(y_test_s).flatten()
    y_generated = np.array(y_generated).flatten()
    
    X_combined = np.concatenate([X_train, X_test_s, X_generated], axis=0)
    y_combined = np.concatenate([y_train, y_test_s, y_generated], axis=0)
    
    n_train = len(X_train)
    n_test = len(X_test_s)
    n_generated = len(X_generated)
    
    y_source = (
        ['Train'] * n_train +
        ['Test'] * n_test +
        ['Generated'] * n_generated
    )
    y_source = np.array(y_source)
    
    # --- 2. Standardize the Combined Data ---
    print("Scaling data...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_combined)
    
    # --- 3. Apply t-SNE (Dimensionality Reduction) ---
    print("Applying t-SNE (may take a moment for large datasets)...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000, n_jobs=-1)
    X_tsne = tsne.fit_transform(X_scaled)
    
    # --- Add Jittering (if jitter_strength > 0) ---
    if jitter_strength > 0:
        print(f"Adding jitter with strength: {jitter_strength}")
        X_tsne += np.random.normal(0, jitter_strength, X_tsne.shape)
    
    # --- 4. Prepare for Plotting ---
    tsne_df = pd.DataFrame(data=X_tsne, columns=['TSNE-1', 'TSNE-2'])
    tsne_df['Class'] = y_combined
    tsne_df['Source'] = y_source
    
    # Color mapping: one color per class
    unique_classes = sorted(np.unique(y_combined))
    base_cmap = cm.get_cmap('tab10')
    class_to_color = {cls: base_cmap(i % base_cmap.N) for i, cls in enumerate(unique_classes)}
    
    # Marker mapping: one marker per source
    source_to_marker = {'Train': 'o', 'Test': 's', 'Generated': '^'}
    unique_sources = sorted(np.unique(y_source))
    
    # --- Decision boundary proxies (in t-SNE space) ---
    boundary = None
    try:
        X_tsne_arr = np.asarray(X_tsne)
        print("Decision boundary: X_tsne_arr shape:", X_tsne_arr.shape)
        print("Decision boundary: X_combined shape:", X_combined.shape)
        # target model predictions on original feature space (try/except for models that need DataFrame)
        try:
            target_preds_all = np.asarray(t_model.predict(X_combined)).flatten()
            print("Decision boundary: target_preds_all shape:", target_preds_all.shape, "unique:", np.unique(target_preds_all))
            proxy_target = knn(n_neighbors=7).fit(X_tsne_arr, target_preds_all)
            used_target = 'model_preds'
        except Exception as e_inner:
            # fallback: use true labels as proxy if model prediction fails
            print(f"Target model predict failed: {e_inner}; falling back to true labels for proxy.")
            target_preds_all = np.asarray(y_combined).flatten()
            proxy_target = knn(n_neighbors=7).fit(X_tsne_arr, target_preds_all)
            used_target = 'true_labels'

        proxy_surrogate = None
        if surrogate_model is not None:
            try:
                surrogate_preds_all = np.asarray(surrogate_model.predict(X_combined)).flatten()
                proxy_surrogate = knn(n_neighbors=7).fit(X_tsne_arr, surrogate_preds_all)
            except Exception as e_sur:
                print(f"Surrogate predict failed: {e_sur}; skipping surrogate boundary.")
        # mesh grid in t-SNE space
        x_min, x_max = X_tsne_arr[:, 0].min() - 1.0, X_tsne_arr[:, 0].max() + 1.0
        y_min, y_max = X_tsne_arr[:, 1].min() - 1.0, X_tsne_arr[:, 1].max() + 1.0
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
        grid = np.c_[xx.ravel(), yy.ravel()]
        Zt = proxy_target.predict(grid).reshape(xx.shape)
        Zs = None
        if proxy_surrogate is not None:
            Zs = proxy_surrogate.predict(grid).reshape(xx.shape)
        boundary = dict(xx=xx, yy=yy, Zt=Zt, used_target=used_target)
        if Zs is not None:
            boundary['Zs'] = Zs
    except Exception as e:
        print(f'Could not compute decision boundaries: {e}')
    
    # --- 5. COMBINED FIGURE: All sources + classes with different shapes ---
    plt.figure(figsize=(16, 10))
    # plot decision boundaries if computed
    if boundary is not None:
        try:
            from matplotlib.patches import Patch
            # Target boundary: map raw predicted labels -> ordered class indices
            Zt_raw = np.asarray(boundary['Zt'])
            # classes seen by the proxy
            class_list = np.sort(np.unique(np.asarray(target_preds_all))) if 'target_preds_all' in locals() else np.sort(np.unique(Zt_raw))
            class_to_idx = {cl: idx for idx, cl in enumerate(class_list)}
            Zt = np.vectorize(class_to_idx.get)(Zt_raw).astype(float)

            # build a colormap matching the scatter point colors for these classes
            colors = [class_to_color.get(cl, base_cmap(i % base_cmap.N)) for i, cl in enumerate(class_list)]
            cmap_t = plt.matplotlib.colors.ListedColormap(colors)
            levels_t = np.arange(len(class_list) + 1) - 0.5
            plt.contourf(boundary['xx'], boundary['yy'], Zt, levels=levels_t, cmap=cmap_t, alpha=0.25, zorder=0)
            plt.contour(boundary['xx'], boundary['yy'], Zt, levels=levels_t, colors='k', linewidths=0.6, alpha=0.6, zorder=0)

            # Surrogate boundary (if present) - align colors if possible
            if 'Zs' in boundary:
                Zs_raw = np.asarray(boundary['Zs'])
                s_class_list = np.sort(np.unique(Zs_raw))
                s_class_to_idx = {cl: idx for idx, cl in enumerate(s_class_list)}
                Zs = np.vectorize(s_class_to_idx.get)(Zs_raw).astype(float)
                s_colors = [class_to_color.get(cl, base_cmap(i % base_cmap.N)) for i, cl in enumerate(s_class_list)]
                cmap_s = plt.matplotlib.colors.ListedColormap(s_colors)
                levels_s = np.arange(len(s_class_list) + 1) - 0.5
                plt.contourf(boundary['xx'], boundary['yy'], Zs, levels=levels_s, cmap=cmap_s, alpha=0.12, zorder=0)
                plt.contour(boundary['xx'], boundary['yy'], Zs, levels=levels_s, colors='k', linewidths=0.3, linestyles='--', alpha=0.5, zorder=0)

            # Add legend patches for class regions (target)
            patches = [Patch(facecolor=colors[i], edgecolor='k', label=f'Class {class_list[i]} (region)') for i in range(len(class_list))]
            # place legend in lower left without overlapping
            plt.legend(handles=patches, loc='lower left', fontsize=8, framealpha=0.8)

            # small annotation about how the proxy was built
            used = boundary.get('used_target', 'model_preds')
            plt.annotate(f"Target boundary proxy: {used}", xy=(0.99, 0.01), xycoords='axes fraction', ha='right', va='bottom', fontsize=8, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        except Exception as e:
            print(f'Decision boundary plotting failed: {e}')
    
    for source in unique_sources:
        for cls in unique_classes:
            mask = (tsne_df['Source'] == source) & (tsne_df['Class'] == cls)
            subset = tsne_df[mask]
            
            if len(subset) > 0:
                marker = source_to_marker.get(source, 'o')
                color = class_to_color.get(cls, '#808080')
                label = f'{source} (Class {cls})'
                
                plt.scatter(
                    subset['TSNE-1'], subset['TSNE-2'], label=label, alpha=alpha_level,
                    s=point_size, color=color, marker=marker, edgecolors='black', linewidth=0.5
                )
    
    plt.title('t-SNE Visualization: All Data\n(Colors = Classes, Shapes = Train(o) / Test(s) / Generated(^))', fontsize=14, fontweight='bold')
    plt.xlabel('t-SNE Component 1', fontsize=12)
    plt.ylabel('t-SNE Component 2', fontsize=12)
    plt.legend(title="Source (Class)", loc='upper left', bbox_to_anchor=(1, 1), fontsize=8, ncol=1)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    os.makedirs('data_visuals', exist_ok=True)
    plt.savefig('data_visuals/tsne_combined_all_sources.png', dpi=150, bbox_inches='tight')
    print("Plot saved as data_visuals/tsne_combined_all_sources.png")
    plt.close()
    
    # --- 6. SEPARATE FIGURES: One per source type (Train, Test, Generated) ---
    for source in unique_sources:
        plt.figure(figsize=(12, 10))
        source_data = tsne_df[tsne_df['Source'] == source]
        
        for cls in unique_classes:
            mask = (source_data['Class'] == cls)
            subset = source_data[mask]
            
            if len(subset) > 0:
                color = class_to_color.get(cls, '#808080')
                label = f'Class {cls}'
                marker = source_to_marker.get(source, 'o')
                
                plt.scatter(
                    subset['TSNE-1'], subset['TSNE-2'], label=label, alpha=alpha_level,
                    s=point_size, color=color, marker=marker, edgecolors='black', linewidth=0.5
                )
        
        plt.title(f't-SNE Visualization: {source} Data Only\n(Colors = Classes)', fontsize=14, fontweight='bold')
        plt.xlabel('t-SNE Component 1', fontsize=12)
        plt.ylabel('t-SNE Component 2', fontsize=12)
        plt.legend(title="Class", loc='upper left', bbox_to_anchor=(1, 1), fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'data_visuals/tsne_{source.lower()}_only.png', dpi=150, bbox_inches='tight')
        print(f"Plot saved as data_visuals/tsne_{source.lower()}_only.png")
        plt.close()

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier

def visualize_samples2(origin_samples, t_model, x_gen, y_gen, which_model, resolution=0.1):
    """
    Visualizes original and generated samples in t-SNE space with 
    proxies for the target model's decision boundaries.
    """
    X_origin = np.array(origin_samples)
    X_gen = np.array(x_gen)
    
    y_origin = t_model.predict(X_origin).flatten()
    y_gen = np.array(y_gen).flatten()
    
    X_combined = np.concatenate([X_origin, X_gen], axis=0)
    y_combined = np.concatenate([y_origin, y_gen], axis=0)
    source_labels = np.array(['Original'] * len(X_origin) + ['Generated'] * len(X_gen))

    print("Computing t-SNE...")
    X_scaled = StandardScaler().fit_transform(X_combined)
    tsne = TSNE(n_components=2, random_state=42, n_jobs=-1)
    X_tsne = tsne.fit_transform(X_scaled)
    X_tsne_df = pd.DataFrame(X_tsne, columns=['TSNE-1', 'TSNE-2'])    

    proxy_clf, _ = load_model(which_model, X_tsne_df, y_combined)

    x_min, x_max = X_tsne[:, 0].min() - 1, X_tsne[:, 0].max() + 1
    y_min, y_max = X_tsne[:, 1].min() - 1, X_tsne[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, resolution),
                         np.arange(y_min, y_max, resolution))
    
    Z = proxy_clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # 4. Plotting
    plt.figure(figsize=(12, 8))
    
    # Plot decision regions
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF', '#F0F0AA']) # Expand colors if > 4 classes
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=cmap_light)

    # Plot Original Samples
    for cls in np.unique(y_origin):
        mask = (source_labels == 'Original') & (y_combined == cls)
        plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1], 
                    label=f'Original Class {cls}', marker='o', 
                    edgecolors='k', s=60, alpha=0.7)

    # Plot Generated Samples
    for cls in np.unique(y_gen):
        mask = (source_labels == 'Generated') & (y_combined == cls)
        plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1], 
                    label=f'Generated Class {cls}', marker='^', 
                    edgecolors='k', s=80, alpha=0.9)

    plt.title("t-SNE: Decision Boundary Proxy & Sample Distribution")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()
    
    os.makedirs('data_visuals', exist_ok=True)
    plt.savefig('data_visuals/tsne_decision_boundary.png', dpi=150)
    plt.show()

    tsne_df = pd.DataFrame(data=X_tsne, columns=['TSNE-1', 'TSNE-2'])
    tsne_df['Class'] = y_combined
    tsne_df['Source'] = source_labels

    # plotting parameters
    point_size = 60
    alpha_level = 0.8

    # Color mapping: one color per class
    unique_classes = sorted(np.unique(y_combined))
    base_cmap = cm.get_cmap('tab10')
    class_to_color = {cls: base_cmap(i % base_cmap.N) for i, cls in enumerate(unique_classes)}

    # Marker mapping for sources
    source_to_marker = {'Original': 'o', 'Generated': '^'}
    unique_sources = sorted(np.unique(source_labels))

    # Combined scatter with decision region proxy already saved above
    plt.figure(figsize=(16, 10))
    for source in unique_sources:
        for cls in unique_classes:
            mask = (tsne_df['Source'] == source) & (tsne_df['Class'] == cls)
            subset = tsne_df[mask]
            if len(subset) > 0:
                marker = source_to_marker.get(source, 'o')
                color = class_to_color.get(cls, '#808080')
                label = f'{source} (Class {cls})'
                plt.scatter(
                    subset['TSNE-1'], subset['TSNE-2'], label=label, alpha=alpha_level,
                    s=point_size, color=color, marker=marker, edgecolors='black', linewidth=0.5
                )

    plt.title('t-SNE Visualization: Original vs. Generated\n(Colors = Classes, Shapes = Original(o) / Generated(^))', fontsize=14, fontweight='bold')
    plt.xlabel('t-SNE Component 1', fontsize=12)
    plt.ylabel('t-SNE Component 2', fontsize=12)
    plt.legend(title="Source (Class)", loc='upper left', bbox_to_anchor=(1, 1), fontsize=8, ncol=1)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    os.makedirs('data_visuals', exist_ok=True)
    plt.savefig('data_visuals/tsne_original_vs_generated.png', dpi=150, bbox_inches='tight')
    print("Plot saved as data_visuals/tsne_original_vs_generated.png")
    plt.close()

    # Separate figures per source
    for source in unique_sources:
        plt.figure(figsize=(12, 10))
        source_data = tsne_df[tsne_df['Source'] == source]
        for cls in unique_classes:
            mask = (source_data['Class'] == cls)
            subset = source_data[mask]
            if len(subset) > 0:
                color = class_to_color.get(cls, '#808080')
                label = f'Class {cls}'
                marker = source_to_marker.get(source, 'o')
                plt.scatter(
                    subset['TSNE-1'], subset['TSNE-2'], label=label, alpha=alpha_level,
                    s=point_size, color=color, marker=marker, edgecolors='black', linewidth=0.5
                )

        plt.title(f't-SNE Visualization: {source} Data Only\n(Colors = Classes)', fontsize=14, fontweight='bold')
        plt.xlabel('t-SNE Component 1', fontsize=12)
        plt.ylabel('t-SNE Component 2', fontsize=12)
        plt.legend(title="Class", loc='upper left', bbox_to_anchor=(1, 1), fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'data_visuals/tsne_{source.lower()}_only_orig_vs_gen.png', dpi=150, bbox_inches='tight')
        print(f"Plot saved as data_visuals/tsne_{source.lower()}_only_orig_vs_gen.png")
        plt.close()
    
def run_attack_auto(wd, wm, et, hms, sss, nfe, ql, so):  # make sure the types are correct
    which_dataset = wd if isinstance(wd, int) else (
        lambda: (_ for _ in ()).throw(TypeError("Only integers are allowed")))()
    which_model = wm if isinstance(wm, int) else (
        lambda: (_ for _ in ()).throw(TypeError("Only integers are allowed")))()
    explanation_tool = et if isinstance(et, int) else (
        lambda: (_ for _ in ()).throw(TypeError("Only integers are allowed")))()
    how_many_sets = hms if isinstance(hms, int) else (
        lambda: (_ for _ in ()).throw(TypeError("Only integers are allowed")))()
    sample_set_sizes = sss if isinstance(sss, list) else (
        lambda: (_ for _ in ()).throw(TypeError("Only lists are allowed")))()
    nfe = nfe if isinstance(nfe, list) else (lambda: (_ for _ in ()).throw(TypeError("Only lists are allowed")))()
    query_limit = ql if isinstance(ql, list) else (lambda: (_ for _ in ()).throw(TypeError("Only lists are allowed")))()
    save_option = so if isinstance(so, bool) else (
        lambda: (_ for _ in ()).throw(TypeError("Only booleans are allowed")))()

    print('DATASET', which_dataset, which_model)
    ## Unpack args
    args1, args2 = load_dataset(which_dataset)
    X_train, X_test, y_train, y_test, X_test_t, X_test_s, y_test_t, y_test_s = args1
    classes, features, n_classes, n_features, isCategorical, epsilon_set, canNegative, classPossibilities, dataset_name, feature_ranges = args2
    t_model, model_name = load_model(which_model, X_train, y_train)
    # t_model = lr(random_state=101, max_iter=10000).fit(X_train, y_train)
    # model_name = 'lr'

    t_accuracy = getModelInfo(t_model, X_train, y_train, X_test_t, y_test_t)
    t_explainer = load_explainer(explanation_tool, t_model, model_name, X_train)
    dataset_dict, model_dict, exp_dict = load_experiment_dicts()
    print('Dataset:  ', dataset_dict.get(which_dataset))
    print('ML Model: ', model_dict.get(which_model))
    print(exp_dict.get(explanation_tool), 'is the explanation tool currently in use\n')

    ## Here goes the attack!!! 
    ## (You can further configure the parameters like lower and upper bounds, relax_factor etc. at your own risk :/) 

    samples_mega = mega_sample_generation(X_test_s.to_numpy(), y_test_s, n_classes, sample_set_sizes, how_many_sets)
    #samples_mega = mega_sample_generation(X_test_s.to_numpy(), y_test_s.to_numpy(), n_classes, sample_set_sizes, how_many_sets)

    accuracies = []
    rtest_sims = []
    prioritizeSim = True

    if model_name == 'nb' or model_name == 'mlp' or model_name == 'lr' or model_name == 'knn':
        repetition = 1
    elif model_name == 'dt':
        repetition = 100
    else:
        repetition = 10

    relax_factor = 0.5
    lb_set = list(map(lambda x: int((x // n_classes) * (1 - relax_factor) + 1), query_limit))
    ub_set = list(map(lambda x: int((x // n_classes) * (n_classes + relax_factor) + 1), query_limit))
    depth = 15
    print('Lower bounds: ', lb_set, ' Upper bounds: ', ub_set, '(per class for both)\n')
    print('-----The attack starts here!-----\n')

    # 1. Generate sample sets
    for f in nfe:
        print('Number of top features allowed to be explored (k):', f)
        for g in range(len(sample_set_sizes)):
            print('\nNumber of samples per class (n):', sample_set_sizes[g])
            max_sim = [False] * how_many_sets
            for h in range(len(lb_set)):  #Queries
                lb, ub = lb_set[h], ub_set[h]
                real_accuracy = []
                sims = []
                for i in range(how_many_sets):  #Sample Sets
                    if max_sim[i]:
                        print('Max similarity reached for sample set', i, ', skipping.')
                        sims += [1]
                        real_accuracy += [t_accuracy]
                    else:
                        if explanation_tool == 0:
                            v_samples_np, v_pred_dec, n_query = traverse_explanations_LIME(samples_mega[i][g],
                                                                                           t_explainer, t_model, lb, ub,
                                                                                           query_limit[h], f, args2)
                        elif explanation_tool == 1:
                            v_samples_np, v_pred_dec, n_query = traverse_explanations_SHAP3(samples_mega[i][g],
                                                                                           t_explainer, t_model, lb, ub,
                                                                                           query_limit[h], f, args2,
                                                                                           model_name, X_train, y_train)
                        else:
                            print('No valid explanation tool selected')
                            break
                        s_accuracy = []
                        sim = []
                        for k in range(repetition):
                            if model_name == 'dt':
                                s_model = dt(random_state=k, max_depth=depth)
                            elif model_name == 'lr':
                                s_model = lr(max_iter=1000, random_state=k)
                            elif model_name == 'nb':
                                s_model = mnb()
                            elif model_name == 'rdf':
                                s_model = rf(max_depth=depth, random_state=k)
                            elif model_name == 'knn':
                                s_model = knn(n_neighbors=n_classes)
                            elif model_name == 'mlp':
                                models = []
                                for layer in range(10):
                                    l = layer + 1
                                    models += [mlp(activation='tanh', hidden_layer_sizes=(10 * l), solver='adam', max_iter=10000)]
                                    models += [mlp(activation='relu', hidden_layer_sizes=(10 * l), solver='adam', max_iter=10000)]
                            else:
                                print('No such model!')
                                break
                            if model_name == 'mlp':
                                for m in models:
                                    m.fit(v_samples_np, v_pred_dec)
                                    sim += [rtest_sim(m, t_model, X_test_t.values)]
                                    s_accuracy += [accuracy_score(y_test_t, m.predict(X_test_t.values))]
                            else:
                                s_model.fit(v_samples_np, v_pred_dec)
                                sim += [rtest_sim(s_model, t_model, X_test_t.values)]
                                s_accuracy += [accuracy_score(y_test_t, s_model.predict(X_test_t.values))]
                        # MEAN over refits, not max (max = test-set selection bias; see the
                        # _build_surrogate_and_eval fix in run_attack_auto_compare). MLP selects
                        # across architectures, not repeats -> left as argmax.
                        if model_name == 'mlp':
                            tmp = np.argmax(sim) if prioritizeSim else np.argmax(s_accuracy)
                            sims += [round(sim[tmp], 4)]
                            real_accuracy += [round(s_accuracy[tmp], 4)]
                        else:
                            sims += [round(float(np.mean(sim)), 4)]
                            real_accuracy += [round(float(np.mean(s_accuracy)), 4)]
                        if sim[tmp] == 1:
                            max_sim[i] = True
                        print('Sample set', i, ', Top similarity:', round(sim[tmp], 4))
                accuracies += [real_accuracy]
                rtest_sims += [sims]
                print('\nQuery limit:', query_limit[h], '| Avg similarity:', round(np.mean(sims), 4), '| All:', sims)

    # Pack up all remaining variables for checking
    args0 = [which_dataset, which_model, explanation_tool]
    args3 = [t_model, model_name, t_accuracy, t_explainer]
    args4 = [how_many_sets, sample_set_sizes, nfe, query_limit]
    other_args = [args0, args1, args2, args3, args4]

    accuracies, rtest_sims = argmaxing(accuracies, rtest_sims, args4)  # Max similarity surrogate model is preserved
    if save_option:
        save_results(dataset_name, model_name, accuracies, rtest_sims, samples_mega)

    return accuracies, rtest_sims, samples_mega, other_args


def run_attack_auto_compare(wd, wm, et, hms, sss, nfe, ql, so, run_shap4=False, main_variant='shap3',
                            seed=None, div_frac=0.15, div_cap=10):
    """Same as run_attack_auto but runs both traverse_explanations_SHAP3 (main) and
    traverse_explanations_SHAP (baseline) on the same sample sets and prints a comparison.
    Returns (accuracies, rtest_sims, accuracies_baseline, rtest_sims_baseline, samples_mega, other_args)
    where the first two correspond to SHAP3 and the second two to the SHAP baseline.
    `seed` (optional): seed random + numpy at entry so the sample sets and traversals are reproducible
    across runs; main and baseline already share the SAME sample sets (paired).
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    which_dataset = wd if isinstance(wd, int) else (
        lambda: (_ for _ in ()).throw(TypeError("Only integers are allowed")))()
    which_model = wm if isinstance(wm, int) else (
        lambda: (_ for _ in ()).throw(TypeError("Only integers are allowed")))()
    explanation_tool = et if isinstance(et, int) else (
        lambda: (_ for _ in ()).throw(TypeError("Only integers are allowed")))()
    how_many_sets = hms if isinstance(hms, int) else (
        lambda: (_ for _ in ()).throw(TypeError("Only integers are allowed")))()
    sample_set_sizes = sss if isinstance(sss, list) else (
        lambda: (_ for _ in ()).throw(TypeError("Only lists are allowed")))()
    nfe = nfe if isinstance(nfe, list) else (lambda: (_ for _ in ()).throw(TypeError("Only lists are allowed")))()
    query_limit = ql if isinstance(ql, list) else (lambda: (_ for _ in ()).throw(TypeError("Only lists are allowed")))()
    save_option = so if isinstance(so, bool) else (
        lambda: (_ for _ in ()).throw(TypeError("Only booleans are allowed")))()

    print('DATASET', which_dataset, which_model)
    args1, args2 = load_dataset(which_dataset)
    X_train, X_test, y_train, y_test, X_test_t, X_test_s, y_test_t, y_test_s = args1
    classes, features, n_classes, n_features, isCategorical, epsilon_set, canNegative, classPossibilities, dataset_name, feature_ranges = args2
    t_model, model_name = load_model(which_model, X_train, y_train)
    t_accuracy = getModelInfo(t_model, X_train, y_train, X_test_t, y_test_t)
    t_explainer = load_explainer(explanation_tool, t_model, model_name, X_train)
    dataset_dict, model_dict, exp_dict = load_experiment_dicts()
    print('Dataset:  ', dataset_dict.get(which_dataset))
    print('ML Model: ', model_dict.get(which_model))
    print(exp_dict.get(explanation_tool), 'is the explanation tool currently in use\n')

    # main = our method; baseline = Autolycus. LIME path (et=0): LIME3 vs base LIME.
    # SHAP path (et=1): SHAP3 vs base SHAP.
    main_label = 'LIME3' if explanation_tool == 0 else 'SHAP3'
    base_label = 'LIME base' if explanation_tool == 0 else 'SHAP base'

    samples_mega = mega_sample_generation(X_test_s.to_numpy(), y_test_s, n_classes, sample_set_sizes, how_many_sets)

    # Main method (SHAP3) results
    accuracies = []
    rtest_sims = []
    # Baseline method (SHAP) results
    accuracies_baseline = []
    rtest_sims_baseline = []
    # SHAP4 (counterfactual-flip) results — only populated when run_shap4=True
    accuracies_shap4 = []
    rtest_sims_shap4 = []
    prioritizeSim = True

    if model_name == 'nb' or model_name == 'mlp' or model_name == 'lr' or model_name == 'knn':
        repetition = 1
    elif model_name == 'dt':
        repetition = 100
    else:
        repetition = 10

    relax_factor = 0.5
    lb_set = list(map(lambda x: int((x // n_classes) * (1 - relax_factor) + 1), query_limit))
    ub_set = list(map(lambda x: int((x // n_classes) * (n_classes + relax_factor) + 1), query_limit))
    depth = 15
    print('Lower bounds: ', lb_set, ' Upper bounds: ', ub_set, '(per class for both)\n')
    print('-----The attack starts here!-----\n')

    def _build_surrogate_and_eval(v_samples_np, v_pred_dec):
        """Fit surrogate models on traversed samples and return best sim/accuracy."""
        if model_name == 'nb':   # MultinomialNB needs non-negative features; +/-eps and bin-edge
            v_samples_np = np.clip(np.asarray(v_samples_np, dtype=float), 0, None)  # steps can dip <0
        s_accuracy = []
        sim = []
        for k in range(repetition):
            if model_name == 'dt':
                s_model = dt(random_state=k, max_depth=depth)
            elif model_name == 'lr':
                s_model = lr(max_iter=1000, random_state=k)
            elif model_name == 'nb':
                s_model = mnb()
            elif model_name == 'rdf':
                s_model = rf(max_depth=depth, random_state=k)
            elif model_name == 'knn':
                s_model = knn(n_neighbors=n_classes)
            elif model_name == 'mlp':
                models = []
                for layer in range(10):
                    l = layer + 1
                    models += [mlp(activation='tanh', hidden_layer_sizes=(10 * l), solver='adam', max_iter=10000)]
                    models += [mlp(activation='relu', hidden_layer_sizes=(10 * l), solver='adam', max_iter=10000)]
            else:
                print('No such model!')
                return 0.0, 0.0
            if model_name == 'mlp':
                for m in models:
                    m.fit(v_samples_np, v_pred_dec)
                    sim += [rtest_sim(m, t_model, X_test_t.values)]
                    s_accuracy += [accuracy_score(y_test_t, m.predict(X_test_t.values))]
            else:
                s_model.fit(v_samples_np, v_pred_dec)
                sim += [rtest_sim(s_model, t_model, X_test_t.values)]
                s_accuracy += [accuracy_score(y_test_t, s_model.predict(X_test_t.values))]
        # Aggregate the `repetition` refits by MEAN, not max. max(sim) selected the luckiest
        # refit ON X_test_t and then reported that same value -> test-set selection bias
        # (winner's curse), inflating dt (reps=100) and rdf (reps=10) the most. Mean is the
        # unbiased estimate of surrogate fidelity, and variance reduction is what `repetition`
        # is actually for. lr/nb/knn use reps=1 so mean == the single value (unchanged).
        # MLP selects across DIFFERENT architectures (not repeats) -> left as argmax; that is a
        # separate selection concern and perceptron is out of scope for this fix.
        if model_name == 'mlp':
            tmp = np.argmax(sim) if prioritizeSim else np.argmax(s_accuracy)
            return round(sim[tmp], 4), round(s_accuracy[tmp], 4)
        return round(float(np.mean(sim)), 4), round(float(np.mean(s_accuracy)), 4)

    for f in nfe:
        print('Number of top features allowed to be explored (k):', f)
        for g in range(len(sample_set_sizes)):
            print('\nNumber of samples per class (n):', sample_set_sizes[g])
            max_sim = [False] * how_many_sets
            max_sim_baseline = [False] * how_many_sets
            max_sim_shap4 = [False] * how_many_sets
            for h in range(len(lb_set)):
                lb, ub = lb_set[h], ub_set[h]
                real_accuracy = []
                sims = []
                real_accuracy_baseline = []
                sims_baseline = []
                real_accuracy_shap4 = []
                sims_shap4 = []
                for i in range(how_many_sets):
                    # --- Main method: our method (LIME3 if et=0, SHAP3 if et=1) ---
                    if max_sim[i]:
                        print('Sample set', i, f" [{main_label}] Max similarity reached, skipping.")
                        sims += [1]
                        real_accuracy += [t_accuracy]
                    else:
                        if explanation_tool == 0:
                            v_samples_np, v_pred_dec, n_query = traverse_explanations_LIME3(samples_mega[i][g],
                                                                                            t_explainer, t_model, lb, ub,
                                                                                            query_limit[h], f, args2,
                                                                                            model_name, X_train, y_train, diverse_method="lime_threshold",
                                                                                            div_frac=div_frac, div_cap=div_cap)
                        elif explanation_tool == 1:
                            # main_variant='best' routes tree targets (dt/rdf) to the boundary-
                            # densification variant SHAP4b(use_diverse=False); everything else uses SHAP3.
                            if main_variant == 'best' and model_name in ('dt', 'rdf'):
                                v_samples_np, v_pred_dec, n_query = traverse_explanations_SHAP4b(
                                    samples_mega[i][g], t_explainer, t_model, lb, ub,
                                    query_limit[h], f, args2, model_name, X_train, y_train, use_diverse=False)
                            else:
                                v_samples_np, v_pred_dec, n_query = traverse_explanations_SHAP3(samples_mega[i][g],
                                                                                                t_explainer, t_model, lb, ub,
                                                                                                query_limit[h], f, args2,
                                                                                                model_name, X_train, y_train)
                        else:
                            print('No valid explanation tool selected')
                            break
                        m_sim, m_acc = _build_surrogate_and_eval(v_samples_np, v_pred_dec)
                        sims += [m_sim]
                        real_accuracy += [m_acc]
                        if m_sim == 1:
                            max_sim[i] = True
                        print(f'[{main_label}]     Sample set {i}, n_queries={n_query}, Set Size = {len(v_samples_np)}, Top similarity={m_sim}')

                    # --- Baseline: Autolycus (base LIME if et=0, base SHAP if et=1) ---
                    if explanation_tool in (0, 1):
                        if max_sim_baseline[i]:
                            print('Sample set', i, f" [{base_label}] Max similarity reached, skipping.")
                            sims_baseline += [1]
                            real_accuracy_baseline += [t_accuracy]
                        else:
                            if explanation_tool == 0:
                                v_samples_np_b, v_pred_dec_b, n_query_b = traverse_explanations_LIME(
                                    samples_mega[i][g], t_explainer, t_model, lb, ub,
                                    query_limit[h], f, args2)
                            else:
                                v_samples_np_b, v_pred_dec_b, n_query_b = traverse_explanations_SHAP(
                                    samples_mega[i][g], t_explainer, t_model, lb, ub,
                                    query_limit[h], f, args2, model_name, X_train, y_train)
                            m_sim_b, m_acc_b = _build_surrogate_and_eval(v_samples_np_b, v_pred_dec_b)
                            sims_baseline += [m_sim_b]
                            real_accuracy_baseline += [m_acc_b]
                            if m_sim_b == 1:
                                max_sim_baseline[i] = True
                            print(f'[{base_label}] Sample set {i}, n_queries={n_query_b}, Set Size = {len(v_samples_np_b)}, Top similarity={m_sim_b}')

                    # --- New method: traverse_explanations_SHAP4 (counterfactual flip) ---
                    if run_shap4 and explanation_tool == 1:
                        if max_sim_shap4[i]:
                            print('Sample set', i, " [SHAP4] Max similarity reached, skipping.")
                            sims_shap4 += [1]
                            real_accuracy_shap4 += [t_accuracy]
                        else:
                            v_samples_np_4, v_pred_dec_4, n_query_4 = traverse_explanations_SHAP4(
                                samples_mega[i][g], t_explainer, t_model, lb, ub,
                                query_limit[h], f, args2, model_name, X_train, y_train)
                            m_sim_4, m_acc_4 = _build_surrogate_and_eval(v_samples_np_4, v_pred_dec_4)
                            sims_shap4 += [m_sim_4]
                            real_accuracy_shap4 += [m_acc_4]
                            if m_sim_4 == 1:
                                max_sim_shap4[i] = True
                            print(f'[SHAP4]     Sample set {i}, n_queries={n_query_4}, Set Size = {len(v_samples_np_4)}, Top similarity={m_sim_4}')

                accuracies += [real_accuracy]
                rtest_sims += [sims]
                accuracies_baseline += [real_accuracy_baseline]
                rtest_sims_baseline += [sims_baseline]
                if run_shap4:
                    accuracies_shap4 += [real_accuracy_shap4]
                    rtest_sims_shap4 += [sims_shap4]

                avg_main = round(np.mean(sims), 4) if sims else 0
                std_main = round(np.std(sims), 4) if sims else 0
                avg_base = round(np.mean(sims_baseline), 4) if sims_baseline else 0
                std_base = round(np.std(sims_baseline), 4) if sims_baseline else 0
                print(f"\n=== Query limit {query_limit[h]} | k={f} | n={sample_set_sizes[g]} ===")
                print(f"  [{main_label} main]    Avg similarity: {avg_main} +/- {std_main}  | All: {sims}")
                if explanation_tool in (0, 1):
                    print(f"  [{base_label}] Avg similarity: {avg_base} +/- {std_base}  | All: {sims_baseline}")
                    print(f"  Difference (main - baseline): {round(avg_main - avg_base, 4):+.4f}\n")
                if run_shap4 and explanation_tool == 1:
                    avg_s4 = round(np.mean(sims_shap4), 4) if sims_shap4 else 0
                    print(f"  [SHAP4 cf-flip] Avg similarity: {avg_s4}  | All: {sims_shap4}")
                    print(f"  Difference (SHAP4 - baseline): {round(avg_s4 - avg_base, 4):+.4f}")
                    print(f"  Difference (SHAP4 - SHAP3):    {round(avg_s4 - avg_main, 4):+.4f}\n")

    args0 = [which_dataset, which_model, explanation_tool]
    args3 = [t_model, model_name, t_accuracy, t_explainer]
    args4 = [how_many_sets, sample_set_sizes, nfe, query_limit]
    other_args = [args0, args1, args2, args3, args4]

    accuracies, rtest_sims = argmaxing(accuracies, rtest_sims, args4)
    if save_option:
        save_results(dataset_name, model_name, accuracies, rtest_sims, samples_mega)

    flat_main = [s for batch in rtest_sims for s in batch]
    print("\n========== FINAL COMPARISON SUMMARY ==========")
    print(f"  [{main_label} main]    Overall avg similarity: {round(np.mean(flat_main), 4)} +/- {round(np.std(flat_main), 4)}")
    if explanation_tool in (0, 1) and rtest_sims_baseline:
        flat_base = [s for batch in rtest_sims_baseline for s in batch]
        print(f"  [{base_label}] Overall avg similarity: {round(np.mean(flat_base), 4)} +/- {round(np.std(flat_base), 4)}")
        print(f"  Overall difference (main - baseline): {round(np.mean(flat_main) - np.mean(flat_base), 4):+.4f}")
    if run_shap4 and explanation_tool == 1:
        flat_s4 = [s for batch in rtest_sims_shap4 for s in batch]
        if flat_s4:   # SHAP4 arm only runs on the SHAP path; skip on LIME to avoid mean([])=nan
            print(f"  [SHAP4 cf-flip] Overall avg similarity: {round(np.mean(flat_s4), 4)}")
    print("===============================================\n")

    if run_shap4 and explanation_tool == 1:   # SHAP4 arm is SHAP-only; keep 6-tuple on LIME path
        return (accuracies, rtest_sims, accuracies_baseline, rtest_sims_baseline,
                samples_mega, other_args,
                {'accuracies': accuracies_shap4, 'rtest_sims': rtest_sims_shap4})
    return accuracies, rtest_sims, accuracies_baseline, rtest_sims_baseline, samples_mega, other_args


def run_attack_auto_v2(wd, wm, et, hms, sss, nfe, ql, so, top_exp=5):  # make sure the types are correct
    which_dataset = wd if isinstance(wd, int) else (
        lambda: (_ for _ in ()).throw(TypeError("Only integers are allowed")))()
    which_model = wm if isinstance(wm, int) else (
        lambda: (_ for _ in ()).throw(TypeError("Only integers are allowed")))()
    explanation_tool = et if isinstance(et, int) else (
        lambda: (_ for _ in ()).throw(TypeError("Only integers are allowed")))()
    how_many_sets = hms if isinstance(hms, int) else (
        lambda: (_ for _ in ()).throw(TypeError("Only integers are allowed")))()
    sample_set_sizes = sss if isinstance(sss, list) else (
        lambda: (_ for _ in ()).throw(TypeError("Only lists are allowed")))()
    nfe = nfe if isinstance(nfe, list) else (lambda: (_ for _ in ()).throw(TypeError("Only lists are allowed")))()
    query_limit = ql if isinstance(ql, list) else (lambda: (_ for _ in ()).throw(TypeError("Only lists are allowed")))()
    save_option = so if isinstance(so, bool) else (
        lambda: (_ for _ in ()).throw(TypeError("Only booleans are allowed")))()

    ## Unpack args
    args1, args2 = load_dataset(which_dataset)
    X_train, X_test, y_train, y_test, X_test_t, X_test_s, y_test_t, y_test_s = args1
    if len(args2) == 10:
        classes, features, n_classes, n_features, isCategorical, epsilon_set, canNegative, classPossibilities, dataset_name, feature_ranges = args2
    else:
        classes, features, n_classes, n_features, isCategorical, epsilon_set, canNegative, classPossibilities, dataset_name = args2
    # explanation_types = ["vanilla", "topk", "random", "zero", "hybrid", "local", "test"]
    explanation_types = ["vanilla"]
    # corr = X_train.corr(numeric_only=True)
    # print(corr)
    # threshold = 0.5
    # high_corr = np.where(abs(corr) > threshold)
    # pairs = [(corr.index[i], corr.columns[j]) 
    #     for i, j in zip(*high_corr) if i != j]

    # print(pairs)
    t_model, model_name = load_model(which_model, X_train, y_train)
    t_accuracy = getModelInfo(t_model, X_train, y_train, X_test_t, y_test_t)
    t_explainer = load_explainer(explanation_tool, t_model, model_name, X_train)
    dataset_dict, model_dict, exp_dict = load_experiment_dicts()
    print('Dataset:  ', dataset_dict.get(which_dataset))
    print('ML Model: ', model_dict.get(which_model))
    print(exp_dict.get(explanation_tool), 'is the explanation tool currently in use\n')

    ## Here goes the attack!!! 
    ## (You can further configure the parameters like lower and upper bounds, relax_factor etc. at your own risk :/) 
    samples_mega = mega_sample_generation(X_test_s.to_numpy(), y_test_s, n_classes, sample_set_sizes, how_many_sets)
    # samples_mega = mega_sample_generation(X_test_s.to_numpy(), y_test_s.to_numpy(), n_classes, sample_set_sizes, how_many_sets)

    rows, cols, dims = len(query_limit), how_many_sets, len(nfe)
    # Create a 3D matrix with dimensions (dims, rows, cols) initialized to 0
    accuracies = [[[0 for _ in range(cols)] for _ in range(rows)] for _ in range(dims)]
    rtest_sims = [[[0 for _ in range(cols)] for _ in range(rows)] for _ in range(dims)]
    prioritizeSim = True

    if model_name == 'nb' or model_name == 'mlp' or model_name == 'lr' or model_name == 'knn':
        repetition = 1
    elif model_name == 'dt':
        repetition = 100
    else:
        repetition = 10

    relax_factor = 0.5
    lb_set = list(map(lambda x: int((x // n_classes) * (1 - relax_factor) + 1), query_limit))
    ub_set = list(map(lambda x: int((x // n_classes) * (n_classes + relax_factor) + 1), query_limit))
    depth = 15
    print('Lower bounds: ', lb_set, ' Upper bounds: ', ub_set, '\n')
    print('-----The attack starts here!-----\n')
    print("samples mega length:", len(samples_mega))
        # Prepare a results container per explanation type (None means default behaviour)
    results_by_type = {etype: {'accuracies': [], 'rtest_sims': [], 'samples_mega': None} for etype in explanation_types}
    # 1. Generate sample sets
    for f in nfe:
        f_idx = nfe.index(f)
        print(f_idx)
        print('Number of top features allowed to be explored (k):', f)
        for g in range(len(sample_set_sizes)):
            print('\nNumber of samples per class (n):', sample_set_sizes[g])
            for i in range(how_many_sets):  #Sample Sets
                print('\nProcessing sample set', i)
                if explanation_tool == 0:
                    v_samples_np, v_pred_dec, n_query = traverse_explanations_LIME(samples_mega[i][g], t_explainer,
                                                                                   t_model, lb_set[-1], ub_set[-1],
                                                                                   query_limit[-1], f, args2)
                elif explanation_tool == 1:
                    v_samples_map = {}
                    for etype in explanation_types:
                        print(f"Processing explanation type: {etype}")
                        if(etype=="local"):
                            v_samples_map[etype] = traverse_explanations_SHAP2(samples_mega[i][g], t_explainer, t_model,
                                                                            lb_set[-1], ub_set[-1], query_limit[-1], f,
                                                                            args2, model_name, X_train, y_train,
                                                                            etype, top_exp)
                            # visualize_samples2(samples_mega[i][g], t_model, v_samples_map[etype][0].copy(), v_samples_map[etype][1].copy(), which_model)
                        else:
                            v_samples_map[etype] = traverse_explanations_SHAP3(samples_mega[i][g], t_explainer, t_model,
                                                                        lb_set[-1], ub_set[-1], query_limit[-1], f,
                                                                         args2, model_name, X_train, y_train,
                                                                         etype, top_exp)
                            # if(etype != "zero"): visualize_samples2(samples_mega[i][g], t_model, v_samples_map[etype][0].copy(), v_samples_map[etype][1].copy(), which_model)

                            # visualize_samples2(samples_mega[i][g], t_model, v_samples_map[etype][0].copy(), v_samples_map[etype][1].copy(), which_model)

                else:
                    print('No valid explanation tool selected')
                    break
                for etype, (v_samples_np, v_pred_dec, n_query) in v_samples_map.items():
                    for h in range(len(query_limit)):
                        print(f"Training a surrogate model for explanation type: {etype} and query limit: {query_limit[h]}")
                        data_x = v_samples_np
                        data_y = v_pred_dec
                        #visualize the samples calling a function
                        #visualize_samples(data_x, data_y, t_model, X_test_t, y_test_t, X_train, y_train)
                        # data_x = v_samples_np[:query_limit[h]]
                        # data_y = v_pred_dec[:query_limit[h]]
                        s_accuracy = []
                        sim = []
                        for k in range(repetition):  #Model building
                            if model_name == 'dt':
                                s_model = dt(random_state=k, max_depth=depth)
                            elif model_name == 'lr':
                                #s_model = lr(solver='sag', max_iter=10000,random_state=k)
                                s_model = lr(max_iter=1000, random_state=k)
                            elif model_name == 'nb':
                                s_model = mnb()
                            elif model_name == 'rdf':
                                s_model = rf(max_depth=depth, random_state=k)
                            elif model_name == 'knn':
                                s_model = knn(n_neighbors=n_classes)
                            elif model_name == 'mlp':
                                #This part is added later
                                models = []
                                for layer in range(10):
                                    l = layer + 1
                                    models += [
                                        mlp(activation='tanh', hidden_layer_sizes=(10 * l), solver='adam', max_iter=10000)]
                                    models += [
                                        mlp(activation='relu', hidden_layer_sizes=(10 * l), solver='adam', max_iter=10000)]
                            else:
                                print('No such model!')
                            if model_name == 'mlp':
                                for m in models:
                                    m.fit(data_x, data_y)
                                    sim += [rtest_sim(m, t_model, X_test_t.values)]
                                    s_accuracy += [accuracy_score(y_test_t, m.predict(X_test_t.values))]
                            else:
                                print("Fitting surrogate model...")
                                print(len(data_x), len(data_y))
                                s_model.fit(data_x, data_y)
                                sim += [rtest_sim(s_model, t_model, X_test_t.values)]
                                s_accuracy += [accuracy_score(y_test_t, s_model.predict(X_test_t.values))]
                    if prioritizeSim:
                        tmp = np.argmax(sim)
                    else:
                        tmp = np.argmax(s_accuracy)
                    m_sim = round(sim[tmp], 4)
                    # if m_sim == 1:
                        # m_sim[i] = True        
                    results_by_type[etype]['accuracies'].append(round(s_accuracy[tmp], 4))
                    results_by_type[etype]['rtest_sims'].append(round(sim[tmp], 4))
                    results_by_type[etype]['samples_mega'] = samples_mega
                print('Sample set', i, ', n_queries =', len(v_pred_dec), ', Similarities by type:')
        for etype in explanation_types:        
            print("For ", etype, " Accuracies: ", results_by_type[etype]['accuracies'], "\nSimilarities: ", results_by_type[etype]['rtest_sims'], "\n")


    # Pack up all remaining variables for checking
    args0 = [which_dataset, which_model, explanation_tool]
    args3 = [t_model, model_name, t_accuracy, t_explainer]
    args4 = [how_many_sets, sample_set_sizes, nfe, query_limit]
    other_args = [args0, args1, args2, args3, args4]

    #accuracies, rtest_sims = argmaxing(accuracies, rtest_sims, args4) # Max similarity surrogate model is preserved
    if save_option:
        save_results(dataset_name, model_name, accuracies, rtest_sims, samples_mega)

    return results_by_type, other_args

def generateNewSamples(sample_set, local_explainer, local_model, nfe, args2, model_name, original_set):
    """
    Generate perturbed samples from `sample_set` by distorting the top-`nfe` features
    (according to the local explanation). Distortions use the per-feature epsilons
    from `args2` (numeric: +/- epsilon, categorical: small category change).

    For each original sample:
    - compute the local explanation (SHAP-aware)
    - take the top `nfe` features
    - for each top feature generate one or two perturbed variants
    - recompute explanation for the perturbed variant and check its top feature
      (highest absolute contribution). If the top feature changed, keep the
      perturbed sample and its label (when available in `sample_set`).

    Returns:
      v_samples_np: numpy array of kept (perturbed) samples
      v_pred_dec: list of labels (if available in sample entries) or None
      n_query: number of explanation queries performed (perturbed samples tested)
    """
    if len(args2) == 10:
        classes, features, n_classes, n_features, isCat, epsilon_set, canNegative, classPossibilities, dataset_name, feature_ranges = args2
    else:
        classes, features, n_classes, n_features, isCat, epsilon_set, canNegative, classPossibilities, dataset_name = args2

    samples = np.asarray(sample_set)
    # detect whether samples have an appended label (common in this repo)
    has_label_col = (samples.ndim == 2 and samples.shape[1] == n_features + 1)

    v_samples = []
    v_pred_dec = []
    n_query = 0

    # helper: robust shap extraction for a single-row input
    def _extract_shap_vector(shap_out):
        arr = np.asarray(shap_out)
        if arr.ndim == 3:
            # common: (1, n_features, n_classes) or (n_classes, 1, n_features)
            if arr.shape[0] == 1 and arr.shape[1] >= n_features:
                return arr[0, :n_features, 0] if arr.shape[2] > 1 else arr[0, :n_features, 0]
            if arr.shape[1] == 1 and arr.shape[2] >= n_features:
                return arr[0, 0, :n_features]
            # fallback: collapse last axis
            return arr.reshape(-1, n_features).mean(axis=0)
        elif arr.ndim == 2:
            # (1, n_features) or (n_features, n_classes)
            if arr.shape[0] == 1:
                return arr[0, :n_features].astype(float).flatten()
            if arr.shape[1] == n_features:
                return arr.mean(axis=0)[:n_features].astype(float).flatten()
            return arr.flatten()[:n_features].astype(float).flatten()
        else:
            return arr.flatten()[:n_features].astype(float).flatten()


    # iterate through provided sample_set
    for idx in range(len(samples)):
        row = samples[idx]
        if has_label_col:
            x = np.asarray(row[:-1], dtype=float).reshape(1, -1)
            label = row[-1]
        else:
            x = np.asarray(row, dtype=float).reshape(1, -1)
            label = None

        # 1) get original explanation vector for x
        # hasattr(local_explainer, 'shap_values'):
        orig_shap = local_explainer.shap_values(x)
        phi = _extract_shap_vector(orig_shap)

        # if explanation is all zeros, skip (no meaningful top feature)
        if np.allclose(phi, 0):
            continue

        orig_top = int(np.argmax(np.abs(phi)))
        # top-n features to perturb
        k = max(1, int(nfe))
        topk = list(np.flip(np.argsort(np.abs(phi)))[:k])

        # for each top feature, create perturbed variants
        for feat in topk:
            perturbed_variants = []
            if isCat[feat]:
                # categorical: change by +1 modulo number of possibilities if available
                maxc = int(classPossibilities[feat]) if classPossibilities and len(classPossibilities) > feat else None
                new_x = x.ravel().copy()
                if maxc and maxc > 1:
                    new_x[feat] = (int(new_x[feat]) + 1) % maxc

                    # fallback: flip between 0/1
                    new_x[feat] = 0 if new_x[feat] != 0 else 1
                perturbed_variants.append(new_x)
            else:
                # numeric: apply +epsilon and -epsilon
                eps = float(epsilon_set[feat]) if epsilon_set and len(epsilon_set) > feat else 0.0
                if eps == 0.0:
                    # small fallback epsilon if none provided
                    eps = 1e-3
                new_x_p = x.ravel().copy(); new_x_p[feat] = new_x_p[feat] + eps
                new_x_m = x.ravel().copy(); new_x_m[feat] = new_x_m[feat] - eps
                perturbed_variants.extend([new_x_p, new_x_m])

            # evaluate each perturbed variant
            for new_x in perturbed_variants:
                n_query += 1
                shap2 = local_explainer.shap_values(new_x.reshape(1, -1))
                phi2 = _extract_shap_vector(shap2)

                # skip if phi2 empty or identical to original
                if np.allclose(phi2, 0):
                    continue

                new_top = int(np.argmax(np.abs(phi2)))
                if new_top != orig_top:
                    # keep sample; append label if available
                    if has_label_col:
                        kept = np.append(new_x, label)
                        v_samples.append(kept)
                        v_pred_dec.append(label)
                    else:
                        v_samples.append(new_x)
                        v_pred_dec.append(label)

    if len(v_samples) == 0:
        # return empty array with appropriate shape (0, n_features (+1 if label present))
        if has_label_col:
            v_samples_np = np.zeros((0, n_features + 1))
        else:
            v_samples_np = np.zeros((0, n_features))
    else:
        v_samples_np = np.asarray(v_samples)

    return v_samples_np, v_pred_dec, n_query

def run_attack_auto_v3(wd, wm, et, hms, sss, nfe, ql, so, top_exp=5):  # make sure the types are correct
    which_dataset = wd if isinstance(wd, int) else (
        lambda: (_ for _ in ()).throw(TypeError("Only integers are allowed")))()
    which_model = wm if isinstance(wm, int) else (
        lambda: (_ for _ in ()).throw(TypeError("Only integers are allowed")))()
    explanation_tool = et if isinstance(et, int) else (
        lambda: (_ for _ in ()).throw(TypeError("Only integers are allowed")))()
    how_many_sets = hms if isinstance(hms, int) else (
        lambda: (_ for _ in ()).throw(TypeError("Only integers are allowed")))()
    sample_set_sizes = sss if isinstance(sss, int) else (
        lambda: (_ for _ in ()).throw(TypeError("Only integers are allowed")))()
    nfe = nfe if isinstance(nfe, int) else (
        lambda: (_ for _ in ()).throw(TypeError("Only integers are allowed")))()
    query_limit = ql if isinstance(ql, int) else (
        lambda: (_ for _ in ()).throw(TypeError("Only integers are allowed")))()
    save_option = so if isinstance(so, bool) else (
        lambda: (_ for _ in ()).throw(TypeError("Only booleans are allowed")))()

    ## Unpack args
    args1, args2 = load_dataset(which_dataset)
    X_train, X_test, y_train, y_test, X_test_t, X_test_s, y_test_t, y_test_s = args1
    if len(args2) == 10:
        classes, features, n_classes, n_features, isCategorical, epsilon_set, canNegative, classPossibilities, dataset_name, feature_ranges = args2
    else:
        classes, features, n_classes, n_features, isCategorical, epsilon_set, canNegative, classPossibilities, dataset_name = args2
    explanation_types = ["vanilla", "topk", "random", "zero"]

    t_model, model_name = load_model(which_model, X_train, y_train)

    t_accuracy = getModelInfo(t_model, X_train, y_train, X_test_t, y_test_t)
    t_explainer = load_explainer(explanation_tool, t_model, model_name, X_train)
    dataset_dict, model_dict, exp_dict = load_experiment_dicts()
    print('Dataset:  ', dataset_dict.get(which_dataset))
    print('ML Model: ', model_dict.get(which_model))
    print(exp_dict.get(explanation_tool), 'is the explanation tool currently in use\n')

    samples_mega = mega_sample_generation(X_test_s.to_numpy(), y_test_s, n_classes, sample_set_sizes, how_many_sets)
    for i in range(len(samples_mega)):
        local_explainer = load_explainer(explanation_tool, t_model, model_name, pd.DataFrame(samples_mega[i]))
        local_model = load_model(which_model, pd.DataFrame(samples_mega[i]), y_test_s)[0]
        
        sample_set = samples_mega[i].copy()
        queried = 0
        while queried < query_limit:
            new_samples, new_samples_pred_local, n_local_query = generateNewSamples(sample_set, local_explainer,
                                                                             nfe, args2, model_name, sample_set)
            batch_size = min(len(new_samples), query_limit - queried)
            batch_samples = new_samples[:batch_size]
            batch_preds_local = new_samples_pred_local[:batch_size]
            
            batch_preds = t_model.predict(batch_samples)
            sample_set = np.vstack([sample_set, batch_samples])
            queried += batch_size
            # Here you can process the batch_samples and batch_preds as needed
    return results_by_type, other_arg

def run_attack_prepared(isFast):
    if isFast:
        which_dataset = 0
        which_model = 1
        explanation_tool = 0
        how_many_sets = 10
        sample_set_sizes = [1]
        nfe = [3]
        query_limit = [0, 10, 25, 50, 100]
    else:
        which_dataset = 2
        which_model = 4
        explanation_tool = 1
        how_many_sets = 10
        sample_set_sizes = [5]
        nfe = [3, 5, 7]
        query_limit = [0, 100, 250, 500, 1000]

    return run_attack_auto(which_dataset, which_model, explanation_tool, how_many_sets, sample_set_sizes, nfe,
                           query_limit, False)


# Save Results
def save_results(dataset_name, model_name, acs, rsims, smegas):
    try:
        pickling(dataset_name, model_name, acs, rsims, smegas)
        print('Save operation successful!')
    except:
        print('Save operation failed!')


def load_results(dataset_name,
                 model_name):  # accuracies, rtest_sims, samples_mega = unpickling(dataset_name, model_name)
    return unpickling(dataset_name, model_name)


# ============================================================================
# SHAP-normal reconstruction diagnostic  (added 2026-07)
# Read-only: adds new functions only, touches no existing attack path.
# See SPEC_shap_normal_diagnostic.md for the full rationale.
# ============================================================================

def _cosine_masked(a, b):
    """Cosine over coords finite in BOTH vectors (NaN coords, e.g. categorical g, dropped)."""
    a = np.asarray(a, float).ravel()
    b = np.asarray(b, float).ravel()
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() == 0:
        return np.nan
    av, bv = a[m], b[m]
    na, nb = np.linalg.norm(av), np.linalg.norm(bv)
    if na == 0 or nb == 0:
        return np.nan
    return float(np.dot(av, bv) / (na * nb))


def reconstruct_normal_single(s, x, x0, eps=1e-8):
    """Single-sample normal  w_i = s_i / (x_i - x0_i).
    0/0 guard: coords with |x_i - x0_i| < eps are masked out (w_i = 0), never divided.
    Returns (w, identifiable_mask)."""
    s = np.asarray(s, float).ravel()
    d = np.asarray(x, float).ravel() - np.asarray(x0, float).ravel()
    w = np.zeros_like(d)
    mask = np.abs(d) >= eps
    w[mask] = s[mask] / d[mask]
    return w, mask


def reconstruct_normal_batch(S, X, x0, var_eps=1e-10):
    """Collection normal: per-feature no-intercept OLS slope of s_i on d_i=(x_i-x0_i)
    across a batch (the separable form of the brief's ElasticNet, without regularization).
        w_i = sum_j d_ij s_ij / sum_j d_ij^2
    Guard: coords with (sum d^2 <= var_eps) OR (var(d_i) <= var_eps) are unidentified —
    set to 0 and reported via `ident`, so poorly-identified coordinates are VISIBLE.
    Returns (w, d_var, ident_mask)."""
    S = np.asarray(S, float)
    X = np.asarray(X, float)
    x0 = np.asarray(x0, float).ravel()
    D = X - x0                       # (m, n)
    denom = np.sum(D * D, axis=0)    # (n,)
    num = np.sum(D * S, axis=0)      # (n,)
    d_var = np.var(D, axis=0)        # (n,) identifiability diagnostic
    w = np.zeros_like(denom)
    ident = (denom > var_eps) & (d_var > var_eps)
    w[ident] = num[ident] / denom[ident]
    return w, d_var, ident


def finite_diff_normal(f, x, feature_ranges, frac=1e-2, cont_mask=None):
    """True local normal g = central-difference gradient of scalar output f at x, in f's
    output space. Step_i = frac * range_i. Categorical axes (cont_mask False) -> NaN
    (a finite difference is meaningless on a discrete axis)."""
    x = np.asarray(x, float).ravel()
    n = x.size
    g = np.full(n, np.nan)
    for i in range(n):
        if cont_mask is not None and not cont_mask[i]:
            continue
        lo, hi = feature_ranges[i]
        step = frac * (hi - lo) if hi > lo else frac
        xp = x.copy(); xp[i] += step
        xm = x.copy(); xm[i] -= step
        g[i] = (float(f(xp.reshape(1, -1))[0]) - float(f(xm.reshape(1, -1))[0])) / (2.0 * step)
    return g


def _lime_normal(lime_expl, x, predict_fn, n_features, label=1, num_samples=1000):
    """Local boundary normal from LIME = coefficients of LIME's locally-weighted linear
    surrogate. LIME fits that surrogate on STANDARDIZED features (lime_tabular.py:453), so
    its coefficients live in scaled space; convert back to RAW feature space via
    w_raw_i = coef_i / scaler.scale_i (verified against the installed lime source). Same
    output space as predict_fn (prob, class-`label`). Explainer MUST be built with
    discretize_continuous=False so the coefficient is an actual per-axis slope (a gradient
    estimate), not a bin-indicator weight. Returns a raw-space vector of length n_features."""
    exp = lime_expl.explain_instance(np.asarray(x, float).ravel(), predict_fn,
                                     labels=(label,), num_features=n_features,
                                     num_samples=num_samples)
    w = np.zeros(n_features, float)
    for fi, wt in exp.local_exp[label]:
        w[fi] = wt
    scale = np.asarray(lime_expl.scaler.scale_, float).ravel()
    scale = np.where(np.abs(scale) < 1e-12, 1.0, scale)
    return w / scale


def _topk_overlap(imp, g, top_k):
    """Feature-SELECTION agreement: fraction of the top-k |imp| features that also fall in
    the top-k |g| set, over coords where g is finite (continuous axes only). This is the
    quantity the ±eps attack actually cares about — does the explanation pick the axes the
    true local normal says matter? NaN if fewer than top_k continuous coords exist."""
    imp = np.asarray(imp, float).ravel()
    g = np.asarray(g, float).ravel()
    idx = np.where(np.isfinite(g) & np.isfinite(imp))[0]
    if idx.size < top_k:
        return np.nan
    gk = idx[np.argsort(-np.abs(g[idx]))[:top_k]]
    ik = idx[np.argsort(-np.abs(imp[idx]))[:top_k]]
    return len(set(gk.tolist()) & set(ik.tolist())) / float(top_k)


def _output_fn_and_explainer(t_model, model_name, output_space, med):
    """Return (f_out, explainer, uses_linear) for the requested output space, all w.r.t.
    the single median baseline `med` (matching the attack's KernelExplainer setup)."""
    med = np.asarray(med, float).reshape(1, -1)
    if output_space == 'prob':
        f_out = lambda X: np.asarray(t_model.predict_proba(np.asarray(X, float)))[:, 1]
        expl = shap.KernelExplainer(f_out, med, normalize=False)
        return f_out, expl, False
    elif output_space == 'margin':
        if model_name == 'lr':
            f_out = lambda X: t_model.decision_function(np.asarray(X, float)).ravel()
            expl = shap.LinearExplainer(t_model, med, feature_perturbation='interventional')
            return f_out, expl, True
        elif model_name == 'nb':
            def f_out(X):
                lp = t_model.predict_log_proba(np.asarray(X, float))
                return lp[:, 1] - lp[:, 0]
            expl = shap.KernelExplainer(f_out, med, normalize=False)
            return f_out, expl, False
        elif model_name == 'mlp':
            def f_out(X):  # logit(prob) pseudo-margin
                p = np.clip(np.asarray(t_model.predict_proba(np.asarray(X, float)))[:, 1], 1e-6, 1 - 1e-6)
                return np.log(p / (1 - p))
            expl = shap.KernelExplainer(f_out, med, normalize=False)
            return f_out, expl, False
        else:
            raise ValueError(f"margin output space not available for model '{model_name}'")
    else:
        raise ValueError(f"unknown output_space '{output_space}'")


def _true_linear_weights(t_model, model_name):
    """Closed-form margin weights for the exact one-shot check (LR / NB only)."""
    if model_name == 'lr':
        return np.asarray(t_model.coef_).ravel()
    if model_name == 'nb':
        flp = np.asarray(t_model.feature_log_prob_)
        return (flp[1] - flp[0]).ravel()
    return None


def run_shap_normal_diagnostic(which_dataset, which_model, output_space='prob',
                               k_sweep=(30, 50, 100, 200), n_eval=150, aux_size=350,
                               top_k=3, fd_frac=1e-2, shap_nsamples=128, seed=0,
                               include_lime=False, lime_num_samples=1000,
                               lime_sample_around=True, verbose=True):
    """Reconstruct the local boundary normal from SHAP and measure (a) how well it matches
    the true FD normal g and (b) how redundant it is with the attack's perturbation
    direction p. Read-only; reuses the harness loaders. Binary targets only.

    Returns a dict of aggregated cosines (overall, per-k, per confidence-bin), the
    one-shot correctness check (margin + LR/NB), and the config used.
    """
    args1, args2 = load_dataset(which_dataset)
    X_train, X_test, y_train, y_test, X_test_t, X_test_s, y_test_t, y_test_s = args1
    (classes, features, n_classes, n_features, isCategorical, epsilon_set, canNegative,
     classPossibilities, dataset_name, feature_ranges) = args2
    if n_classes != 2:
        raise ValueError(f"{dataset_name} is {n_classes}-class; diagnostic is binary-only "
                         f"(class-1-only SHAP is invalid for multiclass).")

    t_model, model_name = load_model(which_model, X_train, y_train)
    Xtr = np.asarray(X_train, float)
    med = np.median(Xtr, axis=0)                 # single baseline x0 (matches attack setup)
    cont_mask = [not c for c in isCategorical]

    f_out, explainer, _ = _output_fn_and_explainer(t_model, model_name, output_space, med)

    # LIME normal (prob space only — LIME's predict_fn is a probability function). Built with
    # discretize_continuous=False so its coefficients are per-axis slopes (a gradient estimate),
    # making them directly comparable to g / w_single. This isolates the mechanism hypothesis:
    # LIME's local linear fit recovers the boundary normal, default SHAP does not.
    lime_expl = lime_predict = None
    do_lime = include_lime and output_space == 'prob'
    if include_lime and output_space != 'prob':
        print(f"  [lime skipped for {dataset_name}/{model_name}] LIME normal is prob-space only "
              f"(output_space={output_space}).")
    if do_lime:
        lime_expl = lime.lime_tabular.LimeTabularExplainer(
            Xtr, mode='classification', discretize_continuous=False,
            sample_around_instance=lime_sample_around, random_state=seed)
        lime_predict = lambda X: np.asarray(t_model.predict_proba(np.asarray(X, float)))

    rng = np.random.default_rng(seed)
    bank_n = min(aux_size, Xtr.shape[0])
    bank_idx = rng.choice(Xtr.shape[0], size=bank_n, replace=False)
    bank = Xtr[bank_idx]
    ks = [k for k in k_sweep if k < bank_n]       # need k neighbors excluding self
    if not ks:
        ks = [max(1, bank_n - 1)]

    # SHAP for the whole bank once (cached; reused for eval points and their neighbors).
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            # l1_reg=0: disable shap's default L1 sparsification, which would zero out
            # features and break the linear-reconstruction identity s_i = w_i*(x_i-x0_i).
            S_bank = np.asarray(explainer.shap_values(bank, nsamples=shap_nsamples,
                                                      l1_reg=0, silent=True))
        except TypeError:  # LinearExplainer has no nsamples/silent/l1_reg kwargs
            S_bank = np.asarray(explainer.shap_values(bank))
    if S_bank.ndim == 3:                          # (m, n, out) -> class-1 / first out
        S_bank = S_bank[:, :, min(1, S_bank.shape[2] - 1)]

    # kNN index in standardized input space.
    scaler = StandardScaler().fit(bank)
    bank_z = scaler.transform(bank)
    max_k = max(ks)
    nn = NearestNeighbors(n_neighbors=min(max_k + 1, bank_n)).fit(bank_z)
    _, nbr_idx = nn.kneighbors(bank_z)             # includes self at col 0

    eval_m = min(n_eval, bank_n)
    eval_rows = rng.choice(bank_n, size=eval_m, replace=False)

    w_true = _true_linear_weights(t_model, model_name) if output_space == 'margin' else None

    rec = {'single_g': [], 'single_p': [], 'batch_g': {k: [] for k in ks},
           'batch_p': {k: [] for k in ks}, 'batch_single': {k: [] for k in ks},
           'conf': [], 'single_g_by_conf': {}, 'ident_frac': [], 'oneshot_cos': [],
           'oneshot_maxabs': [], 'single_ident_frac': [], 'g_p': [],
           # LIME arms (paired with SHAP on the SAME eval points):
           'lime_g': [], 'lime_single': [], 'shap_topk_hit': [], 'lime_topk_hit': []}

    for r in eval_rows:
        x = bank[r]
        s = S_bank[r]
        w_single, id_s = reconstruct_normal_single(s, x, med)
        rec['single_ident_frac'].append(float(id_s.mean()))
        g = finite_diff_normal(f_out, x, feature_ranges, frac=fd_frac, cont_mask=cont_mask)
        # attack direction p = top-k |s| axes (same output space as w by construction)
        p = np.zeros_like(s)
        top = np.argsort(-np.abs(s))[:top_k]
        p[top] = s[top]

        rec['single_g'].append(_cosine_masked(w_single, g))
        rec['single_p'].append(_cosine_masked(w_single, p))
        rec['g_p'].append(_cosine_masked(g, p))   # clean redundancy: true normal vs attack dir
        rec['shap_topk_hit'].append(_topk_overlap(s, g, top_k))

        if do_lime:
            w_lime = _lime_normal(lime_expl, x, lime_predict, n_features,
                                  num_samples=lime_num_samples)
            rec['lime_g'].append(_cosine_masked(w_lime, g))         # LIME recovers true normal?
            rec['lime_single'].append(_cosine_masked(w_lime, w_single))  # LIME vs SHAP normal
            rec['lime_topk_hit'].append(_topk_overlap(w_lime, g, top_k))  # feature-selection vs g

        proba = np.asarray(t_model.predict_proba(x.reshape(1, -1)))[0]
        conf = float(proba.max())
        rec['conf'].append(conf)

        if w_true is not None:
            # correctness on IDENTIFIABLE coords only (x_i==x0_i is a 0/0, uninformative)
            m = id_s & np.isfinite(w_single) & np.isfinite(w_true)
            if m.any():
                rec['oneshot_cos'].append(_cosine_masked(w_single[m], w_true[m]))
                rec['oneshot_maxabs'].append(float(np.max(np.abs(w_single[m] - w_true[m]))))
            else:
                rec['oneshot_cos'].append(np.nan)
                rec['oneshot_maxabs'].append(np.nan)

        for k in ks:
            nb = nbr_idx[r][1:k + 1]               # drop self
            w_b, d_var, ident = reconstruct_normal_batch(S_bank[nb], bank[nb], med)
            rec['batch_g'][k].append(_cosine_masked(w_b, g))
            rec['batch_p'][k].append(_cosine_masked(w_b, p))
            rec['batch_single'][k].append(_cosine_masked(w_b, w_single))
            if k == max_k:
                rec['ident_frac'].append(float(ident.mean()))

    def _m(v):
        v = np.asarray(v, float)
        v = v[np.isfinite(v)]
        return (float(np.mean(v)), float(np.std(v)), int(v.size)) if v.size else (np.nan, np.nan, 0)

    conf = np.asarray(rec['conf'])
    bins = [(0.5, 0.7), (0.7, 0.9), (0.9, 1.01)]
    single_g = np.asarray(rec['single_g'])
    by_conf = {}
    for lo, hi in bins:
        sel = (conf >= lo) & (conf < hi)
        by_conf[f"{lo:.1f}-{hi:.1f}"] = _m(single_g[sel]) if sel.any() else (np.nan, np.nan, 0)

    out = {
        'dataset': dataset_name, 'model': model_name, 'output_space': output_space,
        'n_eval': eval_m, 'bank_n': bank_n, 'k_sweep': ks, 'top_k': top_k,
        'fd_frac': fd_frac, 'shap_nsamples': shap_nsamples, 'seed': seed,
        'cos_single_g': _m(rec['single_g']),
        'cos_single_p': _m(rec['single_p']),
        'cos_g_p': _m(rec['g_p']),
        'cos_batch_g': {k: _m(rec['batch_g'][k]) for k in ks},
        'cos_batch_p': {k: _m(rec['batch_p'][k]) for k in ks},
        'cos_batch_single': {k: _m(rec['batch_single'][k]) for k in ks},
        'cos_single_g_by_conf': by_conf,
        'ident_frac_maxk': _m(rec['ident_frac']),
        'single_ident_frac': _m(rec['single_ident_frac']),
        'oneshot_cos_w_true': _m(rec['oneshot_cos']) if w_true is not None else None,
        'oneshot_maxabs_err': _m(rec['oneshot_maxabs']) if w_true is not None else None,
        # SHAP top-k feature-selection agreement with the true normal (always computed):
        'topk_hit_shap_g': _m(rec['shap_topk_hit']),
        # LIME arms (None when include_lime is off / not prob space):
        'include_lime': do_lime,
        'lime_num_samples': lime_num_samples if do_lime else None,
        'cos_lime_g': _m(rec['lime_g']) if do_lime else None,
        'cos_lime_single': _m(rec['lime_single']) if do_lime else None,
        'topk_hit_lime_g': _m(rec['lime_topk_hit']) if do_lime else None,
    }
    if verbose:
        print(f"[{dataset_name}/{model_name}/{output_space}] "
              f"cos(w_single,g)={out['cos_single_g'][0]:.3f}  "
              f"cos(w_single,p)={out['cos_single_p'][0]:.3f}  "
              f"cos(w_batch@{max(ks)},g)={out['cos_batch_g'][max(ks)][0]:.3f}  "
              f"cos(w_batch@{max(ks)},p)={out['cos_batch_p'][max(ks)][0]:.3f}")
        if do_lime:
            print(f"    LIME: cos(w_lime,g)={out['cos_lime_g'][0]:.3f}  "
                  f"cos(w_lime,w_single)={out['cos_lime_single'][0]:.3f}  "
                  f"topk_hit(g): lime={out['topk_hit_lime_g'][0]:.2f} vs "
                  f"shap={out['topk_hit_shap_g'][0]:.2f}")
        if w_true is not None:
            print(f"    one-shot: cos(w_single,w_true)={out['oneshot_cos_w_true'][0]:.4f}  "
                  f"max|err|={out['oneshot_maxabs_err'][0]:.3e}")
    return out


def _local_influence(f_prob, x, isCat, classPossibilities, feature_ranges, cont_steps=(0.25, 0.5)):
    """Ground-truth local feature influence on the class-1 probability: for each feature, the
    max |Δ f_prob| achievable by changing ONLY that feature. Categorical -> try every OTHER
    category; continuous -> try +/- cont_steps * range. This is the discrete/local analog of
    |g| ('how much does this feature control the local prediction?') — exactly the property a
    +/-eps perturbation attack wants the features it SELECTS to have. Returns (infl, n_queries)."""
    x = np.asarray(x, float).ravel()
    n = x.size
    base = float(f_prob(x.reshape(1, -1))[0])
    infl = np.zeros(n)
    q = 0
    for i in range(n):
        if isCat[i]:
            cand = [v for v in range(int(classPossibilities[i])) if v != int(round(x[i]))]
        else:
            lo, hi = feature_ranges[i]
            rng = (hi - lo) if hi > lo else 1.0
            cand = []
            for s in cont_steps:
                cand += [x[i] + s * rng, x[i] - s * rng]
        best = 0.0
        for v in cand:
            xp = x.copy(); xp[i] = v
            best = max(best, abs(float(f_prob(xp.reshape(1, -1))[0]) - base))
            q += 1
        infl[i] = best
    return infl, q


def run_lime_shap_selection_diagnostic(which_dataset, which_model, n_eval=60, aux_size=350,
                                       top_k=3, shap_nsamples=128, lime_num_samples=1000,
                                       seed=0, verbose=True):
    """Does LIME's feature SELECTION pick locally-influential features better than SHAP (and
    better than random)? This is the exact channel the Autolycus LIME-vs-random ablation
    measured. Faithful to the attack: BOTH explainers explain the class-1 probability
    (load_explainer uses predict_proba[:,1] for SHAP; LIME's default label is 1), SHAP uses
    its DEFAULT L1 sparsification (as the attack does), LIME is discretize_continuous=True
    (as load_explainer builds it). Ground truth = `_local_influence` per feature.

    Reports, over `n_eval` points: top-k feature-selection agreement with the true local
    influence for LIME vs SHAP vs the random baseline (k/n_features), plus the soft
    'influence captured' ratio. Works for multiclass (class-1 proba is always defined)."""
    args1, args2 = load_dataset(which_dataset)
    X_train = args1[0]
    (classes, features, n_classes, n_features, isCategorical, epsilon_set, canNegative,
     classPossibilities, dataset_name, feature_ranges) = args2
    t_model, model_name = load_model(which_model, X_train, args1[2])
    Xtr = np.asarray(X_train, float)
    med = np.median(Xtr, axis=0).reshape(1, -1)

    f_prob1 = lambda X: np.asarray(t_model.predict_proba(np.asarray(X, float)))[:, 1]
    shap_expl = shap.KernelExplainer(f_prob1, med, normalize=False)
    lime_expl = lime.lime_tabular.LimeTabularExplainer(Xtr, discretize_continuous=True,
                                                       random_state=seed)

    rng = np.random.default_rng(seed)
    bank_n = min(aux_size, Xtr.shape[0])
    bank = Xtr[rng.choice(Xtr.shape[0], size=bank_n, replace=False)]
    eval_rows = rng.choice(bank_n, size=min(n_eval, bank_n), replace=False)

    feat_type = ('categorical' if all(isCategorical) else
                 'continuous' if not any(isCategorical) else 'mixed')

    hit_l, hit_s, capt_l, capt_s = [], [], [], []
    for r in eval_rows:
        x = bank[r]
        infl, _ = _local_influence(f_prob1, x, isCategorical, classPossibilities, feature_ranges)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s = np.asarray(shap_expl.shap_values(x.reshape(1, -1), nsamples=shap_nsamples,
                                                 silent=True)).ravel()   # DEFAULT l1_reg (attack)
        imp_shap = np.abs(s)

        exp = lime_expl.explain_instance(x, t_model.predict_proba, num_features=n_features,
                                         num_samples=lime_num_samples)
        key = list(exp.as_map().keys())[0]
        imp_lime = np.zeros(n_features)
        for fi, wt in exp.local_exp[key]:
            imp_lime[fi] = abs(wt)

        hit_l.append(_topk_overlap(imp_lime, infl, top_k))
        hit_s.append(_topk_overlap(imp_shap, infl, top_k))
        true_top = np.argsort(-infl)[:top_k]
        denom = infl[true_top].sum()
        if denom > 0:
            capt_l.append(infl[np.argsort(-imp_lime)[:top_k]].sum() / denom)
            capt_s.append(infl[np.argsort(-imp_shap)[:top_k]].sum() / denom)

    def _m(v):
        v = np.asarray(v, float); v = v[np.isfinite(v)]
        return (float(np.mean(v)), float(np.std(v)), int(v.size)) if v.size else (np.nan, np.nan, 0)

    out = {'dataset': dataset_name, 'model': model_name, 'feat_type': feat_type,
           'n_eval': int(eval_rows.size), 'top_k': top_k, 'n_features': n_features,
           'hit_lime': _m(hit_l), 'hit_shap': _m(hit_s),
           'hit_random_expected': top_k / float(n_features),
           'capt_lime': _m(capt_l), 'capt_shap': _m(capt_s)}
    if verbose:
        print(f"[{dataset_name}/{model_name}/{feat_type}] "
              f"hit(infl): lime={out['hit_lime'][0]:.2f} shap={out['hit_shap'][0]:.2f} "
              f"rand~{out['hit_random_expected']:.2f}  |  "
              f"captured: lime={out['capt_lime'][0]:.2f} shap={out['capt_shap'][0]:.2f}")
    return out


def _parse_lime_rule(rule):
    """Extract (low, high) numeric bin edges from a LIME discretization rule string, mirroring
    `explanation_parser`. Forms: 'f <= v' / 'f < v' -> high; 'f > v' / 'f >= v' -> low;
    'a < f <= b' -> (a, b). Returns (low_or_None, high_or_None)."""
    txt = rule.split(' ')
    low = high = None
    try:
        if len(txt) >= 5:                       # a < f <= b
            low, high = float(txt[0]), float(txt[-1])
        elif len(txt) >= 3:                      # f <op> v
            val = float(txt[-1])
            if txt[-2] in ('<=', '<'):
                high = val
            else:
                low = val
    except ValueError:
        pass
    return low, high


def run_lime_threshold_diagnostic(which_dataset, which_model, n_eval=60, aux_size=350,
                                  top_k=3, epsilon=1.0, lime_num_samples=1000, seed=0,
                                  verbose=True):
    """Explain the MAGNITUDE of the LIME-vs-random ablation by decomposing the LIME attack move
    into two channels, measured as boundary-CROSSING rate (a perturbation that flips the predicted
    class = a boundary-informative training sample). Faithful to `traverse_explanations_LIME`:
    LIME picks top-k features by |weight| and snaps each to its bin edge (high+eps / low-eps);
    the random ablation picks random features and steps current +/- eps.

      cr_lime_bin   = LIME features, snapped to bin edge +/- eps   (full LIME move)
      cr_lime_naive = LIME features, current +/- eps               (feature choice only)
      cr_rand_naive = random features, current +/- eps             (the random ablation)
      total   = cr_lime_bin - cr_rand_naive   (mirrors the ablation)
      feat_ch = cr_lime_naive - cr_rand_naive (value of LIME's feature SELECTION)
      thr_ch  = cr_lime_bin - cr_lime_naive   (value of LIME's bin-edge THRESHOLD)

    Candidate validity replicates the traversal's check (0 <= v < classPossibilities[f])."""
    args1, args2 = load_dataset(which_dataset)
    X_train = args1[0]
    (classes, features, n_classes, n_features, isCategorical, epsilon_set, canNegative,
     classPossibilities, dataset_name, feature_ranges) = args2
    t_model, model_name = load_model(which_model, X_train, args1[2])
    Xtr = np.asarray(X_train, float)
    lime_expl = lime.lime_tabular.LimeTabularExplainer(Xtr, discretize_continuous=True,
                                                       random_state=seed)
    rng = np.random.default_rng(seed)
    bank_n = min(aux_size, Xtr.shape[0])
    bank = Xtr[rng.choice(Xtr.shape[0], size=bank_n, replace=False)]
    eval_rows = rng.choice(bank_n, size=min(n_eval, bank_n), replace=False)
    feat_type = ('categorical' if all(isCategorical) else
                 'continuous' if not any(isCategorical) else 'mixed')

    def crossed(x, f, v, base):
        if v < 0 or v >= classPossibilities[f]:
            return None                          # invalid candidate (as the traversal rejects)
        xp = x.copy(); xp[f] = v
        return 1.0 if int(t_model.predict(xp.reshape(1, -1))[0]) != base else 0.0

    cr_lime_bin, cr_lime_naive, cr_rand_naive = [], [], []
    for r in eval_rows:
        x = bank[r]
        base = int(t_model.predict(x.reshape(1, -1))[0])
        exp = lime_expl.explain_instance(x, t_model.predict_proba, num_features=n_features,
                                         num_samples=lime_num_samples)
        key = list(exp.as_map().keys())[0]
        order = exp.local_exp[key]               # (feat_idx, weight), importance-sorted
        rules = [rr for rr, _ in exp.as_list(key)]  # same order, rule strings with bin edges
        for i in range(min(top_k, len(order))):
            f = int(order[i][0])
            low, high = _parse_lime_rule(rules[i]) if i < len(rules) else (None, None)
            for edge, sign in ((high, +1.0), (low, -1.0)):   # LIME move: bin edge +/- eps
                if edge is not None:
                    c = crossed(x, f, edge + sign * epsilon, base)
                    if c is not None:
                        cr_lime_bin.append(c)
            for sign in (+1.0, -1.0):                        # same features, naive step
                c = crossed(x, f, x[f] + sign * epsilon, base)
                if c is not None:
                    cr_lime_naive.append(c)
        for f in rng.choice(n_features, size=min(top_k, n_features), replace=False):  # random ablation
            for sign in (+1.0, -1.0):
                c = crossed(x, int(f), x[int(f)] + sign * epsilon, base)
                if c is not None:
                    cr_rand_naive.append(c)

    def _m(v):
        v = np.asarray(v, float)
        return (float(np.mean(v)), int(v.size)) if v.size else (np.nan, 0)

    crb, cln, crn = _m(cr_lime_bin), _m(cr_lime_naive), _m(cr_rand_naive)
    out = {'dataset': dataset_name, 'model': model_name, 'feat_type': feat_type,
           'n_eval': int(eval_rows.size), 'top_k': top_k, 'epsilon': epsilon,
           'cr_lime_bin': crb, 'cr_lime_naive': cln, 'cr_rand_naive': crn,
           'total': crb[0] - crn[0], 'feat_ch': cln[0] - crn[0], 'thr_ch': crb[0] - cln[0]}
    if verbose:
        print(f"[{dataset_name}/{model_name}/{feat_type}] cross: bin={crb[0]:.3f} "
              f"limeNaive={cln[0]:.3f} rand={crn[0]:.3f}  |  total={out['total']:+.3f} "
              f"feat_ch={out['feat_ch']:+.3f} thr_ch={out['thr_ch']:+.3f}")
    return out


def _fit_one_surrogate(model_name, v_samples_np, v_pred_dec, n_classes, depth=15, seed=0):
    """Fit a SINGLE surrogate of the target's family on traversed samples, mirroring the
    per-family choices in `_build_surrogate_and_eval` (dt/rdf depth=15; knn k=n_classes; etc.)
    but returning the fitted model instead of a score. mlp is out of scope (architecture
    selection) and raises."""
    if model_name == 'dt':
        s_model = dt(random_state=seed, max_depth=depth)
    elif model_name == 'lr':
        s_model = lr(max_iter=1000, random_state=seed)
    elif model_name == 'nb':
        s_model = mnb()
    elif model_name == 'rdf':
        s_model = rf(max_depth=depth, random_state=seed)
    elif model_name == 'knn':
        s_model = knn(n_neighbors=n_classes)
    else:
        raise ValueError(f"disagreement diagnostic does not support model '{model_name}'")
    s_model.fit(v_samples_np, v_pred_dec)
    return s_model


def run_disagreement_diagnostic(which_dataset, which_model, query_limit=500, nfe=3,
                                set_size=5, seed=0, conf_thr=0.8, n_eval=500,
                                verbose=True, save_plot=True):
    """Are the surrogate/target DISAGREEMENTS coverage errors or boundary errors?

    Motivation (observed): points the surrogate labels differently from the target often carry
    HIGH target confidence. A high-confidence point sits deep in a target class region, far from
    the target's boundary -- boundary search cannot fix it, only better COVERAGE can. This
    diagnostic decides which failure mode dominates, i.e. which half of the two-part method
    (diverse generation vs boundary search) to invest in.

    Method: run the real attack (SHAP3) to build ONE surrogate, then over the held-out target
    test set X_test_t measure, per point:
      - conf  = target predict_proba.max()                         (how interior the point is)
      - d_opp = normalized distance (categorical-aware, via `_compute_dist_matrix`) to the
                nearest test point of a DIFFERENT target-predicted class                (boundary proximity)
    A disagreement is INTERIOR/coverage if conf > conf_thr AND d_opp is above the agreement-set
    median (farther from the boundary than a typical correct point); otherwise BOUNDARY.
    d_opp reuses the harness's own metric (Hamming on categoricals, range-normalized L2 on
    continuous), so it is meaningful on categorical data where a continuous bisection would go
    off-manifold. Writes disagreement_diag.json (+ a conf-vs-d_opp scatter).
    """
    random.seed(seed)
    np.random.seed(seed)

    args1, args2 = load_dataset(which_dataset)
    X_train, X_test, y_train, y_test, X_test_t, X_test_s, y_test_t, y_test_s = args1
    (classes, features, n_classes, n_features, isCategorical, epsilon_set, canNegative,
     classPossibilities, dataset_name, feature_ranges) = args2
    t_model, model_name = load_model(which_model, X_train, y_train)
    t_explainer = load_explainer(1, t_model, model_name, X_train)  # SHAP, as SHAP3 expects

    # --- run the real attack (SHAP3) to produce the training set, then fit one surrogate ---
    samples_mega = mega_sample_generation(X_test_s.to_numpy(), y_test_s, n_classes, [set_size], 1)
    relax_factor = 0.5
    lb = int((query_limit // n_classes) * (1 - relax_factor) + 1)
    ub = int((query_limit // n_classes) * (n_classes + relax_factor) + 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        v_samples_np, v_pred_dec, n_query = traverse_explanations_SHAP3(
            samples_mega[0][0], t_explainer, t_model, lb, ub, query_limit, nfe, args2,
            model_name, X_train, y_train)
    s_model = _fit_one_surrogate(model_name, v_samples_np, v_pred_dec, n_classes, seed=seed)

    # --- evaluate on the target test set ---
    Xt = np.asarray(X_test_t.values, dtype=float)
    rng = np.random.default_rng(seed)
    if Xt.shape[0] > n_eval:
        sel = rng.choice(Xt.shape[0], size=n_eval, replace=False)
        Xt = Xt[sel]
    fidelity = float(np.mean(np.argmax(t_model.predict_proba(Xt), axis=1) ==
                             np.argmax(s_model.predict_proba(Xt), axis=1)))

    t_pred = np.asarray(t_model.predict(Xt))
    s_pred = np.asarray(s_model.predict(Xt))
    conf = t_model.predict_proba(Xt).max(axis=1)
    disagree = (t_pred != s_pred)

    # d_opp: nearest DIFFERENT-target-class distance, categorical-aware, self excluded.
    D = _compute_dist_matrix(Xt, Xt, n_features, isCategorical, feature_ranges)
    D = np.sqrt(np.maximum(D, 0.0))
    same_class = (t_pred[:, None] == t_pred[None, :])
    D[same_class] = np.inf          # keep only opposite-class candidates
    d_opp = D.min(axis=1)           # inf only if a class is a singleton in the eval set
    finite = np.isfinite(d_opp)

    ag = finite & ~disagree
    dg = finite & disagree
    agree_med_d = float(np.median(d_opp[ag])) if ag.any() else float('nan')

    interior = dg & (conf > conf_thr) & (d_opp > agree_med_d)
    boundary = dg & ~((conf > conf_thr) & (d_opp > agree_med_d))

    def _stats(mask):
        if not np.any(mask):
            return {'n': 0, 'conf_mean': None, 'conf_med': None, 'frac_highconf': None,
                    'd_med': None}
        c, d = conf[mask], d_opp[mask]
        return {'n': int(mask.sum()), 'conf_mean': round(float(c.mean()), 4),
                'conf_med': round(float(np.median(c)), 4),
                'frac_highconf': round(float(np.mean(c > conf_thr)), 4),
                'd_med': round(float(np.median(d[np.isfinite(d)])), 4) if np.isfinite(d).any() else None}

    n_dis = int(dg.sum())
    out = {
        'dataset': dataset_name, 'model': model_name,
        'feat_type': ('categorical' if all(isCategorical) else
                      'continuous' if not any(isCategorical) else 'mixed'),
        'n_eval': int(finite.sum()), 'n_query': int(n_query), 'fidelity': round(fidelity, 4),
        'conf_thr': conf_thr, 'agree_median_d_opp': round(agree_med_d, 4),
        'disagree': _stats(dg), 'agree': _stats(ag),
        'interior': _stats(interior), 'boundary': _stats(boundary),
        'frac_disagree_interior': round(interior.sum() / n_dis, 4) if n_dis else None,
        'verdict': None,
    }
    # Verdict: coverage-dominant if most disagreements are interior high-confidence.
    if n_dis:
        fi = interior.sum() / n_dis
        out['verdict'] = ('coverage-dominant' if fi >= 0.5 else
                          'boundary-dominant' if fi <= 0.25 else 'mixed')

    if verbose:
        print(f"\n[{dataset_name}/{model_name}] fidelity={fidelity:.3f}  n_query={n_query}  "
              f"eval={out['n_eval']}  disagreements={n_dis}")
        print(f"  agree:    conf_med={out['agree']['conf_med']}  d_med={out['agree']['d_med']}")
        print(f"  disagree: conf_med={out['disagree']['conf_med']}  d_med={out['disagree']['d_med']}  "
              f"frac_highconf={out['disagree']['frac_highconf']}")
        print(f"  -> interior(coverage)={out['interior']['n']}  boundary={out['boundary']['n']}  "
              f"frac_interior={out['frac_disagree_interior']}  VERDICT={out['verdict']}")

    if save_plot and finite.any():
        plt.figure(figsize=(7, 5))
        plt.scatter(d_opp[ag], conf[ag], s=14, c='tab:blue', alpha=0.4, label='agree')
        plt.scatter(d_opp[boundary], conf[boundary], s=26, c='tab:orange', alpha=0.8,
                    label='disagree-boundary')
        plt.scatter(d_opp[interior], conf[interior], s=34, c='tab:red', alpha=0.9,
                    marker='^', label='disagree-interior(coverage)')
        plt.axhline(conf_thr, ls='--', c='gray', lw=1)
        plt.axvline(agree_med_d, ls='--', c='gray', lw=1)
        plt.xlabel('d_opp: normalized distance to nearest opposite-class point (boundary proximity)')
        plt.ylabel('target confidence  predict_proba.max()')
        plt.title(f"{dataset_name}/{model_name}: disagreements  "
                  f"(interior={out['interior']['n']}, boundary={out['boundary']['n']})")
        plt.legend(loc='lower right', fontsize=8)
        fname = f"disagreement_{dataset_name}_{model_name}.png"
        plt.tight_layout()
        plt.savefig(fname, dpi=140)
        plt.close()
        out['plot'] = fname
        if verbose:
            print(f"  wrote {fname}")

    return out


def _run_one_traversal(method, sample_set, t_explainer, t_model, lb, ub, query_limit, nfe,
                       args2, model_name, X_train, y_train):
    """Dispatch to base Autolycus (`traverse_explanations_SHAP`, no diverse / no boundary search)
    or the two-part method (`traverse_explanations_SHAP3`, diverse + boundary bisection)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if method == 'base':
            return traverse_explanations_SHAP(sample_set, t_explainer, t_model, lb, ub,
                                              query_limit, nfe, args2, model_name, X_train, y_train)
        elif method == 'shap3':
            return traverse_explanations_SHAP3(sample_set, t_explainer, t_model, lb, ub,
                                               query_limit, nfe, args2, model_name, X_train, y_train)
        elif method == 'shap3_nodiv':   # boundary bisection only (Phase 1 diverse OFF)
            return traverse_explanations_SHAP3(sample_set, t_explainer, t_model, lb, ub,
                                               query_limit, nfe, args2, model_name, X_train, y_train,
                                               use_diverse=False)
        elif method in ('lime3', 'lime3_nodiv'):   # LIME analog of SHAP3 (t_explainer ignored)
            lime_expl = lime.lime_tabular.LimeTabularExplainer(X_train.values, discretize_continuous=True)
            return traverse_explanations_LIME3(sample_set, lime_expl, t_model, lb, ub,
                                               query_limit, nfe, args2, model_name, X_train, y_train,
                                               use_diverse=(method == 'lime3'))
        raise ValueError(f"unknown method '{method}'")


def run_confidence_gap_diagnostic(which_dataset, which_model, how_many_sets=10, query_limit=500,
                                  nfe=3, set_size=5, seed=0, methods=('base', 'shap3'),
                                  verbose=True):
    """Reproduce the 'disagreed vs agreed avg TARGET confidence' table per sample set, and compare
    base Autolycus (no diverse / no boundary search) against the two-part method (SHAP3).

    The motivating observation was measured on BASE Autolycus: on some combos (e.g. nursery/lr) the
    surrogate/target DISAGREEMENTS carry HIGHER target confidence than the AGREEMENTS -> a negative
    (agreed - disagreed) gap = the surrogate is confidently wrong in interior regions it never
    covered. The question this answers: does adding diverse generation + boundary search (SHAP3)
    shrink disagreed confidence / turn the gap positive (errors pushed back onto the boundary)?

    Confidence is the TARGET's predict_proba.max() (a fixed property of each test point, method-
    independent); only the surrogate (hence the disagree mask) changes with method. Per set it
    reports: disagreed avg conf, agreed avg conf, gap = agreed - disagreed, similarity (fidelity).
    Writes confidence_gap_diag.json.
    """
    random.seed(seed)
    np.random.seed(seed)

    args1, args2 = load_dataset(which_dataset)
    X_train, X_test, y_train, y_test, X_test_t, X_test_s, y_test_t, y_test_s = args1
    (classes, features, n_classes, n_features, isCategorical, epsilon_set, canNegative,
     classPossibilities, dataset_name, feature_ranges) = args2
    t_model, model_name = load_model(which_model, X_train, y_train)
    t_explainer = load_explainer(1, t_model, model_name, X_train)

    samples_mega = mega_sample_generation(X_test_s.to_numpy(), y_test_s, n_classes,
                                          [set_size], how_many_sets)   # SAME sets for all methods
    relax_factor = 0.5
    lb = int((query_limit // n_classes) * (1 - relax_factor) + 1)
    ub = int((query_limit // n_classes) * (n_classes + relax_factor) + 1)

    Xt = np.asarray(X_test_t.values, dtype=float)
    t_pred = np.asarray(t_model.predict(Xt))
    t_conf = t_model.predict_proba(Xt).max(axis=1)     # fixed, method-independent

    feat_type = ('categorical' if all(isCategorical) else
                 'continuous' if not any(isCategorical) else 'mixed')
    out = {'dataset': dataset_name, 'model': model_name, 'feat_type': feat_type,
           'how_many_sets': how_many_sets, 'query_limit': query_limit, 'methods': {}}

    for method in methods:
        rows = []
        for i in range(how_many_sets):
            v_samples, v_preds, n_query = _run_one_traversal(
                method, samples_mega[i][0], t_explainer, t_model, lb, ub,
                query_limit, nfe, args2, model_name, X_train, y_train)
            s_model = _fit_one_surrogate(model_name, v_samples, v_preds, n_classes, seed=seed)
            s_pred = np.asarray(s_model.predict(Xt))
            dis = (t_pred != s_pred)
            dg = float(t_conf[dis].mean()) if dis.any() else float('nan')
            ag = float(t_conf[~dis].mean()) if (~dis).any() else float('nan')
            rows.append({'set': i, 'n_query': int(n_query), 'n_dis': int(dis.sum()),
                         'disagreed_conf': round(dg, 4), 'agreed_conf': round(ag, 4),
                         'gap': round(ag - dg, 4), 'similarity': round(float((~dis).mean()), 4)})
        # averages across sets
        arr = lambda key: np.array([r[key] for r in rows], float)
        avg = {'disagreed_conf': round(float(np.nanmean(arr('disagreed_conf'))), 4),
               'agreed_conf': round(float(np.nanmean(arr('agreed_conf'))), 4),
               'gap': round(float(np.nanmean(arr('gap'))), 4),
               'similarity': round(float(arr('similarity').mean()), 4),
               'n_neg_gap': int((arr('gap') < 0).sum())}
        out['methods'][method] = {'rows': rows, 'avg': avg}

        if verbose:
            print(f"\n[{dataset_name}/{model_name}]  method={method}")
            print(f"  {'set':>3}{'#dis':>6}{'disagreed':>11}{'agreed':>9}{'gap':>9}{'sim':>8}")
            for r in rows:
                print(f"  {r['set']:>3}{r['n_dis']:>6}{r['disagreed_conf']:>11.4f}"
                      f"{r['agreed_conf']:>9.4f}{r['gap']:>+9.4f}{r['similarity']:>8.4f}")
            print(f"  AVG    disagreed={avg['disagreed_conf']:.4f}  agreed={avg['agreed_conf']:.4f}"
                  f"  gap={avg['gap']:+.4f}  sim={avg['similarity']:.4f}  neg-gap sets={avg['n_neg_gap']}/{how_many_sets}")

    if 'base' in out['methods'] and 'shap3' in out['methods']:
        b, s = out['methods']['base']['avg'], out['methods']['shap3']['avg']
        out['shap3_minus_base'] = {'disagreed_conf': round(s['disagreed_conf'] - b['disagreed_conf'], 4),
                                   'gap': round(s['gap'] - b['gap'], 4),
                                   'similarity': round(s['similarity'] - b['similarity'], 4)}
        if verbose:
            d = out['shap3_minus_base']
            print(f"\n  SHAP3 - base:  disagreed_conf {d['disagreed_conf']:+.4f}  "
                  f"gap {d['gap']:+.4f}  similarity {d['similarity']:+.4f}")
    return out
