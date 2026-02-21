import itertools
import lime
import lime.lime_tabular
import numpy as np
import pandas as pd
import random
import seaborn as sns
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
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.ensemble import RandomForestClassifier as rf
from sklearn.neural_network import MLPClassifier as mlp
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import os, sys
from matplotlib import cm


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


def traverse_explanations_LIME(sample_set, explainer, model, n_visits_lb, n_visits_ub, upper_limit, n_f_e, args2):
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
            # 2. Get the explanation about the current sample
            exp = explainer.explain_instance(curr, model.predict_proba)  #, top_labels=1)
            exp_map = exp.as_map()
            key = list(exp_map.keys())[0]
            exp_list = exp.as_list(key)
            exp_parsed = explanation_parser(exp_map, exp_list, key, features)

            # 3. Generate new samples and check if they were visited before
            tmp_exps, indices, cpys = [], [], []
            for i in range(k):
                tmp_exps += [exp_parsed[i]]
            for i in range(k):
                #print(tmp_exps[i][0])
                indices += [index for index, content in enumerate(features) if tmp_exps[i][0] in content]  #[0]
            for i in range(2 * k):
                cpys += [np.copy(curr)]
            for i in range(k):
                cpys[2 * i][indices[i]] = tmp_exps[i][2]
                cpys[2 * i + 1][indices[i]] = tmp_exps[i][1]
            for i in range(2 * k):
                ind_i = int(i / 2)
                #tmp = (any((cpys[i]==x).all() for x in visited_samples) or any((cpys[i]==x).all() for x in samples))
                #if (tmp and (cpys[i][indices[ind_i]] >= 0)):
                if (cpys[i][indices[ind_i]] >= 0):
                    if i % 2 == 0:
                        cpys[i][indices[ind_i]] += epsilon
                    else:
                        cpys[i][indices[ind_i]] -= epsilon
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
                               model_name, X_train=None, y_train=None, explanation_type='vanilla', num_exp = 5):
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
    return visited_samples, preds, query

def traverse_explanations_SHAP2(sample_set, explainer, model, n_visits_lb, n_visits_ub, upper_limit, n_f_e, args2,
                               model_name, X_train=None, y_train=None, explanation_type='vanilla', num_exp = 5):
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

import numpy as np

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
def create_manifold_aware_diverse_samples(samples, classPossibilities, feature_ranges, isCat, 
                                         num_desired_samples=10, pool_size=1000, mutation_rate=0.7):
    # 1. SANITIZE INPUTS: Ensure no NaNs exist in the input samples or ranges
    current_pool = np.array(samples, dtype=float)
    if np.any(np.isnan(current_pool)):
        # Fill NaNs with 0 or mean as a fallback
        current_pool = np.nan_to_num(current_pool)
        
    n_features = len(classPossibilities)
    new_diverse_samples = []

    for _ in range(num_desired_samples):
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
                        # 2. GUARD: If range is 0 or NaN, use a small default epsilon or skip
                        if np.isnan(low) or np.isnan(high) or low == high:
                            # If feature is constant, don't mutate it
                            continue 
                            
                        std_dev = (high - low) * 0.1
                        noise = np.random.normal(0, std_dev)
                        candidate[i] = np.clip(candidate[i] + noise, low, high)
            
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

def find_boundary_point(target_model, x_a, x_b, iterations=10):
    """
    Finds a point close to the decision boundary between x_a and x_b.
    """
    # Get initial predictions
    label_a = np.argmax(target_model.predict(x_a.reshape(1, -1)))
    label_b = np.argmax(target_model.predict(x_b.reshape(1, -1)))
    
    if label_a == label_b:
        raise ValueError("Points x_a and x_b must have different predicted classes.")

    low = 0.0
    high = 1.0
    boundary_sample = x_a
    query_count = 0

    for _ in range(iterations):
        mid = (low + high) / 2
        # Interpolate between A and B
        x_mid = x_a + mid * (x_b - x_a)
        
        # Query the target model
        current_label = np.argmax(target_model.predict(x_mid.reshape(1, -1)))
        query_count += 1
        
        if current_label == label_a:
            # We are still on the "A" side, move closer to B
            low = mid
            boundary_sample = x_mid
        else:
            # We crossed the boundary, move back toward A to refine
            high = mid
            
    return boundary_sample, query_count

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

#   version 3: adding diverse and target-model-confident samples for categorical datasets (nursey and mushroom) in the beginning
def traverse_explanations_SHAP3(sample_set, explainer, model, n_visits_lb, n_visits_ub, upper_limit, n_f_e, args2,
                               model_name, X_train=None, y_train=None, explanation_type='vanilla', num_exp = 5):
    classes, features, n_classes, n_features, isCat, epsilon_set, canNegative, classPossibilities, dataset_name, feature_ranges = args2
    print("Dataset name in traverse_explanations_SHAP3:", dataset_name)
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

    # print("Generating diverse samples for categorical dataset:", dataset_name)
    # for categorical datasets, create more diverse initial samples
    
    diverse_samples = create_manifold_aware_diverse_samples(samples, classPossibilities, feature_ranges, isCat)
    while len(diverse_samples) > 0:
        current_diverse  = diverse_samples.pop(0)
        query += 1
        if(model.predict_proba([current_diverse]).max() > 0.8): # only add samples that are confidently classified
            print("Diverse samples left to process:", len(diverse_samples))
            samples += [current_diverse]

    # middle_samples = generate_shap_informed_samples(model, samples, explainer, isCat, feature_ranges, num_new_samples=10, top_k=5)
    # samples += middle_samples
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
    return visited_samples, preds, query

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
    confidence_scores = [target_result[i][np.argmax(target_result[i])] for i in range(len(target_result)) if np.argmax(shadow_result[i]) != np.argmax(target_result[i])]
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
    
    avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
    print("Average confidence score for samples classified differently by surrogate model:", avg_confidence)

    correct_confidences = [target_result[i][np.argmax(target_result[i])] for i in range(len(target_result)) if np.argmax(shadow_result[i]) == np.argmax(target_result[i])]
    avg_correct_confidence = sum(correct_confidences) / len(correct_confidences) if correct_confidences else 0
    print("Average confidence score for samples classified same by surrogate model:", avg_correct_confidence)
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
def load_dataset(which_dataset):
    if which_dataset == 0:
        #iris = sklearn.datasets.load_iris()
        #X = iris.data
        #y = iris.target
        X, y = shap.datasets.iris()
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y,
                                                                                    test_size=0.25)  #, random_state=42)
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
                                                                                    random_state=42)
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
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
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
                                                                                    stratify=bc.target)
        X_test_t, X_test_s, y_test_t, y_test_s = sklearn.model_selection.train_test_split(X_test, y_test,
                                                                                          train_size=0.60,
                                                                                          stratify=y_test)
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
                                                                                    random_state=42)
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
                                                                                    random_state=42)
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
                                                                                    random_state=42)
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
    
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42, stratify=y)
        X_test_t, X_test_s, y_test_t, y_test_s = sklearn.model_selection.train_test_split(X_test, y_test,
                                                                                            train_size=0.60,
                                                                                            stratify=y_test,
                                                                                            random_state=21)
        features = [f'pixel_{i}' for i in range(X.shape[1])]
        classes = sorted(list(set(y)))
        n_features = len(features)
        n_classes = len(classes)
        targets = {i: str(i) for i in classes}
        isCategorical = [False] * n_features
        canNegative = [False] * n_features
        epsilon_set = [16] * n_features  # assuming pixel values range from 0 to 16
        dataset_name = 'digits'
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

    top_exp = 5  # Number of top explanations to consider in SHAP
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
                        print('Sample set', i, " Max similarity reached, no need for traversal!")
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
                                                                                           model_name, top_exp, X_train,
                                                                                           y_train)
                        else:
                            print('No valid explanation tool selected')
                            break
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
                                    models += [mlp(activation='tanh', hidden_layer_sizes=(10 * l), solver='adam',
                                                   max_iter=10000)]
                                    models += [mlp(activation='relu', hidden_layer_sizes=(10 * l), solver='adam',
                                                   max_iter=10000)]
                                    #models += [mlp(activation='tanh', hidden_layer_sizes=(10*l,10*l), solver='adam', max_iter=10000)]
                                    #models += [mlp(activation='relu', hidden_layer_sizes=(10*l,10*l), solver='adam', max_iter=10000)]
                                #models = [s_model, s_model1]
                                #models = [s_model, s_model1, s_model2, s_model3]
                            else:
                                print('No such model!')
                            if model_name == 'mlp':
                                for m in models:
                                    m.fit(v_samples_np, v_pred_dec)
                                    sim += [rtest_sim(m, t_model, X_test_t.values)]
                                    s_accuracy += [accuracy_score(y_test_t, m.predict(X_test_t.values))]
                            else:
                                s_model.fit(v_samples_np, v_pred_dec)
                                sim += [rtest_sim(s_model, t_model, X_test_t.values)]
                                s_accuracy += [accuracy_score(y_test_t, s_model.predict(X_test_t.values))]
                        if prioritizeSim:
                            tmp = np.argmax(sim)
                        else:
                            tmp = np.argmax(s_accuracy)
                        m_sim = round(sim[tmp], 4)
                        sims += [m_sim]
                        real_accuracy += [round(s_accuracy[tmp], 4)]
                        if m_sim == 1:
                            max_sim[i] = True
                        print('Sample set', i, ', n_queries =', len(v_pred_dec), ', Top similarity =', m_sim)
                accuracies += [real_accuracy]
                rtest_sims += [sims]
                print("Accuracy: ", real_accuracy, "\nSimilarity: ", sims, "\n")

    # Pack up all remaining variables for checking
    args0 = [which_dataset, which_model, explanation_tool]
    args3 = [t_model, model_name, t_accuracy, t_explainer]
    args4 = [how_many_sets, sample_set_sizes, nfe, query_limit]
    other_args = [args0, args1, args2, args3, args4]

    accuracies, rtest_sims = argmaxing(accuracies, rtest_sims, args4)  # Max similarity surrogate model is preserved
    if save_option:
        save_results(dataset_name, model_name, accuracies, rtest_sims, samples_mega)

    return accuracies, rtest_sims, samples_mega, other_args



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
                        print(f"\Training a surrogate model for explanation type: {etype} and query limit: {query_limit[h]}")
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
        try:
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
        except Exception:
            return np.zeros(n_features, dtype=float)

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
        try:
            # hasattr(local_explainer, 'shap_values'):
            orig_shap = local_explainer.shap_values(x)
            phi = _extract_shap_vector(orig_shap)
        except Exception:
            phi = np.zeros(n_features, dtype=float)

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
                try:
                    maxc = int(classPossibilities[feat]) if classPossibilities and len(classPossibilities) > feat else None
                except Exception:
                    maxc = None
                new_x = x.ravel().copy()
                if maxc and maxc > 1:
                    try:
                        new_x[feat] = (int(new_x[feat]) + 1) % maxc
                    except Exception:
                        new_x[feat] = new_x[feat]
                else:
                    # fallback: flip between 0/1
                    new_x[feat] = 0 if new_x[feat] != 0 else 1
                perturbed_variants.append(new_x)
            else:
                # numeric: apply +epsilon and -epsilon
                try:
                    eps = float(epsilon_set[feat]) if epsilon_set and len(epsilon_set) > feat else 0.0
                except Exception:
                    eps = 0.0
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
