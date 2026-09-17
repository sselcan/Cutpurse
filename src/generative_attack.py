# from ctgan import CTGANSynthesizer
# from sdv.single_table import TabularPresetSynthesizer
# from sdv.metadata import SingleTableMetadata
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
from sklearn.metrics import jaccard_score

import os, sys

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
        for j in range(sizes):
            sample_set_sui = sample_set_generation(test_cpy, n, sizes)
            sample_sets += [sample_set_sui.copy()]
        samples_mega += [sample_sets]
    return samples_mega

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
    elif which_dataset == 6:
        print("HERE")
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
    else:
        print('there is no such dataset')

    classPossibilities = []
    for i in range(n_features):
        uniques, counts = np.unique(X_train.iloc[:, i], return_counts=True)
        classPossibilities.append(len(uniques))

    args1 = [X_train, X_test, y_train, y_test, X_test_t, X_test_s, y_test_t, y_test_s]
    args2 = [classes, features, n_classes, n_features, isCategorical, epsilon_set, canNegative, classPossibilities,
             dataset_name]
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

def load_experiment_dicts():
    dataset_dict = {0: 'Iris', 1: 'Crop', 2: 'Adult Income', 3: 'Breast Cancer', 4: 'Nursery', 5: 'Mushroom'}
    model_dict = {0: 'Decision Tree', 1: 'Logistic Regression', 2: 'Multinomial Naive Bayes', 3: 'K Nearest Neighbor',
                  4: 'Random Forest', 5: 'Multilayer Perceptron'}
    exp_dict = {0: 'LIME', 1: 'SHAP'}
    return dataset_dict, model_dict, exp_dict

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


def train_gen_model(df):
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(df)

    gen = TabularPresetSynthesizer(
        metadata=metadata,
        name="FAST_ML"
    )
    gen.fit(df)
    return gen

# def train_local_exp_model():
#     # train a local explainable model 


def generate_samples(gen_model, k):
    return gen_model.sample(k)


def run_attack(wd, wm, et, hms, sss, ql, so, num_exp, bs):  # make sure the types are correct
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
    query_limit = ql if isinstance(ql, int) else (
        lambda: (_ for _ in ()).throw(TypeError("Only integers are allowed")))()
    save_option = so if isinstance(so, bool) else (
        lambda: (_ for _ in ()).throw(TypeError("Only booleans are allowed")))()
    num_exp = num_exp if isinstance(num_exp, int) else (
        lambda: (_ for _ in ()).throw(TypeError("Only integers are allowed")))()
    batch_size = bs if isinstance(bs, int) else (   
        lambda: (_ for _ in ()).throw(TypeError("Only integers are allowed")))()
    verbose: bool = True

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

    ## Here goes the attack!!! 
    ## (You can further configure the parameters like lower and upper bounds, relax_factor etc. at your own risk :/) 
    samples_mega = mega_sample_generation(X_test_s.to_numpy(), y_test_s, n_classes, sample_set_sizes, how_many_sets)

    prioritizeSim = True

    if model_name == 'nb' or model_name == 'mlp' or model_name == 'lr' or model_name == 'knn':
        repetition = 1
    elif model_name == 'dt':
        repetition = 100
    else:
        repetition = 10

    for i in range(how_many_sets):
        print(f"\n########## Starting Experiment Set {i + 1}/{how_many_sets} ##########\n")
        seed_data = samples_mega[i]

        # Copy so we don’t mutate original
        labeled_data = seed_data.copy()

        # Train initial generative model
        if verbose:
            print("🔧 Training initial generative model...")

        gen_model = train_gen_model(labeled_data)

        total_queries = 0
        iteration = 0

        while total_queries < ql:
            iteration += 1
            if verbose:
                print(f"\n=== Iteration {iteration} ===")
                print(f"Queries used: {total_queries}/{ql}")
            # Adjust final batch if near budget
            k = min(batch_size, ql - total_queries)

            ##################################################
            # 1. Generate K samples
            ##################################################
            synthetic_samples = generate_samples(gen_model, k)

            ##################################################
            # 2. Query labels for synthetic samples
            ##################################################
            labels = []
            for i in range(k):
                label = t_model.predict(synthetic_samples.iloc[i])
                labels.append(label)

            synthetic_samples["label"] = labels

            total_queries += k

            ##################################################
            # 3. Add newly labeled synthetic samples to training set
            ##################################################
            labeled_data = pd.concat([labeled_data, synthetic_samples], ignore_index=True)

            ##################################################
            # 4. Retrain generative model on expanded dataset
            ##################################################
            if verbose:
                print("🔄 Retraining generative model with expanded dataset...")
            gen_model = train_gen_model(labeled_data)

    print("\n########## Attack Finished ##########\n")
    # Train a student model on the labeled data
    student_model, student_model_name = load_model(which_model, labeled_data.drop(columns=["label"]), labeled_data["label"])
    student_accuracy = getModelInfo(student_model, labeled_data.drop(columns=["label"]), labeled_data["label"], X_test_t, y_test_t)
    print(f"Student model '{student_model_name}' trained on {len(labeled_data)} samples achieved accuracy: {student_accuracy}")

    # Measure similarity between target and student model
    target_preds = t_model.predict(X_test_t.values)
    student_preds = student_model.predict(X_test_t.values)

    if n_classes == 2:
        sim_score = jaccard_score(target_preds, student_preds)
    else:
        sim_score = jaccard_score(target_preds, student_preds, average='macro')

    print(f"Jaccard similarity between target and student model predictions: {sim_score}")
