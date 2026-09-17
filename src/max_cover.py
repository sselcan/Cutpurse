import shap
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def diagnosticity(F, faithful_explanations, random_explanations):
    correct = 0
    total = len(faithful_explanations)
    for u, v in zip(faithful_explanations, random_explanations):
        score_u = F(u)
        score_v = F(v)
        # If F(u) > F(v) means more faithful for this metric
        if score_u > score_v:
            correct += 1
    return correct / total

# 1. Train model
X, y = load_breast_cancer(return_X_y=True, as_frame=True)
feature_names = X.columns
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print(len(X_train))

explainer = shap.TreeExplainer(model, X_train)
shap_values = explainer.shap_values(X_train)[:, 0] #
print(shap_values[0].shape)
print(shap_values[1].shape)

n_instances, n_features = shap_values.shape
print(shap_values.shape)

# Define top-k threshold (e.g., top 3 features per instance)
top_k = 3 #todo
coverage_sets = [set() for _ in range(n_features)]  # one set per feature

for j in range(n_instances):
    top_features_idx = np.argsort(-np.abs(shap_values[j]))[:top_k]  # top-k by absolute SHAP
    for i in top_features_idx:
        coverage_sets[i].add(j)  # feature i covers instance j

K = 5  # number of features to select globally
selected_features = []
covered_instances = set()

for _ in range(K):
    best_feature, best_gain = None, 0
    for i in range(n_features):
        if i in selected_features:
            continue
        gain = len(coverage_sets[i] - covered_instances)
        if gain > best_gain:
            best_gain, best_feature = gain, i
    if best_feature is None:
        break
    selected_features.append(best_feature)
    covered_instances |= coverage_sets[best_feature]

# 5. Results
print("Selected global features (top coverage):")
for f in selected_features:
    print(f"- {feature_names[f]}")
print(f"Number of instances covered: {len(covered_instances)} / {n_instances}")
