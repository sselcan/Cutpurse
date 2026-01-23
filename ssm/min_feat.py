# ------------------------------------------------------------
# Local SHAP-based Greedy Feature Selection (submodular analogy)
# ------------------------------------------------------------

import shap
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# 1. Train a small model
X, y = load_breast_cancer(return_X_y=True, as_frame=True)
feature_names = X.columns
# print("number of features", len(feature_names))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 2. Pick one query (q)
x_q = X_test.iloc[[0]]

# Compute SHAP values
explainer = shap.TreeExplainer(model, X_train)
shap_values = explainer.shap_values(x_q)
# print(shap_values)

phi_temp = shap_values[0]
# Select the SHAP values for the positive class (Class 1 is column index 1)
phi = phi_temp[:, 1]
base_value = explainer.expected_value     # works for scalar

print("phi shape:", phi.shape)
print("base_value:", explainer.expected_value)


# Define the characteristic function v(S)
# We'll estimate it by masking features not in S and using model predictions.
def v_x(S, x=x_q, background=X_train, model=model):
    """Compute E[f(X)|X_S=x_S] by replacing other features with background samples."""
    X_masked = background.copy()
    for i in S:
        X_masked[feature_names[i]] = x.iloc[0, i]
    preds = model.predict_proba(X_masked)[:, 1]
    return preds.mean()

# Greedy selection of features
target_value = model.predict_proba(x_q)[0, 1]
baseline_value = v_x([])  # expected value
print("baseline_value:", baseline_value)
k_max = len(feature_names)
selected, gains, scores = [], [], []

current_S = set()
current_value = baseline_value

for _ in range(k_max):
    best_gain, best_i = 0, None
    for i in range(len(feature_names)):
        if i in current_S:
            continue
        candidate_value = v_x(current_S | {i})
        gain = candidate_value - current_value
        if gain > best_gain:
            best_gain, best_i = gain, i

    if best_i is None:
        break  # no further improvement
    current_S.add(best_i)
    current_value = v_x(current_S)
    selected.append(best_i)
    gains.append(best_gain)
    scores.append(current_value)

    # stop if we are within tolerance of true prediction
    if abs(current_value - target_value) < 0.01:
        break

# 6. Display results
print("\nGreedy selected features (approx e_add):")
for i, f in enumerate(selected, start=1):
    print(f"{i}. {feature_names[f]} (gain={gains[i-1]:.4f})")

print(f"\nPredicted probability (model): {target_value:.4f}")
print(f"Baseline (expected): {baseline_value:.4f}")
print(f"Approximation with selected subset: {current_value:.4f}")
print(f"Number of features used: {len(selected)}")


# --- Compute and Rank by Absolute SHAP ---
abs_phi = np.abs(phi)
shap_ranking = np.argsort(abs_phi)[::-1] # indices of features, sorted by descending absolute SHAP value
shap_ranked_features = feature_names[shap_ranking]
shap_ranked_values = abs_phi[shap_ranking]
# print(len(shap_ranked_features))
# print(shap_ranked_values)

print("\n--- SHAP-Based Ranking ---")
print("Base Value:", base_value)
print(f"Target Value: {target_value:.4f}")
print("Sum of SHAP values + Base Value:", np.sum(phi) + base_value)

print("\nFeature | Absolute SHAP Value | Contribution")
print("-" * 50)
for i in range(len(feature_names)):
    f_idx = shap_ranking[i]
    print(f"{feature_names[f_idx]:<10} | {abs_phi[f_idx]:<19.4f} | {phi[f_idx]:.4f}")


# --- Compare Top K Features ---
k = len(selected) # Use the number of features selected by the greedy method
greedy_selected_names = [feature_names[i] for i in selected]
shap_top_k_names = shap_ranked_features[:k].tolist()

print(f"\n--- Comparison of Top {k} Features ---")
print(f"Greedy Selected Features: {greedy_selected_names}")
print(f"SHAP Top {k} Features:    {shap_top_k_names}")

# --- 3. Compare Feature-by-Feature Gain vs. SHAP Value ---
print("\n--- Feature-by-Feature Comparison (Order & Value) ---")
print(f"{'#':<3} | {'Greedy Feature':<20} | {'Greedy Gain':<15} | {'SHAP Rank Feature':<20} | {'SHAP Value':<15}")
print("-" * 80)

# Iterate up to the max length of either list for a side-by-side view
max_len = max(len(selected), len(shap_ranking))
for i in range(max_len):
    greedy_feat_name = feature_names[selected[i]] if i < len(selected) else "N/A"
    greedy_gain = f"{gains[i]:.4f}" if i < len(gains) else "N/A"

    shap_feat_name = shap_ranked_features[i]
    shap_value = f"{phi[shap_ranking[i]]:.4f}"

    print(f"{i+1:<3} | {greedy_feat_name:<20} | {greedy_gain:<15} | {shap_feat_name:<20} | {shap_value:<15}")

# NOTE: The Greedy Gain is a marginal change in P(y=1) based on averaging (v_x).
#       The SHAP Value is the feature's precise contribution based on a game-theoretic average.
#       While related, they are not expected to be identical, but features with high SHAP are
#       often selected early by the greedy approach due to the submodular-like property.