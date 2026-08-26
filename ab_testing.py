import random
import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# ============================================================
# 1. LOAD MODELS AND THEIR CORRESPONDING VECTORIZERS
# ============================================================

# Model A: Baseline Logistic Regression
model_a = joblib.load("baseline_model.pkl")

# Model B: Best Random Forest
model_b = joblib.load("best_model.pkl")

# Vectorizer used by the baseline Logistic Regression
vectorizer_a = joblib.load("multi_model_tfidf_vectorizer.pkl")

# Vectorizer used by the best Random Forest
vectorizer_b = joblib.load("tfidf_vectorizer.pkl")

print("Models and vectorizers loaded successfully.")

print(
    "Model A:",
    type(model_a).__name__,
    "| Features:",
    len(vectorizer_a.get_feature_names_out())
)

print(
    "Model B:",
    type(model_b).__name__,
    "| Features:",
    len(vectorizer_b.get_feature_names_out())
)


# ============================================================
# 2. LOAD DATASET
# ============================================================

data = pd.read_csv("training_data_balanced.csv")

print(f"\nDataset loaded: {len(data)} samples")
print(data["label"].value_counts())


# ============================================================
# 3. A/B TRAFFIC ROUTER
# ============================================================

def select_model():
    """
    Randomly route each request.

    Model A = 50% traffic
    Model B = 50% traffic
    """

    if random.random() < 0.5:
        return "A", model_a, vectorizer_a, "Logistic Regression"

    return "B", model_b, vectorizer_b, "Random Forest"


# ============================================================
# 4. RUN A/B TEST
# ============================================================

results = []

for _, row in data.iterrows():

    text = row["text"]
    actual_label = row["label"]

    # Select model using the 50/50 traffic split
    model_version, selected_model, selected_vectorizer, model_name = (
        select_model()
    )

    # Convert text into the correct TF-IDF representation
    text_features = selected_vectorizer.transform([text])

    # Generate prediction
    prediction = selected_model.predict(text_features)[0]

    # Check whether prediction is correct
    correct = prediction == actual_label

    # Store result
    results.append({
        "model_version": model_version,
        "model_name": model_name,
        "actual_label": actual_label,
        "predicted_label": prediction,
        "correct": correct
    })


# ============================================================
# 5. CONVERT RESULTS TO DATAFRAME
# ============================================================

results_df = pd.DataFrame(results)

print("\n========================================")
print("A/B TEST INDIVIDUAL RESULTS")
print("========================================")

print(results_df.to_string(index=False))


# ============================================================
# 6. CALCULATE METRICS FOR EACH MODEL
# ============================================================

comparison_results = []

for version in ["A", "B"]:

    model_results = results_df[
        results_df["model_version"] == version
    ]

    if len(model_results) == 0:
        continue

    y_true = model_results["actual_label"]
    y_pred = model_results["predicted_label"]

    accuracy = accuracy_score(y_true, y_pred)

    precision = precision_score(
        y_true,
        y_pred,
        pos_label="plagiarized",
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        pos_label="plagiarized",
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        pos_label="plagiarized",
        zero_division=0
    )

    comparison_results.append({
        "Model Version": version,
        "Model": model_results["model_name"].iloc[0],
        "Requests": len(model_results),
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })


# ============================================================
# 7. DISPLAY COMPARISON
# ============================================================

comparison_df = pd.DataFrame(comparison_results)

print("\n========================================")
print("A/B TEST MODEL COMPARISON")
print("========================================")

if len(comparison_df) > 0:
    print(
        comparison_df.to_string(index=False)
    )
else:
    print("No model results were generated.")


# ============================================================
# 8. SAVE RESULTS
# ============================================================

results_df.to_csv(
    "ab_test_individual_results.csv",
    index=False
)

comparison_df.to_csv(
    "ab_test_results.csv",
    index=False
)

print("\n========================================")
print("A/B TEST COMPLETED")
print("========================================")

print("Individual results saved as:")
print("ab_test_individual_results.csv")

print("\nComparison results saved as:")
print("ab_test_results.csv")