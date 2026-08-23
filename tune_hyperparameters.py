import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# ============================================================
# 1. LOAD TRAINING AND TESTING DATA
# ============================================================

train_data = pd.read_csv("training_data_balanced.csv")
test_data = pd.read_csv("test_data.csv")

X_train = train_data["text"].fillna("")
y_train = train_data["label"]

X_test = test_data["text"].fillna("")
y_test = test_data["label"]

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# 2. TF-IDF FEATURE EXTRACTION
# ============================================================

vectorizer = TfidfVectorizer(stop_words="english")

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("TF-IDF training shape:", X_train_tfidf.shape)
print("TF-IDF testing shape:", X_test_tfidf.shape)


# ============================================================
# 3. DEFINE MODELS AND HYPERPARAMETER GRIDS
# ============================================================

models_and_parameters = {

    "Logistic Regression": (
        LogisticRegression(
            random_state=42,
            max_iter=1000
        ),
        {
            "C": [0.1, 1, 10]
        }
    ),

    "SVM": (
        SVC(
            kernel="linear",
            random_state=42
        ),
        {
            "C": [0.1, 1, 10]
        }
    ),

    "Random Forest": (
        RandomForestClassifier(
            random_state=42
        ),
        {
            "n_estimators": [50, 100],
            "max_depth": [None, 5]
        }
    )
}


# ============================================================
# 4. HYPERPARAMETER TUNING
# ============================================================

results = []
best_models = {}

for name, (model, parameters) in models_and_parameters.items():

    print("\n========================================")
    print("Tuning:", name)
    print("========================================")

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=parameters,
        scoring="f1_macro",
        cv=3,
        n_jobs=-1
    )

    grid_search.fit(X_train_tfidf, y_train)

    best_model = grid_search.best_estimator_

    best_models[name] = best_model

    print("Best Parameters:", grid_search.best_params_)
    print("Best Cross-Validation Score:",
          round(grid_search.best_score_, 4))

    # Predictions on untouched test data
    y_pred = best_model.predict(X_test_tfidf)

    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        pos_label="plagiarized",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        pos_label="plagiarized",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        pos_label="plagiarized",
        zero_division=0
    )

    results.append({
        "Model": name,
        "Best Parameters": str(grid_search.best_params_),
        "CV F1 Score": grid_search.best_score_,
        "Test Accuracy": accuracy,
        "Test Precision": precision,
        "Test Recall": recall,
        "Test F1 Score": f1
    })


# ============================================================
# 5. SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(results)

print("\n\n===== HYPERPARAMETER TUNING RESULTS =====")
print(results_df.to_string(index=False))

results_df.to_csv(
    "tuning_results.csv",
    index=False
)


# ============================================================
# 6. SAVE BEST PARAMETERS
# ============================================================

with open("best_model_parameters.txt", "w") as file:

    file.write("BEST HYPERPARAMETERS\n")
    file.write("====================\n\n")

    for name, model in best_models.items():

        file.write(f"{name}\n")
        file.write("-" * len(name) + "\n")
        file.write(str(model.get_params()))
        file.write("\n\n")


# ============================================================
# 7. SAVE TF-IDF VECTORIZER
# ============================================================

joblib.dump(
    vectorizer,
    "tuned_tfidf_vectorizer.pkl"
)


print("\nTuning results saved as: tuning_results.csv")
print("Best parameters saved as: best_model_parameters.txt")
print("TF-IDF vectorizer saved as: tuned_tfidf_vectorizer.pkl")