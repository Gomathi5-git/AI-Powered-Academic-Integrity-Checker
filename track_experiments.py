"""
track_experiments.py
====================
This script integrates MLflow experiment tracking into the Academic Integrity Checker pipeline.

Why MLflow is used:
-------------------
MLflow is an open-source platform for managing the machine learning lifecycle. It helps us:
1. Track and record model parameters, performance metrics, and trained artifacts for complete reproducibility.
2. Compare multiple candidate model architectures (Logistic Regression, SVM, Random Forest) side-by-side.
3. Keep an organized audit trail of experiments without losing history.

Logged Components:
------------------
- Parameters: Model name, model hyperparameters (e.g., C, kernel, n_estimators, max_depth), TF-IDF feature counts.
- Metrics   : Accuracy, Precision, Recall, and F1 Score (with zero_division=0 handling for small datasets).
- Artifacts : Trained scikit-learn models, fitted TF-IDF vectorizer, and comparison CSV results.

Note on Small Dataset Performance:
----------------------------------
The prototype dataset contains only a small number of training and test examples.
As a result, evaluation metrics (such as Precision, Recall, and F1) may be low or unstable.
MLflow enables us to track these baseline runs reliably so we can measure improvements when a larger dataset is introduced.
"""

import pandas as pd
import joblib
import mlflow
import mlflow.sklearn

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def main():
    # 1. Load training_data.csv and test_data.csv using pandas
    print("Loading training and testing datasets...")
    train_data = pd.read_csv("training_data.csv")
    test_data = pd.read_csv("test_data.csv")

    # 2. Identify text column and label column
    X_train_text = train_data["text"].fillna("")
    y_train = train_data["label"]

    X_test_text = test_data["text"].fillna("")
    y_test = test_data["label"]

    # 3. Use the existing text classification approach: TfidfVectorizer
    vectorizer = TfidfVectorizer()

    # 4 & 5. Fit the TF-IDF vectorizer ONLY on the training data, then transform training and testing data
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)

    # 6 & 7. Print training/testing sample counts and TF-IDF feature dimensions
    print(f"Number of training samples : {len(train_data)}")
    print(f"Number of testing samples  : {len(test_data)}")
    print(f"TF-IDF feature dimensions  : {X_train.shape[1]}")

    # 18. Save the TF-IDF vectorizer as a local pickle file
    vectorizer_filename = "tfidf_vectorizer.pkl"
    joblib.dump(vectorizer, vectorizer_filename)
    print(f"Saved local vectorizer to: {vectorizer_filename}")

    # 8. Define the three models and their parameters to log
    models = {
        "Logistic Regression": (
            LogisticRegression(C=0.1, max_iter=1000, random_state=42),
            {
                "model_name": "Logistic Regression",
                "C": 0.1,
                "max_iter": 1000,
                "solver": "lbfgs",
                "random_state": 42,
                "vectorizer_num_features": X_train.shape[1]
            }
        ),
        "SVM": (
            SVC(C=1.0, kernel="linear", random_state=42),
            {
                "model_name": "SVM",
                "C": 1.0,
                "kernel": "linear",
                "random_state": 42,
                "vectorizer_num_features": X_train.shape[1]
            }
        ),
        "Random Forest": (
            RandomForestClassifier(n_estimators=50, max_depth=None, random_state=42),
            {
                "model_name": "Random Forest",
                "n_estimators": 50,
                "max_depth": "None",
                "random_state": 42,
                "vectorizer_num_features": X_train.shape[1]
            }
        )
    }

    # 9. Create or use an experiment named "Academic Integrity Checker"
    experiment_name = "Academic Integrity Checker"
    mlflow.set_experiment(experiment_name)

    results = []

    # 10. Create a separate MLflow run for EACH model
    for model_name, (model, params) in models.items():
        print(f"\nTraining: {model_name}")

        try:
            with mlflow.start_run(run_name=model_name):
                # Log PARAMETERS: model_name, model hyperparameters, TF-IDF configuration
                for param_key, param_value in params.items():
                    mlflow.log_param(param_key, param_value)

                # Fit the model on the TF-IDF feature matrix
                model.fit(X_train, y_train)

                # Predict on test data
                y_pred = model.predict(X_test)

                # 14. Calculate METRICS using zero_division=0 to handle zero positive predictions safely
                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred, pos_label="plagiarized", zero_division=0)
                rec = recall_score(y_test, y_pred, pos_label="plagiarized", zero_division=0)
                f1 = f1_score(y_test, y_pred, pos_label="plagiarized", zero_division=0)

                # Log METRICS to MLflow
                mlflow.log_metric("accuracy", acc)
                mlflow.log_metric("precision", prec)
                mlflow.log_metric("recall", rec)
                mlflow.log_metric("f1_score", f1)

                # Log TF-IDF vectorizer artifact
                mlflow.log_artifact(vectorizer_filename)

                # 19. Save each trained model as an MLflow artifact using skops_trusted_types
                mlflow.sklearn.log_model(
                    model,
                    name="model",
                    skops_trusted_types=["scipy.sparse._csr.csr_matrix"]
                )

                # Append results for CSV output
                results.append({
                    "Model": model_name,
                    "Accuracy": acc,
                    "Precision": prec,
                    "Recall": rec,
                    "F1 Score": f1
                })

                # 15. Print results in requested format
                print(f"Accuracy : {acc:.2f}")
                print(f"Precision: {prec:.2f}")
                print(f"Recall   : {rec:.2f}")
                print(f"F1 Score : {f1:.2f}")

        except Exception as e:
            # 21. Do NOT stop the entire experiment if one model fails
            print(f"Error training {model_name}: {e}")
            print("Continuing with next model...")

    # 16. Create pandas DataFrame
    results_df = pd.DataFrame(results)

    # 17. Save DataFrame as mlflow_experiment_results.csv
    results_csv = "mlflow_experiment_results.csv"
    results_df.to_csv(results_csv, index=False)

    # 24. Print completion banner and MLflow UI instruction
    print("\n========================================")
    print("MLflow experiment tracking completed!")
    print("========================================")
    print(f"\nResults saved as: {results_csv}")
    print("Open MLflow UI using: mlflow ui\n")


# 23. Include main section
if __name__ == "__main__":
    main()