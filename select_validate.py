"""
select_validate.py
==================
Phase 2 - Task 5: Select & Validate

This script performs statistical validation of candidate machine learning models
(Logistic Regression, SVM, Random Forest) using Stratified K-Fold Cross-Validation.

Objectives:
-----------
1. Evaluate model generalization and stability across cross-validation folds.
2. Measure Mean and Standard Deviation for Accuracy, Precision, Recall, and F1 Score.
3. Select the best performing model based on Mean F1 Score while accounting for dataset limitations.
4. Generate validation_results.csv, model_selection_report.txt, and model_validation_comparison.png.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def load_dataset():
    """Load the existing training dataset from the project directory."""
    if os.path.exists("training_data_balanced.csv"):
        data_path = "training_data_balanced.csv"
    elif os.path.exists("training_data.csv"):
        data_path = "training_data.csv"
    else:
        raise FileNotFoundError("Neither training_data_balanced.csv nor training_data.csv was found.")

    df = pd.read_csv(data_path)
    X_text = df["text"].fillna("")
    y = df["label"]
    return df, data_path, X_text, y


def perform_cross_validation(models, X_tfidf, y, n_splits):
    """Perform Stratified K-Fold cross validation and compute mean/std for metrics."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    results = []

    for name, model in models.items():
        acc_scores = []
        prec_scores = []
        rec_scores = []
        f1_scores = []

        for train_idx, val_idx in skf.split(X_tfidf, y):
            X_tr, X_val = X_tfidf[train_idx], X_tfidf[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # Fit model on training fold
            model.fit(X_tr, y_tr)

            # Predict on validation fold
            y_pred = model.predict(X_val)

            # Compute fold metrics safely with zero_division=0
            acc = accuracy_score(y_val, y_pred)
            prec = precision_score(y_val, y_pred, pos_label="plagiarized", zero_division=0)
            rec = recall_score(y_val, y_pred, pos_label="plagiarized", zero_division=0)
            f1 = f1_score(y_val, y_pred, pos_label="plagiarized", zero_division=0)

            acc_scores.append(acc)
            prec_scores.append(prec)
            rec_scores.append(rec)
            f1_scores.append(f1)

        results.append({
            "Model": name,
            "Accuracy Mean": np.mean(acc_scores),
            "Accuracy Std": np.std(acc_scores),
            "Precision Mean": np.mean(prec_scores),
            "Precision Std": np.std(prec_scores),
            "Recall Mean": np.mean(rec_scores),
            "Recall Std": np.std(rec_scores),
            "F1 Mean": np.mean(f1_scores),
            "F1 Std": np.std(f1_scores)
        })

    return pd.DataFrame(results)


def plot_results(results_df, output_path="model_validation_comparison.png"):
    """Generate error-bar bar plot comparing mean F1 scores across models."""
    plt.figure(figsize=(8, 5))
    models = results_df["Model"]
    f1_means = results_df["F1 Mean"]
    f1_stds = results_df["F1 Std"]

    bars = plt.bar(models, f1_means, yerr=f1_stds, capsize=6, color=["#3498db", "#2ecc71", "#e74c3c"], alpha=0.85, edgecolor="black")

    plt.title("Cross-Validation Model Comparison (Mean F1 Score ± Std Dev)", fontsize=13, fontweight="bold")
    plt.ylabel("Mean F1 Score", fontsize=11)
    plt.ylim(0, 1.15)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Add numeric labels above bars
    for bar, mean_val, std_val in zip(bars, f1_means, f1_stds):
        plt.text(bar.get_x() + bar.get_width() / 2.0, mean_val + std_val + 0.02, f"{mean_val:.2f} (±{std_val:.2f})", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def generate_report(df, data_path, num_classes, n_splits, models, results_df, best_model_name, output_path="model_selection_report.txt"):
    """Write comprehensive validation and selection report to text file."""
    best_row = results_df.loc[results_df["Model"] == best_model_name].iloc[0]

    with open(output_path, "w") as f:
        f.write("==================================================\n")
        f.write("MODEL SELECTION & VALIDATION REPORT\n")
        f.write("==================================================\n\n")

        f.write("1. DATASET INFORMATION\n")
        f.write("----------------------\n")
        f.write(f"Source file       : {data_path}\n")
        f.write(f"Total samples     : {len(df)}\n")
        f.write(f"Number of classes : {num_classes}\n")
        f.write(f"Class counts      :\n{df['label'].value_counts().to_string()}\n")
        f.write(f"Validation method : Stratified K-Fold Cross-Validation\n")
        f.write(f"Folds used        : {n_splits}\n\n")

        f.write("2. MODELS EVALUATED & HYPERPARAMETERS\n")
        f.write("-------------------------------------\n")
        f.write("1. Logistic Regression: C=0.1, max_iter=1000, random_state=42\n")
        f.write("2. SVM: C=1.0, kernel='linear', random_state=42\n")
        f.write("3. Random Forest: n_estimators=50, max_depth=None, random_state=42\n\n")

        f.write("3. CROSS-VALIDATION RESULTS\n")
        f.write("---------------------------\n")
        f.write(results_df.to_string(index=False))
        f.write("\n\n")

        f.write("4. FINAL MODEL SELECTION\n")
        f.write("------------------------\n")
        f.write(f"Selected Model: {best_model_name}\n")
        f.write("Selection Reason:\n")
        f.write(f"- {best_model_name} achieved the highest Mean F1 Score of {best_row['F1 Mean']:.4f} (±{best_row['F1 Std']:.4f}) across the cross-validation folds.\n")
        f.write("- F1 score prioritizes a balance between Precision (avoiding false plagiarism flags) and Recall (detecting true plagiarism).\n\n")

        f.write("5. DATASET LIMITATIONS & STABILITY WARNING\n")
        f.write("------------------------------------------\n")
        f.write("- The training dataset is intentionally small (prototype stage with 12 samples).\n")
        f.write("- Cross-validation on small sample sizes leads to high metric variance and potential over-optimistic performance estimates.\n")
        f.write("- The reported metrics represent a preliminary baseline benchmark, NOT production-level performance.\n\n")

        f.write("6. RECOMMENDED FUTURE IMPROVEMENTS\n")
        f.write("----------------------------------\n")
        f.write("1. Collect a significantly larger, diverse academic plagiarism dataset.\n")
        f.write("2. Evaluate performance across balanced original and plagiarized samples.\n")
        f.write("3. Implement advanced semantic similarity techniques and transformer architectures (e.g. BERT/RoBERTa).\n")
        f.write("4. Perform continuous monitoring and validation using MLflow experiment tracking.\n")


def main():
    print("==================================================")
    print("MODEL SELECTION & VALIDATION")
    print("==================================================")

    # 1. Load dataset
    df, data_path, X_text, y = load_dataset()
    num_samples = len(df)
    num_classes = y.nunique()
    min_class_count = y.value_counts().min()

    # Automatically choose safe number of folds based on smallest class count
    n_splits = min(5, min_class_count)
    if n_splits < 2:
        n_splits = 2

    print("\nDataset information\n")
    print(f"Dataset file         : {data_path}")
    print(f"Number of samples    : {num_samples}")
    print(f"Number of classes    : {num_classes}")
    print(f"Cross-validation folds: {n_splits}")

    # 2. Text Preprocessing & Feature Extraction
    vectorizer = TfidfVectorizer(stop_words="english")
    X_tfidf = vectorizer.fit_transform(X_text)

    # 3. Models definition
    models = {
        "Logistic Regression": LogisticRegression(C=0.1, max_iter=1000, random_state=42),
        "SVM": SVC(C=1.0, kernel="linear", random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=50, max_depth=None, random_state=42)
    }

    # 4. Perform Cross Validation
    results_df = perform_cross_validation(models, X_tfidf, y, n_splits)

    print("\n" + "-" * 50)
    for _, row in results_df.iterrows():
        print(f"\nValidating: {row['Model']}")
        print(f"Mean Accuracy        : {row['Accuracy Mean']:.4f}")
        print(f"Mean Precision       : {row['Precision Mean']:.4f}")
        print(f"Mean Recall          : {row['Recall Mean']:.4f}")
        print(f"Mean F1              : {row['F1 Mean']:.4f}")
        print(f"F1 Standard Deviation: {row['F1 Std']:.4f}")

    # 5. Model Selection based on F1 Mean
    best_row = results_df.loc[results_df["F1 Mean"].idxmax()]
    best_model_name = best_row["Model"]

    print("\n" + "=" * 50)
    print("FINAL MODEL SELECTION")
    print("=" * 50)
    print(f"Selected Model: {best_model_name}")
    print(f"Reason        : Highest Mean F1 Score ({best_row['F1 Mean']:.4f} ± {best_row['F1 Std']:.4f}) during {n_splits}-fold Stratified Cross-Validation.")
    print("Note          : Due to the small dataset size, this selection is a preliminary prototype benchmark.")

    # 6. Save Output Files
    csv_file = "validation_results.csv"
    report_file = "model_selection_report.txt"
    plot_file = "model_validation_comparison.png"

    results_df.to_csv(csv_file, index=False)
    plot_results(results_df, plot_file)
    generate_report(df, data_path, num_classes, n_splits, models, results_df, best_model_name, report_file)

    print(f"\nResults saved to: {csv_file}")
    print(f"Selection report saved to: {report_file}")
    print(f"Comparison plot saved to: {plot_file}")
    print("=" * 50)


if __name__ == "__main__":
    main()
