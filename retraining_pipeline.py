import os
import json
import time
import shutil
import joblib
import pandas as pd

from datetime import datetime

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "training_data_balanced.csv"

CURRENT_MODEL_PATH = "best_model.pkl"
CURRENT_VECTORIZER_PATH = "tfidf_vectorizer.pkl"

NEW_MODEL_PATH = "retrained_model.pkl"
NEW_VECTORIZER_PATH = "retrained_tfidf_vectorizer.pkl"

LOG_PATH = "retraining_log.json"

# Validation gates
MIN_ACCURACY = 0.80
MIN_F1 = 0.80


# ============================================================
# LOGGING
# ============================================================

def save_log(log_data):
    with open(LOG_PATH, "w", encoding="utf-8") as file:
        json.dump(log_data, file, indent=4)


# ============================================================
# LOAD DATA
# ============================================================

def load_training_data():

    print("\nLoading training data...")

    df = pd.read_csv(DATA_PATH)

    df["text"] = df["text"].fillna("").astype(str)

    print(f"Dataset loaded: {len(df)} samples")

    print("\nClass distribution:")
    print(df["label"].value_counts())

    return df


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_data(df):

    X = df["text"]
    y = df["label"]

    return X, y


# ============================================================
# TRAIN NEW MODEL
# ============================================================

def train_new_model(X, y):

    print("\nTraining new model...")

    vectorizer = TfidfVectorizer(
        stop_words="english"
    )

    X_features = vectorizer.fit_transform(X)

    model = RandomForestClassifier(
        n_estimators=50,
        random_state=42
    )

    model.fit(X_features, y)

    print("New Random Forest model trained successfully.")

    return model, vectorizer


# ============================================================
# VALIDATE MODEL
# ============================================================

def validate_model(model, vectorizer, X, y):

    print("\nRunning validation gate...")

    X_features = vectorizer.transform(X)

    predictions = model.predict(X_features)

    accuracy = accuracy_score(y, predictions)

    precision = precision_score(
        y,
        predictions,
        pos_label="plagiarized",
        zero_division=0
    )

    recall = recall_score(
        y,
        predictions,
        pos_label="plagiarized",
        zero_division=0
    )

    f1 = f1_score(
        y,
        predictions,
        pos_label="plagiarized",
        zero_division=0
    )

    print("\nValidation Metrics")
    print("-" * 50)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    # Validation gates
    accuracy_pass = accuracy >= MIN_ACCURACY
    f1_pass = f1 >= MIN_F1

    gate_passed = accuracy_pass and f1_pass

    print("\nValidation Gate")
    print("-" * 50)

    print(
        f"Accuracy >= {MIN_ACCURACY:.2f}: "
        f"{'PASS' if accuracy_pass else 'FAIL'}"
    )

    print(
        f"F1 Score >= {MIN_F1:.2f}: "
        f"{'PASS' if f1_pass else 'FAIL'}"
    )

    print(
        f"\nOverall Gate: "
        f"{'PASSED' if gate_passed else 'FAILED'}"
    )

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "accuracy_gate": accuracy_pass,
        "f1_gate": f1_pass,
        "overall_gate": gate_passed
    }

    return metrics


# ============================================================
# SAVE RETRAINED MODEL
# ============================================================

def save_retrained_model(model, vectorizer):

    joblib.dump(
        model,
        NEW_MODEL_PATH
    )

    joblib.dump(
        vectorizer,
        NEW_VECTORIZER_PATH
    )

    print("\nRetrained model saved:")
    print(f"  {NEW_MODEL_PATH}")
    print(f"  {NEW_VECTORIZER_PATH}")


# ============================================================
# DEPLOY MODEL AFTER VALIDATION
# ============================================================

def deploy_model():

    # Backup existing model if available
    if os.path.exists(CURRENT_MODEL_PATH):

        backup_path = "best_model_backup.pkl"

        shutil.copy2(
            CURRENT_MODEL_PATH,
            backup_path
        )

        print(
            f"\nExisting model backed up to: "
            f"{backup_path}"
        )

    if os.path.exists(CURRENT_VECTORIZER_PATH):

        backup_vectorizer = (
            "tfidf_vectorizer_backup.pkl"
        )

        shutil.copy2(
            CURRENT_VECTORIZER_PATH,
            backup_vectorizer
        )

    shutil.copy2(
        NEW_MODEL_PATH,
        CURRENT_MODEL_PATH
    )

    shutil.copy2(
        NEW_VECTORIZER_PATH,
        CURRENT_VECTORIZER_PATH
    )

    print("\nNew model deployed successfully.")


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():

    start_time = time.time()

    print("=" * 60)
    print("AUTOMATED MODEL RETRAINING PIPELINE")
    print("=" * 60)

    timestamp = datetime.now().isoformat()

    log = {
        "timestamp": timestamp,
        "pipeline_status": "started",
        "dataset": DATA_PATH,
        "model_type": "RandomForestClassifier",
        "dataset_size": 0,
        "metrics": {},
        "validation_gate": {},
        "deployment_status": "not_deployed"
    }

    try:

        # Step 1: Load data
        df = load_training_data()

        log["dataset_size"] = len(df)

        # Step 2: Prepare data
        X, y = prepare_data(df)

        # Step 3: Train
        model, vectorizer = train_new_model(
            X,
            y
        )

        # Step 4: Validate
        metrics = validate_model(
            model,
            vectorizer,
            X,
            y
        )

        log["metrics"] = {
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1_score": metrics["f1_score"]
        }

        log["validation_gate"] = {
            "minimum_accuracy": MIN_ACCURACY,
            "minimum_f1": MIN_F1,
            "accuracy_passed": metrics["accuracy_gate"],
            "f1_passed": metrics["f1_gate"],
            "overall_passed": metrics["overall_gate"]
        }

        # Step 5: Deploy only if gate passes
        if metrics["overall_gate"]:

            save_retrained_model(
                model,
                vectorizer
            )

            deploy_model()

            log["deployment_status"] = "deployed"
            log["pipeline_status"] = "completed"

        else:

            print(
                "\nValidation failed."
            )

            print(
                "New model will NOT be deployed."
            )

            log["deployment_status"] = "rejected"
            log["pipeline_status"] = "completed_with_rejection"

        # Step 6: Runtime
        runtime = time.time() - start_time

        log["runtime_seconds"] = runtime

        save_log(log)

        print("\n" + "=" * 60)
        print("RETRAINING PIPELINE COMPLETED")
        print("=" * 60)

        print(
            f"Pipeline status: "
            f"{log['pipeline_status']}"
        )

        print(
            f"Deployment status: "
            f"{log['deployment_status']}"
        )

        print(
            f"Runtime: "
            f"{runtime:.2f} seconds"
        )

        print(
            f"\nLog saved to: {LOG_PATH}"
        )

    except Exception as error:

        log["pipeline_status"] = "failed"
        log["error"] = str(error)

        save_log(log)

        print("\nPipeline failed.")
        print("Error:", error)


if __name__ == "__main__":
    main()