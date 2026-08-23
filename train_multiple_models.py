import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# 1. Load training and testing data
train_data = pd.read_csv("training_data_balanced.csv")
test_data = pd.read_csv("test_data.csv")

X_train = train_data["text"].fillna("")
y_train = train_data["label"]

X_test = test_data["text"].fillna("")
y_test = test_data["label"]


# 2. Convert text into TF-IDF features
vectorizer = TfidfVectorizer(stop_words="english")

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


# 3. Define multiple machine learning models
models = {
    "Logistic Regression": LogisticRegression(
        random_state=42,
        max_iter=1000
    ),

    "SVM": SVC(
        kernel="linear",
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
}


# 4. Train and evaluate each model
results = []

for name, model in models.items():

    print("\nTraining:", name)

    # Train the model
    model.fit(X_train_tfidf, y_train)

    # Make predictions
    y_pred = model.predict(X_test_tfidf)

    # Calculate metrics
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

    # Store results
    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })

    print("Accuracy :", round(accuracy, 2))
    print("Precision:", round(precision, 2))
    print("Recall   :", round(recall, 2))
    print("F1 Score :", round(f1, 2))


# 5. Create comparison table
results_df = pd.DataFrame(results)

print("\n===== MODEL COMPARISON =====")
print(results_df.to_string(index=False))


# 6. Save the comparison results
results_df.to_csv("model_comparison_results.csv", index=False)

# Save the TF-IDF vectorizer
joblib.dump(vectorizer, "multi_model_tfidf_vectorizer.pkl")


print("\nModel comparison saved as: model_comparison_results.csv")
print("TF-IDF vectorizer saved as: multi_model_tfidf_vectorizer.pkl")