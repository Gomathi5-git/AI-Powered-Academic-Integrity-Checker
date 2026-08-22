import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# 1. Load the saved baseline model and TF-IDF vectorizer
model = joblib.load("baseline_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# 2. Load the test dataset
test_data = pd.read_csv("test_data.csv")

# 3. Separate text and actual labels
X_test = test_data["text"].fillna("")
y_test = test_data["label"]

# 4. Convert test text into TF-IDF features
X_test_tfidf = vectorizer.transform(X_test)

# 5. Generate predictions
y_pred = model.predict(X_test_tfidf)

# 6. Calculate performance metrics
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

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=["original", "plagiarized"]
)

# 7. Display the results
print("===== BASELINE MODEL EVALUATION =====")

print(f"Accuracy:  {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall:    {recall:.2f}")
print(f"F1 Score:  {f1:.2f}")

print("\nConfusion Matrix:")
print(cm)

print("\nActual Labels:")
print(list(y_test))

print("\nPredicted Labels:")
print(list(y_pred))

# 8. Save the evaluation results
with open("baseline_evaluation_report.txt", "w") as file:

    file.write("TASK 5: BASELINE MODEL EVALUATION\n\n")

    file.write("Performance Metrics:\n")
    file.write(f"Accuracy: {accuracy:.2%}\n")
    file.write(f"Precision: {precision:.2%}\n")
    file.write(f"Recall: {recall:.2%}\n")
    file.write(f"F1 Score: {f1:.2%}\n\n")

    file.write("Confusion Matrix:\n")
    file.write(str(cm))
    file.write("\n\n")

    file.write("Actual Labels:\n")
    file.write(str(list(y_test)))
    file.write("\n\n")

    file.write("Predicted Labels:\n")
    file.write(str(list(y_pred)))
    file.write("\n")

print("\nEvaluation report saved as baseline_evaluation_report.txt")