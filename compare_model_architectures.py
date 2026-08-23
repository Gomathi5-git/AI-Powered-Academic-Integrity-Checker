import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the prepared datasets
train_data = pd.read_csv("training_data_balanced.csv")
test_data = pd.read_csv("test_data.csv")

# Load the TF-IDF features created during previous tasks
vectorizer = joblib.load("tuned_tfidf_vectorizer.pkl")

X_train = vectorizer.transform(train_data["text"])
X_test = vectorizer.transform(test_data["text"])

y_train = train_data["label"]
y_test = test_data["label"]

# Define the models
models = {
    "Logistic Regression": LogisticRegression(C=0.1, max_iter=1000),
    "SVM": SVC(C=1),
    "Random Forest": RandomForestClassifier(
        n_estimators=50,
        max_depth=None,
        random_state=42
    )
}

results = []

print("=" * 50)
print("MODEL ARCHITECTURE COMPARISON")
print("=" * 50)

for name, model in models.items():

    print(f"\nEvaluating: {name}")

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    predictions = model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions, pos_label="plagiarized", zero_division=0
    )
    recall = recall_score(
        y_test, predictions, pos_label="plagiarized", zero_division=0
    )
    f1 = f1_score(
        y_test, predictions, pos_label="plagiarized", zero_division=0
    )

    print(f"Accuracy : {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall   : {recall:.2f}")
    print(f"F1 Score : {f1:.2f}")

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })

# Create comparison table
results_df = pd.DataFrame(results)

print("\n" + "=" * 50)
print("MODEL COMPARISON RESULTS")
print("=" * 50)
print(results_df.to_string(index=False))

# Save results
results_df.to_csv("model_architecture_comparison.csv", index=False)

# Find best model based on F1 score
best_model = results_df.loc[results_df["F1 Score"].idxmax()]

with open("model_architecture_report.txt", "w") as file:
    file.write("MODEL ARCHITECTURE COMPARISON REPORT\n")
    file.write("=" * 45 + "\n\n")

    file.write("Models evaluated:\n")
    file.write("1. Logistic Regression - linear classification model\n")
    file.write("2. SVM - margin-based classification model\n")
    file.write("3. Random Forest - ensemble tree-based model\n\n")

    file.write("Evaluation metrics:\n")
    file.write("Accuracy, Precision, Recall, and F1 Score\n\n")

    file.write("Results:\n")
    file.write(results_df.to_string(index=False))
    file.write("\n\n")

    file.write(
        f"Best model based on test F1 score: "
        f"{best_model['Model']}\n"
    )

    file.write(
        "\nDataset limitation:\n"
        "The dataset is very small, with only 12 training samples "
        "and 2 test samples. Therefore, the results should be "
        "considered preliminary and should be validated using a "
        "larger real-world dataset.\n"
    )

# Create visualization
metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]

results_df.set_index("Model")[metrics].plot(
    kind="bar",
    figsize=(10, 6)
)

plt.title("Model Architecture Performance Comparison")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=0)
plt.legend(title="Metrics")
plt.tight_layout()

plt.savefig("model_architecture_comparison.png", dpi=300)
plt.close()

print("\nResults saved as: model_architecture_comparison.csv")
print("Report saved as: model_architecture_report.txt")
print("Visualization saved as: model_architecture_comparison.png")