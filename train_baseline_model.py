import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix


# 1. Load the prepared training and testing data
train_data = pd.read_csv("training_data_balanced.csv")
test_data = pd.read_csv("test_data.csv")

print("Training data:")
print(train_data[["text", "label"]])

print("\nTesting data:")
print(test_data[["text", "label"]])


# 2. Separate text and labels
X_train = train_data["text"].fillna("")
y_train = train_data["label"]

X_test = test_data["text"].fillna("")
y_test = test_data["label"]


# 3. Convert text into TF-IDF features
vectorizer = TfidfVectorizer(stop_words="english")

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


print("\nTF-IDF feature shape:")
print("Training:", X_train_tfidf.shape)
print("Testing:", X_test_tfidf.shape)


# 4. Create the baseline Logistic Regression model
model = LogisticRegression(random_state=42, max_iter=1000)


# 5. Train the model
model.fit(X_train_tfidf, y_train)


# 6. Make predictions on the test data
y_pred = model.predict(X_test_tfidf)


# 7. Calculate evaluation metrics
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

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=["original", "plagiarized"]
)


# 8. Display results
print("\n===== BASELINE MODEL RESULTS =====")
print(f"Accuracy:  {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall:    {recall:.2f}")

print("\nConfusion Matrix:")
print(cm)


# 9. Save the trained model and vectorizer
joblib.dump(model, "baseline_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("\nModel saved as: baseline_model.pkl")
print("Vectorizer saved as: tfidf_vectorizer.pkl")