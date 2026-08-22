import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Load academic submission data
df = pd.read_csv("sample_submissions.csv")

print("Original data:")
print(df)

# Handle missing text values
df["text"] = df["text"].fillna("")

# Convert text into TF-IDF features
vectorizer = TfidfVectorizer(stop_words="english")
features = vectorizer.fit_transform(df["text"])

# Convert features into a DataFrame
feature_df = pd.DataFrame(
    features.toarray(),
    columns=vectorizer.get_feature_names_out()
)

print("\nExtracted features:")
print(feature_df)

# Save processed features
feature_df.to_csv("processed_features.csv", index=False)

print("\nFeature extraction completed successfully.")
print("Saved as: processed_features.csv")