import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler

# 1. Load the dataset
df = pd.read_csv("sample_submissions.csv")

print("Original Dataset:")
print(df)

# 2. Check the original class distribution
print("\nOriginal Class Distribution:")
print(df["label"].value_counts())

# 3. Split the dataset into training and testing sets
train_data, test_data = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    stratify=df["label"]
)

print("\nBefore Balancing - Training Distribution:")
print(train_data["label"].value_counts())

# 4. Separate features and labels
X_train = train_data.drop(columns=["label"])
y_train = train_data["label"]

# 5. Balance ONLY the training data
oversampler = RandomOverSampler(random_state=42)

X_train_balanced, y_train_balanced = oversampler.fit_resample(
    X_train,
    y_train
)

# 6. Recreate the balanced training dataset
balanced_train = X_train_balanced.copy()
balanced_train["label"] = y_train_balanced

# 7. Display the balanced distribution
print("\nAfter Balancing - Training Distribution:")
print(balanced_train["label"].value_counts())

# 8. Display the test distribution
print("\nTesting Distribution:")
print(test_data["label"].value_counts())

# 9. Save the datasets
balanced_train.to_csv("training_data_balanced.csv", index=False)
test_data.to_csv("test_data.csv", index=False)

print("\nTraining and testing datasets saved successfully!")