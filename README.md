# AI-Powered Academic Integrity Checker

## 1. Project Overview

The AI-Powered Academic Integrity Checker is a machine learning project designed to help identify possible plagiarism in student submissions.

Academic institutions receive a large number of assignments and other written submissions. Manually checking every submission for copied or highly similar content can be time-consuming. This project aims to use Natural Language Processing (NLP) and Machine Learning techniques to analyze submitted text and identify potentially plagiarized content.

The system will provide a plagiarism prediction and a similarity/plagiarism score to help instructors review suspicious submissions more efficiently.

---

## 2. Problem Statement

Plagiarism occurs when a person uses another person's words, ideas, or work without giving proper credit.

The main problem is to automatically analyze student submissions and identify whether the content is likely to be original or potentially plagiarized.

The ML system should be able to process a student's submission, compare its textual characteristics with reference content, and produce an understandable result.

---

## 3. ML Problem Definition

This project can be formulated as a supervised machine learning problem when labeled training data is available.

The model will learn from examples of:

- Original submissions
- Plagiarized submissions
- Similar or partially copied submissions

The main ML task is classification.

### ML Task

**Supervised Learning → Classification**

The model will predict whether a given submission belongs to an original or potentially plagiarized class.

A similarity score can also be generated to provide additional information about the degree of similarity.

---

## 4. Input

The main input to the system is a student's written submission.

Examples of input:

- Assignment text
- Essay
- Report
- Answer document
- Text extracted from a document

The text will be processed before being given to the machine learning model.

### Example Input

```text
Machine learning is a branch of artificial intelligence
that enables computers to learn patterns from data.

## Phase 2 - Task 3: Model Architecture Comparison

### Objective

The objective of this task was to compare multiple machine learning
architectures for classifying academic submissions as original or
plagiarized.

### Implementation

The prepared text dataset was converted into numerical features using
TF-IDF. Three machine learning algorithms were evaluated:

- **Logistic Regression** - linear classification algorithm
- **SVM** - margin-based classification algorithm
- **Random Forest** - ensemble model based on multiple decision trees

The models were trained using the prepared training data and evaluated
on the independent test dataset using Accuracy, Precision, Recall, and
F1 Score.

### Model Comparison

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.50 | 0.00 | 0.00 | 0.00 |
| SVM | 0.50 | 0.00 | 0.00 | 0.00 |
| Random Forest | 0.50 | 0.00 | 0.00 | 0.00 |

### Hyperparameter Tuning Comparison

During the previous hyperparameter tuning task, Random Forest achieved
the highest 3-fold cross-validation F1 score of **91.11%** using
`n_estimators=50` and `max_depth=None`.

Logistic Regression achieved a cross-validation F1 score of **65.56%**,
while SVM also achieved **65.56%**.

### Findings

Random Forest showed the strongest cross-validation performance among
the tested models and is therefore the current candidate for further
development.

However, all three models achieved 50% accuracy on the independent test
set. This result should not be interpreted as reliable production
performance because the current dataset is extremely small.

### Dataset Limitation

The current dataset contains only **12 training samples and 2 test
samples**. With only two test samples, a single incorrect prediction
changes the accuracy by 50 percentage points.

Therefore, the current model comparison is preliminary. A larger and
more representative plagiarism dataset is required before making a
final production model selection.

### Generated Files

- `compare_model_architectures.py` - model comparison implementation
- `model_architecture_comparison.csv` - comparison metrics
- `model_architecture_comparison.png` - performance visualization
- `model_architecture_report.txt` - detailed findings and recommendation

---

## Phase 2 – Task 4: Experiment Tracking with MLflow

### Overview

In Task 4, MLflow was integrated into the existing training pipeline to manage experiment tracking, record hyperparameters, measure evaluation metrics, and store model artifacts for full reproducibility.

Three machine learning models were tracked and evaluated:
- **Logistic Regression**
- **SVM**
- **Random Forest**

### Experiment Tracking Workflow

```text
Dataset
   ↓
TF-IDF
   ↓
Model Training
   ↓
MLflow Run
   ↓
Parameters + Metrics + Artifacts
   ↓
MLflow UI
```

### Implementation & Logging Details

- **Feature Extraction**: TF-IDF (`TfidfVectorizer`) was fitted exclusively on `training_data.csv` and transformed both training and `test_data.csv` text samples.
- **Metrics Logged**: Accuracy, Precision, Recall, and F1 Score (evaluated with `zero_division=0` to ensure stability on tiny test sets).
- **Parameters Logged**: Model names, model hyperparameters (`C`, `max_iter`, `kernel`, `n_estimators`, `max_depth`, `random_state`), and TF-IDF feature dimensions.
- **Artifacts Logged**: Trained scikit-learn models (using safe deserialization configuration `skops_trusted_types=["scipy.sparse._csr.csr_matrix"]`) and the TF-IDF vectorizer.
- **Results Storage**: Experiment results were saved locally to `mlflow_experiment_results.csv`.

### Baseline Results

The table below shows the actual evaluation results tracked in MLflow for each model architecture evaluated on `test_data.csv`:

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.50 | 0.00 | 0.00 | 0.00 |
| SVM | 0.50 | 0.00 | 0.00 | 0.00 |
| Random Forest | 0.50 | 0.00 | 0.00 | 0.00 |

*Note: The current dataset is intentionally very small, so the reported metrics represent a baseline/prototype benchmark. Future work will involve evaluating the system on a larger, more representative dataset.*

### How to Run

1. Make sure MLflow is installed.
2. Ensure training_data.csv and test_data.csv are present.
3. Run:

python track_experiments.py

4. Start MLflow:

mlflow ui

5. Open:

http://127.0.0.1:5000

### Limitations

- **Very Small Dataset**: The current dataset contains only a small number of samples (8 training samples and 2 test samples), which limits statistical validity.
- **Limited Generalization**: Patterns learned from a tiny text corpus do not generalize well to unseen academic texts.
- **Unstable Metrics**: A single misclassification on a 2-sample test set results in a 50 percentage point drop in accuracy.
- **Larger Dataset Required**: A larger, diverse dataset is required for reliable evaluation and production assessment.

### Future Improvements

- **Larger Plagiarism Dataset**: Expand training and evaluation datasets with diverse academic writing samples.
- **Better Class Balance**: Collect balanced original and plagiarized samples to prevent class bias.
- **Cross-Validation**: Incorporate k-fold cross-validation across larger datasets for more robust performance estimates.
- **Advanced NLP Models**: Explore n-gram feature expansion, word embeddings, and semantic similarity techniques.
- **Transformer-Based Models**: Investigate domain-specific transformer models (e.g., BERT, RoBERTa) for deep contextual plagiarism detection.
- **Production Monitoring**: Implement continuous monitoring and model registry tracking in MLflow for production deployment.

## Phase 2 – Task 4: Track Experiments with MLflow

### Objective

MLflow was integrated into the existing AI-Powered Academic Integrity Checker training pipeline to track machine learning experiments. The purpose of this task is to record model parameters, evaluation metrics, and trained model artifacts so that different experiments can be compared and reproduced.

### Implementation

The experiment tracking pipeline is implemented in `track_experiments.py`.

The workflow is:

Dataset
   ↓
Training and Testing Data
   ↓
TF-IDF Vectorization
   ↓
Model Training
   ↓
MLflow Experiment Run
   ↓
Parameters + Metrics + Artifacts
   ↓
MLflow UI

The pipeline uses the existing `training_data.csv` and `test_data.csv` files.

TF-IDF is fitted only on the training data and then used to transform both the training and testing data.

Three machine learning algorithms are tracked:

1. Logistic Regression
2. Support Vector Machine (SVM)
3. Random Forest

For every model, an independent MLflow run is created.

### Parameters Tracked

MLflow records the important model configuration used for each experiment.

#### Logistic Regression
- C = 0.1
- max_iter = 1000
- solver = lbfgs
- random_state = 42

#### SVM
- C = 1.0
- kernel = linear
- random_state = 42

#### Random Forest
- n_estimators = 50
- max_depth = None
- random_state = 42

The TF-IDF feature configuration is also recorded.

### Metrics Tracked

The following evaluation metrics are logged for every model:

- Accuracy
- Precision
- Recall
- F1 Score

These metrics allow the models to be compared using the same evaluation criteria.

### Experiment Results

The current experiment produced the following results:

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.50 | 0.00 | 0.00 | 0.00 |
| SVM | 0.50 | 0.00 | 0.00 | 0.00 |
| Random Forest | 0.50 | 0.00 | 0.00 | 0.00 |

The results are stored in:

`mlflow_experiment_results.csv`

### Model Artifacts

MLflow also stores the trained model and TF-IDF vectorizer as artifacts for each experiment run.

The tracked artifacts include:

- Trained scikit-learn model
- `tfidf_vectorizer.pkl`

Saving these artifacts makes it possible to reproduce or reuse a trained model without retraining it from scratch.

### Relationship with Hyperparameter Tuning

This task builds on the previous Phase 2 hyperparameter tuning task.

During hyperparameter tuning, different parameter configurations were evaluated for Logistic Regression, SVM, and Random Forest.

The best configurations identified during tuning were:

- Logistic Regression: C = 0.1
- SVM: C = 1
- Random Forest: n_estimators = 50, max_depth = None

MLflow provides a structured way to record these configurations together with their resulting metrics and model artifacts.

### How to Run

Make sure the following files are present in the project directory:

- `training_data.csv`
- `test_data.csv`
- `track_experiments.py`

Run the experiment tracking script:

```bash
python track_experiments.py

