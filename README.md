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
