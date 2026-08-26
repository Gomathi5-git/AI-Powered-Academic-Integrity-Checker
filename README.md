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
```

---

## Phase 2 – Task 5: Select & Validate

### Objective

The objective of Task 5 is to perform statistical validation across candidate machine learning models using K-Fold Cross-Validation, enabling a reliable model-selection decision while accounting for dataset size and metric stability.

### Context & Connections with Previous Phase 2 Tasks

Task 5 integrates insights from all previous Phase 2 milestones:

1. **Model Training**: Initial baseline training established the NLP text classification approach using TF-IDF feature vectors.
2. **Hyperparameter Tuning**: Grid search identified optimal model hyperparameters (`C=0.1` for Logistic Regression, `C=1.0` linear kernel for SVM, and `n_estimators=50`, `max_depth=None` for Random Forest).
3. **Model Architecture Comparison**: Candidate model families (linear vs. margin-based vs. decision tree ensemble) were evaluated.
4. **Experiment Tracking**: MLflow was implemented in `track_experiments.py` to record model runs, parameters, metrics, and model artifacts.
5. **Selection & Validation**: Stratified K-Fold Cross-Validation is now applied in `select_validate.py` to evaluate model variance across multiple folds and select the optimal model.

### Preprocessing & Validation Methodology

- **Dataset**: `training_data_balanced.csv` (12 text samples, balanced binary classification).
- **Text Feature Extraction**: `TfidfVectorizer` (English stop-word removal).
- **Cross-Validation Strategy**: 5-Fold Stratified K-Fold Cross-Validation (`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`), automatically configured based on the minor class count to prevent data leakage and handle class distribution safely.

### Cross-Validation Results

The table below summarizes the mean and standard deviation for Accuracy, Precision, Recall, and F1 Score across 5 cross-validation folds:

| Model | Accuracy Mean (±Std) | Precision Mean (±Std) | Recall Mean (±Std) | F1 Mean (±Std) |
|---|---|---|---|---|
| **Logistic Regression** | 0.5333 (±0.2449) | 0.4667 (±0.3232) | 0.8000 (±0.4000) | 0.5667 (±0.3266) |
| **SVM** | 0.6667 (±0.1826) | 0.6000 (±0.3742) | 0.7000 (±0.4000) | 0.6000 (±0.3266) |
| **Random Forest** | **0.9000 (±0.2000)** | **0.9000 (±0.2000)** | **1.0000 (±0.0000)** | **0.9333 (±0.1333)** |

### Model Selection & Rationale

* **Selected Model**: **Random Forest** (`n_estimators=50`, `max_depth=None`, `random_state=42`)
* **Selection Reason**:
  - Random Forest achieved the highest **Mean F1 Score of 0.9333 (±0.1333)** across 5 cross-validation folds.
  - F1 Score was chosen as the primary selection metric because academic plagiarism detection requires a balance between Precision (avoiding false accusations) and Recall (catching actual plagiarized submissions).
  - Random Forest exhibited lower standard deviation in F1 score (0.1333) compared to Logistic Regression (0.3266) and SVM (0.3266), indicating superior stability on the prototype dataset.

### Small Dataset Limitation & Validation Warning

* **Baseline Benchmark Status**: The current dataset contains only 12 training samples and 2 test samples.
* **Metric Instability**: Cross-validation on small sample sizes yields high standard deviations. The reported metrics serve strictly as a preliminary prototype benchmark and must **NOT** be interpreted as production-ready accuracy.

### How to Run Validation

1. Open PowerShell in the project directory: `l:\mlproject`.
2. Ensure required Python packages (`pandas`, `scikit-learn`, `matplotlib`, `numpy`) are installed.
3. Run the validation script:

```bash
python select_validate.py
```

4. The script will output validation metrics to the terminal and automatically generate the following 3 files:
   - `validation_results.csv`: Contains tabular mean and standard deviation metrics for all models across all folds.
   - `model_selection_report.txt`: Detailed text report summarizing dataset stats, hyperparameters, fold metrics, selection rationale, and limitations.
   - `model_validation_comparison.png`: High-resolution error-bar chart visualizing Mean F1 Scores (±Std Dev) across candidate models.

### Future Improvements

- **Larger Plagiarism Dataset**: Expand sample collection with real-world student essays and reference text sources.
- **Advanced Semantic Features**: Incorporate n-gram embeddings, sentence similarity measures, and transformer models (e.g., BERT/RoBERTa).
- **Production MLflow Deployment**: Register the selected Random Forest model in the MLflow Model Registry for automated deployment and monitoring.

---

## Phase 3 – Task 1: Build Serving Layer

### Purpose & Overview

The Model Serving Layer provides a production-ready REST API built with **FastAPI** to expose the trained plagiarism classification model for real-time inference. It accepts student submission text via HTTP requests, validates the input using Pydantic, extracts TF-IDF numerical features, passes the vector to the selected **Random Forest** model, and returns structured JSON predictions containing classification labels, plagiarism flags, and confidence scores.

### System Architecture & Data Flow

```text
Client Request (HTTP POST /predict)
   ↓
FastAPI Application (app.py)
   ↓
Pydantic Request Validation (PredictionRequest)
   ↓
TF-IDF Preprocessing (TfidfVectorizer from tfidf_vectorizer.pkl)
   ↓
Random Forest Classification (random_forest_model.pkl)
   ↓
Confidence Calculation (predict_proba)
   ↓
Structured JSON Response (PredictionResponse)
```

### Model & Feature Extractor Artifacts

- **Selected Model**: Random Forest Classifier (`n_estimators=50`, `max_depth=None`, `random_state=42`), loaded from `random_forest_model.pkl` (or `best_model.pkl`).
- **Feature Extractor**: Fitted `TfidfVectorizer` (Vocabulary size: 35 features), loaded from `tfidf_vectorizer.pkl`.
- **Target Classes**: `original` (Non-plagiarized) and `plagiarized` (Plagiarized submission).

### API Endpoints Summary

| Method | Endpoint | Description | Request Body | Success Response | Error Code |
|---|---|---|---|---|---|
| `GET` | `/health` | Service Health Check & Model Metadata | None | `{"status": "healthy", "model": "Random Forest"}` | `503` |
| `POST` | `/predict` | Academic Plagiarism Prediction | `{"text": "..."}` | `{"prediction": "...", "is_plagiarized": bool, "confidence": float}` | `422` / `500` |

### Input Validation Rules (Pydantic Schema)

All incoming requests to `/predict` are validated using Pydantic v2 schemas (`PredictionRequest`):
1. **Type Check**: `text` must be a valid string.
2. **Non-Empty**: `text` cannot be an empty string (`""`).
3. **Whitespace Check**: `text` cannot consist solely of spaces, tabs, or newlines (`"   "`).
4. **Length Constraint**: `text` must not exceed `10,000` characters.
5. **Rejection**: Any violation automatically returns HTTP `422 Unprocessable Entity` with field-level diagnostic error details.

### How to Install Dependencies & Run the Server

1. **Open PowerShell** in the project directory:
   ```bash
   cd l:\mlproject
   ```

2. **Install Required Packages**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Start the FastAPI Server with Uvicorn**:
   ```bash
   uvicorn app:app --reload
   ```
   *The server will start on `http://127.0.0.1:8000`.*

4. **Access Interactive Swagger / OpenAPI Documentation**:
   Open your browser to:
   ```text
   http://127.0.0.1:8000/docs
   ```

### How to Run Automated API Tests

To execute the test suite (8 unit & integration tests validating health checks, inference responses, and all validation failure cases):

```bash
python -m pytest test_api.py
```

### Example API Request & Response

#### 1. Health Check Request (`GET /health`)
```powershell
curl -X GET "http://127.0.0.1:8000/health"
```
**Response (`200 OK`)**:
```json
{
  "status": "healthy",
  "model": "Random Forest"
}
```

#### 2. Plagiarism Inference Request (`POST /predict`)
```powershell
curl -X POST "http://127.0.0.1:8000/predict" `
     -H "Content-Type: application/json" `
     -d '{"text": "Natural language processing helps computers understand human language."}'
```
**Response (`200 OK`)**:
```json
{
  "prediction": "plagiarized",
  "is_plagiarized": true,
  "confidence": 0.7895
}
```

#### 3. Validation Failure Example (`POST /predict` with whitespace text)
```powershell
curl -X POST "http://127.0.0.1:8000/predict" `
     -H "Content-Type: application/json" `
     -d '{"text": "   "}'
```
**Response (`422 Unprocessable Entity`)**:
```json
{
  "detail": [
    {
      "type": "value_error",
      "loc": ["body", "text"],
      "msg": "Value error, Input text must not contain only whitespace.",
      "input": "   "
    }
  ]
}
```

### Small Dataset Limitation & Serving vs. ML Performance Disclaimer

> **Important Distinction**: 
> - **Serving Layer Functionality**: The FastAPI service architecture, REST endpoints, Pydantic validation, schema handling, and unit test suite are fully operational and robust.
> - **Model Detection Performance**: The underlying Random Forest model was trained on a prototype dataset of 12 samples. Therefore, returned predictions and confidence scores serve strictly as an integration demonstration. Production deployment requires retraining on a large, representative plagiarism dataset.

## Phase 3 – A/B Testing

### Task Objective

Implemented A/B testing to compare different machine learning model versions for the Academic Integrity Checker.

### Models Compared

- **Model A:** Logistic Regression
- **Model B:** Random Forest

The application uses approximately a **50/50 traffic split** between the two model versions.

### A/B Test Results

| Model | Requests | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 6 | 1.00 | 1.00 | 1.00 | 1.00 |
| Random Forest | 6 | 0.8333 | 0.80 | 1.00 | 0.8889 |

In this particular A/B experiment, Logistic Regression achieved the higher F1 score and accuracy. However, the experiment used only 12 samples, so these results should be considered a preliminary prototype comparison rather than statistically significant evidence.

### Implementation

The following files were created:

- `ab_testing.py` – Performs A/B traffic splitting and model prediction.
- `ab_test_results.csv` – Stores model-level performance metrics.
- `ab_test_individual_results.csv` – Stores individual predictions and correctness.
- `ab_test_report.txt` – Documents the A/B testing methodology, results, limitations, and future improvements.
- `ab_test_comparison.png` – Visual comparison of model performance.
- `create_ab_report.py` – Generates the A/B testing report and comparison chart.

### Relationship to Model Validation

The A/B testing results are interpreted together with the previous cross-validation results.

The previous validation task selected **Random Forest** based on its highest mean F1 score during 5-fold Stratified Cross-Validation. The A/B test produced different results because it used a very small number of requests and each model received a different subset of the data.

Therefore, the A/B test demonstrates the model comparison and traffic-splitting workflow, while cross-validation remains the stronger basis for model selection.

### Limitations

The current experiment uses only 12 balanced samples. A larger dataset and a larger number of real-world requests would be required for statistically reliable A/B testing.

Future improvements include:

- Larger real-world datasets
- Longer A/B testing periods
- Statistical significance testing
- Latency and resource monitoring
- Continuous result collection
- MLflow integration
- Dynamic traffic allocation

## Phase 3 – Model Monitoring

### Task Objective

Implemented a model monitoring system for the deployed Random Forest plagiarism detection model. The system monitors prediction behaviour, prediction latency, model accuracy, and prediction distribution drift.

### Monitoring Metrics

The monitoring system tracks:

- Prediction distribution
- Prediction latency
- Model accuracy
- Prediction distribution drift
- Configured monitoring thresholds
- Alerts when thresholds are exceeded

### Monitoring Results

The monitoring system was tested using 12 samples.

| Metric | Result |
|---|---:|
| Model | Random Forest |
| Monitoring samples | 12 |
| Accuracy | 91.67% |
| Original predictions | 41.67% |
| Plagiarized predictions | 58.33% |
| Average latency | 8.44 ms |
| Maximum latency | 15.36 ms |
| Latency threshold | 100 ms |
| Maximum distribution drift | 8.33% |
| Drift threshold | 20% |
| Alerts | None |

The prediction distribution remained within the configured drift threshold, and the observed prediction latency remained below the configured latency threshold.

### Drift Detection

A 50/50 baseline distribution was used for the prototype:

- Original: 50%
- Plagiarized: 50%

The maximum observed difference was 8.33%, which is below the configured 20% drift threshold. Therefore, no prediction distribution drift alert was generated.

### Monitoring Architecture

The monitoring workflow is:

```text
Model Serving API
       ↓
Model Prediction
       ↓
Monitoring Layer
   ↙    ↓     ↘
Prediction  Latency  Accuracy
Distribution
       ↓
Drift Detection
       ↓
Threshold Checks
       ↓
Alerts
       ↓
JSON / CSV Logs

## Phase 4 – Automated Model Retraining Pipeline

### Task Objective

Implemented an automated retraining pipeline for the plagiarism detection model with validation gates to ensure that a new model meets minimum performance requirements before deployment.

### Retraining Workflow

```text
Training Data
     ↓
Data Preparation
     ↓
TF-IDF Vectorization
     ↓
Random Forest Training
     ↓
Model Validation
     ↓
Validation Gates
   ↙       ↘
 PASS      FAIL
  ↓          ↓
Deploy     Reject
 Model      Model
  ↓
Backup Previous Model
  ↓
Retraining Log

