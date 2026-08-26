import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# LOAD A/B TEST RESULTS
# ============================================================

results = pd.read_csv("ab_test_results.csv")


# ============================================================
# FIND BEST MODEL IN THIS A/B EXPERIMENT
# ============================================================

best_row = results.loc[results["F1 Score"].idxmax()]

best_model = best_row["Model"]
best_f1 = best_row["F1 Score"]


# ============================================================
# CREATE TEXT REPORT
# ============================================================

report = f"""
============================================================
A/B TESTING REPORT
AI-POWERED ACADEMIC INTEGRITY CHECKER
============================================================

1. OBJECTIVE
------------------------------------------------------------

The objective of this A/B testing experiment was to compare
two machine learning model versions for academic integrity
classification.

The experiment demonstrates traffic splitting and allows the
performance of both model versions to be compared using
accuracy, precision, recall, and F1 score.


2. MODEL VERSIONS
------------------------------------------------------------

Model A:
    Logistic Regression
    Role: Baseline model

Model B:
    Random Forest
    Role: Selected/best model from previous validation


3. TRAFFIC ALLOCATION
------------------------------------------------------------

The A/B router was configured to distribute incoming requests
approximately equally between the two model versions.

Target allocation:

    Model A: 50%
    Model B: 50%

Actual requests in this experiment:

    Model A: {int(results.loc[results["Model Version"] == "A", "Requests"].iloc[0])}
    Model B: {int(results.loc[results["Model Version"] == "B", "Requests"].iloc[0])}

Total requests:

    {int(results["Requests"].sum())}


4. PERFORMANCE RESULTS
------------------------------------------------------------

"""

for _, row in results.iterrows():
    report += f"""
{row["Model"]} ({row["Model Version"]})

    Requests : {int(row["Requests"])}
    Accuracy : {row["Accuracy"]:.4f}
    Precision: {row["Precision"]:.4f}
    Recall   : {row["Recall"]:.4f}
    F1 Score : {row["F1 Score"]:.4f}

"""


report += f"""
5. A/B TEST OBSERVATION
------------------------------------------------------------

Based on this particular A/B experiment, {best_model} achieved
the highest F1 score of {best_f1:.4f}.

The experiment therefore indicates that {best_model} performed
better on the requests assigned during this test.


6. RELATION TO PREVIOUS VALIDATION
------------------------------------------------------------

The A/B experiment should be interpreted together with the
cross-validation results from the previous model selection task.

The previous 5-fold Stratified Cross-Validation selected
Random Forest based on its highest mean F1 score.

The A/B experiment produced different observed results because
it used only a small number of requests and each model received
a different subset of those requests.

Therefore, the A/B experiment does not replace cross-validation.


7. LIMITATIONS
------------------------------------------------------------

This is a prototype A/B testing implementation.

The dataset contains only 12 balanced samples.

Because of this small sample size:

- The results are not statistically significant.
- The results should not be considered representative of
  production traffic.
- A larger real-world dataset would be required for reliable
  model comparison.
- More requests and longer observation periods would provide
  stronger evidence.


8. FUTURE IMPROVEMENTS
------------------------------------------------------------

Future versions could:

- Use a much larger validation dataset.
- Run the A/B test over real application traffic.
- Store results continuously.
- Use statistical significance testing.
- Monitor latency and resource usage.
- Dynamically adjust traffic allocation based on performance.
- Integrate the A/B testing system with MLflow.


============================================================
END OF REPORT
============================================================
"""

with open("ab_test_report.txt", "w", encoding="utf-8") as file:
    file.write(report)


# ============================================================
# CREATE VISUALIZATION
# ============================================================

metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]

x = range(len(results))

width = 0.35

plt.figure(figsize=(10, 6))

for i, metric in enumerate(metrics):

    values = results[metric].tolist()

    positions = [
        pos + i * width
        for pos in x
    ]

    plt.bar(
        positions,
        values,
        width=width,
        label=metric
    )

plt.xticks(
    [
        pos + width * 1.5
        for pos in x
    ],
    results["Model"]
)

plt.ylabel("Score")
plt.title("A/B Testing Model Performance Comparison")
plt.ylim(0, 1.1)
plt.legend()

plt.tight_layout()

plt.savefig(
    "ab_test_comparison.png",
    dpi=150
)

plt.close()


print("A/B testing report created:")
print("ab_test_report.txt")

print("A/B testing visualization created:")
print("ab_test_comparison.png")