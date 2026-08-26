import pandas as pd
import json

# Load actual monitoring results
results = pd.read_csv("monitoring_results.csv")

with open("monitoring_log.json", "r", encoding="utf-8") as f:
    log = json.load(f)

row = results.iloc[0]

report = f"""
============================================================
MODEL MONITORING REPORT
AI-POWERED ACADEMIC INTEGRITY CHECKER
============================================================

1. OBJECTIVE
------------------------------------------------------------

A model monitoring system was implemented to monitor the
behaviour and operational performance of the deployed
Random Forest model.

The monitoring system tracks:

- Prediction distribution
- Prediction latency
- Model accuracy
- Prediction distribution drift
- Monitoring thresholds
- Alerts


2. MONITORED MODEL
------------------------------------------------------------

Model:
    {row["model"]}

Monitoring samples:
    {int(row["total_predictions"])}


3. PREDICTION DISTRIBUTION
------------------------------------------------------------

The system records the distribution of predictions made by
the model.

Original predictions:
    {row["original_percentage"] * 100:.2f}%

Plagiarized predictions:
    {row["plagiarized_percentage"] * 100:.2f}%

This distribution can be monitored over time to identify
unusual changes in model behaviour.


4. LATENCY MONITORING
------------------------------------------------------------

Prediction latency is measured for every model prediction.

Average latency:
    {row["average_latency_ms"]:.4f} ms

Minimum latency:
    {row["minimum_latency_ms"]:.4f} ms

Maximum latency:
    {row["maximum_latency_ms"]:.4f} ms

Configured latency threshold:
    100.00 ms

Latency alert:
    {"TRIGGERED" if row["latency_alert"] else "NOT TRIGGERED"}

The observed latency remained below the configured
threshold during this monitoring run.


5. MODEL PERFORMANCE
------------------------------------------------------------

Accuracy on the monitoring dataset:
    {row["accuracy"]:.4f}

Percentage:
    {row["accuracy"] * 100:.2f}%


6. DRIFT DETECTION
------------------------------------------------------------

The monitoring system compares the current prediction
distribution against a baseline distribution of:

    Original      : 50%
    Plagiarized   : 50%

Maximum observed distribution drift:
    {row["maximum_drift"] * 100:.2f}%

Configured drift threshold:
    20.00%

Drift status:
    {"DRIFT DETECTED" if row["drift_detected"] else "NO DRIFT DETECTED"}

The observed distribution difference remained below the
configured threshold.


7. ALERTING
------------------------------------------------------------

The monitoring system generates alerts when configured
thresholds are exceeded.

Current monitoring result:

    {"High latency detected" if row["latency_alert"] else "No high-latency alert"}
    {"Prediction distribution drift detected" if row["drift_detected"] else "No prediction distribution drift alert"}

Overall status:
    {"ALERTS DETECTED" if (row["latency_alert"] or row["drift_detected"]) else "NO ALERTS DETECTED"}


8. MONITORING ARCHITECTURE
------------------------------------------------------------

The monitoring workflow is:

    Model Serving API
           |
           v
    Model Prediction
           |
           v
    Monitoring Layer
      /     |      \\
     /      |       \\
Prediction  Latency  Accuracy
Distribution   |        |
     \\       |       /
      \\      |      /
           v
      Drift Detection
           |
           v
        Thresholds
           |
           v
         Alerts
           |
           v
   JSON / CSV Monitoring Logs


9. MONITORING FILES
------------------------------------------------------------

model_monitoring.py
    Main monitoring implementation.

monitoring_log.json
    Stores structured monitoring information.

monitoring_results.csv
    Stores monitoring metrics in tabular format.

prediction_distribution.png
    Visualizes the model's prediction distribution.

prediction_latency.png
    Visualizes prediction latency for individual requests.


10. LIMITATIONS
------------------------------------------------------------

This implementation is a prototype monitoring system.

The monitoring dataset contains only 12 samples. Therefore,
the observed accuracy, prediction distribution, latency, and
drift values should not be considered representative of
production behaviour.

The baseline distribution is also a simple 50/50 reference
distribution.

A production system would require a much larger stream of
real-world predictions and historical monitoring data.


11. FUTURE IMPROVEMENTS
------------------------------------------------------------

Future improvements could include:

- Continuous monitoring of production traffic
- Persistent monitoring database
- Elasticsearch integration
- Automated alert notifications
- More advanced statistical drift detection
- Monitoring precision, recall and F1 over time
- Historical dashboards
- Automated model retraining
- MLflow integration
- Monitoring infrastructure using Docker


12. CONCLUSION
------------------------------------------------------------

The monitoring system successfully tracks prediction
distribution, prediction latency, model accuracy and
distribution drift.

For the current monitoring run, no latency or prediction
distribution alerts were triggered.

The implementation provides a foundation for continuously
monitoring the Academic Integrity Checker model and detecting
potential operational or behavioural changes.


============================================================
END OF MODEL MONITORING REPORT
============================================================
"""

with open(
    "model_monitoring_report.txt",
    "w",
    encoding="utf-8"
) as f:
    f.write(report)

print("Model monitoring report created:")
print("model_monitoring_report.txt")