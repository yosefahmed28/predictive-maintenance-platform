# Deep Learning & Anomaly Detection Track — Handoff Document

## 1. Module Scope & Objectives
This track audited, evaluated, and benchmarked Deep Learning architectures for the AI4I 2020 Predictive Maintenance dataset. It covers Supervised Neural Networks (Multi-Layer Perceptrons) and Unsupervised Anomaly Detection paradigms (Feature-Weighted Autoencoders and Deep SVDD), evaluating their performance against unseen test data (`X_test`).

---

## 2. Track Artifacts & File Locations

* **Audited Notebook:** `notebook/predictive_maintenance.ipynb`
* **Supervised Model Artifact:** `models/tabular_mlp_tuned.keras`
* **Unsupervised Model Artifact:** `models/autoencoder_anomaly_detector_tuned.keras`
* **Benchmark Metrics Export:** `reports/ml_vs_dl_comparison.csv`

---

## 3. Performance Summary (`X_test` Evaluation)

| Architecture / Model Checkpoint | Optimal Decision Threshold | Test PR-AUC | Test ROC-AUC | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Supervised MLP (`tabular_mlp_tuned.keras`)** | **0.5000** | **0.8418** | **0.9862** | **0.6667** | **0.7843** | **0.7207** |
| Tuned Autoencoder (`autoencoder_anomaly_detector_tuned.keras`) | 0.6541 | 0.1618 | 0.8385 | 0.1783 | 0.4510 | 0.2556 |
| Deep SVDD (Neural Hypersphere Boundary) | Optimal Radius | 0.1475 | N/A | 0.4074 | 0.2157 | 0.2821 |

---

## 4. Key Engineering Insights

1. **Supervised Dominance:** The **Tuned Supervised MLP** leveraging Binary Cross-Entropy is the top-performing deep learning model. Target failure labels provide explicit boundary information necessary to achieve strong precision without triggering excessive false alarms.
2. **Unsupervised Feature Overlap Limits:** Unsupervised models struggle on this continuous tabular dataset. High normal operational swings in stress/wear features generate reconstruction errors and distance metrics that mimic true failure modes, leading to high false-positive rates (precision < 18% for Autoencoders).
3. **Deep SVDD vs. Autoencoder:** Deep SVDD successfully eliminated the decoder shortcut, driving precision up to ~40.7%, but overall recall remains capped due to normal operational variance covering a wider radius than the gap separating normal and failure states.

---

## 5. Instructions for Loading & Inference

To load and run predictions using the audited deep learning artifacts:

```python
import numpy as np
import tensorflow as tf

# Load saved production model
mlp_model = tf.keras.models.load_model('models/tabular_mlp_tuned.keras')

# Predict failure probabilities on new 14-feature tabular sample batch
# Expected input shape: (batch_size, 14)
probabilities = mlp_model.predict(X_new).ravel()
binary_predictions = (probabilities >= 0.50).astype(int)
```
