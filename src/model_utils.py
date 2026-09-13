# src/model_utils.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, precision_recall_curve, average_precision_score,
    f1_score, recall_score, precision_score, confusion_matrix
)

def train_and_evaluate(model, X_train, y_train, X_val, y_val, model_name,
                         threshold=0.5, already_fitted=False, verbose=True):
    '''
    this function takes the model and returns its threshold, metrics,
    confusion matrix items & classification report
    '''
    if not already_fitted:
        model.fit(X_train, y_train)

    y_proba = model.predict_proba(X_val)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()

    results = {
        "model": model_name,
        "threshold": threshold,
        "precision": precision_score(y_val, y_pred, zero_division=0),
        "recall": recall_score(y_val, y_pred, zero_division=0),
        "f1": f1_score(y_val, y_pred, zero_division=0),
        "pr_auc": average_precision_score(y_val, y_proba),
        "false_negative_rate": fn / (fn + tp) if (fn + tp) > 0 else np.nan,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "fitted_model": model,
        "y_val": y_val,
        "y_pred": y_pred,
        "y_proba": y_proba,
    }

    if verbose:
        print(f"=== {model_name} (threshold={threshold}) ===")
        for k, v in results.items():
            if k not in ("fitted_model", "y_val", "y_pred", "y_proba"):
                print(f"{k}: {v}")
        print("\nFull report:\n", classification_report(y_val, y_pred, digits=3))

    return results


def plot_confusion_matrix(results):
    cm = confusion_matrix(results["y_val"], results["y_pred"])
    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["No Failure", "Failure"])
    ax.set_yticklabels(["No Failure", "Failure"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"{results['model']} — Confusion Matrix")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max()/2 else "black", fontsize=14)
    plt.colorbar(im)
    plt.tight_layout()
    plt.show()


def plot_pr_curve(results, ax=None, show=True):
    precision, recall, _ = precision_recall_curve(results["y_val"], results["y_proba"])
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(recall, precision, label=f"{results['model']} (PR-AUC={results['pr_auc']:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend()
    if show:
        plt.tight_layout()
        plt.show()
    return ax


def plot_feature_importance(results, feature_names):
    model = results["fitted_model"]
    if not hasattr(model, "feature_importances_"):
        print(f"{results['model']} has no feature_importances_ — skipping.")
        return
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh([feature_names[i] for i in sorted_idx][::-1], importances[sorted_idx][::-1])
    ax.set_xlabel("Feature Importance")
    ax.set_title(f"{results['model']} — Feature Importance")
    plt.tight_layout()
    plt.show()


def predict_new_sample(model, preprocessor, zscorer, raw_sample_df, threshold=0.5):
    """
    Takes a RAW (unprocessed) single-row or multi-row DataFrame with the
    original column names/values, applies the SAME preprocessing pipeline
    used in training (zscorer, then preprocessor), and returns predictions.
    This is the function Role 6 will reuse for live/simulated scoring.
    """
    from src.features import engineer_features
    df_fe = engineer_features(raw_sample_df)
    df_fe = zscorer.transform(df_fe)
    X_ready = preprocessor.transform(df_fe)

    proba = model.predict_proba(X_ready)[:, 1]
    pred = (proba >= threshold).astype(int)

    return pd.DataFrame({
        "failure_probability": proba,
        "predicted_failure": pred,
    })