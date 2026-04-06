"""Model evaluation and metrics."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
) -> dict:
    """Evaluate model and return metrics dict."""
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    importances = model.feature_importances_
    feature_importance = [
        {"feature": name, "importance": round(float(imp), 4)}
        for name, imp in sorted(zip(feature_names, importances), key=lambda x: -x[1])
    ]

    return {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "confusion_matrix": cm.tolist(),
        "feature_importance": feature_importance,
    }


def find_optimal_threshold(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """Find the probability threshold that maximizes F1 score.

    Instead of the default 0.5, finds the threshold where
    precision and recall are best balanced for the specific dataset.
    """
    probabilities = model.predict_proba(X_test)[:, 1]

    best_f1 = 0
    best_threshold = 0.5
    results = []

    for threshold in np.arange(0.1, 0.9, 0.05):
        preds = (probabilities >= threshold).astype(int)
        p = precision_score(y_test, preds, zero_division=0)
        r = recall_score(y_test, preds, zero_division=0)
        f = f1_score(y_test, preds, zero_division=0)

        results.append({
            "threshold": round(float(threshold), 2),
            "precision": round(float(p), 4),
            "recall": round(float(r), 4),
            "f1": round(float(f), 4),
        })

        if f > best_f1:
            best_f1 = f
            best_threshold = threshold

    return {
        "optimal_threshold": round(float(best_threshold), 2),
        "optimal_f1": round(float(best_f1), 4),
        "default_threshold": 0.5,
        "threshold_curve": results,
    }
