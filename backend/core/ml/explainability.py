"""SHAP-based model explainability for individual predictions."""

import numpy as np
import shap
from sklearn.ensemble import RandomForestClassifier


def explain_prediction(
    model: RandomForestClassifier,
    X_single: np.ndarray,
    feature_names: list[str],
) -> dict:
    """Compute SHAP values for a single prediction."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_single)

    prediction = int(model.predict(X_single)[0])
    if isinstance(shap_values, list):
        # Older SHAP: list of arrays, one per class
        sv = shap_values[prediction][0]
        base = float(explainer.expected_value[prediction])
    elif shap_values.ndim == 3:
        # Newer SHAP: single array with shape (n_samples, n_features, n_classes)
        sv = shap_values[0, :, prediction]
        base = float(explainer.expected_value[prediction])
    else:
        sv = shap_values[0]
        base = float(explainer.expected_value) if np.isscalar(explainer.expected_value) else float(explainer.expected_value[prediction])

    contributions = [
        {
            "feature": name,
            "value": round(float(X_single[0, i]), 4),
            "shap_value": round(float(sv[i]), 4),
        }
        for i, name in enumerate(feature_names)
    ]
    contributions.sort(key=lambda x: -abs(x["shap_value"]))

    return {
        "prediction": prediction,
        "prediction_label": "High Risk" if prediction == 1 else "Low Risk",
        "base_value": round(base, 4),
        "shap_values": contributions,
    }


def explain_summary_text(
    explanation: dict,
    x_raw: np.ndarray,
    feature_names: list[str],
) -> str:
    """Generate a deterministic, templated explainability summary."""
    label = explanation["prediction_label"]
    contribs = explanation["shap_values"]
    top = contribs[:3]

    parts = []
    for c in top:
        sign = "+" if c["shap_value"] >= 0 else ""
        parts.append(f"{c['feature']} ({c['value']}, {sign}{c['shap_value']})")

    drivers = ", ".join(parts)
    return f"{label} — driven by: {drivers}"
