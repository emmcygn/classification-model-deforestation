"""Model training and persistence."""

from __future__ import annotations

import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


def train_model(
    X_train,
    y_train,
    n_estimators: int = 100,
    max_depth: int | None = 10,
    min_samples_split: int = 5,
    min_samples_leaf: int = 2,
    random_state: int = 42,
    model_type: str = "random_forest",
) -> RandomForestClassifier | "XGBClassifier":
    """Train a classifier. Supports 'random_forest' and 'xgboost'."""
    if model_type == "xgboost" and HAS_XGBOOST:
        # Compute scale_pos_weight for imbalanced classes
        n_pos = int(y_train.sum())
        n_neg = len(y_train) - n_pos
        scale = n_neg / max(n_pos, 1)

        model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth or 6,
            learning_rate=0.1,
            scale_pos_weight=scale,
            random_state=random_state,
            n_jobs=-1,
            eval_metric="logloss",
        )
    else:
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            class_weight="balanced",
            n_jobs=-1,
        )

    model.fit(X_train, y_train)
    return model


def save_model(model, directory: Path, name: str) -> Path:
    """Save model to disk. Returns the path to the saved file."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.joblib"
    joblib.dump(model, path)
    return path


def load_model(path: Path):
    """Load a saved model from disk."""
    return joblib.load(path)
