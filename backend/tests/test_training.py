import pytest
import numpy as np
from core.ml.dataset import load_dataset, prepare_features, split_data
from core.ml.training import train_model, save_model, load_model
from core.ml.evaluation import evaluate_model

@pytest.fixture
def data():
    df = load_dataset()
    X, y, feature_names = prepare_features(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    return X_train, X_test, y_train, y_test, feature_names

def test_train_model(data):
    X_train, X_test, y_train, y_test, feature_names = data
    model = train_model(X_train, y_train, n_estimators=10, max_depth=5)
    preds = model.predict(X_test)
    assert len(preds) == len(y_test)
    assert set(preds).issubset({0, 1})

def test_evaluate_model(data):
    X_train, X_test, y_train, y_test, feature_names = data
    model = train_model(X_train, y_train, n_estimators=10, max_depth=5)
    metrics = evaluate_model(model, X_test, y_test, feature_names)
    assert "accuracy" in metrics
    assert "f1" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "confusion_matrix" in metrics
    assert "feature_importance" in metrics
    assert len(metrics["feature_importance"]) == len(feature_names)

    # Metrics should be valid numbers in [0, 1]
    for key in ["accuracy", "f1", "precision", "recall"]:
        assert 0 <= metrics[key] <= 1, f"{key}={metrics[key]} out of range"

    # Confusion matrix should be 2x2 and sum to test set size
    cm = metrics["confusion_matrix"]
    assert len(cm) == 2 and len(cm[0]) == 2
    assert sum(sum(row) for row in cm) == len(y_test)

def test_save_load_model(data, tmp_path):
    X_train, _, y_train, _, _ = data
    model = train_model(X_train, y_train, n_estimators=10, max_depth=5)
    path = save_model(model, tmp_path, "test_model")
    loaded = load_model(path)
    assert np.array_equal(model.predict(X_train[:5]), loaded.predict(X_train[:5]))
