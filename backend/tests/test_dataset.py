import pytest
from core.ml.dataset import load_dataset, prepare_features

def test_load_dataset():
    df = load_dataset()
    assert len(df) > 0
    assert "lat" in df.columns
    assert "high_risk" in df.columns

def test_prepare_features():
    df = load_dataset()
    X, y, feature_names = prepare_features(df)
    assert X.shape[0] == len(df)
    assert X.shape[1] == len(feature_names)
    assert len(y) == len(df)
    assert "lat" not in feature_names
    assert "high_risk" not in feature_names
    assert X.shape[1] >= 8  # At least base features
