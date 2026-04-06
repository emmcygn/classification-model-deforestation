import pytest
from core.ml.dataset import load_dataset, prepare_features, split_data
from core.ml.training import train_model
from core.ml.explainability import explain_prediction, explain_summary_text

@pytest.fixture
def trained():
    df = load_dataset()
    X, y, feature_names = prepare_features(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    model = train_model(X_train, y_train, n_estimators=20, max_depth=5)
    return model, X_test, feature_names

def test_explain_prediction(trained):
    model, X_test, feature_names = trained
    result = explain_prediction(model, X_test[0:1], feature_names)
    assert "shap_values" in result
    assert "base_value" in result
    assert "prediction" in result
    assert len(result["shap_values"]) == len(feature_names)

def test_explain_summary_text(trained):
    model, X_test, feature_names = trained
    explanation = explain_prediction(model, X_test[0:1], feature_names)
    text = explain_summary_text(explanation, X_test[0], feature_names)
    assert isinstance(text, str)
    assert len(text) > 20
