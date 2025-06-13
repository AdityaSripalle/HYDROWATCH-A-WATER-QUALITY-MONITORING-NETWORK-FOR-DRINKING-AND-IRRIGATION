# test_app.py

from app import train_models

def test_train_classification_models():
    results, best_model_name, status = train_models("classification")
    assert results is not None, "Classification training results should not be None"
    assert best_model_name in results['Model Name'].values, "Best model should be in the results"

def test_train_regression_models():
    results, best_model_name, status = train_models("regression")
    assert results is not None, "Regression training results should not be None"
    assert best_model_name in results['Model Name'].values, "Best model should be in the results"
