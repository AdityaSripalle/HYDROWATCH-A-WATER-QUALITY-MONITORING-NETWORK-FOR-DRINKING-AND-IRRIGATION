import pytest
from app import load_data, train_models, predict_water_quality

# Test classification data loading
def test_load_data_classification():
    df, features, target, label_encoder, status = load_data("classification")
    assert df is not None
    assert "Water Quality Classification" in df.columns
    assert label_encoder is not None
    assert status.strip().startswith("Data Loaded")

# Test regression data loading
def test_load_data_regression():
    df, features, target, label_encoder, status = load_data("regression")
    assert df is not None
    assert "WQI" in df.columns
    assert status.strip().startswith("Data Loaded")

# Test model training for classification
def test_train_models_classification():
    results_df, best_model_name, status = train_models("classification")
    assert not results_df.empty
    assert "Model Name" in results_df.columns
    assert "Best Model" in status or "🏆" in status

# Test prediction with dummy input after training
def test_predict_water_quality_classification():
    train_models("classification")
    result, result2 = predict_water_quality(*[7.0]*15)
    assert "Predicted Water Quality Class" in result or "Water Quality Class" in result2
