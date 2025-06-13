from app import train_models

def test_train_classification_models():
    results, best_model_name, status = train_models("classification")
    assert results is not None
    assert not results.empty
    assert best_model_name in results['Model Name'].values
