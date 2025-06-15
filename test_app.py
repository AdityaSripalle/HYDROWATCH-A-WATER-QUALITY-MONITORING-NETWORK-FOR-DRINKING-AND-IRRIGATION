import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch
from io import StringIO

# Import your functions from the app (adjust the import as needed)
from app import load_data, train_models

# Sample CSV data
MOCK_CSV = StringIO("""
pH,EC,CO3,HCO3,Cl,SO4,NO3,TH,Ca,Mg,Na,K,F,TDS,Water Quality Classification,WQI
7.0,250,12,100,25,30,1.5,120,30,12,5,1,0.3,500,Excellent,80
6.8,240,15,105,26,28,1.2,110,28,11,4,1.2,0.2,480,Good,75
7.2,260,13,110,27,32,1.7,125,32,13,6,0.8,0.4,520,Poor,60
""")

@patch("builtins.open")
@patch("pandas.read_csv")
def test_load_data_classification(mock_read_csv, mock_open):
    mock_read_csv.return_value = pd.read_csv(MOCK_CSV)

    df, features, target, encoder, status = load_data("classification")

    assert df is not None
    assert target == "Water Quality Classification"
    assert "Water Quality Classification" not in df.columns  # Encoded
    assert status.strip() == "Data Loaded Successfully!"

@patch("builtins.open")
@patch("pandas.read_csv")
def test_load_data_regression(mock_read_csv, mock_open):
    mock_read_csv.return_value = pd.read_csv(MOCK_CSV)

    df, features, target, encoder, status = load_data("regression")

    assert df is not None
    assert target == "WQI"
    assert "WQI" in df.columns
    assert status.strip() == "Data Loaded Successfully!"

@patch("builtins.open")
@patch("pandas.read_csv")
def test_train_models_classification(mock_read_csv, mock_open):
    mock_read_csv.return_value = pd.read_csv(MOCK_CSV)

    results_df, best_model_name, status = train_models("classification")

    assert results_df is not None
    assert best_model_name in results_df['Model Name'].values
    assert "Best Model Selected" in status

@patch("builtins.open")
@patch("pandas.read_csv")
def test_train_models_regression(mock_read_csv, mock_open):
    mock_read_csv.return_value = pd.read_csv(MOCK_CSV)

    results_df, best_model_name, status = train_models("regression")

    assert results_df is not None
    assert best_model_name in results_df['Model Name'].values
    assert "Best Model Selected" in status
