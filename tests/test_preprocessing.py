"""
Unit tests for the preprocessing module.
"""
import pytest
import pandas as pd
import numpy as np
from src.churn_pipeline.data.preprocessing import (
    BinaryEncoder,
    DropColumns,
    prepare_data,
    build_preprocessor,
    BINARY_COLS,
    COLS_TO_DROP,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_df():
    """Minimal valid DataFrame mimicking the Telco dataset structure."""
    return pd.DataFrame({
        "customerID": ["1111-AAA", "2222-BBB", "3333-CCC"],
        "gender": ["Male", "Female", "Male"],
        "SeniorCitizen": [0, 1, 0],
        "Partner": ["Yes", "No", "Yes"],
        "Dependents": ["No", "No", "Yes"],
        "tenure": [12, 1, 45],
        "PhoneService": ["Yes", "Yes", "No"],
        "MultipleLines": ["No", "Yes", "No phone service"],
        "InternetService": ["DSL", "Fiber optic", "DSL"],
        "OnlineSecurity": ["Yes", "No", "Yes"],
        "OnlineBackup": ["No", "Yes", "No"],
        "DeviceProtection": ["Yes", "No", "Yes"],
        "TechSupport": ["No", "No", "Yes"],
        "StreamingTV": ["No", "Yes", "No"],
        "StreamingMovies": ["Yes", "No", "Yes"],
        "Contract": ["Month-to-month", "One year", "Two year"],
        "PaperlessBilling": ["Yes", "No", "Yes"],
        "PaymentMethod": ["Electronic check", "Mailed check", "Bank transfer (automatic)"],
        "MonthlyCharges": [52.55, 20.25, 80.85],
        "TotalCharges": [630.6, 20.25, 3634.9],
        "Churn": ["No", "Yes", "No"],
        "Churn_binary": [0, 1, 0],
    })


# ── DropColumns tests ─────────────────────────────────────────────────────────

def test_drop_columns_removes_specified_columns(sample_df):
    """DropColumns should remove exactly the specified columns."""
    transformer = DropColumns(columns=["customerID", "gender"])
    result = transformer.transform(sample_df)
    assert "customerID" not in result.columns
    assert "gender" not in result.columns


def test_drop_columns_keeps_remaining_columns(sample_df):
    """DropColumns should not remove any unspecified columns."""
    transformer = DropColumns(columns=["customerID"])
    result = transformer.transform(sample_df)
    assert "tenure" in result.columns
    assert "Churn" in result.columns


def test_drop_columns_ignores_missing_columns(sample_df):
    """DropColumns should not raise if a column to drop does not exist."""
    transformer = DropColumns(columns=["customerID", "nonexistent_column"])
    result = transformer.transform(sample_df)
    assert "customerID" not in result.columns


# ── BinaryEncoder tests ───────────────────────────────────────────────────────

def test_binary_encoder_converts_yes_to_1(sample_df):
    """BinaryEncoder should convert Yes to 1."""
    encoder = BinaryEncoder(columns=["Partner"])
    result = encoder.transform(sample_df)
    assert result["Partner"].iloc[0] == 1


def test_binary_encoder_converts_no_to_0(sample_df):
    """BinaryEncoder should convert No to 0."""
    encoder = BinaryEncoder(columns=["Partner"])
    result = encoder.transform(sample_df)
    assert result["Partner"].iloc[1] == 0


def test_binary_encoder_does_not_modify_other_columns(sample_df):
    """BinaryEncoder should only modify specified columns."""
    encoder = BinaryEncoder(columns=["Partner"])
    result = encoder.transform(sample_df)
    assert result["tenure"].tolist() == sample_df["tenure"].tolist()


def test_binary_encoder_handles_missing_column_gracefully(sample_df):
    """BinaryEncoder should skip columns that do not exist."""
    encoder = BinaryEncoder(columns=["Partner", "nonexistent_column"])
    result = encoder.transform(sample_df)
    assert "Partner" in result.columns


# ── prepare_data tests ────────────────────────────────────────────────────────

def test_prepare_data_returns_tuple(sample_df):
    """prepare_data should return a tuple of (X, y)."""
    X, y = prepare_data(sample_df)
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)


def test_prepare_data_drops_target_column(sample_df):
    """prepare_data should not include Churn in X."""
    X, y = prepare_data(sample_df)
    assert "Churn" not in X.columns


def test_prepare_data_drops_useless_columns(sample_df):
    """prepare_data should drop customerID and gender."""
    X, y = prepare_data(sample_df)
    for col in COLS_TO_DROP:
        assert col not in X.columns


def test_prepare_data_target_is_binary(sample_df):
    """y should only contain 0 and 1."""
    _, y = prepare_data(sample_df)
    assert set(y.unique()).issubset({0, 1})


def test_prepare_data_target_length_matches_features(sample_df):
    """X and y should have the same number of rows."""
    X, y = prepare_data(sample_df)
    assert len(X) == len(y)


def test_prepare_data_binary_cols_are_numeric(sample_df):
    """Binary columns should be 0/1 integers after prepare_data."""
    X, y = prepare_data(sample_df)
    for col in ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]:
        if col in X.columns:
            assert X[col].isin([0, 1]).all(), f"{col} contains non-binary values"