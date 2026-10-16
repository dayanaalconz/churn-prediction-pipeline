"""
Unit tests for the feature engineering module.
"""
import pandas as pd
import pytest

from src.churn_pipeline.features.engineering import FeatureEngineer

# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_df():
    """Minimal DataFrame to test feature engineering."""
    return pd.DataFrame({
        "tenure": [1, 15, 30, 60],
        "MonthlyCharges": [20.0, 50.0, 80.0, 100.0],
        "TotalCharges": [20.0, 750.0, 2400.0, 6000.0],
        "PhoneService": ["Yes", "No", "Yes", "Yes"],
        "OnlineSecurity": ["No", "Yes", "No", "Yes"],
        "OnlineBackup": ["No", "No", "Yes", "Yes"],
        "DeviceProtection": ["No", "No", "No", "Yes"],
        "TechSupport": ["No", "No", "No", "Yes"],
        "StreamingTV": ["No", "Yes", "No", "Yes"],
        "StreamingMovies": ["No", "No", "Yes", "Yes"],
        "InternetService": ["DSL", "DSL", "Fiber optic", "Fiber optic"],
    })


# ── tenure_group tests ────────────────────────────────────────────────────────

def test_tenure_group_is_created(sample_df):
    """tenure_group column should be created."""
    fe = FeatureEngineer()
    result = fe.transform(sample_df)
    assert "tenure_group" in result.columns


def test_tenure_group_new_customer(sample_df):
    """Customer with tenure=1 should be in 0-12m group."""
    fe = FeatureEngineer()
    result = fe.transform(sample_df)
    assert result.loc[0, "tenure_group"] == "0-12m"


def test_tenure_group_loyal_customer(sample_df):
    """Customer with tenure=60 should be in 49-72m group."""
    fe = FeatureEngineer()
    result = fe.transform(sample_df)
    assert result.loc[3, "tenure_group"] == "49-72m"


# ── services_count tests ──────────────────────────────────────────────────────

def test_services_count_is_created(sample_df):
    """services_count column should be created."""
    fe = FeatureEngineer()
    result = fe.transform(sample_df)
    assert "services_count" in result.columns


def test_services_count_range(sample_df):
    """services_count should be between 0 and 7."""
    fe = FeatureEngineer()
    result = fe.transform(sample_df)
    assert result["services_count"].between(0, 7).all()


def test_services_count_loyal_customer_has_more(sample_df):
    """Loyal customer (index 3) should have more services than new one (index 0)."""
    fe = FeatureEngineer()
    result = fe.transform(sample_df)
    assert result.loc[3, "services_count"] > result.loc[0, "services_count"]


# ── is_high_value tests ───────────────────────────────────────────────────────

def test_is_high_value_is_created(sample_df):
    """is_high_value column should be created."""
    fe = FeatureEngineer()
    result = fe.transform(sample_df)
    assert "is_high_value" in result.columns


def test_is_high_value_is_binary(sample_df):
    """is_high_value should only contain 0 and 1."""
    fe = FeatureEngineer()
    result = fe.transform(sample_df)
    assert result["is_high_value"].isin([0, 1]).all()


def test_is_high_value_high_charge(sample_df):
    """Customer with highest charge should be high value."""
    fe = FeatureEngineer()
    result = fe.transform(sample_df)
    assert result.loc[3, "is_high_value"] == 1


# ── has_no_support tests ──────────────────────────────────────────────────────

def test_has_no_support_is_created(sample_df):
    """has_no_support column should be created."""
    fe = FeatureEngineer()
    result = fe.transform(sample_df)
    assert "has_no_support" in result.columns


def test_has_no_support_is_binary(sample_df):
    """has_no_support should only contain 0 and 1."""
    fe = FeatureEngineer()
    result = fe.transform(sample_df)
    assert result["has_no_support"].isin([0, 1]).all()


def test_has_no_support_vulnerable_customer(sample_df):
    """Customer with no security and no support should be flagged."""
    fe = FeatureEngineer()
    result = fe.transform(sample_df)
    assert result.loc[0, "has_no_support"] == 1


# ── is_fiber_high_charge tests ────────────────────────────────────────────────

def test_is_fiber_high_charge_is_created(sample_df):
    """is_fiber_high_charge column should be created."""
    fe = FeatureEngineer()
    result = fe.transform(sample_df)
    assert "is_fiber_high_charge" in result.columns


def test_is_fiber_high_charge_is_binary(sample_df):
    """is_fiber_high_charge should only contain 0 and 1."""
    fe = FeatureEngineer()
    result = fe.transform(sample_df)
    assert result["is_fiber_high_charge"].isin([0, 1]).all()


def test_is_fiber_high_charge_correct_flag(sample_df):
    """Customer with Fiber optic and high charge should be flagged."""
    fe = FeatureEngineer()
    result = fe.transform(sample_df)
    assert result.loc[3, "is_fiber_high_charge"] == 1


# ── General tests ─────────────────────────────────────────────────────────────

def test_transform_does_not_modify_original(sample_df):
    """transform should not modify the original DataFrame."""
    fe = FeatureEngineer()
    original_cols = sample_df.columns.tolist()
    fe.transform(sample_df)
    assert sample_df.columns.tolist() == original_cols


def test_transform_returns_dataframe(sample_df):
    """transform should return a DataFrame."""
    fe = FeatureEngineer()
    result = fe.transform(sample_df)
    assert isinstance(result, pd.DataFrame)