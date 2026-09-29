"""
Preprocessing pipeline for the Churn Prediction dataset.

Decisions based on EDA findings:
- Drop customerID (identifier) and gender (no predictive power)
- Ordinal encoding for Contract (natural order)
- One-hot encoding for remaining categoricals
- TotalCharges already imputed with 0 in loader
- class_weight='balanced' handled at model level
"""
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import logging

logger = logging.getLogger(__name__)

# ── Constantes ────────────────────────────────────────────────────────────────

COLS_TO_DROP = ["customerID", "gender"]

TARGET_COL = "Churn"

ORDINAL_COLS = ["Contract"]
ORDINAL_CATEGORIES = [["Month-to-month", "One year", "Two year"]]

ONEHOT_COLS = [
    "InternetService", "PaymentMethod", "MultipleLines",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies"
]

BINARY_COLS = [
    "Partner", "Dependents", "PhoneService", "PaperlessBilling"
]

NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]


# ── Transformers custom ───────────────────────────────────────────────────────

class BinaryEncoder(BaseEstimator, TransformerMixin):
    """Encodes Yes/No columns as 1/0."""

    def __init__(self, columns: list[str]):
        self.columns = columns

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in self.columns:
            if col in X.columns:
                X[col] = (X[col] == "Yes").astype(int)
        logger.info(f"Binary encoded: {self.columns}")
        return X


class DropColumns(BaseEstimator, TransformerMixin):
    """Drops specified columns."""

    def __init__(self, columns: list[str]):
        self.columns = columns

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        cols_to_drop = [c for c in self.columns if c in X.columns]
        logger.info(f"Dropped columns: {cols_to_drop}")
        return X.drop(columns=cols_to_drop)


class TargetEncoder(BaseEstimator, TransformerMixin):
    """Encodes target column Yes/No as 1/0 and returns series."""

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.Series:
        return (X[TARGET_COL] == "Yes").astype(int)


# ── Pipeline principal ────────────────────────────────────────────────────────

def build_preprocessor() -> ColumnTransformer:
    """
    Build the preprocessing ColumnTransformer.

    Returns:
        Fitted-ready ColumnTransformer with all transformations.
    """
    ordinal_transformer = OrdinalEncoder(
        categories=ORDINAL_CATEGORIES,
        handle_unknown="use_encoded_value",
        unknown_value=-1
    )

    onehot_transformer = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,
        drop="first"
    )

    numeric_transformer = StandardScaler()

    preprocessor = ColumnTransformer(
        transformers=[
            ("ordinal", ordinal_transformer, ORDINAL_COLS),
            ("onehot", onehot_transformer, ONEHOT_COLS),
            ("numeric", numeric_transformer, NUMERIC_COLS),
        ],
        remainder="passthrough"
    )

    return preprocessor


def prepare_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Full preprocessing pipeline: clean, encode, split X and y.

    Args:
        df: Raw DataFrame loaded with loader.load_raw_data()

    Returns:
        Tuple of (X, y) ready for model training.
    """
    logger.info(f"Starting preprocessing — shape: {df.shape}")

    # 1. Encode target
    y = (df[TARGET_COL] == "Yes").astype(int)
    logger.info(f"Target encoded — churn rate: {y.mean():.2%}")

    # 2. Drop target + useless columns
    drop_encoder = DropColumns(columns=COLS_TO_DROP + [TARGET_COL, "Churn_binary"])
    X = drop_encoder.transform(df)

    # 3. Binary encode Yes/No columns
    binary_encoder = BinaryEncoder(columns=BINARY_COLS)
    X = binary_encoder.transform(X)

    logger.info(f"Preprocessing complete — X shape: {X.shape}")
    return X, y
    