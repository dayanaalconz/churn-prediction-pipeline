"""
Feature engineering for the Churn Prediction Pipeline.

New features based on EDA findings:
- tenure_group: segment customers by seniority
- services_count: total number of contracted services
- is_high_value: high monthly charge flag
- has_no_support: no tech support + no online security flag
- is_fiber_high_charge: Fiber optic + high monthly charge interaction
"""
import logging

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger(__name__)


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Creates new features from raw Telco churn data based on EDA insights.
    All transformations are deterministic — no fitting required.
    """

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        # 1. Tenure group — segment by seniority
        X["tenure_group"] = pd.cut(
            X["tenure"],
            bins=[0, 12, 24, 48, 72],
            labels=["0-12m", "13-24m", "25-48m", "49-72m"],
            include_lowest=True
        ).astype(str)

        # 2. Services count — total contracted services
        service_cols = [
            "PhoneService", "OnlineSecurity", "OnlineBackup",
            "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"
        ]
        for col in service_cols:
            X[col] = (X[col].astype(str) == "Yes").astype(int)

        X["services_count"] = X[service_cols].sum(axis=1)

        # 3. High value customer flag
        monthly_median = X["MonthlyCharges"].median()
        X["is_high_value"] = (X["MonthlyCharges"] > monthly_median).astype(int)

        # 4. No support flag — vulnerable customers
        no_security = (X["OnlineSecurity"] == 0) | (X["OnlineSecurity"] == "No")
        no_support = (X["TechSupport"] == 0) | (X["TechSupport"] == "No")
        X["has_no_support"] = (no_security & no_support).astype(int)

        # 5. Fiber + high charge interaction
        is_fiber = X["InternetService"] == "Fiber optic"
        is_expensive = X["MonthlyCharges"] > monthly_median
        X["is_fiber_high_charge"] = (is_fiber & is_expensive).astype(int)

        logger.info(f"Feature engineering complete — new shape: {X.shape}")
        logger.info("New features: tenure_group, services_count, is_high_value, has_no_support, is_fiber_high_charge")

        return X