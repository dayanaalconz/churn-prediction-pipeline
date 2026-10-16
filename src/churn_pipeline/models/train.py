"""
Model training with MLflow tracking for the Churn Prediction Pipeline.

Models trained:
- Baseline: Logistic Regression
- Final: XGBoost with hyperparameter tuning
"""
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score,
    recall_score, classification_report
)
from xgboost import XGBClassifier
import logging

from src.churn_pipeline.data.loader import load_raw_data
from src.churn_pipeline.data.preprocessing import prepare_data, build_preprocessor
from src.churn_pipeline.features.engineering import FeatureEngineer

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

RANDOM_STATE = 42
TEST_SIZE = 0.2
MLFLOW_EXPERIMENT = "churn-prediction"
MODELS_DIR = Path("models")


# ── Data preparation ──────────────────────────────────────────────────────────

def load_and_prepare(data_path: str | Path) -> tuple:
    """
    Full data preparation pipeline: load → feature engineering → preprocessing.

    Returns:
        X_train, X_test, y_train, y_test
    """
    # Load
    df = load_raw_data(data_path)

    # Feature engineering
    fe = FeatureEngineer()
    df = fe.transform(df)

    # Preprocessing
    X, y = prepare_data(df)

    # Train/test split — stratified to maintain class balance
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    logger.info(f"Train: {X_train.shape} | Test: {X_test.shape}")
    logger.info(f"Train churn rate: {y_train.mean():.2%} | Test churn rate: {y_test.mean():.2%}")

    return X_train, X_test, y_train, y_test


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred, y_prob) -> dict:
    """Compute all evaluation metrics."""
    return {
        "roc_auc": roc_auc_score(y_true, y_prob),
        "f1": f1_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
    }


# ── Training functions ────────────────────────────────────────────────────────

def train_baseline(X_train, X_test, y_train, y_test) -> dict:
    """
    Train Logistic Regression baseline with MLflow tracking.
    """
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name="logistic_regression_baseline"):
        # Log parameters
        params = {
            "model": "LogisticRegression",
            "C": 1.0,
            "max_iter": 1000,
            "class_weight": "balanced",
            "random_state": RANDOM_STATE,
        }
        mlflow.log_params(params)

        # Build pipeline
        preprocessor = build_preprocessor()
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", LogisticRegression(
                C=params["C"],
                max_iter=params["max_iter"],
                class_weight=params["class_weight"],
                random_state=params["random_state"]
            ))
        ])

        # Train
        pipeline.fit(X_train, y_train)

        # Predict
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        # Metrics
        metrics = compute_metrics(y_test, y_pred, y_prob)
        mlflow.log_metrics(metrics)

        # Cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        cv_auc = cross_val_score(pipeline, X_train, y_train, 
                                  cv=cv, scoring="roc_auc").mean()
        mlflow.log_metric("cv_roc_auc", cv_auc)

        # Log model
        mlflow.sklearn.log_model(
        pipeline, 
        "model",
        skops_trusted_types=["numpy.dtype"]
        )

        logger.info(f"Baseline — ROC-AUC: {metrics['roc_auc']:.4f} | F1: {metrics['f1']:.4f}")
        print(f"\nBaseline Logistic Regression:")
        print(f"  ROC-AUC:  {metrics['roc_auc']:.4f}")
        print(f"  F1:       {metrics['f1']:.4f}")
        print(f"  Precision:{metrics['precision']:.4f}")
        print(f"  Recall:   {metrics['recall']:.4f}")
        print(f"  CV AUC:   {cv_auc:.4f}")

        return metrics


def train_xgboost(X_train, X_test, y_train, y_test) -> dict:
    """
    Train XGBoost with MLflow tracking.
    """
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name="xgboost_tuned"):

        # Parameters — tuned based on dataset size and class imbalance
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        params = {
            "model": "XGBoost",
            "n_estimators": 300,
            "max_depth": 4,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "scale_pos_weight": round(scale_pos_weight, 2),
            "random_state": RANDOM_STATE,
            "eval_metric": "auc",
        }
        mlflow.log_params(params)

        # Build pipeline
        preprocessor = build_preprocessor()
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", XGBClassifier(
                n_estimators=params["n_estimators"],
                max_depth=params["max_depth"],
                learning_rate=params["learning_rate"],
                subsample=params["subsample"],
                colsample_bytree=params["colsample_bytree"],
                scale_pos_weight=params["scale_pos_weight"],
                random_state=params["random_state"],
                eval_metric=params["eval_metric"],
                verbosity=0,
            ))
        ])

        # Train
        pipeline.fit(X_train, y_train)

        # Predict
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        # Metrics
        metrics = compute_metrics(y_test, y_pred, y_prob)
        mlflow.log_metrics(metrics)

        # Cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        cv_auc = cross_val_score(pipeline, X_train, y_train,
                                  cv=cv, scoring="roc_auc").mean()
        mlflow.log_metric("cv_roc_auc", cv_auc)

        # Log model
        import joblib
        joblib.dump(pipeline, MODELS_DIR / "xgboost_pipeline.joblib")
        mlflow.log_artifact(str(MODELS_DIR / "xgboost_pipeline.joblib"))

        # Save pipeline locally
        MODELS_DIR.mkdir(exist_ok=True)
        import joblib
        joblib.dump(pipeline, MODELS_DIR / "xgboost_pipeline.joblib")
        mlflow.log_artifact(str(MODELS_DIR / "xgboost_pipeline.joblib"))

        logger.info(f"XGBoost — ROC-AUC: {metrics['roc_auc']:.4f} | F1: {metrics['f1']:.4f}")
        print(f"\nXGBoost:")
        print(f"  ROC-AUC:  {metrics['roc_auc']:.4f}")
        print(f"  F1:       {metrics['f1']:.4f}")
        print(f"  Precision:{metrics['precision']:.4f}")
        print(f"  Recall:   {metrics['recall']:.4f}")
        print(f"  CV AUC:   {cv_auc:.4f}")
        print(f"\n{classification_report(y_test, y_pred, target_names=['No Churn', 'Churn'])}")

        return metrics, pipeline


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    DATA_PATH = Path("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")

    print("Loading and preparing data...")
    X_train, X_test, y_train, y_test = load_and_prepare(DATA_PATH)

    print("\nTraining baseline model...")
    baseline_metrics = train_baseline(X_train, X_test, y_train, y_test)

    print("\nTraining XGBoost model...")
    xgb_metrics, pipeline = train_xgboost(X_train, X_test, y_train, y_test)

    print("\nTraining complete. Run 'mlflow ui' to see experiments.")
