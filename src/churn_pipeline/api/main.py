"""
FastAPI application for the Churn Prediction Pipeline.

Endpoints:
- GET  /health   — health check
- POST /predict  — predict churn for one customer
"""
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
import joblib
import pandas as pd
from pathlib import Path
import sys
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.churn_pipeline.features.engineering import FeatureEngineer
from src.churn_pipeline.api.schemas import (
    CustomerFeatures, PredictionResponse, HealthResponse
)

logger = logging.getLogger(__name__)

# ── Model loading ─────────────────────────────────────────────────────────────

MODEL_PATH = Path(__file__).resolve().parents[3] / "models" / "xgboost_pipeline.joblib"
THRESHOLD = 0.50

model_store = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, release on shutdown."""
    logger.info(f"Loading model from {MODEL_PATH}")
    try:
        model_store["pipeline"] = joblib.load(MODEL_PATH)
        model_store["feature_engineer"] = FeatureEngineer()
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        model_store["pipeline"] = None
    yield
    model_store.clear()


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Churn Prediction API",
    description="Predicts customer churn probability for a telco company.",
    version="1.0.0",
    lifespan=lifespan
)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
def health_check():
    """Check API and model status."""
    return HealthResponse(
        status="ok",
        model_loaded=model_store.get("pipeline") is not None,
        version="1.0.0"
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerFeatures):
    """
    Predict churn probability for a single customer.

    Returns churn probability, binary prediction and risk level.
    """
    if model_store.get("pipeline") is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Convert input to DataFrame
        input_data = pd.DataFrame([customer.model_dump()])

        # Apply binary encoding (Yes/No → 1/0)
        from src.churn_pipeline.data.preprocessing import BinaryEncoder, BINARY_COLS
        binary_encoder = BinaryEncoder(columns=BINARY_COLS)
        input_data = binary_encoder.transform(input_data)

        # Apply feature engineering
        fe = model_store["feature_engineer"]
        input_engineered = fe.transform(input_data)

        # Predict
        pipeline = model_store["pipeline"]
        churn_prob = pipeline.predict_proba(input_engineered)[:, 1][0]
        churn_pred = bool(churn_prob >= THRESHOLD)

        # Risk level
        if churn_prob < 0.3:
            risk_level = "LOW"
        elif churn_prob < 0.6:
            risk_level = "MEDIUM"
        else:
            risk_level = "HIGH"

        return PredictionResponse(
            churn_probability=round(float(churn_prob), 4),
            churn_prediction=churn_pred,
            risk_level=risk_level,
            threshold_used=THRESHOLD
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def root():
    return {
        "message": "Churn Prediction API",
        "docs": "/docs",
        "health": "/health"
    }