"""
Pydantic schemas for the Churn Prediction API.
"""
from pydantic import BaseModel, Field


class CustomerFeatures(BaseModel):
    """Input schema — features required to predict churn for one customer."""

    # Demographics
    SeniorCitizen: int = Field(..., ge=0, le=1, description="1 if senior citizen, 0 otherwise")
    Partner: str = Field(..., description="Yes or No")
    Dependents: str = Field(..., description="Yes or No")

    # Account
    tenure: int = Field(..., ge=0, description="Months as a customer")
    Contract: str = Field(..., description="Month-to-month, One year, or Two year")
    PaperlessBilling: str = Field(..., description="Yes or No")
    PaymentMethod: str = Field(..., description="Payment method")

    # Services
    PhoneService: str = Field(..., description="Yes or No")
    MultipleLines: str = Field(..., description="Yes, No, or No phone service")
    InternetService: str = Field(..., description="DSL, Fiber optic, or No")
    OnlineSecurity: str = Field(..., description="Yes, No, or No internet service")
    OnlineBackup: str = Field(..., description="Yes, No, or No internet service")
    DeviceProtection: str = Field(..., description="Yes, No, or No internet service")
    TechSupport: str = Field(..., description="Yes, No, or No internet service")
    StreamingTV: str = Field(..., description="Yes, No, or No internet service")
    StreamingMovies: str = Field(..., description="Yes, No, or No internet service")

    # Billing
    MonthlyCharges: float = Field(..., ge=0, description="Monthly charge in dollars")
    TotalCharges: float = Field(..., ge=0, description="Total charges in dollars")

    class Config:
        json_schema_extra = {
            "example": {
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 1,
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "PhoneService": "No",
                "MultipleLines": "No phone service",
                "InternetService": "DSL",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "MonthlyCharges": 29.85,
                "TotalCharges": 29.85
            }
        }


class PredictionResponse(BaseModel):
    """Output schema — churn prediction result."""
    churn_probability: float = Field(..., description="Probability of churning (0-1)")
    churn_prediction: bool = Field(..., description="True if predicted to churn")
    risk_level: str = Field(..., description="LOW, MEDIUM, or HIGH")
    threshold_used: float = Field(..., description="Decision threshold")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    version: str = "1.0.0"