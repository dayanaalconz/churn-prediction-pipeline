# Churn Prediction Pipeline

> End-to-end machine learning pipeline to predict customer churn for a telco company — from raw data to a production-ready REST API.

![CI](https://github.com/dayanaalconz/churn-prediction-pipeline/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11-blue)
![Coverage](https://img.shields.io/badge/coverage-87%25-brightgreen)

---

## Business Problem

A telecommunications company needs to identify customers at risk of churning **30 days before it happens**, so the retention team can act proactively. Predicting churn in advance can significantly reduce the cost of acquiring new customers.

**Model results:**
- ROC-AUC: **0.84** on held-out test set
- Recall: **78%** — catches 4 out of 5 churners
- Business impact: up to **€133,600 net savings** per 1,409 customers contacted

---

## Architecture
raw data → feature engineering → preprocessing → model training → REST API
↓ ↓ ↓ ↓ ↓
loader.py engineering.py preprocessing.py train.py main.py
MLflow FastAPI


---

## Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| ML | scikit-learn, XGBoost |
| Experiment tracking | MLflow |
| API | FastAPI + Uvicorn |
| Testing | pytest + pytest-cov |
| Linting | ruff + black |
| CI/CD | GitHub Actions |
| Containerisation | Docker |

---

## Project Structure

churn-prediction-pipeline/
├── .github/workflows/ # CI pipeline
├── data/
│ ├── raw/ # raw data (gitignored)
│ └── processed/ # EDA plots and processed files
├── models/ # serialised model (gitignored)
├── notebooks/
│ ├── 01_eda.ipynb # exploratory data analysis
│ └── 02_feature_engineering.ipynb # feature engineering walkthrough
├── src/churn_pipeline/
│ ├── api/ # FastAPI app
│ ├── data/ # data loading and preprocessing
│ ├── features/ # feature engineering
│ └── models/ # training and evaluation
├── tests/ # unit tests
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
└── requirements.txt


---

## Quickstart

### 1. Clone and set up environment

```bash
git clone https://github.com/dayanaalconz/churn-prediction-pipeline.git
cd churn-prediction-pipeline
python -m venv .venv
source .venv/Scripts/activate  # Windows
pip install -r requirements.txt
```

### 2. Train the model

```bash
python -m src.churn_pipeline.models.train
```

### 3. Run the API

```bash
uvicorn src.churn_pipeline.api.main:app --reload --port 8000
```

### 4. Run with Docker

```bash
docker-compose up --build
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Health check and model status |
| POST | `/predict` | Predict churn for a single customer |
| GET | `/docs` | Interactive API documentation |

### Example request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 1,
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "MonthlyCharges": 70.70,
    "TotalCharges": 70.70
  }'
```

### Example response

```json
{
  "churn_probability": 0.9219,
  "churn_prediction": true,
  "risk_level": "HIGH",
  "threshold_used": 0.5
}
```

---

## Model Performance

| Model | ROC-AUC | F1 | Recall | CV AUC |
|---|---|---|---|---|
| Logistic Regression (baseline) | 0.8425 | 0.6167 | 0.7914 | 0.8464 |
| XGBoost | 0.8401 | 0.6266 | 0.7807 | 0.8439 |

**Key insight:** The baseline Logistic Regression performs comparably to XGBoost — a sign that the feature engineering added significant signal and the problem is well-represented by linear relationships.

---

## Feature Engineering

Five new features created from EDA insights:

| Feature | Logic | Churn rate impact |
|---|---|---|
| `tenure_group` | Segment tenure into 4 bands | 47.4% (new) vs 9.5% (loyal) |
| `services_count` | Total contracted services | 43.8% (0 services) vs 5.8% (7 services) |
| `is_high_value` | MonthlyCharges > median | +17.3pp |
| `has_no_support` | No security AND no tech support | +16.3pp |
| `is_fiber_high_charge` | Fiber optic AND high charge | +24.8pp |

---

## Tests

```bash
pytest tests/ -v --cov=src
```

**Coverage: 87%** across tested modules (preprocessing: 82%, feature engineering: 96%).

---

## Dataset

IBM Telco Customer Churn — 7,043 customers, 21 variables.  
Source: [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)