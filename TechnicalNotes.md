# Technical Notes — Churn Prediction Pipeline

> A detailed walkthrough of every module, design decision, and technical concept in this project.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Data Pipeline](#2-data-pipeline)
3. [Feature Engineering](#3-feature-engineering)
4. [Preprocessing Pipeline](#4-preprocessing-pipeline)
5. [Model Training](#5-model-training)
6. [Model Evaluation](#6-model-evaluation)
7. [REST API](#7-rest-api)
8. [Testing Strategy](#8-testing-strategy)
9. [CI/CD Pipeline](#9-cicd-pipeline)
10. [Key Design Decisions](#10-key-design-decisions)

---

## 1. Project Overview

### Problem Statement
A telecommunications company loses revenue every time a customer churns. The cost of acquiring a new customer is significantly higher than retaining an existing one. This pipeline predicts which customers are likely to churn in the next 30 days, giving the retention team time to act.

### Approach
Binary classification problem: `Churn = Yes/No`.

The pipeline follows a production-first approach from day one:
- Modular code in `src/` — not just notebooks
- Every transformation is encapsulated in sklearn-compatible classes
- MLflow tracks all experiments
- FastAPI serves predictions via REST
- GitHub Actions runs tests on every push

### Dataset
IBM Telco Customer Churn — 7,043 customers, 21 variables covering demographics, contracted services, billing and churn status.

**Class distribution:** 73.5% No Churn / 26.5% Churn — moderate imbalance handled via `scale_pos_weight` in XGBoost and `class_weight='balanced'` in Logistic Regression.

---

## 2. Data Pipeline

### `src/churn_pipeline/data/loader.py`

**Purpose:** Load raw CSV and perform basic validation before any transformation.

**Key function: `load_raw_data(path)`**
```python
df = load_raw_data("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
```

**What it does:**
1. Checks the file exists — raises `FileNotFoundError` if not
2. Loads the CSV with pandas
3. Validates all expected columns are present — raises `ValueError` if any are missing
4. Converts `TotalCharges` from `str` to `float` using `pd.to_numeric(errors='coerce')`

**Why `TotalCharges` comes as string:**
Customers with `tenure=0` have a blank space `" "` instead of a number. Pandas reads it as string. `errors='coerce'` converts blanks to `NaN`, which we then impute with `0` (these are new customers with no accumulated charges).

**Design decision:** Validation happens at load time, not later. If data is malformed, we fail fast before any expensive computation.

---

## 3. Feature Engineering

### `src/churn_pipeline/features/engineering.py`

**Purpose:** Create new features from raw variables based on EDA insights.

**Class: `FeatureEngineer`**

Inherits from `BaseEstimator` and `TransformerMixin` — this means it behaves like any sklearn transformer and can be inserted into a `Pipeline`.

```python
fe = FeatureEngineer()
df_engineered = fe.transform(df_raw)
```

**Five new features:**

#### `tenure_group`
```python
pd.cut(X["tenure"], bins=[0, 12, 24, 48, 72], 
       labels=["0-12m", "13-24m", "25-48m", "49-72m"])
```
Segments tenure into business bands. EDA showed churn rate drops from 47.4% (0-12m) to 9.5% (49-72m) — a non-linear relationship that raw tenure doesn't capture as cleanly.

#### `services_count`
```python
X[service_cols].sum(axis=1)
```
Total contracted services. More services = more integrated = harder to leave. Churn drops from 43.8% (0 services) to 5.8% (7 services).

#### `is_high_value`
```python
(X["MonthlyCharges"] > X["MonthlyCharges"].median()).astype(int)
```
Binary flag: customers paying above median monthly charges. High-value customers have 35.2% churn vs 17.9% — they're more price-sensitive and more likely to find alternatives.

#### `has_no_support`
```python
(no_security & no_support).astype(int)
```
Flags customers with no online security AND no tech support — the most vulnerable segment with no retention anchors. Churn rate: 33.4% vs 17.1%.

#### `is_fiber_high_charge`
```python
(is_fiber & is_expensive).astype(int)
```
Interaction feature: Fiber optic + high monthly charge. The highest-risk combination identified in EDA: 41.0% churn vs 16.2%. A 2.5x multiplier.

**Why custom sklearn transformers?**

Because they can be serialised, chained in pipelines, and applied consistently to new data at prediction time. A plain function would require manual re-application every time.

---

## 4. Preprocessing Pipeline

### `src/churn_pipeline/data/preprocessing.py`

**Purpose:** Transform raw features into model-ready format.

**Key components:**

#### `DropColumns`
Removes `customerID` (identifier, no predictive value) and `gender` (no significant churn rate difference between categories).

#### `BinaryEncoder`
Converts Yes/No string columns to 1/0 integers:
```python
X[col] = (X[col] == "Yes").astype(int)
```
Applied to: `Partner`, `Dependents`, `PhoneService`, `PaperlessBilling`.

#### `build_preprocessor()` — ColumnTransformer
Applies different transformations to different column types simultaneously:

OrdinalEncoder → Contract (natural order: month < one year < two year)
OneHotEncoder → InternetService, PaymentMethod, MultipleLines,
OnlineSecurity, OnlineBackup, DeviceProtection,
TechSupport, StreamingTV, StreamingMovies, tenure_group
StandardScaler → tenure, MonthlyCharges, TotalCharges, SeniorCitizen

SimpleImputer → handles TotalCharges nulls (median strategy)


**Why OrdinalEncoder for Contract?**
Because Month-to-month < One year < Two year has a real order — a two-year contract customer is categorically more committed than a month-to-month customer. OrdinalEncoder preserves this information; OneHot would lose it.

**Why `drop='first'` in OneHotEncoder?**
To avoid multicollinearity (the dummy variable trap). With k categories, we only need k-1 columns to represent all information.

#### `prepare_data(df)`
Orchestrates the full preprocessing sequence:
1. Encode target: `Churn` → `y` (0/1)
2. Drop `customerID`, `gender`, `Churn`
3. Binary encode Yes/No columns
4. Return `(X, y)` ready for modelling

---

## 5. Model Training

### `src/churn_pipeline/models/train.py`

**Purpose:** Train models with full MLflow experiment tracking.

**Data split strategy:**
```python
train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
```
`stratify=y` ensures both train and test sets maintain the 73/27 class ratio — critical for imbalanced datasets.

**Model 1: Logistic Regression (baseline)**
- `class_weight='balanced'` — automatically adjusts weights inversely proportional to class frequencies
- `C=1.0` — default regularisation
- Serves as the performance floor — any more complex model must beat this

**Model 2: XGBoost**
- `scale_pos_weight = negative_count / positive_count` — equivalent to class_weight for XGBoost
- `n_estimators=300`, `learning_rate=0.05` — slower learning, better generalisation
- `subsample=0.8`, `colsample_bytree=0.8` — reduces overfitting via stochasticity

**MLflow tracking:**
Every run logs:
- Parameters: model type, hyperparameters, random state
- Metrics: ROC-AUC, F1, Precision, Recall, CV AUC
- Artifacts: serialised model

**Why ROC-AUC as primary metric?**
Because it measures the model's ability to rank customers by churn probability regardless of threshold. With class imbalance, accuracy is misleading — a model predicting always "No Churn" gets 73.5% accuracy but catches zero churners.

**Cross-validation:**
`StratifiedKFold(n_splits=5)` — gives a more robust estimate of generalisation performance than a single train/test split.

---

## 6. Model Evaluation

### `src/churn_pipeline/models/evaluate.py`

**Purpose:** Translate model performance into business value.

**Plots generated:**

#### ROC Curve
Shows the trade-off between True Positive Rate and False Positive Rate at all thresholds. AUC = 0.84 means the model correctly ranks a churner above a non-churner 84% of the time.

#### Precision-Recall Curve
More informative than ROC for imbalanced datasets. Shows the trade-off between catching churners (recall) and not wasting retention budget on false alarms (precision).

#### Confusion Matrix
Shows absolute counts and percentages of TP, TN, FP, FN at the default threshold (0.5).

#### Threshold Analysis
Plots Precision, Recall and F1 across all thresholds (0.1 to 0.9). The optimal threshold by F1 is 0.50 for this model.

**This is critical for business:** the default 0.5 threshold is rarely optimal. A retention team might prefer higher recall (catch more churners, accept more false alarms) or higher precision (only contact high-confidence churners).

#### Business Impact Analysis

Net savings = (True Positives × churn_cost) - (Contacted × retention_cost)

Assumptions:
- `churn_cost = €500` — revenue lost per churned customer
- `retention_cost = €50` — cost of contacting a customer

**Result:** Optimal threshold 0.15, max net savings **€133,600** per 1,409 test customers.

---

## 7. REST API

### `src/churn_pipeline/api/`

**Framework:** FastAPI — chosen for automatic OpenAPI docs, Pydantic validation, and async support.

**Startup:** Model is loaded once at startup via `lifespan` context manager — not on every request. This is critical for performance.

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    model_store["pipeline"] = joblib.load(MODEL_PATH)
    yield
    model_store.clear()
```

**Endpoints:**

#### `GET /health`
Returns model load status and API version. Used by load balancers and monitoring tools to check service health.

#### `POST /predict`
1. Receives `CustomerFeatures` (Pydantic validates all fields automatically)
2. Converts to DataFrame
3. Applies `BinaryEncoder` (Yes/No → 1/0)
4. Applies `FeatureEngineer` (creates 5 new features)
5. Runs `pipeline.predict_proba()` (preprocessing + model in one step)
6. Returns probability, binary prediction, risk level

**Risk levels:**
- `LOW`: probability < 0.30
- `MEDIUM`: 0.30 ≤ probability < 0.60
- `HIGH`: probability ≥ 0.60

**Pydantic schemas (`schemas.py`):**
Input validation happens automatically — if a required field is missing or has the wrong type, FastAPI returns a 422 error before the prediction code even runs.

---

## 8. Testing Strategy

**Framework:** pytest + pytest-cov

**Coverage: 87%** on tested modules.

### Unit test philosophy
Each test has three parts — Arrange, Act, Assert:

```python
def test_binary_encoder_converts_yes_to_1(sample_df):
    # Arrange
    encoder = BinaryEncoder(columns=["Partner"])
    # Act
    result = encoder.transform(sample_df)
    # Assert
    assert result["Partner"].iloc[0] == 1
```

### What we test
- **`test_preprocessing.py`** — 13 tests covering DropColumns, BinaryEncoder and prepare_data
- **`test_features.py`** — 17 tests covering all 5 engineered features

### What we don't test (yet)
- `train.py` — requires real data and MLflow; integration test territory
- `evaluate.py` — plot generation; difficult to unit test
- `api/main.py` — requires a running server; needs `httpx` + `TestClient`

These will be added in future iterations.

---

## 9. CI/CD Pipeline

### `.github/workflows/ci.yml`

Triggered on every push to `main` and every pull request.

**Steps:**
1. **Checkout** — clone the repo on a fresh Ubuntu runner
2. **Python setup** — install Python 3.11
3. **Dependencies** — install all requirements + `pip install -e .` (editable install so `src` is importable)
4. **Lint** — `ruff check src/ tests/` — enforces import order, unused variables, style
5. **Tests** — `pytest tests/ -v --cov=src` — runs all 30 tests with coverage
6. **Coverage threshold** — fails if coverage drops below 75% on tested modules

**Why editable install (`pip install -e .`)?**
The CI runner doesn't have the project directory in its Python path. Installing the package in editable mode makes `from src.churn_pipeline.xxx import yyy` work correctly on any machine.

**Coverage configuration (`pyproject.toml`):**
Modules not yet covered by tests are excluded from the coverage calculation — honest reporting of what is actually tested.

---

## 10. Key Design Decisions

### Why sklearn-compatible transformers?
Every custom class (`FeatureEngineer`, `BinaryEncoder`, `DropColumns`) inherits from `BaseEstimator` and `TransformerMixin`. This means:
- They can be inserted into sklearn `Pipeline` objects
- They are serialisable with `joblib`
- `fit()` and `transform()` are separate — parameters learned on train data are applied to test data without leakage

### Why separate feature engineering from preprocessing?
Feature engineering creates new columns from domain knowledge. Preprocessing encodes and scales existing columns for the model. Keeping them separate makes each step independently testable and replaceable.

### Why MLflow?
Experiment tracking is non-negotiable in production ML. Without it, you cannot answer "which model is in production?", "what hyperparameters did it use?", or "how does it compare to last week's model?". MLflow provides this out of the box.

### Why FastAPI over Flask?
- Automatic input validation via Pydantic
- Auto-generated interactive docs at `/docs`
- Native async support for high-throughput scenarios
- Type hints throughout — better developer experience

### Why XGBoost + Logistic Regression?
Starting with a simple baseline is a professional practice. If Logistic Regression performs comparably to XGBoost (as it does here — 0.8425 vs 0.8401 ROC-AUC), it's worth considering the simpler model for production: easier to explain, faster to serve, and more robust to distribution shift.

### Threshold optimisation
The default 0.5 threshold is arbitrary. The optimal threshold depends on the business cost of false positives (wasting retention budget) versus false negatives (missing churners). Our business impact analysis shows the optimal threshold is 0.15 — meaning we flag customers as at-risk even when the model is only 15% confident, because the cost of missing a churner (€500) far outweighs the cost of a false alarm (€50).