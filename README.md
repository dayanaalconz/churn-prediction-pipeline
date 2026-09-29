# Churn Prediction Pipeline

Predictor de churn para telco con pipeline de producción completo.

## Stack
- Python 3.11 · scikit-learn · XGBoost · MLflow · FastAPI · Docker

## Estructura

churn-prediction-pipeline/
├── data/ # datos (raw ignorado en git)
├── notebooks/ # EDA y exploración
├── src/ # código de producción
├── tests/ # tests unitarios
└── models/ # modelos serializados


## Cómo ejecutar
```bash
# Activar entorno
source .venv/Scripts/activate

# Tests
pytest
```