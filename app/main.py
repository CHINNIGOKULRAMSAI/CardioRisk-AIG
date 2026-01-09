from fastapi import FastAPI
import pandas as pd

from app.schemas import PatientData
from app.utils import (
    load_model,
    load_preprocessor,
    apply_feature_engineering
)

app = FastAPI(
    title="Cardiovascular Disease Prediction API",
    description="ML-powered heart disease risk prediction",
    version="1.0.0"
)

model = load_model()
preprocessor = load_preprocessor()

RISK_THRESHOLD = 0.40


@app.get("/")
def health_check():
    return {"status": "API is running"}


@app.post("/predict")
def predict(data: PatientData):
    # Convert input to DataFrame
    df = pd.DataFrame([data.dict()])

    # Feature engineering
    df = apply_feature_engineering(df)

    # The preprocessor was trained with specific columns including 'id'
    # Add 'id' column if missing to match training data structure
    if 'id' not in df.columns:
        df.insert(0, 'id', 0)
    
    # Reorder columns to match preprocessor's expected order
    # Expected order: id, age, gender, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active, bmi
    expected_columns = ['id', 'age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'bmi']
    df = df[expected_columns]

    # Preprocess
    X = preprocessor.transform(df)

    # Predict probability
    probability = model.predict_proba(X)[0][1]
    prediction = int(probability >= RISK_THRESHOLD)

    result = {
        "prediction": "High Risk (Disease)" if prediction == 1 else "Low Risk (No Disease)",
        "confidence": round(probability * 100, 2),
        "threshold_used": RISK_THRESHOLD,
        "risk_level": (
            "HIGH" if probability >= 0.70
            else "MEDIUM" if probability >= 0.40
            else "LOW"
        )
    }

    return result
