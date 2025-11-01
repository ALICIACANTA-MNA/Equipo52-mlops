
"""
Legacy FastAPI service - Simplified version for basic testing.
Para la implementación completa, usar src/api/serve.py
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import time
from datetime import datetime

app = FastAPI(title="Obesity Prediction API - Legacy")

class PatientBasic(BaseModel):
    """Modelo básico para datos del paciente"""
    Age: float = Field(..., ge=14, le=80, description="Edad en años")
    Height: float = Field(..., ge=1.40, le=2.10, description="Altura en metros") 
    Weight: float = Field(..., ge=35, le=200, description="Peso en kilogramos")
    Gender: str = Field(default="Male", regex="^(Male|Female)$")

@app.get("/")
def root():
    return {
        "message": "Obesity Prediction API - Legacy Version",
        "note": "Para funcionalidad completa usar src/api/serve.py",
        "version": "legacy-1.0"
    }

@app.get("/health")
def health():
    return {
        "status": "ok",
        "version": "legacy",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.post("/predict")
def predict(patient: PatientBasic):
    """Predicción básica basada en BMI"""
    try:
        # Calcular BMI
        bmi = patient.Weight / (patient.Height ** 2)
        
        # Clasificación simple por BMI
        if bmi < 18.5:
            prediction = "Insufficient_Weight"
        elif bmi < 25:
            prediction = "Normal_Weight"
        elif bmi < 30:
            prediction = "Overweight_Level_I"
        elif bmi < 35:
            prediction = "Overweight_Level_II"
        else:
            prediction = "Obesity_Type_I"
        
        return {
            "prediction": prediction,
            "bmi": round(bmi, 2),
            "note": "Predicción básica. Para análisis completo usar API principal."
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
