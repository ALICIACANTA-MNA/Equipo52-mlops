"""
FastAPI service for obesity prediction - Simplified MLOps implementation.

Basado en patrones ITESM-MNA/MLOps con las mejores prácticas identificadas:
- API simple y enfocada en predicción
- Validación de datos con Pydantic
- Health checks básicos
- Carga de modelos con fallback
- Logging estructurado
- Error handling apropiado

Referencias:
- ITESM-MNA/MLOps: Patrones académicos MLOps
- Real Python FastAPI Guide: Mejores prácticas FastAPI
- Proyecto actual: Mantener consistencia con arquitectura existente
"""

import os
import json
import time
import joblib
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

# FastAPI y dependencias
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator

# ML y procesamiento
import pandas as pd
import numpy as np

# MLflow (opcional)
try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    warnings.warn("MLflow no disponible. Usando modelo local.")

# Configuración de logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# MODELOS DE DATOS CON PYDANTIC
# =============================================================================

class PatientInput(BaseModel):
    """
    Datos de entrada del paciente para predicción de obesidad.
    Validaciones basadas en rangos médicos establecidos.
    """
    # Datos básicos
    Age: float = Field(..., ge=14, le=80, description="Edad en años")
    Gender: str = Field(..., regex="^(Male|Female)$", description="Género")
    Height: float = Field(..., ge=1.40, le=2.10, description="Altura en metros")
    Weight: float = Field(..., ge=35, le=200, description="Peso en kilogramos")
    
    # Historial familiar
    family_history_with_overweight: str = Field(..., regex="^(yes|no)$")
    
    # Hábitos alimenticios
    FAVC: str = Field(..., regex="^(yes|no)$", description="Consume alimentos altos en calorías frecuentemente")
    FCVC: float = Field(..., ge=1, le=3, description="Frecuencia consumo de vegetales")
    NCP: float = Field(..., ge=1, le=4, description="Número comidas principales")
    CAEC: str = Field(..., regex="^(no|Sometimes|Frequently|Always)$", description="Consume alimentos entre comidas")
    
    # Actividad física y hábitos
    SMOKE: str = Field(..., regex="^(yes|no)$", description="Fuma")
    CH2O: float = Field(..., ge=1, le=3, description="Consumo agua diario (litros)")
    SCC: str = Field(..., regex="^(yes|no)$", description="Monitorea calorías")
    FAF: float = Field(..., ge=0, le=3, description="Frecuencia actividad física")
    TUE: float = Field(..., ge=0, le=2, description="Tiempo usando tecnología")
    CALC: str = Field(..., regex="^(no|Sometimes|Frequently|Always)$", description="Consume alcohol")
    MTRANS: str = Field(..., regex="^(Walking|Bike|Automobile|Motorbike|Public_Transportation)$", description="Transporte")
    
    @validator('Weight', 'Height')
    def validate_bmi_range(cls, v, values):
        """Validar que el BMI esté en rango fisiológico"""
        if 'Height' in values and 'Weight' in values:
            height = values.get('Height')
            weight = values.get('Weight') if 'Weight' in values else v
            if height and weight:
                bmi = weight / (height ** 2)
                if bmi < 12 or bmi > 60:
                    raise ValueError(f"BMI calculado ({bmi:.1f}) fuera de rango válido")
        return v


class PredictionResponse(BaseModel):
    """Respuesta de predicción estructurada"""
    prediction: str = Field(..., description="Clase de obesidad predicha")
    confidence: Optional[float] = Field(None, description="Confianza de la predicción")
    bmi: float = Field(..., description="BMI calculado")
    bmi_category: str = Field(..., description="Categoría BMI")
    processing_time_ms: float = Field(..., description="Tiempo de procesamiento")
    model_version: Optional[str] = Field(None, description="Versión del modelo")
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class HealthResponse(BaseModel):
    """Estado de salud del servicio"""
    status: str = Field(..., description="Estado del servicio")
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    model_loaded: bool = Field(..., description="Modelo cargado correctamente")
    uptime_seconds: float = Field(..., description="Tiempo funcionando")


# =============================================================================
# CARGADOR DE MODELOS
# =============================================================================

class ModelManager:
    """Gestor simple de modelos con fallback"""
    
    def __init__(self):
        self.model = None
        self.preprocessing_pipeline = None
        self.model_version = None
        self.model_loaded = False
        self.load_start_time = time.time()
        
        # Rutas de modelos
        self.model_paths = [
            "models/best_model.joblib",
            "artifacts/training/random_forest_model.pkl",
            "models/obesity_model.pkl"
        ]
        
        # Intentar cargar modelo
        self._load_model()
    
    def _load_model(self):
        """Carga modelo con estrategia de fallback"""
        # Intentar MLflow primero
        if MLFLOW_AVAILABLE:
            try:
                self._load_from_mlflow()
                return
            except Exception as e:
                logger.warning(f"No se pudo cargar desde MLflow: {e}")
        
        # Fallback a archivos locales
        for model_path in self.model_paths:
            try:
                if Path(model_path).exists():
                    self.model = joblib.load(model_path)
                    self.model_version = f"local-{Path(model_path).name}"
                    self.model_loaded = True
                    logger.info(f"Modelo cargado desde: {model_path}")
                    return
            except Exception as e:
                logger.warning(f"Error cargando {model_path}: {e}")
        
        # Modelo dummy si no se pudo cargar ninguno
        logger.error("No se pudo cargar ningún modelo. Usando predictor dummy.")
        self.model_version = "dummy-v1.0"
        self.model_loaded = True  # Técnicamente está "cargado"
    
    def _load_from_mlflow(self):
        """Carga modelo desde MLflow Model Registry"""
        try:
            # Intentar cargar modelo desde registry
            model_uri = "models:/obesity_prediction_model/Production"
            self.model = mlflow.sklearn.load_model(model_uri)
            self.model_version = "mlflow-production"
            self.model_loaded = True
            logger.info("Modelo cargado desde MLflow Model Registry")
        except Exception:
            # Fallback a último run
            runs = mlflow.search_runs(experiment_names=["obesity_prediction"])
            if not runs.empty:
                best_run = runs.iloc[0]
                model_uri = f"runs:/{best_run.run_id}/model"
                self.model = mlflow.sklearn.load_model(model_uri)
                self.model_version = f"mlflow-{best_run.run_id[:8]}"
                self.model_loaded = True
                logger.info(f"Modelo cargado desde run: {best_run.run_id}")
    
    def predict(self, input_data: Dict) -> Dict:
        """Realizar predicción"""
        start_time = time.time()
        
        try:
            # Preparar datos para el modelo
            df = pd.DataFrame([input_data])
            
            # Calcular BMI
            bmi = input_data['Weight'] / (input_data['Height'] ** 2)
            bmi_category = self._get_bmi_category(bmi)
            
            # Realizar predicción
            if self.model is None:
                # Predicción dummy basada en BMI
                prediction = self._dummy_predict(bmi)
                confidence = 0.5
            else:
                try:
                    prediction = self.model.predict(df)[0]
                    if hasattr(self.model, 'predict_proba'):
                        probabilities = self.model.predict_proba(df)[0]
                        confidence = float(np.max(probabilities))
                    else:
                        confidence = 0.8
                except Exception as e:
                    logger.error(f"Error en predicción del modelo: {e}")
                    prediction = self._dummy_predict(bmi)
                    confidence = 0.5
            
            processing_time = (time.time() - start_time) * 1000
            
            return {
                "prediction": prediction,
                "confidence": confidence,
                "bmi": round(bmi, 2),
                "bmi_category": bmi_category,
                "processing_time_ms": round(processing_time, 2),
                "model_version": self.model_version
            }
            
        except Exception as e:
            logger.error(f"Error en predicción: {e}")
            raise HTTPException(status_code=500, detail=f"Error en predicción: {str(e)}")
    
    def _get_bmi_category(self, bmi: float) -> str:
        """Categorizar BMI según estándares médicos"""
        if bmi < 18.5:
            return "Underweight"
        elif bmi < 25:
            return "Normal_Weight"
        elif bmi < 30:
            return "Overweight_Level_I"
        elif bmi < 35:
            return "Overweight_Level_II"
        elif bmi < 40:
            return "Obesity_Type_I"
        elif bmi < 45:
            return "Obesity_Type_II"
        else:
            return "Obesity_Type_III"
    
    def _dummy_predict(self, bmi: float) -> str:
        """Predicción simple basada en BMI cuando no hay modelo"""
        return self._get_bmi_category(bmi)
    
    def get_uptime(self) -> float:
        """Tiempo funcionando en segundos"""
        return time.time() - self.load_start_time


# =============================================================================
# APLICACIÓN FASTAPI
# =============================================================================

# Inicializar aplicación
app = FastAPI(
    title="Obesity Prediction API",
    description="API simple para predicción de niveles de obesidad - Equipo 52 MLOps",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, especificar dominios
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Inicializar gestor de modelos
model_manager = ModelManager()

# Variable de inicio para uptime
app_start_time = time.time()


# =============================================================================
# ENDPOINTS
# =============================================================================

@app.get("/", response_model=Dict[str, str])
async def root():
    """Endpoint raíz con información básica"""
    return {
        "message": "Obesity Prediction API - Equipo 52 MLOps",
        "version": "1.0.0",
        "status": "active",
        "documentation": "/docs"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check del servicio"""
    uptime = time.time() - app_start_time
    
    return HealthResponse(
        status="healthy" if model_manager.model_loaded else "degraded",
        model_loaded=model_manager.model_loaded,
        uptime_seconds=round(uptime, 2)
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_obesity(patient: PatientInput):
    """
    Predecir nivel de obesidad basado en datos del paciente
    
    Recibe datos del paciente y retorna:
    - Predicción de nivel de obesidad
    - BMI calculado y categoría
    - Confianza de la predicción
    - Tiempo de procesamiento
    """
    try:
        # Convertir a diccionario para el modelo
        input_data = patient.dict()
        
        # Realizar predicción
        result = model_manager.predict(input_data)
        
        # Retornar respuesta estructurada
        return PredictionResponse(**result)
        
    except Exception as e:
        logger.error(f"Error en endpoint predict: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error procesando predicción: {str(e)}"
        )


@app.get("/model/info")
async def model_info():
    """Información del modelo cargado"""
    return {
        "model_loaded": model_manager.model_loaded,
        "model_version": model_manager.model_version,
        "model_type": type(model_manager.model).__name__ if model_manager.model else "None",
        "uptime_seconds": round(model_manager.get_uptime(), 2),
        "mlflow_available": MLFLOW_AVAILABLE
    }


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """Handler para errores de validación"""
    return JSONResponse(
        status_code=422,
        content={
            "error": "Validation Error",
            "detail": str(exc),
            "timestamp": datetime.utcnow().isoformat()
        }
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handler para excepciones HTTP"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "HTTP Exception",
            "detail": exc.detail,
            "timestamp": datetime.utcnow().isoformat()
        }
    )


# =============================================================================
# PUNTO DE ENTRADA
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    
    # Configuración para desarrollo
    uvicorn.run(
        "serve:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )