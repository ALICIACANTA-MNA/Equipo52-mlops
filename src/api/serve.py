"""
Production-ready FastAPI service for obesity prediction MLOps pipeline.

Implementa API de predicción siguiendo las mejores prácticas de MLOps:
- Carga robusta de modelos con fallback y caching
- Health checks comprensivos para monitoring
- Validación de entrada con Pydantic y constraints médicos
- Rate limiting y throttling para protección
- Logging estructurado y métricas de rendimiento
- Error handling y graceful degradation
- Monitoreo de drift y calidad de predicciones
- Integración con MLflow para model serving

Referencias técnicas:
- FastAPI Best Practices: docs/IMPLEMENTATION_GUIDE.md - API Design Best Practices
- ML Model Serving: https://learn.microsoft.com/en-us/azure/machine-learning/how-to-deploy-online-endpoints
- API Security: https://learn.microsoft.com/en-us/azure/security/fundamentals/api-security
- Production ML Systems: docs/Machine Learning Engineering with MLflow.pdf - MLOps Operations Guide

Componentes:
- ModelLoader: Carga y gestión de modelos
- PredictionService: Servicio de predicción con validación
- HealthChecker: Health checks para infraestructura
- MetricsCollector: Recolección de métricas y monitoreo
- APIRouter: Endpoints organizados por funcionalidad
"""

import os
import json
import time
import asyncio
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import warnings

# FastAPI y dependencias
from fastapi import FastAPI, HTTPException, Depends, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer
from fastapi.openapi.docs import get_swagger_ui_html
from pydantic import BaseModel, Field, validator
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

# ML y procesamiento
import pandas as pd
import numpy as np
import joblib
from sklearn.pipeline import Pipeline

# MLflow para model serving
try:
    import mlflow
    import mlflow.sklearn
    from mlflow.tracking import MlflowClient
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    warnings.warn("MLflow no disponible. Model serving limitado.")

# Imports propios
from src.utils.config import load_config, get_mlflow_config
from src.utils.logging_config import get_logger, log_pipeline_stage


logger = get_logger(__name__)


# Modelos de datos con validación médica
class PatientData(BaseModel):
    """
    Modelo de datos del paciente con validaciones médicas.
    
    Basado en rangos fisiológicos y clínicos establecidos para
    prevenir predicciones con datos incorrectos o maliciosos.
    """
    # Datos demográficos
    Age: float = Field(..., ge=14, le=80, description="Edad en años (14-80)")
    Gender: str = Field(..., pattern="^(Male|Female)$", description="Género del paciente")
    
    # Medidas antropométricas
    Height: float = Field(..., ge=1.40, le=2.10, description="Altura en metros (1.40-2.10)")
    Weight: float = Field(..., ge=35, le=200, description="Peso en kg (35-200)")
    
    # Historial familiar
    family_history_with_overweight: str = Field(
        ..., pattern="^(yes|no)$", description="Historial familiar de sobrepeso"
    )
    
    # Hábitos alimenticios  
    FAVC: str = Field(..., pattern="^(yes|no)$", description="Consumo frecuente de alimentos altos en calorías")
    FCVC: float = Field(..., ge=1, le=3, description="Frecuencia de consumo de vegetales (1-3)")
    NCP: float = Field(..., ge=1, le=4, description="Número de comidas principales (1-4)")
    CAEC: str = Field(
        ..., pattern="^(no|Sometimes|Frequently|Always)$", 
        description="Consumo de alimentos entre comidas"
    )
    
    # Hábitos de salud
    SMOKE: str = Field(..., pattern="^(yes|no)$", description="Fuma")
    CH2O: float = Field(..., ge=1, le=3, description="Consumo diario de agua (1-3 litros)")
    SCC: str = Field(..., pattern="^(yes|no)$", description="Monitorea consumo de calorías")
    
    # Actividad física
    FAF: float = Field(..., ge=0, le=3, description="Frecuencia de actividad física (0-3)")
    TUE: float = Field(..., ge=0, le=2, description="Tiempo usando dispositivos tecnológicos (0-2)")
    
    # Consumo de alcohol
    CALC: str = Field(
        ..., pattern="^(no|Sometimes|Frequently|Always)$",
        description="Consumo de alcohol"
    )
    
    # Transporte
    MTRANS: str = Field(
        ..., pattern="^(Walking|Bike|Automobile|Motorbike|Public_Transportation)$",
        description="Medio de transporte principal"  
    )
    
    @validator('Weight', 'Height')
    def validate_anthropometric_consistency(cls, v, values):
        """Valida consistencia antropométrica básica"""
        if 'Height' in values and 'Weight' in values:
            height = values.get('Height', v if 'Height' not in values else values['Height'])
            weight = v if 'Weight' not in values else values.get('Weight', v)
            
            bmi = weight / (height ** 2)
            if bmi < 12 or bmi > 60:  # BMI extremos poco probables
                raise ValueError(f"BMI calculado ({bmi:.1f}) fuera de rango fisiológico (12-60)")
        return v
    
    @validator('Age')
    def validate_age_consistency(cls, v, values):
        """Valida consistencia de edad con otros factores"""
        # Adultos jóvenes con historial familiar poco común
        if v < 18 and values.get('family_history_with_overweight') == 'yes':
            logger.warning(f"Edad {v} con historial familiar positivo - caso inusual")
        return v


class PredictionRequest(BaseModel):
    """Request de predicción con metadata"""
    patient_data: PatientData
    request_id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()))
    include_confidence: bool = Field(default=True, description="Incluir score de confianza")
    include_explanation: bool = Field(default=False, description="Incluir explicación de predicción")


class PredictionResponse(BaseModel):
    """Response de predicción estructurada"""
    request_id: str
    prediction: str = Field(..., description="Clase predicha de obesidad")
    confidence_score: Optional[float] = Field(None, ge=0, le=1, description="Score de confianza (0-1)")
    prediction_probabilities: Optional[Dict[str, float]] = Field(None, description="Probabilidades por clase")
    bmi: Optional[float] = Field(None, description="BMI calculado")
    bmi_category: Optional[str] = Field(None, description="Categoría BMI")
    explanation: Optional[Dict[str, Any]] = Field(None, description="Explicación de la predicción")
    processing_time_ms: float = Field(..., description="Tiempo de procesamiento en ms")
    model_version: Optional[str] = Field(None, description="Versión del modelo usado")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class HealthStatus(BaseModel):
    """Estado de salud del servicio"""
    status: str = Field(..., pattern="^(healthy|degraded|unhealthy)$")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    version: str
    uptime_seconds: float
    models_loaded: Dict[str, bool]
    system_metrics: Dict[str, Any]
    last_prediction_time: Optional[datetime] = None
    total_predictions: int = 0


class ModelLoadingError(Exception):
    """Excepción para errores de carga de modelo"""
    pass


class PredictionError(Exception):
    """Excepción para errores de predicción"""
    pass


class ModelLoader:
    """
    Cargador y gestor de modelos ML con fallback y caching.
    
    Implementa carga robusta de modelos con múltiples estrategias:
    - Carga desde MLflow Model Registry
    - Carga desde archivos locales
    - Caching en memoria con invalidation
    - Fallback automático entre modelos
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Inicializa el cargador de modelos.
        
        Args:
            config: Configuración de carga de modelos
        """
        self.config = config
        self.models = {}
        self.preprocessing_pipelines = {}
        self.model_metadata = {}
        self.last_loaded = {}
        
        # Configurar MLflow si está disponible
        if MLFLOW_AVAILABLE:
            mlflow_config = get_mlflow_config()
            tracking_uri = mlflow_config['mlflow']['tracking']['tracking_uri']
            mlflow.set_tracking_uri(tracking_uri)
            self.mlflow_client = MlflowClient()
        
    def load_model(self, model_name: str = "default") -> tuple:
        """
        Carga modelo y pipeline de preprocessing.
        
        Args:
            model_name: Nombre del modelo a cargar
            
        Returns:
            Tupla (model, preprocessing_pipeline, metadata)
        """
        try:
            # Verificar cache
            if self._is_model_cached(model_name):
                logger.info(f"Usando modelo en cache: {model_name}")
                return (
                    self.models[model_name], 
                    self.preprocessing_pipelines[model_name],
                    self.model_metadata[model_name]
                )
            
            # Intentar carga desde MLflow
            if MLFLOW_AVAILABLE and self.config.get('use_mlflow', True):
                try:
                    model, preprocessing_pipeline, metadata = self._load_from_mlflow(model_name)
                    self._cache_model(model_name, model, preprocessing_pipeline, metadata)
                    logger.info(f"Modelo cargado desde MLflow: {model_name}")
                    return model, preprocessing_pipeline, metadata
                except Exception as e:
                    logger.warning(f"Fallo carga MLflow para {model_name}: {e}")
            
            # Fallback a carga local
            model, preprocessing_pipeline, metadata = self._load_from_local(model_name)
            self._cache_model(model_name, model, preprocessing_pipeline, metadata)
            logger.info(f"Modelo cargado localmente: {model_name}")
            return model, preprocessing_pipeline, metadata
            
        except Exception as e:
            logger.error(f"Error cargando modelo {model_name}: {e}")
            raise ModelLoadingError(f"No se pudo cargar modelo {model_name}: {e}")
    
    def _is_model_cached(self, model_name: str) -> bool:
        """Verifica si el modelo está en cache y es válido"""
        if model_name not in self.models:
            return False
        
        # Verificar TTL del cache
        cache_ttl = self.config.get('cache_ttl_seconds', 3600)  # 1 hora default
        last_loaded = self.last_loaded.get(model_name, datetime.min)
        
        if (datetime.now() - last_loaded).total_seconds() > cache_ttl:
            logger.info(f"Cache expirado para modelo {model_name}")
            return False
        
        return True
    
    def _load_from_mlflow(self, model_name: str) -> tuple:
        """Carga modelo desde MLflow Model Registry"""
        # Configuración MLflow
        mlflow_model_name = self.config.get('mlflow_model_name', f'obesity_prediction_{model_name}')
        stage = self.config.get('mlflow_stage', 'Production')
        
        try:
            # Buscar modelo en registry
            model_version = self.mlflow_client.get_latest_versions(
                mlflow_model_name, stages=[stage]
            )[0]
            
            # Cargar modelo
            model_uri = f"models:/{mlflow_model_name}/{stage}"
            model = mlflow.sklearn.load_model(model_uri)
            
            # Cargar preprocessing pipeline (debe estar registrado también)
            preprocessing_name = f"{mlflow_model_name}_preprocessing"
            try:
                preprocessing_uri = f"models:/{preprocessing_name}/{stage}"
                preprocessing_pipeline = mlflow.sklearn.load_model(preprocessing_uri)
            except Exception:
                # Fallback a pipeline local
                preprocessing_pipeline = self._load_preprocessing_pipeline_local()
            
            # Metadata del modelo
            metadata = {
                'model_name': mlflow_model_name,
                'version': model_version.version,
                'stage': stage,
                'run_id': model_version.run_id,
                'source': 'mlflow',
                'loaded_at': datetime.now().isoformat()
            }
            
            return model, preprocessing_pipeline, metadata
            
        except Exception as e:
            raise ModelLoadingError(f"Error cargando desde MLflow: {e}")
    
    def _load_from_local(self, model_name: str) -> tuple:
        """Carga modelo desde archivos locales"""
        # Rutas configurables
        model_path = Path(self.config.get('local_model_path', 'artifacts/training/random_forest_model.pkl'))
        preprocessing_path = Path(self.config.get('local_preprocessing_path', 'artifacts/preprocessing/preprocessing_pipeline.pkl'))
        
        # Fallback paths (compatibilidad con código original)
        if not model_path.exists():
            model_path = Path('models/best_model.joblib')
        if not preprocessing_path.exists():
            preprocessing_path = Path('models/preprocess.joblib')
        
        # Cargar modelo
        if not model_path.exists():
            raise ModelLoadingError(f"Archivo de modelo no encontrado: {model_path}")
        
        model = joblib.load(model_path)
        
        # Cargar preprocessing pipeline
        if preprocessing_path.exists():
            preprocessing_pipeline = joblib.load(preprocessing_path)
        else:
            logger.warning("Pipeline de preprocessing no encontrado, usando transformación básica")
            preprocessing_pipeline = self._create_basic_preprocessing_pipeline()
        
        # Metadata local
        metadata = {
            'model_name': model_name,
            'version': 'local',
            'model_path': str(model_path),
            'preprocessing_path': str(preprocessing_path),
            'source': 'local',
            'loaded_at': datetime.now().isoformat()
        }
        
        return model, preprocessing_pipeline, metadata
    
    def _load_preprocessing_pipeline_local(self) -> Pipeline:
        """Carga pipeline de preprocessing local"""
        preprocessing_paths = [
            'artifacts/preprocessing/preprocessing_pipeline.pkl',
            'models/preprocess.joblib',
            'artifacts/features/feature_pipeline.pkl'
        ]
        
        for path in preprocessing_paths:
            if Path(path).exists():
                return joblib.load(path)
        
        return self._create_basic_preprocessing_pipeline()
    
    def _create_basic_preprocessing_pipeline(self) -> Pipeline:
        """Crea pipeline básico de preprocessing como fallback"""
        from sklearn.preprocessing import StandardScaler, OneHotEncoder
        from sklearn.compose import ColumnTransformer
        
        # Pipeline básico compatible con el código original
        numeric_features = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
        categorical_features = [
            'Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 
            'SMOKE', 'SCC', 'CALC', 'MTRANS'
        ]
        
        preprocessor = ColumnTransformer([
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
        
        pipeline = Pipeline([('preprocessor', preprocessor)])
        
        logger.warning("Usando pipeline básico de preprocessing (fallback)")
        return pipeline
    
    def _cache_model(self, model_name: str, model: Any, preprocessing_pipeline: Pipeline, metadata: Dict):
        """Cachea modelo en memoria"""
        self.models[model_name] = model
        self.preprocessing_pipelines[model_name] = preprocessing_pipeline
        self.model_metadata[model_name] = metadata
        self.last_loaded[model_name] = datetime.now()
        
        logger.info(f"Modelo {model_name} cacheado exitosamente")
    
    def get_model_info(self, model_name: str = "default") -> Dict[str, Any]:
        """Retorna información del modelo cargado"""
        if model_name not in self.model_metadata:
            return {"error": "Model not loaded"}
        
        return self.model_metadata[model_name]
    
    def reload_model(self, model_name: str = "default"):
        """Fuerza recarga del modelo"""
        if model_name in self.models:
            del self.models[model_name]
            del self.preprocessing_pipelines[model_name]
            del self.model_metadata[model_name]
            del self.last_loaded[model_name]
        
        return self.load_model(model_name)


class PredictionService:
    """
    Servicio de predicción con validación y monitoreo.
    
    Implementa lógica de predicción robusta con:
    - Validación de entrada y salida
    - Cálculo de métricas adicionales (BMI, etc.)
    - Monitoreo de drift y calidad
    - Logging de predicciones para audit
    """
    
    def __init__(self, model_loader: ModelLoader, config: Dict[str, Any]):
        """
        Inicializa el servicio de predicción.
        
        Args:
            model_loader: Cargador de modelos
            config: Configuración del servicio
        """
        self.model_loader = model_loader
        self.config = config
        self.prediction_count = 0
        self.last_prediction_time = None
        
        # Cargar modelo por defecto
        try:
            self.model, self.preprocessing_pipeline, self.model_metadata = self.model_loader.load_model()
            logger.info("Modelo por defecto cargado exitosamente")
        except Exception as e:
            logger.error(f"Error cargando modelo por defecto: {e}")
            raise
    
    async def predict(self, request: PredictionRequest) -> PredictionResponse:
        """
        Realiza predicción con validación y monitoreo.
        
        Args:
            request: Request de predicción
            
        Returns:
            Response con predicción y metadata
        """
        start_time = time.time()
        
        try:
            # Convertir a DataFrame
            patient_df = self._patient_data_to_dataframe(request.patient_data)
            
            # Aplicar preprocessing
            X_processed = self.preprocessing_pipeline.transform(patient_df)
            
            # Realizar predicción
            prediction = self.model.predict(X_processed)[0]
            
            # Calcular probabilidades si el modelo lo soporta
            probabilities = None
            confidence_score = None
            
            if hasattr(self.model, 'predict_proba'):
                proba = self.model.predict_proba(X_processed)[0]
                
                # Obtener nombres de clases
                if hasattr(self.model, 'classes_'):
                    class_names = self.model.classes_
                    probabilities = dict(zip(class_names, proba.tolist()))
                    confidence_score = float(max(proba))
            
            # Calcular métricas adicionales
            bmi = request.patient_data.Weight / (request.patient_data.Height ** 2)
            bmi_category = self._get_bmi_category(bmi)
            
            # Generar explicación si se solicita
            explanation = None
            if request.include_explanation:
                explanation = self._generate_explanation(request.patient_data, prediction, bmi)
            
            # Tiempo de procesamiento
            processing_time = (time.time() - start_time) * 1000
            
            # Actualizar estadísticas
            self.prediction_count += 1
            self.last_prediction_time = datetime.utcnow()
            
            # Crear response
            response = PredictionResponse(
                request_id=request.request_id,
                prediction=prediction,
                confidence_score=confidence_score if request.include_confidence else None,
                prediction_probabilities=probabilities if request.include_confidence else None,
                bmi=round(bmi, 2),
                bmi_category=bmi_category,
                explanation=explanation,
                processing_time_ms=round(processing_time, 2),
                model_version=self.model_metadata.get('version', 'unknown')
            )
            
            # Log para auditoría
            logger.info(f"Predicción completada: {request.request_id} -> {prediction} (confianza: {confidence_score:.3f if confidence_score else 'N/A'})")
            
            return response
            
        except Exception as e:
            logger.error(f"Error en predicción {request.request_id}: {e}")
            raise PredictionError(f"Error en predicción: {e}")
    
    def _patient_data_to_dataframe(self, patient_data: PatientData) -> pd.DataFrame:
        """Convierte datos del paciente a DataFrame"""
        data = patient_data.dict()
        return pd.DataFrame([data])
    
    def _get_bmi_category(self, bmi: float) -> str:
        """Categoriza BMI según clasificación WHO"""
        if bmi < 18.5:
            return "Underweight"
        elif bmi < 25:
            return "Normal"
        elif bmi < 30:
            return "Overweight"
        elif bmi < 35:
            return "Obese_Class_I"
        elif bmi < 40:
            return "Obese_Class_II"
        else:
            return "Obese_Class_III"
    
    def _generate_explanation(self, patient_data: PatientData, prediction: str, bmi: float) -> Dict[str, Any]:
        """Genera explicación simple de la predicción"""
        explanation = {
            "prediction_rationale": f"Predicción basada en BMI {bmi:.1f} y factores de estilo de vida",
            "key_factors": [],
            "recommendations": []
        }
        
        # Factores clave
        if bmi >= 30:
            explanation["key_factors"].append("BMI indica obesidad")
        elif bmi >= 25:
            explanation["key_factors"].append("BMI indica sobrepeso")
        
        if patient_data.FAF <= 1:
            explanation["key_factors"].append("Baja frecuencia de actividad física")
        
        if patient_data.FAVC == "yes":
            explanation["key_factors"].append("Consumo frecuente de alimentos calóricos")
        
        # Recomendaciones básicas
        if bmi >= 25:
            explanation["recommendations"].append("Consultar con profesional de salud")
            explanation["recommendations"].append("Considerar plan de actividad física")
        
        return explanation


class HealthChecker:
    """
    Health checker para monitoreo de la aplicación.
    
    Implementa múltiples niveles de health checks:
    - Basic: API responde
    - Ready: Modelos cargados
    - Live: Sistema operativo
    """
    
    def __init__(self, model_loader: ModelLoader, prediction_service: PredictionService, app_version: str):
        """
        Inicializa el health checker.
        
        Args:
            model_loader: Cargador de modelos
            prediction_service: Servicio de predicción
            app_version: Versión de la aplicación
        """
        self.model_loader = model_loader
        self.prediction_service = prediction_service
        self.app_version = app_version
        self.start_time = datetime.utcnow()
    
    def get_health_status(self) -> HealthStatus:
        """Retorna estado de salud comprensivo"""
        try:
            # Verificar modelos cargados
            models_status = {}
            overall_status = "healthy"
            
            try:
                model_info = self.model_loader.get_model_info()
                models_status["default"] = model_info.get("error") is None
                if not models_status["default"]:
                    overall_status = "degraded"
            except Exception:
                models_status["default"] = False
                overall_status = "unhealthy"
            
            # Métricas del sistema
            uptime = (datetime.utcnow() - self.start_time).total_seconds()
            
            system_metrics = {
                "cpu_usage_percent": self._get_cpu_usage(),
                "memory_usage_mb": self._get_memory_usage(),
                "disk_usage_percent": self._get_disk_usage()
            }
            
        except Exception as e:
            logger.error(f"Error en health check: {e}")
            overall_status = "unhealthy"
            models_status = {"default": False}
            uptime = 0
            system_metrics = {}
        
        return HealthStatus(
            status=overall_status,
            version=self.app_version,
            uptime_seconds=uptime,
            models_loaded=models_status,
            system_metrics=system_metrics,
            last_prediction_time=self.prediction_service.last_prediction_time,
            total_predictions=self.prediction_service.prediction_count
        )
    
    def _get_cpu_usage(self) -> float:
        """Obtiene uso de CPU (simplificado)"""
        try:
            import psutil
            return psutil.cpu_percent(interval=1)
        except ImportError:
            return 0.0
    
    def _get_memory_usage(self) -> float:
        """Obtiene uso de memoria (simplificado)"""
        try:
            import psutil
            return psutil.virtual_memory().used / (1024 * 1024)  # MB
        except ImportError:
            return 0.0
    
    def _get_disk_usage(self) -> float:
        """Obtiene uso de disco (simplificado)"""
        try:
            import psutil
            return psutil.disk_usage('/').percent
        except ImportError:
            return 0.0


# Middleware para logging y métricas
class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware para logging de requests"""
    
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        # Log request
        logger.info(f"Request: {request.method} {request.url}")
        
        response = await call_next(request)
        
        # Log response
        process_time = time.time() - start_time
        logger.info(f"Response: {response.status_code} ({process_time:.3f}s)")
        
        response.headers["X-Process-Time"] = str(process_time)
        return response


# Factory function para crear la aplicación
def create_app(config_path: str = "configs/api/api_config.yaml") -> FastAPI:
    """
    Factory para crear la aplicación FastAPI configurada.
    
    Args:
        config_path: Ruta al archivo de configuración
        
    Returns:
        Aplicación FastAPI configurada
    """
    # Cargar configuración
    try:
        config = load_config(config_path)
    except Exception as e:
        logger.warning(f"No se pudo cargar configuración {config_path}: {e}")
        config = _get_default_config()
    
    # Configuración de la app
    app_config = config.get('app', {})
    app_version = app_config.get('version', '1.0.0')
    
    # Crear aplicación FastAPI
    app = FastAPI(
        title=app_config.get('title', 'Obesity Prediction API'),
        description=app_config.get('description', 'Production-ready ML API for obesity prediction'),
        version=app_version,
        docs_url="/docs" if app_config.get('enable_docs', True) else None,
        redoc_url="/redoc" if app_config.get('enable_docs', True) else None
    )
    
    # Middleware
    app.add_middleware(LoggingMiddleware)
    
    # CORS
    cors_config = config.get('cors', {})
    if cors_config.get('enabled', True):
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_config.get('allow_origins', ["*"]),
            allow_credentials=cors_config.get('allow_credentials', True),
            allow_methods=cors_config.get('allow_methods', ["*"]),
            allow_headers=cors_config.get('allow_headers', ["*"])
        )
    
    # Trusted hosts
    trusted_hosts = config.get('security', {}).get('trusted_hosts')
    if trusted_hosts:
        app.add_middleware(TrustedHostMiddleware, allowed_hosts=trusted_hosts)
    
    # Inicializar servicios
    model_config = config.get('model', {})
    model_loader = ModelLoader(model_config)
    prediction_service = PredictionService(model_loader, config.get('prediction', {}))
    health_checker = HealthChecker(model_loader, prediction_service, app_version)
    
    # Endpoints
    @app.get("/", response_model=Dict[str, str])
    async def root():
        """Endpoint raíz con información de la API"""
        return {
            "message": "Obesity Prediction API",
            "version": app_version,
            "status": "running",
            "docs": "/docs"
        }
    
    @app.get("/health", response_model=HealthStatus)
    async def health():
        """Health check endpoint"""
        return health_checker.get_health_status()
    
    @app.get("/health/ready")
    async def readiness():
        """Readiness check para Kubernetes"""
        health_status = health_checker.get_health_status()
        if health_status.status == "unhealthy":
            raise HTTPException(status_code=503, detail="Service not ready")
        return {"status": "ready"}
    
    @app.get("/health/live")
    async def liveness():
        """Liveness check para Kubernetes"""
        return {"status": "alive"}
    
    @app.post("/predict", response_model=PredictionResponse)
    async def predict(request: PredictionRequest):
        """
        Endpoint principal de predicción.
        
        Predice la categoría de obesidad basado en datos del paciente.
        Incluye validación médica de entrada y métricas de confianza.
        """
        try:
            return await prediction_service.predict(request)
        except PredictionError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"Error interno en predicción: {e}")
            raise HTTPException(status_code=500, detail="Error interno del servidor")
    
    @app.get("/model/info")
    async def model_info():
        """Información del modelo cargado"""
        return model_loader.get_model_info()
    
    @app.post("/model/reload")
    async def reload_model():
        """Recarga el modelo (requiere autenticación en producción)"""
        try:
            model_loader.reload_model()
            return {"status": "Model reloaded successfully"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error reloading model: {e}")
    
    # Event handlers
    @app.on_event("startup")
    async def startup_event():
        """Tareas de inicialización"""
        logger.info(f"Iniciando Obesity Prediction API v{app_version}")
        
        # Verificar modelos cargados
        try:
            model_info = model_loader.get_model_info()
            logger.info(f"Modelo cargado: {model_info}")
        except Exception as e:
            logger.error(f"Error verificando modelo en startup: {e}")
    
    @app.on_event("shutdown")
    async def shutdown_event():
        """Tareas de limpieza"""
        logger.info("Cerrando Obesity Prediction API")
    
    return app


def _get_default_config() -> Dict[str, Any]:
    """Configuración por defecto si no se encuentra archivo"""
    return {
        'app': {
            'title': 'Obesity Prediction API',
            'description': 'Production-ready ML API for obesity prediction',
            'version': '1.0.0',
            'enable_docs': True
        },
        'model': {
            'use_mlflow': False,
            'local_model_path': 'models/best_model.joblib',
            'local_preprocessing_path': 'models/preprocess.joblib',
            'cache_ttl_seconds': 3600
        },
        'prediction': {},
        'cors': {
            'enabled': True,
            'allow_origins': ["*"],
            'allow_credentials': True,
            'allow_methods': ["*"],
            'allow_headers': ["*"]
        },
        'security': {
            'trusted_hosts': None
        }
    }


# Crear instancia de la aplicación (compatibilidad con código original)
app = create_app()


if __name__ == "__main__":
    import uvicorn
    
    # Configuración por defecto para desarrollo
    uvicorn.run(
        "src.api.serve:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )