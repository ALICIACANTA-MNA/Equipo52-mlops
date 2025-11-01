"""
Configuración global para testing del proyecto MLOps Obesidad.
Este módulo contiene fixtures compartidas y configuración de testing.
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import Mock, MagicMock
import tempfile
import json
from typing import Dict, Any, Generator

# Importar MLflow solo si está disponible
try:
    import mlflow
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False

# Agregar src al path para imports
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

# Variables de entorno para testing
os.environ["ENVIRONMENT"] = "test"
os.environ["MLFLOW_TRACKING_URI"] = "file:///tmp/mlflow-test"
os.environ["LOG_LEVEL"] = "WARNING"

@pytest.fixture(scope="session")
def project_root() -> Path:
    """Ruta raíz del proyecto."""
    return Path(__file__).parent.parent

@pytest.fixture(scope="session") 
def test_data_dir(project_root: Path) -> Path:
    """Directorio de datos de prueba."""
    return project_root / "tests" / "fixtures" / "data"

@pytest.fixture(scope="session")
def config_dir(project_root: Path) -> Path:
    """Directorio de configuración."""
    return project_root / "configs"

@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Directorio temporal para tests."""  
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)

@pytest.fixture
def sample_obesity_data() -> pd.DataFrame:
    """Dataset de muestra para testing de obesidad."""
    np.random.seed(42)
    n_samples = 100
    
    data = {
        'Gender': np.random.choice(['Male', 'Female'], n_samples),
        'Age': np.random.randint(16, 80, n_samples),
        'Height': np.random.normal(1.70, 0.15, n_samples),
        'Weight': np.random.normal(70, 20, n_samples),
        'family_history_with_overweight': np.random.choice(['yes', 'no'], n_samples),
        'FAVC': np.random.choice(['yes', 'no'], n_samples),
        'FCVC': np.random.randint(1, 4, n_samples),
        'NCP': np.random.randint(1, 5, n_samples),
        'CAEC': np.random.choice(['no', 'Sometimes', 'Frequently', 'Always'], n_samples),
        'SMOKE': np.random.choice(['yes', 'no'], n_samples),
        'CH2O': np.random.randint(1, 4, n_samples),
        'SCC': np.random.choice(['yes', 'no'], n_samples),
        'FAF': np.random.randint(0, 4, n_samples),
        'TUE': np.random.randint(0, 3, n_samples),
        'CALC': np.random.choice(['no', 'Sometimes', 'Frequently', 'Always'], n_samples),
        'MTRANS': np.random.choice(['Walking', 'Public_Transportation', 'Automobile', 'Bike', 'Motorbike'], n_samples),
        'NObeyesdad': np.random.choice([
            'Insufficient_Weight', 'Normal_Weight', 'Obesity_Type_I',
            'Obesity_Type_II', 'Obesity_Type_III', 'Overweight_Level_I', 'Overweight_Level_II'
        ], n_samples)
    }
    
    return pd.DataFrame(data)

@pytest.fixture
def sample_processed_features() -> pd.DataFrame:
    """Features procesadas de muestra."""
    np.random.seed(42)
    n_samples = 100
    
    return pd.DataFrame({
        'bmi': np.random.normal(25, 5, n_samples),
        'age_normalized': np.random.normal(0, 1, n_samples),
        'height_normalized': np.random.normal(0, 1, n_samples),
        'weight_normalized': np.random.normal(0, 1, n_samples),
        'gender_encoded': np.random.choice([0, 1], n_samples),
        'family_history_encoded': np.random.choice([0, 1], n_samples),
        'favc_encoded': np.random.choice([0, 1], n_samples),
        'lifestyle_score': np.random.normal(0.5, 0.2, n_samples)
    })

@pytest.fixture
def sample_prediction_input() -> Dict[str, Any]:
    """Input de muestra para predicciones."""
    return {
        "gender": "Female",
        "age": 25,
        "height": 1.65,
        "weight": 65.0,
        "family_history_with_overweight": "yes",
        "favc": "no",
        "fcvc": 2,
        "ncp": 3,
        "caec": "Sometimes",
        "smoke": "no",
        "ch2o": 2,
        "scc": "no",
        "faf": 1,
        "tue": 1,
        "calc": "Sometimes",
        "mtrans": "Public_Transportation"
    }

@pytest.fixture
def mock_mlflow_client():
    """Cliente MLflow mockeado para testing."""
    with pytest.MonkeyPatch.context() as mp:
        mock_client = Mock()
        mock_client.search_experiments.return_value = []
        mock_client.create_experiment.return_value = "test_experiment_id"
        mock_client.search_runs.return_value = []
        mock_client.log_metric = Mock()
        mock_client.log_param = Mock()
        mock_client.log_artifacts = Mock()
        
        mp.setattr("mlflow.MlflowClient", lambda: mock_client)
        yield mock_client

@pytest.fixture 
def mock_model():
    """Modelo ML mockeado para testing."""
    mock_model = Mock()
    mock_model.predict.return_value = np.array(['Normal_Weight'] * 10)
    mock_model.predict_proba.return_value = np.random.rand(10, 7)
    mock_model.score.return_value = 0.85
    return mock_model

@pytest.fixture
def api_test_client():
    """Cliente de testing para FastAPI."""
    try:
        from fastapi.testclient import TestClient
        from src.api.serve import app
        
        with TestClient(app) as test_client:
            yield test_client
    except ImportError:
        # Si FastAPI no está disponible, retornar mock
        yield Mock()

@pytest.fixture(autouse=True)
def cleanup_mlflow():
    """Limpieza automática de MLflow después de cada test."""
    yield
    # Limpiar tracking URI temporal
    mlflow.end_run()

@pytest.fixture
def test_config() -> Dict[str, Any]:
    """Configuración de prueba."""
    return {
        "data": {
            "raw_path": "tests/fixtures/data/raw_sample.csv",
            "processed_path": "tests/fixtures/data/processed_sample.csv",
            "target_column": "NObeyesdad",
            "test_size": 0.2,
            "random_state": 42
        },
        "model": {
            "algorithm": "random_forest",
            "hyperparameters": {
                "n_estimators": 10,
                "max_depth": 5,
                "random_state": 42
            }
        },
        "api": {
            "host": "localhost",
            "port": 8000,
            "timeout": 30
        },
        "logging": {
            "level": "WARNING",
            "format": "simple"
        }
    }

@pytest.fixture
def mock_redis_client():
    """Cliente Redis mockeado."""
    mock_redis = Mock()
    mock_redis.get.return_value = None
    mock_redis.set.return_value = True
    mock_redis.delete.return_value = True
    mock_redis.ping.return_value = True
    return mock_redis

# Hooks de pytest para personalización
def pytest_configure(config):
    """Configuración inicial de pytest."""
    # Crear directorio de reportes si no existe
    reports_dir = Path("tests/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)