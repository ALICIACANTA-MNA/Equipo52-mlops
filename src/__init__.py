"""
Obesity Prediction MLOps Project - Equipo 52

Este paquete implementa un pipeline MLOps completo para la predicción de niveles de obesidad
basado en características biométricas y hábitos de vida.

Arquitectura modular basada en Microsoft MLOps Best Practices:
- src.data: Ingesta, validación y procesamiento de datos
- src.features: Feature engineering y transformaciones
- src.models: Entrenamiento, evaluación y registro de modelos
- src.api: Servicio REST para inferencia
- src.utils: Utilidades compartidas

Referencias:
- MLOps Best Practices: docs/MLOPS_THEORY.md - MLOps Best Practices
- Repository Structure: docs/IMPLEMENTATION_GUIDE.md - Data Science Best Practices
"""

__version__ = "2.0.0"
__author__ = "Equipo 52"

# Importaciones principales para facilitar el uso
from src.utils.config import load_config
from src.utils.logging_config import setup_logging

__all__ = [
    "load_config",
    "setup_logging"
]