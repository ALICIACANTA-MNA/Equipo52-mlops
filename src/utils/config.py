"""
Configuration management utilities for MLOps pipeline.

Implementa carga centralizada de configuraciones YAML siguiendo
las mejores prácticas de MLOps para parametrización externa.

Referencias:
- Infrastructure as Code Best Practices
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import os
from dataclasses import dataclass


@dataclass
class ConfigPaths:
    """Rutas de configuración centralizadas"""
    DATA_CONFIG = "configs/data/data_config.yaml"
    MODEL_CONFIG = "configs/model/model_config.yaml"
    MLFLOW_CONFIG = "configs/mlflow/mlflow_config.yaml"
    PARAMS = "params.yaml"


class ConfigurationError(Exception):
    """Excepción para errores de configuración"""
    pass


def load_config(config_path: str, section: Optional[str] = None) -> Dict[str, Any]:
    """
    Carga configuración desde archivo YAML.
    
    Args:
        config_path: Ruta al archivo de configuración
        section: Sección específica a cargar (opcional)
        
    Returns:
        Diccionario con la configuración cargada
        
    Raises:
        ConfigurationError: Si el archivo no existe o hay errores de formato
        
    Example:
        >>> config = load_config("configs/data/data_config.yaml")
        >>> data_paths = load_config("configs/data/data_config.yaml", "data")
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise ConfigurationError(f"Archivo de configuración no encontrado: {config_path}")
    
    try:
        with open(config_file, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
            
        if section and section in config:
            return config[section]
        elif section and section not in config:
            raise ConfigurationError(f"Sección '{section}' no encontrada en {config_path}")
        
        return config
        
    except yaml.YAMLError as e:
        raise ConfigurationError(f"Error parsing YAML en {config_path}: {e}")
    except Exception as e:
        raise ConfigurationError(f"Error cargando configuración desde {config_path}: {e}")


def load_params(params_path: str = "params.yaml") -> Dict[str, Any]:
    """
    Carga parámetros del pipeline DVC.
    
    Args:
        params_path: Ruta al archivo params.yaml
        
    Returns:
        Diccionario con todos los parámetros
    """
    return load_config(params_path)


def get_data_config() -> Dict[str, Any]:
    """Carga configuración de datos"""
    return load_config(ConfigPaths.DATA_CONFIG)


def get_model_config() -> Dict[str, Any]:
    """Carga configuración de modelos"""
    return load_config(ConfigPaths.MODEL_CONFIG)


def get_mlflow_config() -> Dict[str, Any]:
    """Carga configuración de MLflow"""
    return load_config(ConfigPaths.MLFLOW_CONFIG)


def get_environment_variable(var_name: str, default: Optional[str] = None) -> str:
    """
    Obtiene variable de entorno con fallback.
    
    Args:
        var_name: Nombre de la variable de entorno
        default: Valor por defecto si no existe
        
    Returns:
        Valor de la variable de entorno
        
    Raises:
        ConfigurationError: Si la variable no existe y no hay default
    """
    value = os.getenv(var_name, default)
    if value is None:
        raise ConfigurationError(f"Variable de entorno requerida no encontrada: {var_name}")
    return value


def setup_mlflow_tracking(config: Optional[Dict[str, Any]] = None) -> str:
    """
    Configura MLflow tracking URI desde configuración.
    
    Args:
        config: Configuración MLflow (opcional, se carga automáticamente)
        
    Returns:
        URI de tracking configurado
    """
    import mlflow
    
    if config is None:
        config = get_mlflow_config()
    
    tracking_uri = config['mlflow']['tracking']['tracking_uri']
    mlflow.set_tracking_uri(tracking_uri)
    
    return tracking_uri


def validate_paths(config: Dict[str, Any], section: str = "paths") -> None:
    """
    Valida que los directorios necesarios existan.
    
    Args:
        config: Configuración con paths
        section: Sección que contiene los paths
        
    Raises:
        ConfigurationError: Si algún directorio requerido no existe
    """
    if section not in config:
        return
    
    paths_config = config[section]
    for path_name, path_value in paths_config.items():
        if isinstance(path_value, str):
            path_obj = Path(path_value)
            if not path_obj.parent.exists():
                path_obj.parent.mkdir(parents=True, exist_ok=True)
        elif isinstance(path_value, dict):
            # Recursively validate nested paths
            validate_paths({section: path_value}, section)


if __name__ == "__main__":
    # Test configuration loading
    try:
        print("Testeando carga de configuraciones...")
        
        # Test params loading
        params = load_params()
        print(f"Parámetros cargados: {len(params)} secciones")
        
        # Test MLflow config
        mlflow_config = get_mlflow_config()
        print(f"Configuración MLflow cargada")
        
        print("Todas las configuraciones cargadas exitosamente")
        
    except ConfigurationError as e:
        print(f"Error de configuración: {e}")
    except Exception as e:
        print(f"Error inesperado: {e}")