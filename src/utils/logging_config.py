"""
Logging configuration utilities for MLOps pipeline.

Implementa configuración centralizada de logging siguiendo
las mejores prácticas de observabilidad en MLOps.

Referencias:
- Observability Best Practices: docs/MLOPS_THEORY.md - Operational Excellence
- MLOps Monitoring: docs/Machine Learning Engineering with MLflow.pdf - MLOps Operations Guide
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime
import json


class MLOpsFormatter(logging.Formatter):
    """Formatter personalizado para logs MLOps con contexto estructurado"""
    
    def format(self, record):
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Agregar contexto MLOps si está disponible
        if hasattr(record, 'experiment_id'):
            log_entry['experiment_id'] = record.experiment_id
        if hasattr(record, 'run_id'):
            log_entry['run_id'] = record.run_id
        if hasattr(record, 'model_name'):
            log_entry['model_name'] = record.model_name
        if hasattr(record, 'pipeline_stage'):
            log_entry['pipeline_stage'] = record.pipeline_stage
            
        return json.dumps(log_entry, ensure_ascii=False)


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    enable_console: bool = True,
    enable_structured: bool = False
) -> logging.Logger:
    """
    Configura logging centralizado para el pipeline MLOps.
    
    Args:
        log_level: Nivel de logging (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Archivo de log (opcional)
        enable_console: Habilitar logging a consola
        enable_structured: Usar formato JSON estructurado
        
    Returns:
        Logger configurado
        
    Example:
        >>> logger = setup_logging(log_level="DEBUG", log_file="logs/mlops.log")
        >>> logger.info("Pipeline iniciado", extra={"pipeline_stage": "data_ingestion"})
    """
    
    # Crear directorio de logs si no existe
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Configurar logger root
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Limpiar handlers existentes
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Formatter
    if enable_structured:
        formatter = MLOpsFormatter()
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(module)s:%(funcName)s:%(lineno)d] - %(message)s'
        )
    
    # Console handler
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Configurar loggers específicos
    configure_library_loggers()
    
    return root_logger


def configure_library_loggers():
    """Configura logging para librerías externas"""
    
    # MLflow logging
    logging.getLogger("mlflow").setLevel(logging.WARNING)
    
    # Sklearn logging
    logging.getLogger("sklearn").setLevel(logging.WARNING)
    
    # Matplotlib logging
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    
    # Requests logging
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Obtiene logger con nombre específico.
    
    Args:
        name: Nombre del logger (usar __name__)
        
    Returns:
        Logger configurado
    """
    return logging.getLogger(name)


class MLOpsLoggerAdapter(logging.LoggerAdapter):
    """
    Adapter para agregar contexto MLOps automáticamente a los logs.
    
    Example:
        >>> adapter = MLOpsLoggerAdapter(logger, {
        ...     'experiment_id': 'exp_123',
        ...     'pipeline_stage': 'training'
        ... })
        >>> adapter.info("Modelo entrenado exitosamente")
    """
    
    def process(self, msg, kwargs):
        # Agregar contexto extra
        if 'extra' not in kwargs:
            kwargs['extra'] = {}
        
        kwargs['extra'].update(self.extra)
        return msg, kwargs


def log_pipeline_stage(stage_name: str):
    """
    Decorator para logging automático de stages del pipeline.
    
    Args:
        stage_name: Nombre del stage del pipeline
        
    Example:
        >>> @log_pipeline_stage("data_preprocessing")
        ... def preprocess_data():
        ...     pass
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger = get_logger(func.__module__)
            logger.info(f"Iniciando stage: {stage_name}")
            
            try:
                result = func(*args, **kwargs)
                logger.info(f"Stage completado exitosamente: {stage_name}")
                return result
            except Exception as e:
                logger.error(f"Error en stage {stage_name}: {str(e)}")
                raise
                
        return wrapper
    return decorator


def log_function_execution(func):
    """
    Decorator para logging automático de ejecución de funciones.
    
    Example:
        >>> @log_function_execution
        ... def train_model():
        ...     pass
    """
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        logger.debug(f"Ejecutando función: {func.__name__}")
        
        try:
            result = func(*args, **kwargs)
            logger.debug(f"Función ejecutada exitosamente: {func.__name__}")
            return result
        except Exception as e:
            logger.error(f"Error en función {func.__name__}: {str(e)}")
            raise
            
    return wrapper


# Configurar logging por defecto al importar el módulo
default_logger = setup_logging(
    log_level="INFO",
    log_file="logs/mlops.log",
    enable_console=True,
    enable_structured=False
)


if __name__ == "__main__":
    # Test logging configuration
    logger = setup_logging(log_level="DEBUG", enable_structured=True)
    
    # Test basic logging
    logger.info("Test básico de logging")
    
    # Test with MLOps context
    adapter = MLOpsLoggerAdapter(logger, {
        'experiment_id': 'exp_test_123',
        'pipeline_stage': 'testing'
    })
    adapter.info("Test con contexto MLOps")
    
    # Test decorator
    @log_pipeline_stage("test_stage")
    def test_function():
        return "success"
    
    result = test_function()
    logger.info(f"Resultado del test: {result}")
    
    print("Configuración de logging testeada exitosamente")