"""
Servidor de producción para la API de predicción de obesidad.

Script para lanzar la API con configuración optimizada para diferentes entornos:
- Desarrollo: Servidor de desarrollo con auto-reload
- Testing: Configuración para tests automatizados  
- Producción: Servidor multi-worker con Gunicorn/Uvicorn

Uso:
    python -m src.api.server --env development
    python -m src.api.server --env production --workers 4
    python -m src.api.server --config configs/api/production.yaml

Referencias:
- FastAPI Deployment: https://fastapi.tiangolo.com/deployment/
- Uvicorn Deployment: https://www.uvicorn.org/deployment/
- Gunicorn with Uvicorn: https://www.uvicorn.org/deployment/#gunicorn
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional

# Importaciones principales
try:
    import uvicorn
    import gunicorn.app.base
    from gunicorn.six import iteritems
except ImportError as e:
    print(f"Error: Dependencias de servidor no instaladas: {e}")
    print("Instalar con: pip install uvicorn gunicorn")
    sys.exit(1)

# Imports propios
from src.utils.config import load_config
from src.utils.logging_config import get_logger
from src.api.serve import create_app


logger = get_logger(__name__)


class GunicornApp(gunicorn.app.base.BaseApplication):
    """
    Aplicación Gunicorn personalizada para FastAPI.
    
    Permite configurar Gunicorn programáticamente con configuración YAML.
    """
    
    def __init__(self, app, options=None):
        """
        Inicializa la aplicación Gunicorn.
        
        Args:
            app: Aplicación ASGI (FastAPI)
            options: Opciones de configuración
        """
        self.options = options or {}
        self.application = app
        super().__init__()
    
    def load_config(self):
        """Carga configuración de Gunicorn"""
        config = {key: value for key, value in iteritems(self.options)
                 if key in self.cfg.settings and value is not None}
        for key, value in iteritems(config):
            self.cfg.set(key.lower(), value)
    
    def load(self):
        """Carga la aplicación"""
        return self.application


def get_server_config(config_path: str, environment: str) -> Dict[str, Any]:
    """
    Carga configuración del servidor.
    
    Args:
        config_path: Ruta al archivo de configuración
        environment: Entorno (development, testing, production)
        
    Returns:
        Configuración del servidor
    """
    try:
        config = load_config(config_path)
    except Exception as e:
        logger.warning(f"No se pudo cargar configuración {config_path}: {e}")
        config = get_default_server_config()
    
    # Combinar configuración base con específica del entorno
    server_config = config.get('server', {})
    env_config = config.get(environment, {})
    
    # Merge configurations
    merged_config = {**server_config, **env_config}
    
    # Override con variables de entorno
    merged_config.update({
        'host': os.getenv('HOST', merged_config.get('host', '0.0.0.0')),
        'port': int(os.getenv('PORT', merged_config.get('port', 8000))),
        'workers': int(os.getenv('WORKERS', merged_config.get('workers', 1))),
        'log_level': os.getenv('LOG_LEVEL', merged_config.get('log_level', 'info')),
    })
    
    return merged_config


def get_default_server_config() -> Dict[str, Any]:
    """Configuración por defecto del servidor"""
    return {
        'server': {
            'host': '0.0.0.0',
            'port': 8000,
            'workers': 1,
            'log_level': 'info',
            'reload': False,
            'access_log': True
        },
        'development': {
            'reload': True,
            'log_level': 'debug',
            'workers': 1
        },
        'production': {
            'workers': 4,
            'worker_class': 'uvicorn.workers.UvicornWorker',
            'worker_connections': 1000,
            'keepalive': 2,
            'max_requests': 1000,
            'max_requests_jitter': 50,
            'preload_app': True,
            'graceful_timeout': 30
        }
    }


def run_development_server(config: Dict[str, Any], app_module: str = "src.api.serve:app"):
    """
    Ejecuta servidor de desarrollo con Uvicorn.
    
    Args:
        config: Configuración del servidor
        app_module: Módulo de la aplicación
    """
    logger.info("Iniciando servidor de desarrollo...")
    
    uvicorn.run(
        app_module,
        host=config['host'],
        port=config['port'],
        log_level=config['log_level'],
        reload=config.get('reload', True),
        access_log=config.get('access_log', True),
        reload_dirs=["src/"] if config.get('reload', True) else None,
        reload_excludes=["*.pyc", "*.pyo", "__pycache__"] if config.get('reload', True) else None
    )


def run_production_server(config: Dict[str, Any], app_factory):
    """
    Ejecuta servidor de producción con Gunicorn.
    
    Args:
        config: Configuración del servidor
        app_factory: Factory de la aplicación
    """
    logger.info(f"Iniciando servidor de producción con {config['workers']} workers...")
    
    # Configuración de Gunicorn
    gunicorn_config = {
        'bind': f"{config['host']}:{config['port']}",
        'workers': config['workers'],
        'worker_class': config.get('worker_class', 'uvicorn.workers.UvicornWorker'),
        'worker_connections': config.get('worker_connections', 1000),
        'keepalive': config.get('keepalive', 2),
        'max_requests': config.get('max_requests', 1000),
        'max_requests_jitter': config.get('max_requests_jitter', 50),
        'preload_app': config.get('preload_app', True),
        'timeout': config.get('timeout', 30),
        'graceful_timeout': config.get('graceful_timeout', 30),
        'loglevel': config['log_level'],
        'access_log_format': '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s',
        'accesslog': '-' if config.get('access_log', True) else None,
        'errorlog': '-'
    }
    
    # SSL configuration si está disponible
    if config.get('ssl_keyfile') and config.get('ssl_certfile'):
        gunicorn_config.update({
            'keyfile': config['ssl_keyfile'],
            'certfile': config['ssl_certfile'],
            'ca_certs': config.get('ssl_ca_cert')
        })
        logger.info("SSL/TLS habilitado")
    
    # Crear y ejecutar aplicación Gunicorn
    app = app_factory()
    GunicornApp(app, gunicorn_config).run()


def run_testing_server(config: Dict[str, Any], app_factory):
    """
    Ejecuta servidor para testing.
    
    Args:
        config: Configuración del servidor
        app_factory: Factory de la aplicación
    """
    logger.info("Iniciando servidor de testing...")
    
    # Configuración específica para tests
    test_config = {
        **config,
        'workers': 1,  # Single worker para tests
        'log_level': 'warning',  # Reducir logging
        'reload': False,
        'access_log': False
    }
    
    app = app_factory()
    
    uvicorn.run(
        app,
        host=test_config['host'],
        port=test_config['port'],
        log_level=test_config['log_level'],
        access_log=test_config['access_log']
    )


def validate_environment(environment: str) -> bool:
    """
    Valida que el entorno sea válido.
    
    Args:
        environment: Nombre del entorno
        
    Returns:
        True si es válido
    """
    valid_environments = ['development', 'testing', 'production']
    
    if environment not in valid_environments:
        logger.error(f"Entorno inválido: {environment}")
        logger.error(f"Entornos válidos: {', '.join(valid_environments)}")
        return False
    
    return True


def setup_logging(config: Dict[str, Any]):
    """
    Configura logging para el servidor.
    
    Args:
        config: Configuración completa
    """
    logging_config = config.get('logging', {})
    
    # Configurar nivel de logging
    log_level = logging_config.get('level', 'INFO').upper()
    logging.getLogger().setLevel(getattr(logging, log_level))
    
    # Configurar formato
    log_format = logging_config.get('format', 'text')
    if log_format == 'json':
        # Configurar logging JSON (implementar si es necesario)
        pass
    
    logger.info(f"Logging configurado: nivel={log_level}, formato={log_format}")


def check_dependencies():
    """Verifica que las dependencias necesarias estén instaladas"""
    missing_deps = []
    
    try:
        import fastapi
    except ImportError:
        missing_deps.append('fastapi')
    
    try:
        import uvicorn
    except ImportError:
        missing_deps.append('uvicorn')
    
    try:
        import pydantic
    except ImportError:
        missing_deps.append('pydantic')
    
    if missing_deps:
        logger.error(f"Dependencias faltantes: {', '.join(missing_deps)}")
        logger.error("Instalar con: pip install " + " ".join(missing_deps))
        return False
    
    return True


def main():
    """Función principal del servidor"""
    parser = argparse.ArgumentParser(description='Servidor de API de predicción de obesidad')
    
    parser.add_argument(
        '--env', '--environment',
        choices=['development', 'testing', 'production'],
        default='development',
        help='Entorno de ejecución'
    )
    
    parser.add_argument(
        '--config', '--config-file',
        type=str,
        default='configs/api/api_config.yaml',
        help='Archivo de configuración'
    )
    
    parser.add_argument(
        '--host',
        type=str,
        help='Host del servidor (override config)'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        help='Puerto del servidor (override config)'
    )
    
    parser.add_argument(
        '--workers',
        type=int,
        help='Número de workers (solo producción)'
    )
    
    parser.add_argument(
        '--reload',
        action='store_true',
        help='Habilitar auto-reload (solo desarrollo)'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['debug', 'info', 'warning', 'error', 'critical'],
        help='Nivel de logging'
    )
    
    args = parser.parse_args()
    
    # Validar entorno
    if not validate_environment(args.env):
        sys.exit(1)
    
    # Verificar dependencias
    if not check_dependencies():
        sys.exit(1)
    
    # Cargar configuración
    try:
        config = get_server_config(args.config, args.env)
        
        # Override con argumentos de línea de comandos
        if args.host:
            config['host'] = args.host
        if args.port:
            config['port'] = args.port
        if args.workers and args.env == 'production':
            config['workers'] = args.workers
        if args.reload and args.env == 'development':
            config['reload'] = args.reload
        if args.log_level:
            config['log_level'] = args.log_level
            
    except Exception as e:
        logger.error(f"Error cargando configuración: {e}")
        sys.exit(1)
    
    # Configurar logging
    setup_logging(config)
    
    # Factory para crear la aplicación
    def app_factory():
        return create_app(args.config)
    
    # Ejecutar servidor según el entorno
    try:
        if args.env == 'development':
            run_development_server(config)
        elif args.env == 'testing':
            run_testing_server(config, app_factory)
        elif args.env == 'production':
            run_production_server(config, app_factory)
            
    except KeyboardInterrupt:
        logger.info("Servidor detenido por el usuario")
    except Exception as e:
        logger.error(f"Error ejecutando servidor: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()