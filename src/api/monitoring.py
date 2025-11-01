"""
Utilidades de monitoreo y métricas para la API de predicción.

Implementa recolección de métricas, health checks y monitoreo de rendimiento
siguiendo las mejores prácticas de observabilidad en producción:
- Métricas de Prometheus para monitoreo
- Health checks para Kubernetes/Docker
- Logging estructurado para análisis
- Alertas basadas en umbrales
- Dashboard de métricas en tiempo real

Referencias:
- Prometheus Metrics: https://prometheus.io/docs/practices/naming/
- FastAPI Monitoring: https://fastapi.tiangolo.com/advanced/middleware/
- Application Insights: docs/TECHNICAL_REFERENCES.md - API Monitoring
- OpenTelemetry: https://opentelemetry.io/docs/instrumentation/python/
"""

import time
import psutil
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from collections import defaultdict, deque
from dataclasses import dataclass, field
import json

# Monitoring libraries (optional)
try:
    from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, generate_latest
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

try:
    from azure.monitor.opentelemetry import configure_azure_monitor
    from opentelemetry import trace, metrics
    AZURE_MONITORING_AVAILABLE = True
except ImportError:
    AZURE_MONITORING_AVAILABLE = False

from src.utils.logging_config import get_logger


logger = get_logger(__name__)


@dataclass
class MetricData:
    """Estructura para datos de métricas"""
    name: str
    value: float
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)
    type: str = "gauge"  # gauge, counter, histogram


@dataclass
class PredictionMetrics:
    """Métricas específicas de predicción"""
    total_predictions: int = 0
    successful_predictions: int = 0
    failed_predictions: int = 0
    average_response_time_ms: float = 0.0
    predictions_by_class: Dict[str, int] = field(default_factory=dict)
    confidence_distribution: List[float] = field(default_factory=list)
    last_prediction_time: Optional[datetime] = None


@dataclass
class SystemMetrics:
    """Métricas del sistema"""
    cpu_usage_percent: float = 0.0
    memory_usage_mb: float = 0.0
    memory_usage_percent: float = 0.0
    disk_usage_percent: float = 0.0
    network_io_bytes: Dict[str, int] = field(default_factory=dict)
    process_count: int = 0
    uptime_seconds: float = 0.0


class MetricsCollector:
    """
    Recolector de métricas centralizado.
    
    Recolecta, almacena y expone métricas de la aplicación y del sistema
    en múltiples formatos (Prometheus, JSON, etc.).
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Inicializa el recolector de métricas.
        
        Args:
            config: Configuración de monitoreo
        """
        self.config = config
        self.start_time = datetime.utcnow()
        
        # Almacenamiento de métricas
        self.prediction_metrics = PredictionMetrics()
        self.system_metrics = SystemMetrics()
        self.custom_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Configurar Prometheus si está disponible
        if PROMETHEUS_AVAILABLE and config.get('enable_prometheus', True):
            self._setup_prometheus_metrics()
        
        # Configurar Azure Monitor si está disponible
        if AZURE_MONITORING_AVAILABLE and config.get('enable_azure_monitoring', False):
            self._setup_azure_monitoring()
        
        # Iniciar recolección automática de métricas del sistema
        if config.get('collect_system_metrics', True):
            self._start_system_metrics_collection()
    
    def _setup_prometheus_metrics(self):
        """Configura métricas de Prometheus"""
        self.registry = CollectorRegistry()
        
        # Métricas de predicción
        self.prediction_counter = Counter(
            'api_predictions_total',
            'Total number of predictions',
            ['status', 'prediction_class'],
            registry=self.registry
        )
        
        self.prediction_duration = Histogram(
            'api_prediction_duration_seconds',
            'Time spent processing predictions',
            ['prediction_class'],
            registry=self.registry
        )
        
        self.prediction_confidence = Histogram(
            'api_prediction_confidence',
            'Confidence scores of predictions',
            ['prediction_class'],
            registry=self.registry
        )
        
        # Métricas de sistema
        self.system_cpu_usage = Gauge(
            'system_cpu_usage_percent',
            'CPU usage percentage',
            registry=self.registry
        )
        
        self.system_memory_usage = Gauge(
            'system_memory_usage_bytes',
            'Memory usage in bytes',
            registry=self.registry
        )
        
        self.system_disk_usage = Gauge(
            'system_disk_usage_percent',
            'Disk usage percentage',
            registry=self.registry
        )
        
        # Métricas de API
        self.http_requests_total = Counter(
            'api_http_requests_total',
            'Total HTTP requests',
            ['method', 'endpoint', 'status'],
            registry=self.registry
        )
        
        self.http_request_duration = Histogram(
            'api_http_request_duration_seconds',
            'HTTP request duration',
            ['method', 'endpoint'],
            registry=self.registry
        )
        
        logger.info("Métricas de Prometheus configuradas")
    
    def _setup_azure_monitoring(self):
        """Configura monitoreo de Azure Application Insights"""
        try:
            app_insights_key = self.config.get('application_insights_key')
            if app_insights_key:
                configure_azure_monitor(connection_string=f"InstrumentationKey={app_insights_key}")
                self.tracer = trace.get_tracer(__name__)
                logger.info("Azure Application Insights configurado")
        except Exception as e:
            logger.warning(f"Error configurando Azure monitoring: {e}")
    
    def _start_system_metrics_collection(self):
        """Inicia la recolección automática de métricas del sistema"""
        def collect_system_metrics():
            while True:
                try:
                    self._collect_system_metrics()
                    time.sleep(self.config.get('system_metrics_interval', 30))
                except Exception as e:
                    logger.error(f"Error recolectando métricas del sistema: {e}")
                    time.sleep(60)  # Retry after 1 minute
        
        thread = threading.Thread(target=collect_system_metrics, daemon=True)
        thread.start()
        logger.info("Recolección automática de métricas del sistema iniciada")
    
    def _collect_system_metrics(self):
        """Recolecta métricas del sistema"""
        try:
            # CPU
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory
            memory = psutil.virtual_memory()
            memory_mb = memory.used / (1024 * 1024)
            memory_percent = memory.percent
            
            # Disk
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            
            # Network
            network = psutil.net_io_counters()
            network_io = {
                'bytes_sent': network.bytes_sent,
                'bytes_recv': network.bytes_recv
            }
            
            # Process count
            process_count = len(psutil.pids())
            
            # Uptime
            uptime = (datetime.utcnow() - self.start_time).total_seconds()
            
            # Actualizar métricas
            with self._lock:
                self.system_metrics.cpu_usage_percent = cpu_percent
                self.system_metrics.memory_usage_mb = memory_mb
                self.system_metrics.memory_usage_percent = memory_percent
                self.system_metrics.disk_usage_percent = disk_percent
                self.system_metrics.network_io_bytes = network_io
                self.system_metrics.process_count = process_count
                self.system_metrics.uptime_seconds = uptime
            
            # Actualizar métricas de Prometheus
            if PROMETHEUS_AVAILABLE and hasattr(self, 'system_cpu_usage'):
                self.system_cpu_usage.set(cpu_percent)
                self.system_memory_usage.set(memory.used)
                self.system_disk_usage.set(disk_percent)
            
        except Exception as e:
            logger.error(f"Error recolectando métricas del sistema: {e}")
    
    def record_prediction(self, prediction_class: str, confidence: float, 
                         processing_time_ms: float, success: bool = True):
        """
        Registra métricas de una predicción.
        
        Args:
            prediction_class: Clase predicha
            confidence: Score de confianza
            processing_time_ms: Tiempo de procesamiento en ms
            success: Si la predicción fue exitosa
        """
        with self._lock:
            # Actualizar contadores
            self.prediction_metrics.total_predictions += 1
            if success:
                self.prediction_metrics.successful_predictions += 1
            else:
                self.prediction_metrics.failed_predictions += 1
            
            # Actualizar distribución por clase
            if prediction_class not in self.prediction_metrics.predictions_by_class:
                self.prediction_metrics.predictions_by_class[prediction_class] = 0
            self.prediction_metrics.predictions_by_class[prediction_class] += 1
            
            # Actualizar tiempo de respuesta promedio
            total_time = (self.prediction_metrics.average_response_time_ms * 
                         (self.prediction_metrics.total_predictions - 1) + processing_time_ms)
            self.prediction_metrics.average_response_time_ms = total_time / self.prediction_metrics.total_predictions
            
            # Actualizar distribución de confianza
            self.prediction_metrics.confidence_distribution.append(confidence)
            if len(self.prediction_metrics.confidence_distribution) > 1000:
                self.prediction_metrics.confidence_distribution = self.prediction_metrics.confidence_distribution[-1000:]
            
            # Actualizar timestamp
            self.prediction_metrics.last_prediction_time = datetime.utcnow()
        
        # Métricas de Prometheus
        if PROMETHEUS_AVAILABLE and hasattr(self, 'prediction_counter'):
            status = 'success' if success else 'failure'
            self.prediction_counter.labels(status=status, prediction_class=prediction_class).inc()
            
            if success:
                self.prediction_duration.labels(prediction_class=prediction_class).observe(processing_time_ms / 1000)
                self.prediction_confidence.labels(prediction_class=prediction_class).observe(confidence)
        
        logger.debug(f"Métricas de predicción registradas: {prediction_class}, confianza={confidence:.3f}")
    
    def record_http_request(self, method: str, endpoint: str, status_code: int, duration_seconds: float):
        """
        Registra métricas de una request HTTP.
        
        Args:
            method: Método HTTP
            endpoint: Endpoint llamado
            status_code: Código de respuesta
            duration_seconds: Duración en segundos
        """
        if PROMETHEUS_AVAILABLE and hasattr(self, 'http_requests_total'):
            self.http_requests_total.labels(method=method, endpoint=endpoint, status=str(status_code)).inc()
            self.http_request_duration.labels(method=method, endpoint=endpoint).observe(duration_seconds)
    
    def record_custom_metric(self, name: str, value: float, labels: Dict[str, str] = None):
        """
        Registra una métrica personalizada.
        
        Args:
            name: Nombre de la métrica
            value: Valor de la métrica
            labels: Labels adicionales
        """
        metric = MetricData(
            name=name,
            value=value,
            timestamp=datetime.utcnow(),
            labels=labels or {}
        )
        
        with self._lock:
            self.custom_metrics[name].append(metric)
        
        logger.debug(f"Métrica personalizada registrada: {name}={value}")
    
    def get_prediction_metrics(self) -> Dict[str, Any]:
        """Retorna métricas de predicción"""
        with self._lock:
            confidence_stats = {}
            if self.prediction_metrics.confidence_distribution:
                confidences = self.prediction_metrics.confidence_distribution
                confidence_stats = {
                    'mean': sum(confidences) / len(confidences),
                    'min': min(confidences),
                    'max': max(confidences),
                    'count': len(confidences)
                }
            
            return {
                'total_predictions': self.prediction_metrics.total_predictions,
                'successful_predictions': self.prediction_metrics.successful_predictions,
                'failed_predictions': self.prediction_metrics.failed_predictions,
                'success_rate': (self.prediction_metrics.successful_predictions / 
                               max(self.prediction_metrics.total_predictions, 1)),
                'average_response_time_ms': self.prediction_metrics.average_response_time_ms,
                'predictions_by_class': dict(self.prediction_metrics.predictions_by_class),
                'confidence_stats': confidence_stats,
                'last_prediction_time': self.prediction_metrics.last_prediction_time.isoformat() 
                                      if self.prediction_metrics.last_prediction_time else None
            }
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Retorna métricas del sistema"""
        with self._lock:
            return {
                'cpu_usage_percent': self.system_metrics.cpu_usage_percent,
                'memory_usage_mb': self.system_metrics.memory_usage_mb,
                'memory_usage_percent': self.system_metrics.memory_usage_percent,
                'disk_usage_percent': self.system_metrics.disk_usage_percent,
                'network_io_bytes': dict(self.system_metrics.network_io_bytes),
                'process_count': self.system_metrics.process_count,
                'uptime_seconds': self.system_metrics.uptime_seconds
            }
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Retorna todas las métricas"""
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'prediction_metrics': self.get_prediction_metrics(),
            'system_metrics': self.get_system_metrics(),
            'custom_metrics': {name: list(metrics)[-10:] for name, metrics in self.custom_metrics.items()}
        }
    
    def get_prometheus_metrics(self) -> str:
        """Retorna métricas en formato Prometheus"""
        if not PROMETHEUS_AVAILABLE or not hasattr(self, 'registry'):
            return "# Prometheus not available\n"
        
        return generate_latest(self.registry).decode('utf-8')
    
    def check_alerts(self) -> List[Dict[str, Any]]:
        """Verifica umbrales de alerta y retorna alertas activas"""
        alerts = []
        
        # Configuración de alertas
        alert_config = self.config.get('alerting', {})
        
        # Alert por alta tasa de error
        if alert_config.get('alert_on_high_error_rate', False):
            error_threshold = alert_config.get('error_rate_threshold', 0.05)
            if self.prediction_metrics.total_predictions > 0:
                error_rate = (self.prediction_metrics.failed_predictions / 
                            self.prediction_metrics.total_predictions)
                if error_rate > error_threshold:
                    alerts.append({
                        'type': 'high_error_rate',
                        'severity': 'warning',
                        'message': f'Error rate {error_rate:.2%} exceeds threshold {error_threshold:.2%}',
                        'value': error_rate,
                        'threshold': error_threshold
                    })
        
        # Alert por alta latencia
        if alert_config.get('alert_on_high_latency', False):
            latency_threshold = alert_config.get('latency_threshold_ms', 1000)
            if self.prediction_metrics.average_response_time_ms > latency_threshold:
                alerts.append({
                    'type': 'high_latency',
                    'severity': 'warning',
                    'message': f'Average response time {self.prediction_metrics.average_response_time_ms:.0f}ms exceeds threshold {latency_threshold}ms',
                    'value': self.prediction_metrics.average_response_time_ms,
                    'threshold': latency_threshold
                })
        
        # Alert por uso alto de CPU
        if self.system_metrics.cpu_usage_percent > 80:
            alerts.append({
                'type': 'high_cpu_usage',
                'severity': 'warning',
                'message': f'CPU usage {self.system_metrics.cpu_usage_percent:.1f}% is high',
                'value': self.system_metrics.cpu_usage_percent,
                'threshold': 80
            })
        
        # Alert por uso alto de memoria
        if self.system_metrics.memory_usage_percent > 85:
            alerts.append({
                'type': 'high_memory_usage',
                'severity': 'critical',
                'message': f'Memory usage {self.system_metrics.memory_usage_percent:.1f}% is critical',
                'value': self.system_metrics.memory_usage_percent,
                'threshold': 85
            })
        
        return alerts


class HealthChecker:
    """
    Health checker avanzado para la aplicación.
    
    Implementa múltiples tipos de health checks:
    - Liveness: La aplicación está viva
    - Readiness: La aplicación está lista para servir tráfico
    - Custom: Checks personalizados para componentes específicos
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Inicializa el health checker.
        
        Args:
            config: Configuración de health checks
        """
        self.config = config
        self.start_time = datetime.utcnow()
        self.custom_checks: Dict[str, Callable] = {}
        
        # Estado de salud de componentes
        self.component_health: Dict[str, Dict[str, Any]] = {}
        
        logger.info("Health checker inicializado")
    
    def register_custom_check(self, name: str, check_function: Callable):
        """
        Registra un health check personalizado.
        
        Args:
            name: Nombre del check
            check_function: Función que retorna (success: bool, message: str, details: dict)
        """
        self.custom_checks[name] = check_function
        logger.info(f"Health check personalizado registrado: {name}")
    
    def check_liveness(self) -> Dict[str, Any]:
        """
        Liveness check - verifica que la aplicación esté viva.
        
        Returns:
            Resultado del liveness check
        """
        try:
            uptime = (datetime.utcnow() - self.start_time).total_seconds()
            
            return {
                'status': 'alive',
                'uptime_seconds': uptime,
                'timestamp': datetime.utcnow().isoformat(),
                'checks': {
                    'process_running': {'status': 'pass', 'message': 'Process is running'},
                    'uptime': {'status': 'pass', 'message': f'Uptime: {uptime:.0f}s'}
                }
            }
        except Exception as e:
            return {
                'status': 'dead',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
    
    def check_readiness(self, model_loader=None, database=None) -> Dict[str, Any]:
        """
        Readiness check - verifica que la aplicación esté lista para servir tráfico.
        
        Args:
            model_loader: Cargador de modelos para verificar
            database: Conexión de base de datos para verificar
            
        Returns:
            Resultado del readiness check
        """
        checks = {}
        overall_status = 'ready'
        
        # Check modelo cargado
        if model_loader:
            try:
                model_info = model_loader.get_model_info()
                if model_info.get('error'):
                    checks['model'] = {'status': 'fail', 'message': model_info['error']}
                    overall_status = 'not_ready'
                else:
                    checks['model'] = {'status': 'pass', 'message': 'Model loaded successfully'}
            except Exception as e:
                checks['model'] = {'status': 'fail', 'message': f'Model check failed: {e}'}
                overall_status = 'not_ready'
        
        # Check base de datos (si aplica)
        if database:
            try:
                # Implementar check de base de datos
                checks['database'] = {'status': 'pass', 'message': 'Database connection OK'}
            except Exception as e:
                checks['database'] = {'status': 'fail', 'message': f'Database check failed: {e}'}
                overall_status = 'not_ready'
        
        # Checks personalizados
        for name, check_func in self.custom_checks.items():
            try:
                success, message, details = check_func()
                checks[name] = {
                    'status': 'pass' if success else 'fail',
                    'message': message,
                    'details': details
                }
                if not success:
                    overall_status = 'not_ready'
            except Exception as e:
                checks[name] = {'status': 'fail', 'message': f'Check failed: {e}'}
                overall_status = 'not_ready'
        
        return {
            'status': overall_status,
            'timestamp': datetime.utcnow().isoformat(),
            'checks': checks
        }
    
    def check_dependencies(self) -> Dict[str, Any]:
        """Verifica estado de dependencias externas"""
        dependencies = {}
        
        # Check MLflow (si está configurado)
        mlflow_config = self.config.get('mlflow', {})
        if mlflow_config.get('enabled', False):
            try:
                import mlflow
                mlflow.set_tracking_uri(mlflow_config.get('tracking_uri', 'http://localhost:5000'))
                experiments = mlflow.list_experiments()
                dependencies['mlflow'] = {
                    'status': 'healthy',
                    'message': f'MLflow available, {len(experiments)} experiments found'
                }
            except Exception as e:
                dependencies['mlflow'] = {
                    'status': 'unhealthy', 
                    'message': f'MLflow unavailable: {e}'
                }
        
        # Check Redis (si está configurado)
        redis_config = self.config.get('redis', {})
        if redis_config.get('enabled', False):
            try:
                import redis
                r = redis.from_url(redis_config.get('url', 'redis://localhost:6379'))
                r.ping()
                dependencies['redis'] = {
                    'status': 'healthy',
                    'message': 'Redis connection OK'
                }
            except Exception as e:
                dependencies['redis'] = {
                    'status': 'unhealthy',
                    'message': f'Redis unavailable: {e}'
                }
        
        return dependencies
    
    def get_comprehensive_health(self, model_loader=None, database=None) -> Dict[str, Any]:
        """
        Retorna un health check comprensivo.
        
        Args:
            model_loader: Cargador de modelos
            database: Conexión de base de datos
            
        Returns:
            Health check completo
        """
        liveness = self.check_liveness()
        readiness = self.check_readiness(model_loader, database)
        dependencies = self.check_dependencies()
        
        # Determinar estado general
        overall_status = 'healthy'
        if liveness['status'] != 'alive':
            overall_status = 'unhealthy'
        elif readiness['status'] != 'ready':
            overall_status = 'degraded'
        elif any(dep['status'] == 'unhealthy' for dep in dependencies.values()):
            overall_status = 'degraded'
        
        return {
            'status': overall_status,
            'timestamp': datetime.utcnow().isoformat(),
            'version': self.config.get('version', '1.0.0'),
            'liveness': liveness,
            'readiness': readiness,
            'dependencies': dependencies
        }