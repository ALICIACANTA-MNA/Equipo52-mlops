"""
Utilidades para gestión de errores y validación en la API.

Implementa manejo robusto de errores, validación avanzada y logging estructurado
para facilitar debugging y monitoreo en producción:
- Exception handlers personalizados
- Validadores médicos específicos del dominio
- Logging contextual para trazabilidad
- Rate limiting y throttling
- Sanitización de datos de entrada

Referencias:
- FastAPI Error Handling: https://fastapi.tiangolo.com/tutorial/handling-errors/
- Pydantic Validators: https://docs.pydantic.dev/latest/concepts/validators/
- Python Logging Best Practices: https://docs.python.org/3/howto/logging.html
"""

import re
import time
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
from collections import defaultdict, deque
from contextlib import contextmanager
import traceback
import functools

# FastAPI y Pydantic
try:
    from fastapi import HTTPException, Request, Response
    from fastapi.responses import JSONResponse
    from fastapi.exceptions import RequestValidationError
    from pydantic import ValidationError, validator
    from starlette.exceptions import HTTPException as StarletteHTTPException
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

# Utils propios
from src.utils.logging_config import get_logger


logger = get_logger(__name__)


# Excepciones personalizadas
class APIError(Exception):
    """Excepción base para errores de API"""
    def __init__(self, message: str, error_code: str = "API_ERROR", 
                 status_code: int = 500, details: Dict = None):
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)


class ValidationError(APIError):
    """Error de validación de datos"""
    def __init__(self, message: str, field: str = None, value: Any = None):
        super().__init__(
            message=message,
            error_code="VALIDATION_ERROR",
            status_code=400,
            details={"field": field, "value": str(value) if value is not None else None}
        )


class ModelError(APIError):
    """Error relacionado con el modelo ML"""
    def __init__(self, message: str, model_name: str = None):
        super().__init__(
            message=message,
            error_code="MODEL_ERROR",
            status_code=503,
            details={"model_name": model_name}
        )


class RateLimitError(APIError):
    """Error de rate limiting"""
    def __init__(self, message: str, retry_after: int = None):
        super().__init__(
            message=message,
            error_code="RATE_LIMIT_ERROR",
            status_code=429,
            details={"retry_after": retry_after}
        )


class DataSecurityError(APIError):
    """Error de seguridad de datos"""
    def __init__(self, message: str, security_issue: str = None):
        super().__init__(
            message=message,
            error_code="SECURITY_ERROR",
            status_code=400,
            details={"security_issue": security_issue}
        )


# Validadores médicos específicos del dominio
class MedicalValidators:
    """
    Validadores específicos para datos médicos y antropométricos.
    
    Implementa validaciones basadas en rangos fisiológicos establecidos
    y guidelines médicos para prevenir predicciones con datos inválidos.
    """
    
    # Rangos normales basados en literatura médica
    NORMAL_RANGES = {
        'age': {'min': 14, 'max': 80, 'unit': 'years'},
        'height': {'min': 1.40, 'max': 2.10, 'unit': 'meters'},
        'weight': {'min': 35, 'max': 200, 'unit': 'kg'},
        'bmi': {'min': 12, 'max': 60, 'unit': 'kg/m²'},
        'water_intake': {'min': 1, 'max': 3, 'unit': 'liters'},
        'vegetable_frequency': {'min': 1, 'max': 3, 'unit': 'times/day'},
        'main_meals': {'min': 1, 'max': 4, 'unit': 'meals/day'},
        'physical_activity': {'min': 0, 'max': 3, 'unit': 'frequency'},
        'technology_use': {'min': 0, 'max': 2, 'unit': 'hours'}
    }
    
    # Patrones sospechosos que pueden indicar ataques
    SUSPICIOUS_PATTERNS = [
        r'<script.*?>.*?</script>',  # XSS
        r'union\s+select',  # SQL injection
        r'javascript:',  # JavaScript injection
        r'data:text/html',  # Data URI XSS
        r'vbscript:',  # VBScript injection
    ]
    
    @staticmethod
    def validate_age(age: float, context: Dict = None) -> Tuple[bool, str]:
        """
        Valida edad del paciente.
        
        Args:
            age: Edad en años
            context: Contexto adicional para validación
            
        Returns:
            Tupla (es_válido, mensaje_error)
        """
        ranges = MedicalValidators.NORMAL_RANGES['age']
        
        if not isinstance(age, (int, float)):
            return False, "Edad debe ser un número"
        
        if age < ranges['min'] or age > ranges['max']:
            return False, f"Edad debe estar entre {ranges['min']} y {ranges['max']} años"
        
        # Validaciones contextuales
        if context:
            # Adolescentes con ciertos factores de riesgo
            if age < 18 and context.get('family_history_with_overweight') == 'yes':
                logger.warning(f"Adolescente (edad {age}) con historial familiar positivo")
            
            # Adultos mayores con alta actividad física poco común
            if age > 65 and context.get('FAF', 0) > 2:
                logger.warning(f"Adulto mayor (edad {age}) con alta frecuencia de ejercicio")
        
        return True, ""
    
    @staticmethod
    def validate_anthropometric_data(height: float, weight: float) -> Tuple[bool, str]:
        """
        Valida datos antropométricos y su consistencia.
        
        Args:
            height: Altura en metros
            weight: Peso en kg
            
        Returns:
            Tupla (es_válido, mensaje_error)
        """
        # Validar altura
        height_ranges = MedicalValidators.NORMAL_RANGES['height']
        if height < height_ranges['min'] or height > height_ranges['max']:
            return False, f"Altura debe estar entre {height_ranges['min']} y {height_ranges['max']} metros"
        
        # Validar peso
        weight_ranges = MedicalValidators.NORMAL_RANGES['weight']
        if weight < weight_ranges['min'] or weight > weight_ranges['max']:
            return False, f"Peso debe estar entre {weight_ranges['min']} y {weight_ranges['max']} kg"
        
        # Validar BMI calculado
        bmi = weight / (height ** 2)
        bmi_ranges = MedicalValidators.NORMAL_RANGES['bmi']
        if bmi < bmi_ranges['min'] or bmi > bmi_ranges['max']:
            return False, f"BMI calculado ({bmi:.1f}) está fuera del rango fisiológico ({bmi_ranges['min']}-{bmi_ranges['max']})"
        
        return True, ""
    
    @staticmethod
    def validate_lifestyle_consistency(data: Dict) -> Tuple[bool, str]:
        """
        Valida consistencia en datos de estilo de vida.
        
        Args:
            data: Diccionario con datos del paciente
            
        Returns:
            Tupla (es_válido, mensaje_error)
        """
        warnings = []
        
        # Verificar consistencia entre actividad física y transporte
        faf = data.get('FAF', 0)  # Frecuencia actividad física
        transport = data.get('MTRANS', '')
        
        if faf == 0 and transport == 'Walking':
            warnings.append("Inconsistencia: no hace ejercicio pero camina como transporte principal")
        
        if faf >= 2 and transport == 'Automobile':
            warnings.append("Alta actividad física pero usa automóvil como transporte principal")
        
        # Verificar consistencia en hábitos alimenticios
        favc = data.get('FAVC', 'no')  # Consume alimentos calóricos frecuentemente
        scc = data.get('SCC', 'no')   # Monitorea consumo de calorías
        
        if favc == 'yes' and scc == 'yes':
            warnings.append("Consume alimentos calóricos frecuentemente pero monitorea calorías")
        
        # Verificar consistencia en consumo de agua y alcohol
        ch2o = data.get('CH2O', 0)    # Consumo de agua
        calc = data.get('CALC', 'no') # Consumo de alcohol
        
        if ch2o <= 1 and calc in ['Frequently', 'Always']:
            warnings.append("Bajo consumo de agua pero alto consumo de alcohol")
        
        # Log warnings pero no rechazar
        for warning in warnings:
            logger.warning(f"Inconsistencia en datos de estilo de vida: {warning}")
        
        return True, ""  # Por ahora no rechazamos por inconsistencias
    
    @staticmethod
    def detect_suspicious_input(data: Dict) -> Tuple[bool, str]:
        """
        Detecta patrones sospechosos en la entrada que podrían indicar ataques.
        
        Args:
            data: Diccionario con datos de entrada
            
        Returns:
            Tupla (es_seguro, mensaje_error)
        """
        for field, value in data.items():
            if not isinstance(value, str):
                continue
            
            # Verificar patrones sospechosos
            for pattern in MedicalValidators.SUSPICIOUS_PATTERNS:
                if re.search(pattern, value, re.IGNORECASE):
                    logger.warning(f"Patrón sospechoso detectado en campo {field}: {pattern}")
                    return False, f"Entrada potencialmente maliciosa detectada en campo {field}"
            
            # Verificar longitud excesiva
            if len(value) > 100:  # Límite razonable para nuestros campos
                return False, f"Campo {field} excede la longitud máxima permitida"
            
            # Verificar caracteres no válidos
            if field in ['Gender', 'MTRANS', 'CAEC', 'CALC'] and not value.replace('_', '').isalnum():
                return False, f"Campo {field} contiene caracteres no válidos"
        
        return True, ""
    
    @staticmethod
    def sanitize_input(value: Any, field_type: str = "string") -> Any:
        """
        Sanitiza entrada de usuario.
        
        Args:
            value: Valor a sanitizar
            field_type: Tipo de campo (string, numeric, boolean)
            
        Returns:
            Valor sanitizado
        """
        if value is None:
            return None
        
        if field_type == "string":
            # Convertir a string y limpiar
            clean_value = str(value).strip()
            
            # Remover caracteres de control
            clean_value = ''.join(char for char in clean_value if ord(char) >= 32 or char in '\t\n\r')
            
            # Escapar caracteres HTML
            clean_value = (clean_value
                          .replace('&', '&amp;')
                          .replace('<', '&lt;')
                          .replace('>', '&gt;')
                          .replace('"', '&quot;')
                          .replace("'", '&#x27;'))
            
            return clean_value
        
        elif field_type == "numeric":
            try:
                return float(value)
            except (ValueError, TypeError):
                raise ValidationError(f"Valor debe ser numérico: {value}")
        
        elif field_type == "boolean":
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                return value.lower() in ['true', '1', 'yes', 'on']
            return bool(value)
        
        return value


class RateLimiter:
    """
    Rate limiter simple basado en memoria.
    
    Implementa rate limiting por IP y por usuario para prevenir abuso
    del API y proteger los recursos del modelo.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Inicializa el rate limiter.
        
        Args:
            config: Configuración de rate limiting
        """
        self.config = config
        self.requests: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.blocked_ips: Dict[str, datetime] = {}
        
        # Configuración
        self.max_requests = config.get('rate_limit_requests', 100)
        self.window_seconds = config.get('rate_limit_window', 60)
        self.block_duration = config.get('block_duration_seconds', 300)  # 5 minutos
        
        logger.info(f"Rate limiter configurado: {self.max_requests} requests/{self.window_seconds}s")
    
    def is_allowed(self, client_id: str) -> Tuple[bool, int]:
        """
        Verifica si una request está permitida.
        
        Args:
            client_id: ID del cliente (IP, user ID, etc.)
            
        Returns:
            Tupla (permitido, retry_after_seconds)
        """
        now = datetime.utcnow()
        
        # Verificar si está bloqueado
        if client_id in self.blocked_ips:
            unblock_time = self.blocked_ips[client_id]
            if now < unblock_time:
                retry_after = int((unblock_time - now).total_seconds())
                return False, retry_after
            else:
                # Desbloquear
                del self.blocked_ips[client_id]
        
        # Limpiar requests antigas
        cutoff_time = now - timedelta(seconds=self.window_seconds)
        client_requests = self.requests[client_id]
        
        while client_requests and client_requests[0] < cutoff_time:
            client_requests.popleft()
        
        # Verificar límite
        if len(client_requests) >= self.max_requests:
            # Bloquear cliente
            self.blocked_ips[client_id] = now + timedelta(seconds=self.block_duration)
            logger.warning(f"Cliente bloqueado por rate limit: {client_id}")
            return False, self.block_duration
        
        # Registrar request
        client_requests.append(now)
        return True, 0
    
    def get_client_stats(self, client_id: str) -> Dict[str, Any]:
        """Retorna estadísticas del cliente"""
        client_requests = self.requests[client_id]
        now = datetime.utcnow()
        
        # Contar requests en la ventana actual
        cutoff_time = now - timedelta(seconds=self.window_seconds)
        current_requests = sum(1 for req_time in client_requests if req_time >= cutoff_time)
        
        return {
            'client_id': client_id,
            'requests_in_window': current_requests,
            'max_requests': self.max_requests,
            'window_seconds': self.window_seconds,
            'is_blocked': client_id in self.blocked_ips,
            'remaining_requests': max(0, self.max_requests - current_requests)
        }


class ErrorHandler:
    """
    Gestor centralizado de errores para la API.
    
    Maneja excepciones, logging estructurado y respuestas consistentes
    para facilitar debugging y proporcionar UX adecuada.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Inicializa el gestor de errores.
        
        Args:
            config: Configuración de manejo de errores
        """
        self.config = config
        self.error_counts: Dict[str, int] = defaultdict(int)
        self.recent_errors: deque = deque(maxlen=100)
        
        # Configuración
        self.log_stack_traces = config.get('log_stack_traces', True)
        self.include_details_in_response = config.get('include_error_details', False)
        self.mask_sensitive_data = config.get('mask_sensitive_data', True)
    
    def handle_api_error(self, error: APIError, request: Optional[Any] = None) -> Dict[str, Any]:
        """
        Maneja errores de API personalizados.
        
        Args:
            error: Excepción APIError
            request: Request HTTP (opcional)
            
        Returns:
            Diccionario con respuesta de error
        """
        # Logging
        error_context = {
            'error_code': error.error_code,
            'status_code': error.status_code,
            'message': error.message,
            'details': error.details
        }
        
        if request:
            error_context.update({
                'method': getattr(request, 'method', 'unknown'),
                'url': str(getattr(request, 'url', 'unknown')),
                'client_ip': self._get_client_ip(request)
            })
        
        logger.error(f"API Error: {error.error_code}", extra=error_context)
        
        # Actualizar estadísticas
        self.error_counts[error.error_code] += 1
        self.recent_errors.append({
            'timestamp': datetime.utcnow().isoformat(),
            'error_code': error.error_code,
            'message': error.message
        })
        
        # Construir respuesta
        response = {
            'error': True,
            'error_code': error.error_code,
            'message': error.message,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Incluir detalles si está configurado
        if self.include_details_in_response and error.details:
            response['details'] = error.details
        
        return response
    
    def handle_validation_error(self, error: Union[ValidationError, RequestValidationError], 
                              request: Optional[Any] = None) -> Dict[str, Any]:
        """
        Maneja errores de validación de Pydantic/FastAPI.
        
        Args:
            error: Error de validación
            request: Request HTTP (opcional)
            
        Returns:
            Diccionario con respuesta de error
        """
        if isinstance(error, RequestValidationError):
            # Error de FastAPI/Pydantic
            errors = []
            for pydantic_error in error.errors():
                field = '.'.join(str(loc) for loc in pydantic_error['loc'])
                errors.append({
                    'field': field,
                    'message': pydantic_error['msg'],
                    'type': pydantic_error['type'],
                    'input': str(pydantic_error.get('input', ''))
                })
            
            message = f"Errores de validación en {len(errors)} campo(s)"
        else:
            # Error personalizado
            errors = [{'field': error.details.get('field'), 'message': error.message}]
            message = error.message
        
        # Logging
        logger.warning(f"Validation error: {message}", extra={'errors': errors})
        
        # Respuesta
        return {
            'error': True,
            'error_code': 'VALIDATION_ERROR',
            'message': message,
            'validation_errors': errors,
            'timestamp': datetime.utcnow().isoformat()
        }
    
    def handle_unexpected_error(self, error: Exception, request: Optional[Any] = None) -> Dict[str, Any]:
        """
        Maneja errores inesperados.
        
        Args:
            error: Excepción no manejada
            request: Request HTTP (opcional)
            
        Returns:
            Diccionario con respuesta de error
        """
        # ID único para el error
        error_id = hashlib.md5(f"{datetime.utcnow().isoformat()}{str(error)}".encode()).hexdigest()[:8]
        
        # Logging completo
        error_context = {
            'error_id': error_id,
            'error_type': type(error).__name__,
            'error_message': str(error)
        }
        
        if request:
            error_context.update({
                'method': getattr(request, 'method', 'unknown'),
                'url': str(getattr(request, 'url', 'unknown')),
                'client_ip': self._get_client_ip(request)
            })
        
        if self.log_stack_traces:
            error_context['stack_trace'] = traceback.format_exc()
        
        logger.error(f"Unexpected error: {type(error).__name__}", extra=error_context)
        
        # Actualizar estadísticas
        self.error_counts['INTERNAL_ERROR'] += 1
        self.recent_errors.append({
            'timestamp': datetime.utcnow().isoformat(),
            'error_id': error_id,
            'error_type': type(error).__name__,
            'message': str(error)
        })
        
        # Respuesta genérica (sin exponer detalles internos)
        response = {
            'error': True,
            'error_code': 'INTERNAL_ERROR',
            'message': 'Error interno del servidor',
            'error_id': error_id,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # En desarrollo, incluir más detalles
        if self.config.get('debug_mode', False):
            response.update({
                'error_type': type(error).__name__,
                'error_message': str(error)
            })
        
        return response
    
    def _get_client_ip(self, request) -> str:
        """Extrae IP del cliente del request"""
        if not request:
            return 'unknown'
        
        # Verificar headers de proxy
        forwarded_for = getattr(request.headers, 'x-forwarded-for', None) if hasattr(request, 'headers') else None
        if forwarded_for:
            return forwarded_for.split(',')[0].strip()
        
        real_ip = getattr(request.headers, 'x-real-ip', None) if hasattr(request, 'headers') else None
        if real_ip:
            return real_ip
        
        # IP directa
        client = getattr(request, 'client', None)
        if client and hasattr(client, 'host'):
            return client.host
        
        return 'unknown'
    
    def get_error_stats(self) -> Dict[str, Any]:
        """Retorna estadísticas de errores"""
        return {
            'error_counts': dict(self.error_counts),
            'recent_errors': list(self.recent_errors),
            'total_errors': sum(self.error_counts.values())
        }


# Context manager para logging de operaciones
@contextmanager
def log_operation(operation_name: str, logger_instance=None, **kwargs):
    """
    Context manager para logging automático de operaciones.
    
    Args:
        operation_name: Nombre de la operación
        logger_instance: Logger a usar (por defecto el global)
        **kwargs: Contexto adicional para logging
    """
    log = logger_instance or logger
    start_time = time.time()
    
    # Log inicio
    log.info(f"Iniciando operación: {operation_name}", extra=kwargs)
    
    try:
        yield
        # Log éxito
        duration = time.time() - start_time
        log.info(f"Operación completada: {operation_name} ({duration:.3f}s)", 
                extra={**kwargs, 'duration_seconds': duration, 'status': 'success'})
    except Exception as e:
        # Log error
        duration = time.time() - start_time
        log.error(f"Operación falló: {operation_name} ({duration:.3f}s): {e}", 
                 extra={**kwargs, 'duration_seconds': duration, 'status': 'error', 'error': str(e)})
        raise


# Decorador para retry automático
def retry_on_failure(max_attempts: int = 3, delay: float = 1.0, 
                    backoff_multiplier: float = 2.0, 
                    exceptions: Tuple = (Exception,)):
    """
    Decorador para retry automático en caso de falla.
    
    Args:
        max_attempts: Máximo número de intentos
        delay: Delay inicial entre intentos
        backoff_multiplier: Multiplicador para backoff exponencial
        exceptions: Excepciones que activan el retry
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts - 1:  # No delay en el último intento
                        logger.warning(f"Intento {attempt + 1} falló para {func.__name__}: {e}. "
                                     f"Reintentando en {current_delay}s...")
                        time.sleep(current_delay)
                        current_delay *= backoff_multiplier
                    else:
                        logger.error(f"Todos los intentos fallaron para {func.__name__}: {e}")
            
            raise last_exception
        return wrapper
    return decorator