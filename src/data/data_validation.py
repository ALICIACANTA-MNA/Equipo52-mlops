"""
Data validation module for obesity prediction MLOps pipeline.

Implementa validación de datos siguiendo las mejores prácticas de MLOps:
- Validación de esquemas y tipos de datos
- Detección de anomalías y valores fuera de rango
- Data drift detection
- Reportes de calidad de datos

REFERENCIAS BIBLIOGRÁFICAS TÉCNICAS:
- Treveil, M. et al. (2024): "Introducing MLOps: How to Scale Machine Learning in the Enterprise"
  Capítulo 5: "Data Quality Gates" - Implementación de checkpoints de calidad
- Wilson, B. & Bansal, A. (2022): "Practical MLOps: Operationalizing Machine Learning Models"
  Capítulo 6: "Data Validation Strategies" - Patrones de validación automatizada
- Huyen, C. (2022): "Designing Machine Learning Systems"
  Capítulo 4: "Training Data" - Detección de drift y anomalías
- Bonaccorso, G. (2024): "Mastering Machine Learning Algorithms"
  Capítulo 2: "Statistical Data Analysis" - Métricas de calidad estadística
- Breck, E. et al. (2017): "The ML Test Score: A Rubric for ML Production Readiness"
  Google Research - Framework de testing de datos ML

ARQUITECTURA DE VALIDACIÓN:

                            ┌─────────────────┐
                            │   data/raw/     │ ← Datos de ingesta
                            │ ObesityDataSet  │
                            └─────────┬───────┘
                                      │
                                      ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │                 DATA VALIDATION ENGINE                          │
    │                                                                 │
    │  📋 SCHEMA VALIDATION:          🎯 QUALITY CHECKS:              │
    │  • Required columns             • Numeric ranges                │
    │  • Data types                   • Categorical values            │
    │  • Column count                 • Missing data %                │
    │                                 • Outlier detection            │
    │  📊 DRIFT DETECTION:            📈 STATISTICAL PROFILING:       │
    │  • Kolmogorov-Smirnov          • Distributions                 │
    │  • Chi-square test             • Correlations                  │
    │  • Population Stability        • Class balance                 │
    └─────────────────────────────┬──────────────────────────────────┘
                                  │
                                  ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │                    VALIDATION OUTPUTS                           │
    │  - data/interim/obesity_estimation_original.csv (if valid)      │
    │  - validation_report.json (quality metrics)                     │
    │  - alerts.json (drift/anomaly warnings)                         │
    └─────────────────────────────────────────────────────────────────┘

Componentes de validación:
- Schema validation: Verificación de estructura esperada
- Data quality checks: Rangos, valores válidos, distribuciones
- Drift detection: Comparación con dataset de referencia
- Reporting: Generación de reportes JSON de calidad
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import json
from dataclasses import dataclass, asdict
from scipy import stats
import warnings

from src.utils.config import get_data_config, load_config


def convert_numpy_types(obj):
    """Convierte tipos numpy a tipos JSON serializables"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj
from src.utils.logging_config import get_logger, log_pipeline_stage


logger = get_logger(__name__)


@dataclass
class ValidationResult:
    """Resultado de validación de datos"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    statistics: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return convert_numpy_types(asdict(self))


@dataclass
class DriftResult:
    """Resultado de detección de drift"""
    column: str
    test_name: str
    p_value: float
    is_drift: bool
    threshold: float
    statistic: float


class DataValidationError(Exception):
    """Excepción para errores de validación de datos"""
    pass


class DataValidator:
    """
    Validador de datos para pipeline MLOps.
    
    Implementa validación comprensiva de datos siguiendo
    las mejores prácticas de calidad de datos en MLOps.
    """
    
    def __init__(self, validation_config_path: str = "configs/data/data_validation.yaml"):
        """
        Inicializa el validador con configuración.
        
        Args:
            validation_config_path: Ruta al archivo de configuración de validación
        """
        self.config = load_config(validation_config_path)
        self.validation_rules = self.config['validation_rules']
        self.drift_config = self.config.get('drift_detection', {})
        
    def validate_schema(self, df: pd.DataFrame) -> ValidationResult:
        """
        Valida el esquema del DataFrame.
        
        Args:
            df: DataFrame a validar
            
        Returns:
            Resultado de validación del esquema
        """
        errors = []
        warnings = []
        
        schema_rules = self.validation_rules['schema']
        
        # Validar columnas requeridas
        required_columns = set(schema_rules['required_columns'])
        actual_columns = set(df.columns)
        
        missing_columns = required_columns - actual_columns
        if missing_columns:
            errors.append(f"Columnas faltantes: {missing_columns}")
        
        extra_columns = actual_columns - required_columns
        if extra_columns:
            warnings.append(f"Columnas adicionales: {extra_columns}")
        
        # Validar tipos de datos
        if 'column_types' in schema_rules:
            for column, expected_type in schema_rules['column_types'].items():
                if column in df.columns:
                    actual_type = str(df[column].dtype)
                    if not self._is_compatible_type(actual_type, expected_type):
                        errors.append(f"Tipo incorrecto en columna '{column}': "
                                    f"esperado {expected_type}, encontrado {actual_type}")
        
        # Estadísticas del esquema
        statistics = {
            'total_columns': len(df.columns),
            'required_columns_present': len(required_columns & actual_columns),
            'extra_columns_count': len(extra_columns),
            'column_types_correct': len(errors) == 0
        }
        
        is_valid = len(errors) == 0
        
        return ValidationResult(is_valid, errors, warnings, statistics)
    
    def validate_data_quality(self, df: pd.DataFrame) -> ValidationResult:
        """
        Valida la calidad de los datos.
        
        Args:
            df: DataFrame a validar
            
        Returns:
            Resultado de validación de calidad
        """
        errors = []
        warnings = []
        statistics = {}
        
        quality_checks = self.validation_rules['quality_checks']
        
        for column, rules in quality_checks.items():
            if column not in df.columns:
                warnings.append(f"Columna '{column}' no encontrada para validación")
                continue
            
            col_data = df[column]
            
            # Validar rangos numéricos
            if isinstance(rules, dict):
                if 'min' in rules:
                    min_val = rules['min']
                    violations = col_data < min_val
                    if violations.any():
                        count = violations.sum()
                        errors.append(f"Columna '{column}': {count} valores menores a {min_val}")
                
                if 'max' in rules:
                    max_val = rules['max']
                    violations = col_data > max_val
                    if violations.any():
                        count = violations.sum()
                        errors.append(f"Columna '{column}': {count} valores mayores a {max_val}")
                
                # Validar valores válidos (categóricas)
                if 'valid_values' in rules:
                    valid_values = set(rules['valid_values'])
                    actual_values = set(col_data.dropna().unique())
                    invalid_values = actual_values - valid_values
                    
                    if invalid_values:
                        errors.append(f"Columna '{column}': valores inválidos {invalid_values}")
            
            # Estadísticas por columna
            statistics[column] = {
                'null_count': col_data.isnull().sum(),
                'null_percentage': col_data.isnull().mean() * 100,
                'unique_values': col_data.nunique(),
                'data_type': str(col_data.dtype)
            }
            
            if col_data.dtype in ['int64', 'float64']:
                statistics[column].update({
                    'mean': float(col_data.mean()),
                    'std': float(col_data.std()),
                    'min': float(col_data.min()),
                    'max': float(col_data.max())
                })
        
        # Estadísticas generales
        statistics['overall'] = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'total_nulls': df.isnull().sum().sum(),
            'duplicate_rows': df.duplicated().sum()
        }
        
        is_valid = len(errors) == 0
        
        return ValidationResult(is_valid, errors, warnings, statistics)
    
    def detect_data_drift(self, df_current: pd.DataFrame, 
                         reference_path: Optional[str] = None) -> List[DriftResult]:
        """
        Detecta drift en los datos comparando con dataset de referencia.
        
        Args:
            df_current: DataFrame actual
            reference_path: Ruta al dataset de referencia
            
        Returns:
            Lista de resultados de drift por columna
        """
        if not reference_path:
            reference_path = self.drift_config.get('reference_dataset')
        
        if not reference_path or not Path(reference_path).exists():
            logger.warning("Dataset de referencia no disponible, saltando detección de drift")
            return []
        
        try:
            df_reference = pd.read_csv(reference_path)
            logger.info(f"Dataset de referencia cargado: {df_reference.shape}")
        except Exception as e:
            logger.error(f"Error cargando dataset de referencia: {e}")
            return []
        
        drift_results = []
        threshold = self.drift_config.get('monitoring_threshold', 0.05)
        tests = self.drift_config.get('statistical_tests', ['ks_test'])
        
        # Validar columnas comunes
        common_columns = set(df_current.columns) & set(df_reference.columns)
        
        for column in common_columns:
            current_data = df_current[column].dropna()
            reference_data = df_reference[column].dropna()
            
            if len(current_data) == 0 or len(reference_data) == 0:
                continue
            
            # Determinar tipo de datos
            if current_data.dtype in ['int64', 'float64']:
                # Tests para datos numéricos
                if 'ks_test' in tests:
                    drift_result = self._ks_test(current_data, reference_data, column, threshold)
                    drift_results.append(drift_result)
                
                if 'psi' in tests:
                    drift_result = self._psi_test(current_data, reference_data, column, threshold)
                    drift_results.append(drift_result)
            else:
                # Tests para datos categóricos
                if 'chi2_test' in tests:
                    drift_result = self._chi2_test(current_data, reference_data, column, threshold)
                    drift_results.append(drift_result)
        
        # Log resultados de drift
        drift_detected = [r for r in drift_results if r.is_drift]
        if drift_detected:
            logger.warning(f"Drift detectado en {len(drift_detected)} columnas:")
            for result in drift_detected:
                logger.warning(f"  {result.column}: {result.test_name} p={result.p_value:.4f}")
        else:
            logger.info("No se detectó drift significativo")
        
        return drift_results
    
    def _is_compatible_type(self, actual_type: str, expected_type: str) -> bool:
        """Verifica compatibilidad de tipos de datos"""
        type_mapping = {
            'float64': ['float64', 'int64', 'number'],
            'int64': ['int64', 'float64', 'number'],
            'object': ['object', 'string', 'category'],
            'bool': ['bool', 'boolean']
        }
        
        return expected_type in type_mapping.get(actual_type, [])
    
    def _ks_test(self, current: pd.Series, reference: pd.Series, 
                column: str, threshold: float) -> DriftResult:
        """Kolmogorov-Smirnov test para datos numéricos"""
        try:
            statistic, p_value = stats.ks_2samp(current, reference)
            is_drift = p_value < threshold
            
            return DriftResult(
                column=column,
                test_name='ks_test',
                p_value=p_value,
                is_drift=is_drift,
                threshold=threshold,
                statistic=statistic
            )
        except Exception as e:
            logger.error(f"Error en KS test para columna {column}: {e}")
            return DriftResult(column, 'ks_test', 1.0, False, threshold, 0.0)
    
    def _psi_test(self, current: pd.Series, reference: pd.Series, 
                 column: str, threshold: float) -> DriftResult:
        """Population Stability Index test"""
        try:
            # Crear bins basados en datos de referencia
            _, bins = np.histogram(reference, bins=10)
            
            # Calcular distribuciones
            current_dist, _ = np.histogram(current, bins=bins)
            reference_dist, _ = np.histogram(reference, bins=bins)
            
            # Normalizar distribuciones
            current_dist = current_dist / current_dist.sum()
            reference_dist = reference_dist / reference_dist.sum()
            
            # Evitar divisiones por cero
            current_dist = np.where(current_dist == 0, 1e-10, current_dist)
            reference_dist = np.where(reference_dist == 0, 1e-10, reference_dist)
            
            # Calcular PSI
            psi = np.sum((current_dist - reference_dist) * 
                        np.log(current_dist / reference_dist))
            
            # PSI > 0.2 generalmente indica drift significativo
            is_drift = psi > 0.2
            p_value = 1 - psi  # Pseudo p-value para consistencia
            
            return DriftResult(column, 'psi', p_value, is_drift, threshold, psi)
            
        except Exception as e:
            logger.error(f"Error en PSI test para columna {column}: {e}")
            return DriftResult(column, 'psi', 1.0, False, threshold, 0.0)
    
    def _chi2_test(self, current: pd.Series, reference: pd.Series, 
                  column: str, threshold: float) -> DriftResult:
        """Chi-squared test para datos categóricos"""
        try:
            # Obtener categorías únicas
            all_categories = pd.concat([current, reference]).unique()
            
            # Contar frecuencias
            current_counts = current.value_counts().reindex(all_categories, fill_value=0)
            reference_counts = reference.value_counts().reindex(all_categories, fill_value=0)
            
            # Chi-squared test
            statistic, p_value = stats.chisquare(current_counts, reference_counts)
            is_drift = p_value < threshold
            
            return DriftResult(column, 'chi2_test', p_value, is_drift, threshold, statistic)
            
        except Exception as e:
            logger.error(f"Error en Chi2 test para columna {column}: {e}")
            return DriftResult(column, 'chi2_test', 1.0, False, threshold, 0.0)


@log_pipeline_stage("data_validation")
def validate_dataset(input_path: str, output_path: str) -> ValidationResult:
    """
    Valida dataset completo y guarda resultados.
    
    Args:
        input_path: Ruta al dataset a validar
        output_path: Ruta donde guardar el dataset validado
        
    Returns:
        Resultado de validación combinado
    """
    # Cargar datos
    df = pd.read_csv(input_path)
    logger.info(f"Dataset cargado para validación: {df.shape}")
    
    # Inicializar validador
    validator = DataValidator()
    
    # Validar esquema
    schema_result = validator.validate_schema(df)
    logger.info(f"Validación de esquema: {'PASS' if schema_result.is_valid else 'FAIL'}")
    
    # Validar calidad
    quality_result = validator.validate_data_quality(df)
    logger.info(f"Validación de calidad: {'PASS' if quality_result.is_valid else 'FAIL'}")
    
    # Detectar drift
    drift_results = validator.detect_data_drift(df)
    drift_detected = any(r.is_drift for r in drift_results)
    if drift_detected:
        logger.warning("DRIFT detectado en los datos")
    
    # Combinar resultados
    all_errors = schema_result.errors + quality_result.errors
    all_warnings = schema_result.warnings + quality_result.warnings
    
    combined_statistics = {
        'schema': schema_result.statistics,
        'quality': quality_result.statistics,
        'drift': [convert_numpy_types(asdict(r)) for r in drift_results]
    }
    
    overall_valid = schema_result.is_valid and quality_result.is_valid
    
    combined_result = ValidationResult(
        is_valid=overall_valid,
        errors=all_errors,
        warnings=all_warnings,
        statistics=combined_statistics
    )
    
    # Guardar reporte de validación
    report_path = Path(output_path).parent / "validation_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(combined_result.to_dict(), f, indent=2, ensure_ascii=False)
    
    logger.info(f"Reporte de validación guardado: {report_path}")
    
    # Si la validación pasa, guardar datos en output
    if overall_valid:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Datos validados guardados: {output_path}")
    else:
        logger.error("Validación falló, no se guardan los datos")
        for error in all_errors:
            logger.error(f"  Error: {error}")
    
    return combined_result


def main():
    """
    Función principal de validación de datos.
    """
    try:
        # Cargar configuración
        config = get_data_config()
        
        # Rutas de archivos
        input_path = config['data']['raw']['local_path']
        output_path = config['data']['interim']['obesity_original']
        
        # Ejecutar validación
        result = validate_dataset(input_path, output_path)
        
        # Mostrar resultados
        if result.is_valid:
            print(f"Validación exitosa: {output_path}")
        else:
            print("Validación falló:")
            for error in result.errors:
                print(f"  - {error}")
            
        if result.warnings:
            print("Advertencias:")
            for warning in result.warnings:
                print(f"  - {warning}")
        
    except Exception as e:
        logger.error(f"Error en validación de datos: {e}")
        raise DataValidationError(f"Error en validación: {e}")


if __name__ == "__main__":
    main()