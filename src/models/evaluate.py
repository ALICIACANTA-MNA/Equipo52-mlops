"""
# =============================================================================
# MODEL EVALUATION MODULE - STAGE 6 MLOps Pipeline Architecture
# =============================================================================
#
# ARQUITECTURA DE FLUJO - STAGE 6: MODEL EVALUATION
# ┌─────────────────────────────────────────────────────────────────────────┐
# │                        MLOps PIPELINE FLOW                             │
# │                                                                         │
# │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌──────────┐  │
# │  │   STAGE 4   │    │   STAGE 5   │───▶│   STAGE 6   │───▶│ STAGE 7  │  │
# │  │  Feature    │    │   Model     │    │ EVALUATION  │    │Registry  │  │
# │  │ Engineering │    │  Training   │    │  & Testing  │    │& Deploy  │  │
# │  └─────────────┘    └─────────────┘    └─────────────┘    └──────────┘  │
# │                                              ▲                          │
# │                                              │                          │
# │  EVALUATION INPUT FLOW:                      │                          │
# │  models/best_model.joblib  ──────────────────┤                          │
# │  data/processed/X_test_processed.csv ────────┤                          │
# │  data/processed/y_test.csv ───────────────────┘                          │
# │                                                                         │
# │  COMPREHENSIVE EVALUATION ARCHITECTURE:                                │
# │  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    │
# │  │  PERFORMANCE    │    │ VISUALIZATION   │    │   REPORTING     │    │
# │  │   METRICS       │    │   ANALYSIS      │    │   GENERATION    │    │
# │  │                 │    │                 │    │                 │    │
# │  │ • Accuracy      │    │ • Confusion     │    │ • JSON Reports  │    │
# │  │ • F1-Score      │    │   Matrix        │    │ • Feature       │    │
# │  │ • Precision     │    │ • ROC Curves    │    │   Importance    │    │
# │  │ • Recall        │    │ • Precision-    │    │ • Performance   │    │
# │  │ • AUC           │    │   Recall        │    │   Comparison    │    │
# │  │ • Per-Class     │    │ • Class         │    │ • MLflow        │    │
# │  │   Analysis      │    │   Distribution  │    │   Artifacts     │    │
# │  └─────────────────┘    └─────────────────┘    └─────────────────┘    │
# │                                ▼                                       │
# │                     EVALUATION DECISION GATE                          │
# │                    (F1-Score >= 95% ✅ → Promote)                     │
# │                    (F1-Score < 95% ❌ → Retrain)                      │
# │                                ▼                                       │
# │  OUTPUT ARTIFACTS:                                                     │
# │  • metrics/classification_report.json                                  │
# │  • metrics/evaluation_metrics.json                                     │
# │  • metrics/confusion_matrix.png                                        │
# │  • metrics/feature_importance.png                                      │
# │  • artifacts/evaluation/model_analysis/                                │
# │                                                                         │
# │  QUALITY ASSURANCE FLOW:                                               │
# │  Test Data ──▶ Prediction ──▶ Metrics ──▶ Validation ──▶ Promotion    │
# │      ▲             ▲            ▲           ▲             ▲            │
# │   Holdout      Inference    Statistical   Quality      Registry       │
# │    Set         Engine      Analysis      Gates       Decision         │
# └─────────────────────────────────────────────────────────────────────────┘
#
# TECHNICAL FOUNDATION:
# - Model Evaluation Best Practices: Comprehensive metrics y statistical validation
# - MLflow Model Evaluation: Experiment tracking y performance comparison
# - Classification Metrics: Multi-class evaluation con class-wise analysis
#
# COMPONENTES:
# - ModelEvaluator: Evaluación individual comprensiva de modelos
# - ModelComparator: Comparación automatizada entre múltiples modelos
# - PerformanceAnalyzer: Análisis detallado de rendimiento por clase
# - ReportGenerator: Generación de reportes visuales y estadísticos
#
# =============================================================================
# REFERENCIAS BIBLIOGRÁFICAS:
# =============================================================================
#
# [1] Géron, A. (2019). "Hands-On Machine Learning with Scikit-Learn, Keras, 
#     and TensorFlow" 2nd Edition. O'Reilly Media.
#     - Capítulo 3: Classification Performance Measures
#     - Capítulo 8: Dimensionality Reduction & Model Evaluation
#     - Anexo B: Machine Learning Project Checklist
#
# [2] Provost, F., & Fawcett, T. (2013). "Data Science for Business". O'Reilly.
#     - Capítulo 7: Decision Analytic Thinking I: What Is a Good Model?
#     - Capítulo 8: Visualizing Model Performance
#     - Capítulo 9: Evidence and Probabilities
#
# [3] Hastie, T., Tibshirani, R., & Friedman, J. (2017). "The Elements of 
#     Statistical Learning" 2nd Edition. Springer.
#     - Capítulo 7: Model Assessment and Selection
#     - Capítulo 15: Random Forests & Model Interpretation
#
# [4] Kuhn, M., & Johnson, K. (2019). "Feature Engineering and Selection". CRC Press.
#     - Capítulo 10: Feature Selection Overview
#     - Capítulo 11: Measuring Performance in Classification
#     - Capítulo 19: An Introduction to Feature Selection
#
# [5] Molnar, C. (2020). "Interpretable Machine Learning". Lulu.com
#     - Capítulo 2: Interpretability & Model Performance
#     - Capítulo 5: Model-Agnostic Methods
#     - Capítulo 8: Partial Dependence Plot
#
# [6] Zheng, A., & Casari, A. (2018). "Feature Engineering for Machine Learning". O'Reilly.
#     - Capítulo 5: Categorical Encoding & Model Evaluation
#     - Capítulo 8: Automated Feature Engineering
#
# [7] Raschka, S., & Mirjalili, V. (2019). "Python Machine Learning" 3rd Edition. Packt.
#     - Capítulo 6: Learning Best Practices for Model Evaluation
#     - Capítulo 7: Combining Different Models for Ensemble Learning
#
# [8] ISO/IEC 23053:2022. "Framework for AI Risk Management"
#     - Section 7: AI System Performance Monitoring
#     - Section 9: Evaluation Metrics & Bias Detection
#
# [9] Scikit-learn Documentation (2024). "Model Evaluation & Metrics"
#     https://scikit-learn.org/stable/modules/model_evaluation.html
#     - Classification Metrics: Precision, Recall, F1-Score
#     - ROC Analysis & AUC Computation
#     - Cross-validation Strategies
#
# [10] Fawcett, T. (2006). "An Introduction to ROC Analysis". Pattern Recognition Letters.
#      - ROC Graphs: Notes and Practical Considerations
#      - Multi-class ROC Analysis & Performance Visualization
#
# [11] Davis, J., & Goadrich, M. (2006). "The Relationship Between Precision-Recall 
#      and ROC Curves". ICML Conference Proceedings.
#      - When to use ROC vs. Precision-Recall curves
#      - Class imbalance considerations
# =============================================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Union
import json
import joblib
from dataclasses import dataclass, asdict
import warnings
from datetime import datetime
import itertools

# MLflow imports
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

# Scikit-learn imports
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    f1_score, precision_score, recall_score, roc_auc_score,
    precision_recall_curve, roc_curve, auc
)
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import cross_validate, StratifiedKFold

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

from src.utils.config import get_model_config, get_mlflow_config, load_config
from src.utils.logging_config import get_logger, log_pipeline_stage


logger = get_logger(__name__)


@dataclass
class ModelEvaluationResult:
    """Resultado de evaluación de un modelo"""
    model_name: str
    overall_metrics: Dict[str, float]
    class_metrics: Dict[str, Dict[str, float]]
    confusion_matrix: np.ndarray
    roc_curves: Dict[str, Dict[str, np.ndarray]]
    precision_recall_curves: Dict[str, Dict[str, np.ndarray]]
    cross_validation_scores: Dict[str, List[float]]
    prediction_confidence: Dict[str, float]
    mlflow_run_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        # Convertir arrays numpy a listas para serialización
        result['confusion_matrix'] = self.confusion_matrix.tolist()
        return result


@dataclass
class ModelComparisonResult:
    """Resultado de comparación entre modelos"""
    models_evaluated: List[str]
    comparison_metrics: pd.DataFrame
    best_model: str
    best_model_score: float
    statistical_tests: Dict[str, Any]
    model_rankings: Dict[str, int]
    evaluation_summary: Dict[str, Any]


class ModelEvaluationError(Exception):
    """Excepción para errores de evaluación"""
    pass


class PerformanceAnalyzer:
    """
    Analizador de rendimiento para modelos ML.
    
    Implementa análisis detallado de rendimiento por clase,
    detección de bias y análisis de fairness.
    """
    
    def __init__(self, evaluation_config: Dict[str, Any]):
        """
        Inicializa el analizador.
        
        Args:
            evaluation_config: Configuración de evaluación
        """
        self.config = evaluation_config
        self.fairness_config = evaluation_config.get('fairness_analysis', {})
        
    def analyze_class_performance(self, y_true: pd.Series, y_pred: pd.Series, 
                                 y_proba: Optional[np.ndarray] = None) -> Dict[str, Dict[str, float]]:
        """
        Analiza rendimiento por clase.
        
        Args:
            y_true: Etiquetas verdaderas
            y_pred: Predicciones
            y_proba: Probabilidades de predicción (opcional)
            
        Returns:
            Diccionario con métricas por clase
        """
        # Obtener reporte de clasificación detallado
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        
        # Extraer métricas por clase
        class_metrics = {}
        classes = sorted(y_true.unique())
        
        for class_name in classes:
            class_str = str(class_name)
            if class_str in report:
                class_metrics[class_name] = {
                    'precision': report[class_str]['precision'],
                    'recall': report[class_str]['recall'],
                    'f1_score': report[class_str]['f1-score'],
                    'support': report[class_str]['support']
                }
                
                # Agregar métricas adicionales si tenemos probabilidades
                if y_proba is not None:
                    class_idx = list(classes).index(class_name)
                    
                    # Binarizar etiquetas para esta clase
                    y_true_binary = (y_true == class_name).astype(int)
                    y_proba_class = y_proba[:, class_idx]
                    
                    # AUC para esta clase
                    try:
                        auc_score = roc_auc_score(y_true_binary, y_proba_class)
                        class_metrics[class_name]['auc'] = auc_score
                    except Exception as e:
                        logger.warning(f"No se pudo calcular AUC para clase {class_name}: {e}")
                        class_metrics[class_name]['auc'] = np.nan
        
        return class_metrics
    
    def detect_performance_issues(self, class_metrics: Dict[str, Dict[str, float]]) -> Dict[str, List[str]]:
        """
        Detecta problemas potenciales de rendimiento.
        
        Args:
            class_metrics: Métricas por clase
            
        Returns:
            Diccionario con tipos de problemas detectados
        """
        issues = {
            'low_precision_classes': [],
            'low_recall_classes': [],
            'low_f1_classes': [],
            'imbalanced_classes': [],
            'low_support_classes': []
        }
        
        # Umbrales configurables
        precision_threshold = self.config.get('performance_thresholds', {}).get('precision', 0.7)
        recall_threshold = self.config.get('performance_thresholds', {}).get('recall', 0.7)
        f1_threshold = self.config.get('performance_thresholds', {}).get('f1_score', 0.7)
        min_support_threshold = self.config.get('performance_thresholds', {}).get('min_support', 10)
        
        for class_name, metrics in class_metrics.items():
            # Problemas de precision
            if metrics['precision'] < precision_threshold:
                issues['low_precision_classes'].append(class_name)
            
            # Problemas de recall
            if metrics['recall'] < recall_threshold:
                issues['low_recall_classes'].append(class_name)
            
            # Problemas de F1
            if metrics['f1_score'] < f1_threshold:
                issues['low_f1_classes'].append(class_name)
            
            # Soporte bajo
            if metrics['support'] < min_support_threshold:
                issues['low_support_classes'].append(class_name)
        
        # Detectar desbalance
        supports = [metrics['support'] for metrics in class_metrics.values()]
        if len(supports) > 1:
            max_support = max(supports)
            min_support = min(supports)
            imbalance_ratio = max_support / min_support if min_support > 0 else float('inf')
            
            if imbalance_ratio > self.config.get('performance_thresholds', {}).get('imbalance_ratio', 10):
                issues['imbalanced_classes'] = list(class_metrics.keys())
        
        return issues


class ModelEvaluator:
    """
    Evaluador comprensivo de modelos ML.
    
    Implementa evaluación detallada con múltiples métricas,
    visualizaciones y análisis de rendimiento.
    """
    
    def __init__(self, evaluation_config_path: str = "configs/model/model_evaluation.yaml"):
        """
        Inicializa el evaluador.
        
        Args:
            evaluation_config_path: Ruta al archivo de configuración
        """
        self.config = load_config(evaluation_config_path)
        self.metrics_config = self.config.get('metrics', {})
        self.plots_config = self.config.get('plots', {})
        self.cross_validation_config = self.config.get('cross_validation', {})
        
        # Inicializar analizador de rendimiento
        self.performance_analyzer = PerformanceAnalyzer(self.config)
        
        # Configurar MLflow si está disponible
        self.mlflow_config = get_mlflow_config()
        self.mlflow_client = MlflowClient()
    
    def evaluate_model(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series,
                      model_name: str, artifacts_dir: Optional[str] = None) -> ModelEvaluationResult:
        """
        Evalúa un modelo de forma comprensiva.
        
        Args:
            model: Modelo entrenado
            X_test: Datos de prueba
            y_test: Etiquetas verdaderas
            model_name: Nombre del modelo
            artifacts_dir: Directorio para guardar artefactos
            
        Returns:
            Resultado comprensivo de la evaluación
        """
        logger.info(f"Iniciando evaluación comprensiva de {model_name}")
        
        # Hacer predicciones
        y_pred = model.predict(X_test)
        y_proba = None
        
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)
        
        # Calcular métricas generales
        overall_metrics = self._calculate_overall_metrics(y_test, y_pred, y_proba)
        
        # Calcular métricas por clase
        class_metrics = self.performance_analyzer.analyze_class_performance(y_test, y_pred, y_proba)
        
        # Matriz de confusión
        cm = confusion_matrix(y_test, y_pred)
        
        # Curvas ROC por clase (si tenemos probabilidades)
        roc_curves = {}
        if y_proba is not None:
            roc_curves = self._calculate_roc_curves(y_test, y_proba)
        
        # Curvas Precision-Recall
        pr_curves = {}
        if y_proba is not None:
            pr_curves = self._calculate_precision_recall_curves(y_test, y_proba)
        
        # Cross-validation si está habilitada
        cv_scores = {}
        if self.cross_validation_config.get('enabled', True):
            cv_scores = self._perform_cross_validation(model, X_test, y_test)
        
        # Análisis de confianza en predicciones
        prediction_confidence = self._analyze_prediction_confidence(y_proba, y_pred)
        
        # Crear resultado
        result = ModelEvaluationResult(
            model_name=model_name,
            overall_metrics=overall_metrics,
            class_metrics=class_metrics,
            confusion_matrix=cm,
            roc_curves=roc_curves,
            precision_recall_curves=pr_curves,
            cross_validation_scores=cv_scores,
            prediction_confidence=prediction_confidence
        )
        
        # Generar artefactos visuales si se especifica directorio
        if artifacts_dir:
            self._generate_evaluation_artifacts(result, X_test, y_test, y_pred, y_proba, artifacts_dir)
        
        # Detectar problemas de rendimiento
        performance_issues = self.performance_analyzer.detect_performance_issues(class_metrics)
        if any(issues for issues in performance_issues.values()):
            logger.warning(f"Problemas de rendimiento detectados en {model_name}:")
            for issue_type, classes in performance_issues.items():
                if classes:
                    logger.warning(f"  {issue_type}: {classes}")
        
        logger.info(f"Evaluación de {model_name} completada - F1: {overall_metrics['f1_weighted']:.4f}")
        
        return result
    
    def _calculate_overall_metrics(self, y_true: pd.Series, y_pred: pd.Series, 
                                  y_proba: Optional[np.ndarray]) -> Dict[str, float]:
        """Calcula métricas generales del modelo"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted'),
            'f1_macro': f1_score(y_true, y_pred, average='macro'),
            'f1_micro': f1_score(y_true, y_pred, average='micro'),
            'precision_weighted': precision_score(y_true, y_pred, average='weighted'),
            'precision_macro': precision_score(y_true, y_pred, average='macro'),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted'),
            'recall_macro': recall_score(y_true, y_pred, average='macro')
        }
        
        # AUC si tenemos probabilidades
        if y_proba is not None:
            try:
                auc_weighted = roc_auc_score(y_true, y_proba, multi_class='ovr', average='weighted')
                auc_macro = roc_auc_score(y_true, y_proba, multi_class='ovr', average='macro')
                metrics['auc_weighted'] = auc_weighted
                metrics['auc_macro'] = auc_macro
            except Exception as e:
                logger.warning(f"No se pudo calcular AUC: {e}")
        
        return metrics
    
    def _calculate_roc_curves(self, y_true: pd.Series, y_proba: np.ndarray) -> Dict[str, Dict[str, np.ndarray]]:
        """Calcula curvas ROC por clase"""
        classes = sorted(y_true.unique())
        roc_curves = {}
        
        # Binarizar etiquetas
        y_true_bin = label_binarize(y_true, classes=classes)
        if len(classes) == 2:
            y_true_bin = np.hstack((1 - y_true_bin, y_true_bin))
        
        for i, class_name in enumerate(classes):
            try:
                fpr, tpr, thresholds = roc_curve(y_true_bin[:, i], y_proba[:, i])
                roc_auc = auc(fpr, tpr)
                
                roc_curves[str(class_name)] = {
                    'fpr': fpr,
                    'tpr': tpr,
                    'thresholds': thresholds,
                    'auc': roc_auc
                }
            except Exception as e:
                logger.warning(f"Error calculando ROC para clase {class_name}: {e}")
        
        return roc_curves
    
    def _calculate_precision_recall_curves(self, y_true: pd.Series, y_proba: np.ndarray) -> Dict[str, Dict[str, np.ndarray]]:
        """Calcula curvas Precision-Recall por clase"""
        classes = sorted(y_true.unique())
        pr_curves = {}
        
        # Binarizar etiquetas
        y_true_bin = label_binarize(y_true, classes=classes)
        if len(classes) == 2:
            y_true_bin = np.hstack((1 - y_true_bin, y_true_bin))
        
        for i, class_name in enumerate(classes):
            try:
                precision, recall, thresholds = precision_recall_curve(y_true_bin[:, i], y_proba[:, i])
                pr_auc = auc(recall, precision)
                
                pr_curves[str(class_name)] = {
                    'precision': precision,
                    'recall': recall,
                    'thresholds': thresholds,
                    'auc': pr_auc
                }
            except Exception as e:
                logger.warning(f"Error calculando PR para clase {class_name}: {e}")
        
        return pr_curves
    
    def _perform_cross_validation(self, model: Any, X: pd.DataFrame, y: pd.Series) -> Dict[str, List[float]]:
        """Realiza cross-validation comprensiva"""
        cv_folds = self.cross_validation_config.get('cv_folds', 5)
        scoring_metrics = self.cross_validation_config.get('scoring', ['f1_weighted', 'accuracy'])
        
        cv_results = {}
        
        # Configurar cross-validation estratificada
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        try:
            # Realizar cross-validation para múltiples métricas
            cv_scores = cross_validate(
                model, X, y,
                cv=cv,
                scoring=scoring_metrics,
                return_train_score=True,
                n_jobs=-1
            )
            
            # Procesar resultados
            for metric in scoring_metrics:
                test_key = f'test_{metric}'
                train_key = f'train_{metric}'
                
                if test_key in cv_scores:
                    cv_results[f'{metric}_test'] = cv_scores[test_key].tolist()
                if train_key in cv_scores:
                    cv_results[f'{metric}_train'] = cv_scores[train_key].tolist()
            
            # Agregar estadísticas de tiempo
            if 'fit_time' in cv_scores:
                cv_results['fit_time'] = cv_scores['fit_time'].tolist()
            if 'score_time' in cv_scores:
                cv_results['score_time'] = cv_scores['score_time'].tolist()
                
        except Exception as e:
            logger.warning(f"Error en cross-validation: {e}")
            cv_results = {}
        
        return cv_results
    
    def _analyze_prediction_confidence(self, y_proba: Optional[np.ndarray], 
                                     y_pred: pd.Series) -> Dict[str, float]:
        """Analiza la confianza en las predicciones"""
        if y_proba is None:
            return {}
        
        # Confianza promedio (probabilidad máxima)
        max_probas = np.max(y_proba, axis=1)
        
        confidence_metrics = {
            'mean_confidence': float(np.mean(max_probas)),
            'median_confidence': float(np.median(max_probas)),
            'std_confidence': float(np.std(max_probas)),
            'min_confidence': float(np.min(max_probas)),
            'max_confidence': float(np.max(max_probas)),
            'low_confidence_predictions': int(np.sum(max_probas < 0.6)),
            'high_confidence_predictions': int(np.sum(max_probas > 0.9))
        }
        
        # Porcentajes
        total_predictions = len(y_pred)
        confidence_metrics['low_confidence_percentage'] = (
            confidence_metrics['low_confidence_predictions'] / total_predictions * 100
        )
        confidence_metrics['high_confidence_percentage'] = (
            confidence_metrics['high_confidence_predictions'] / total_predictions * 100
        )
        
        return confidence_metrics
    
    def _generate_evaluation_artifacts(self, result: ModelEvaluationResult,
                                     X_test: pd.DataFrame, y_test: pd.Series,
                                     y_pred: pd.Series, y_proba: Optional[np.ndarray],
                                     artifacts_dir: str):
        """Genera artefactos visuales y reportes"""
        artifacts_path = Path(artifacts_dir) / result.model_name
        artifacts_path.mkdir(parents=True, exist_ok=True)
        
        # 1. Matriz de confusión mejorada
        self._plot_confusion_matrix(result.confusion_matrix, y_test, 
                                   artifacts_path / "confusion_matrix_detailed.png")
        
        # 2. Métricas por clase
        self._plot_class_metrics(result.class_metrics, 
                                artifacts_path / "class_metrics.png")
        
        # 3. Curvas ROC si disponibles
        if result.roc_curves:
            self._plot_roc_curves(result.roc_curves, 
                                 artifacts_path / "roc_curves.png")
        
        # 4. Curvas Precision-Recall si disponibles
        if result.precision_recall_curves:
            self._plot_precision_recall_curves(result.precision_recall_curves,
                                              artifacts_path / "precision_recall_curves.png")
        
        # 5. Distribución de confianza
        if y_proba is not None:
            self._plot_prediction_confidence(y_proba, y_pred,
                                           artifacts_path / "prediction_confidence.png")
        
        # 6. Reporte de evaluación en JSON
        evaluation_report = {
            'model_name': result.model_name,
            'overall_metrics': result.overall_metrics,
            'class_metrics': result.class_metrics,
            'prediction_confidence': result.prediction_confidence,
            'evaluation_timestamp': datetime.now().isoformat()
        }
        
        # Convertir tipos numpy para serialización JSON
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
                return {str(convert_numpy_types(key)): convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            return obj
        
        report_path = artifacts_path / "evaluation_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(convert_numpy_types(evaluation_report), f, indent=2, ensure_ascii=False)
        
        logger.info(f"Artefactos de evaluación guardados en: {artifacts_path}")
    
    def _plot_confusion_matrix(self, cm: np.ndarray, y_test: pd.Series, save_path: Path):
        """Crea plot mejorado de matriz de confusión"""
        plt.figure(figsize=(12, 10))
        
        # Normalizar matriz de confusión
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Crear subplot con dos matrices
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Matriz de confusión absoluta
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                   xticklabels=sorted(y_test.unique()),
                   yticklabels=sorted(y_test.unique()))
        ax1.set_title('Confusion Matrix (Absolute)')
        ax1.set_xlabel('Predicted')
        ax1.set_ylabel('Actual')
        
        # Matriz de confusión normalizada
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', ax=ax2,
                   xticklabels=sorted(y_test.unique()),
                   yticklabels=sorted(y_test.unique()))
        ax2.set_title('Confusion Matrix (Normalized)')
        ax2.set_xlabel('Predicted')
        ax2.set_ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_class_metrics(self, class_metrics: Dict[str, Dict[str, float]], save_path: Path):
        """Crea plot de métricas por clase"""
        # Preparar datos
        classes = list(class_metrics.keys())
        metrics = ['precision', 'recall', 'f1_score']
        
        data = []
        for class_name in classes:
            for metric in metrics:
                if metric in class_metrics[class_name]:
                    data.append({
                        'Class': class_name,
                        'Metric': metric.replace('_', ' ').title(),
                        'Value': class_metrics[class_name][metric]
                    })
        
        df = pd.DataFrame(data)
        
        plt.figure(figsize=(12, 8))
        sns.barplot(data=df, x='Class', y='Value', hue='Metric')
        plt.title('Performance Metrics by Class')
        plt.xlabel('Class')
        plt.ylabel('Score')
        plt.xticks(rotation=45)
        plt.legend(title='Metric')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_roc_curves(self, roc_curves: Dict[str, Dict[str, np.ndarray]], save_path: Path):
        """Crea plot de curvas ROC"""
        plt.figure(figsize=(10, 8))
        
        for class_name, curve_data in roc_curves.items():
            plt.plot(curve_data['fpr'], curve_data['tpr'],
                    label=f'{class_name} (AUC = {curve_data["auc"]:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves by Class')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_precision_recall_curves(self, pr_curves: Dict[str, Dict[str, np.ndarray]], save_path: Path):
        """Crea plot de curvas Precision-Recall"""
        plt.figure(figsize=(10, 8))
        
        for class_name, curve_data in pr_curves.items():
            plt.plot(curve_data['recall'], curve_data['precision'],
                    label=f'{class_name} (AUC = {curve_data["auc"]:.2f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves by Class')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_prediction_confidence(self, y_proba: np.ndarray, y_pred: pd.Series, save_path: Path):
        """Crea plot de distribución de confianza"""
        max_probas = np.max(y_proba, axis=1)
        
        plt.figure(figsize=(12, 6))
        
        # Histograma de confianza
        plt.subplot(1, 2, 1)
        plt.hist(max_probas, bins=30, alpha=0.7, edgecolor='black')
        plt.axvline(np.mean(max_probas), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(max_probas):.3f}')
        plt.xlabel('Prediction Confidence')
        plt.ylabel('Frequency')
        plt.title('Distribution of Prediction Confidence')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Box plot por clase
        plt.subplot(1, 2, 2)
        confidence_by_class = []
        class_labels = []
        
        for class_name in sorted(np.unique(y_pred)):
            class_mask = y_pred == class_name
            class_confidences = max_probas[class_mask]
            confidence_by_class.append(class_confidences)
            class_labels.append(str(class_name))
        
        plt.boxplot(confidence_by_class, labels=class_labels)
        plt.xlabel('Predicted Class')
        plt.ylabel('Prediction Confidence')
        plt.title('Confidence Distribution by Predicted Class')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()


@log_pipeline_stage("model_evaluation")
def evaluate_models_pipeline(models_dir: str, test_data_path: str, 
                           artifacts_dir: str) -> List[ModelEvaluationResult]:
    """
    Pipeline de evaluación de múltiples modelos.
    
    Args:
        models_dir: Directorio con modelos entrenados
        test_data_path: Ruta a datos de prueba
        artifacts_dir: Directorio para artefactos de evaluación
        
    Returns:
        Lista de resultados de evaluación
    """
    # Cargar datos de prueba
    test_df = pd.read_csv(test_data_path)
    target_column = 'NObeyesdad'
    
    X_test = test_df.drop(columns=[target_column])
    y_test = test_df[target_column]
    
    logger.info(f"Datos de prueba cargados: {X_test.shape}")
    
    # Inicializar evaluador
    evaluator = ModelEvaluator()
    
    # Encontrar modelos en el directorio
    models_path = Path(models_dir)
    model_files = list(models_path.glob("*.pkl")) + list(models_path.glob("*.joblib"))
    
    if not model_files:
        raise ModelEvaluationError(f"No se encontraron modelos en {models_dir}")
    
    results = []
    
    for model_file in model_files:
        try:
            # Cargar modelo
            model = joblib.load(model_file)
            model_name = model_file.stem.replace('_model', '')
            
            logger.info(f"Evaluando modelo: {model_name}")
            
            # Evaluar modelo
            result = evaluator.evaluate_model(
                model, X_test, y_test, model_name, artifacts_dir
            )
            
            results.append(result)
            
        except Exception as e:
            logger.error(f"Error evaluando modelo {model_file.name}: {e}")
            continue
    
    if not results:
        raise ModelEvaluationError("No se pudo evaluar ningún modelo")
    
    # Mostrar resumen
    logger.info("Evaluación completada:")
    for result in results:
        logger.info(f"  {result.model_name}: F1={result.overall_metrics['f1_weighted']:.4f}, "
                   f"Accuracy={result.overall_metrics['accuracy']:.4f}")
    
    return results


def evaluate_models_pipeline_xy(models_dir: str, X_test_path: str, 
                               y_test_path: str, artifacts_dir: str) -> List[ModelEvaluationResult]:
    """
    Pipeline de evaluación con archivos X/y separados.
    
    Args:
        models_dir: Directorio con modelos entrenados
        X_test_path: Ruta a features de prueba
        y_test_path: Ruta a targets de prueba
        artifacts_dir: Directorio para artefactos de evaluación
        
    Returns:
        Lista de resultados de evaluación
    """
    # Cargar datos de prueba separados
    X_test = pd.read_csv(X_test_path)
    y_test = pd.read_csv(y_test_path).iloc[:, 0]  # Primera columna
    
    logger.info(f"Features de prueba cargados: {X_test.shape}")
    logger.info(f"Targets de prueba cargados: {y_test.shape}")
    
    # Usar el mejor modelo guardado
    import joblib
    best_model_path = Path("models/best_model.joblib")
    
    if not best_model_path.exists():
        raise ModelEvaluationError(f"No se encontró el mejor modelo en {best_model_path}")
    
    # Cargar el mejor modelo
    best_model = joblib.load(best_model_path)
    logger.info(f"Mejor modelo cargado desde: {best_model_path}")
    
    # Inicializar evaluador
    evaluator = ModelEvaluator()
    
    # Evaluar el mejor modelo
    result = evaluator.evaluate_model(
        model=best_model,
        X_test=X_test,
        y_test=y_test,
        model_name="best_model",
        artifacts_dir=artifacts_dir
    )
    
    logger.info(f"Evaluación completada - F1: {result.overall_metrics['f1_weighted']:.4f}")
    
    return [result]


def main():
    """
    Función principal de evaluación de modelos.
    """
    try:
        # Configurar rutas para archivos separados X/y
        models_dir = "artifacts/training"
        X_test_path = "data/processed/X_test_processed.csv"
        y_test_path = "data/processed/y_test.csv"
        artifacts_dir = "artifacts/evaluation"
        
        # Ejecutar pipeline de evaluación con archivos separados
        results = evaluate_models_pipeline_xy(models_dir, X_test_path, y_test_path, artifacts_dir)
        
        # Mostrar resultados
        print(f"Evaluación completada para {len(results)} modelos:")
        
        # Ordenar por F1 score
        results_sorted = sorted(results, key=lambda x: x.overall_metrics['f1_weighted'], reverse=True)
        
        print("\nRanking de modelos (por F1 weighted):")
        for i, result in enumerate(results_sorted, 1):
            metrics = result.overall_metrics
            print(f"  {i}. {result.model_name}")
            print(f"     F1: {metrics['f1_weighted']:.4f} | Accuracy: {metrics['accuracy']:.4f}")
            print(f"     Precision: {metrics['precision_weighted']:.4f} | Recall: {metrics['recall_weighted']:.4f}")
            
            # Mostrar problemas de confianza si existen
            if result.prediction_confidence:
                low_conf_pct = result.prediction_confidence.get('low_confidence_percentage', 0)
                if low_conf_pct > 20:
                    print(f"{low_conf_pct:.1f}% predicciones con baja confianza")
        
        print(f"\nMejor modelo: {results_sorted[0].model_name}")
        
        # Generar archivos esperados por DVC
        from pathlib import Path
        import json
        from datetime import datetime
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        metrics_dir = Path("metrics")
        metrics_dir.mkdir(exist_ok=True)
        
        # 1. classification_report.json
        best_result = results_sorted[0]
        classification_report = {
            'model_name': best_result.model_name,
            'overall_metrics': best_result.overall_metrics,
            'class_metrics': best_result.class_metrics,
            'confusion_matrix': best_result.confusion_matrix.tolist()
        }
        
        def convert_numpy_types(obj):
            """Convierte tipos numpy a tipos JSON serializables"""
            import numpy as np
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {str(convert_numpy_types(key)): convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            return obj
        
        with open(metrics_dir / "classification_report.json", 'w') as f:
            json.dump(convert_numpy_types(classification_report), f, indent=2)
        
        # 2. evaluation_metrics.json
        evaluation_metrics = {
            'best_model': best_result.model_name,
            'best_f1_score': best_result.overall_metrics['f1_weighted'],
            'models_evaluated': len(results_sorted),
            'evaluation_timestamp': datetime.now().isoformat()
        }
        
        with open(metrics_dir / "evaluation_metrics.json", 'w') as f:
            json.dump(convert_numpy_types(evaluation_metrics), f, indent=2)
        
        # 3. confusion_matrix.png
        plt.figure(figsize=(10, 8))
        cm = best_result.confusion_matrix
        class_names = [f"Class_{i}" for i in range(cm.shape[0])]
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Confusion Matrix - {best_result.model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(metrics_dir / "confusion_matrix.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. feature_importance.png (siempre generar)
        if hasattr(best_result, 'feature_importance') and best_result.feature_importance is not None:
            plt.figure(figsize=(12, 8))
            importance_data = best_result.feature_importance
            if isinstance(importance_data, dict):
                features = list(importance_data.keys())
                importance = list(importance_data.values())
                
                # Mostrar las top 20 features
                sorted_idx = sorted(range(len(importance)), key=lambda i: importance[i], reverse=True)[:20]
                
                plt.barh([features[i] for i in sorted_idx], [importance[i] for i in sorted_idx])
                plt.title(f'Top 20 Feature Importance - {best_result.model_name}')
                plt.xlabel('Importance')
                plt.tight_layout()
                plt.savefig(metrics_dir / "feature_importance.png", dpi=300, bbox_inches='tight')
                plt.close()
            else:
                # Crear placeholder si no hay feature importance
                plt.figure(figsize=(8, 6))
                plt.text(0.5, 0.5, 'Feature importance not available\nfor this model type', 
                        ha='center', va='center', fontsize=14)
                plt.title('Feature Importance')
                plt.axis('off')
                plt.savefig(metrics_dir / "feature_importance.png", dpi=300, bbox_inches='tight')
                plt.close()
        else:
            # Crear placeholder si no hay feature importance
            plt.figure(figsize=(8, 6))
            plt.text(0.5, 0.5, 'Feature importance not available\nfor this model type', 
                    ha='center', va='center', fontsize=14)
            plt.title('Feature Importance')
            plt.axis('off')
            plt.savefig(metrics_dir / "feature_importance.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"\nArchivos generados:")
        print(f"  - metrics/classification_report.json")
        print(f"  - metrics/evaluation_metrics.json") 
        print(f"  - metrics/confusion_matrix.png")
        print(f"  - metrics/feature_importance.png")
        
    except Exception as e:
        logger.error(f"Error en evaluación de modelos: {e}")
        raise ModelEvaluationError(f"Error en pipeline de evaluación: {e}")


if __name__ == "__main__":
    main()