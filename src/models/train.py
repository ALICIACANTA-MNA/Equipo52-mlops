"""
# =============================================================================
# MODEL TRAINING MODULE - STAGE 5 MLOps Pipeline Architecture
# =============================================================================
#
# ARQUITECTURA DE FLUJO - STAGE 5: MODEL TRAINING
# ┌─────────────────────────────────────────────────────────────────────────┐
# │                        MLOps PIPELINE FLOW                              │
# │                                                                         │
# │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌──────────┐  │
# │  │   STAGE 4   │──▶│   STAGE 5   │───▶│   STAGE 6   │───▶│ STAGE 7 │  │
# │  │  Feature    │    │   MODEL     │    │  Evaluation │    │Registry  │  │
# │  │ Engineering │    │  TRAINING   │    │   & Test    │    │& Deploy  │  │
# │  └─────────────┘    └─────────────┘    └─────────────┘    └──────────┘  │
# │                           ▲                                             │
# │                           │                                             │
# │  INPUT DATA FLOW:         │                                             │
# │  X_train_processed.csv ───┤                                             │
# │  X_test_processed.csv  ───┤                                             │
# │  y_train.csv           ───┤                                             │
# │  y_test.csv            ───┘                                             │
# │                                                                         │
# │  TRAINING ARCHITECTURE:                                                 │
# │  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐      │
# │  │ Random Forest   │    │ Gradient Boost  │    │ Logistic Regr.  │      │
# │  │ Ensemble: 300   │    │ Learning Rate   │    │ Multi-class     │      │
# │  │ Max Depth: 12   │    │ N_estimators    │    │ Regularization  │      │
# │  └─────────────────┘    └─────────────────┘    └─────────────────┘      │
# │                                ▼                                        │
# │                        MODEL SELECTION                                  │
# │                    (Best F1-Score > 95%)                                │
# │                                ▼                                        │
# │  OUTPUT ARTIFACTS:                                                      │
# │  • models/best_model.joblib                                             │
# │  • metrics/train_metrics.json                                           │
# │  • MLflow: Experiment tracking + Model registry versions                │
# │                                                                         │
# │  MLFLOW INTEGRATION FLOW:                                               │
# │  Training ──▶ Experiment ──▶ Run Logging ──▶ Model Registry ──▶ S3    │
# │     ▲              ▲              ▲              ▲             ▲        │
# │  Params        Metrics       Artifacts      Versioning    Backup        │
# └─────────────────────────────────────────────────────────────────────────┘
#
# TECHNICAL FOUNDATION:
# - MLflow Model Training: Experiment tracking y model versioning
# - Model Training Best Practices: Hyperparameter tuning y cross-validation
# - Scientific approach: Multiple algorithms con statistical validation
#
# COMPONENTES:
# - ModelTrainer: Clase principal para entrenamiento multi-algoritmo
# - HyperparameterTuner: Optimización automática de hiperparámetros  
# - ModelRegistry: Gestión de modelos y versiones en MLflow
# - Training Pipeline: Pipeline completo con validation y selection
#
# =============================================================================
# REFERENCIAS BIBLIOGRÁFICAS:
# =============================================================================
#
# [1] Géron, A. (2019). "Hands-On Machine Learning with Scikit-Learn, Keras, 
#     and TensorFlow" 2nd Edition. O'Reilly Media.
#     - Capítulo 2: End-to-End Machine Learning Project
#     - Capítulo 3: Classification & Model Selection
#     - Capítulo 4: Training Models & Hyperparameter Tuning
#
# [2] Kleppmann, M. (2017). "Designing Data-Intensive Applications". O'Reilly.
#     - Capítulo 1: Reliable, Scalable, and Maintainable Applications
#     - Capítulo 10: Batch Processing & ML Pipelines
#
# [3] Lakshmanan, V., Robinson, S., & Munn, M. (2020). "Machine Learning Design 
#     Patterns". O'Reilly Media.
#     - Pattern 1: Transform - Feature Engineering Patterns
#     - Pattern 2: Checkpoints - Model Versioning & Reproducibility
#     - Pattern 3: Feature Store - Data Pipeline Patterns
#
# [4] Gift, N., & Deza, A. (2021). "Practical MLOps". O'Reilly Media.
#     - Capítulo 3: MLOps Foundations & Model Training
#     - Capítulo 4: Continuous Integration for Machine Learning
#     - Capítulo 5: Model Deployment & Monitoring
#
# [5] Huyen, C. (2022). "Designing Machine Learning Systems". O'Reilly Media.
#     - Capítulo 6: Model Development and Offline Evaluation
#     - Capítulo 7: Model Deployment and Prediction Service
#     - Capítulo 8: Data Distribution Shifts and Monitoring
#
# [6] Burkov, A. (2019). "The Hundred-Page Machine Learning Book".
#     - Capítulo 3: Fundamental Algorithms (Gradient Boosting, Random Forest)
#     - Capítulo 5: Model Performance Assessment
#
# [7] MLflow Documentation (2024). "MLflow: A Machine Learning Lifecycle Platform"
#     https://mlflow.org/docs/latest/
#     - Tracking: Experiment management & metrics logging
#     - Models: Model packaging & versioning
#     - Model Registry: Centralized model store
#
# [8] Scikit-learn Documentation (2024). "Machine Learning in Python"
#     https://scikit-learn.org/stable/
#     - Model Selection: Cross-validation & hyperparameter tuning
#     - Ensemble Methods: RandomForest, GradientBoosting
#     - Preprocessing: Feature scaling & transformation
#
# [9] IEEE Standards (2017). "IEEE 2857-2021 - Privacy Engineering for ML Systems"
#     - Section 4: Data Processing & Feature Engineering
#     - Section 6: Model Training & Validation Practices
#
# [10] Google AI (2020). "Rules of Machine Learning: Best Practices for ML Engineering"
#      https://developers.google.com/machine-learning/guides/rules-of-ml
#      - Rule #3: Choose machine learning over a complex heuristic
#      - Rule #4: Keep the first model simple and get the infrastructure right
#      - Rule #6: Be careful about dropped data when copying pipelines
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

# MLflow imports
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

# Scikit-learn imports
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, 
    f1_score, precision_score, recall_score, roc_auc_score
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils.config import get_model_config, get_mlflow_config, load_config
from src.utils.logging_config import get_logger, log_pipeline_stage


logger = get_logger(__name__)


@dataclass
class ModelTrainingResult:
    """Resultado del entrenamiento de un modelo"""
    model_name: str
    model: Any
    metrics: Dict[str, float]
    best_params: Dict[str, Any]
    cross_val_scores: List[float]
    mlflow_run_id: str
    model_artifact_path: str
    training_time: float
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        # No serializar el modelo directamente
        result.pop('model', None)
        return result


@dataclass
class TrainingSession:
    """Sesión de entrenamiento con múltiples modelos"""
    experiment_name: str
    session_id: str
    models_results: List[ModelTrainingResult]
    best_model_name: str
    best_model_score: float
    training_data_shape: Tuple[int, int]
    target_classes: List[str]
    feature_names: List[str]
    session_config: Dict[str, Any]
    
    def get_best_model_result(self) -> Optional[ModelTrainingResult]:
        """Retorna el resultado del mejor modelo"""
        for result in self.models_results:
            if result.model_name == self.best_model_name:
                return result
        return None


class ModelTrainingError(Exception):
    """Excepción para errores de entrenamiento"""
    pass


class HyperparameterTuner:
    """
    Optimizador de hiperparámetros para modelos ML.
    
    Implementa búsqueda de hiperparámetros con cross-validation
    y métricas optimizadas para clasificación multiclase.
    """
    
    def __init__(self, tuning_config: Dict[str, Any]):
        """
        Inicializa el optimizador.
        
        Args:
            tuning_config: Configuración de hyperparameter tuning
        """
        self.config = tuning_config
        self.search_method = tuning_config.get('search_method', 'grid')
        self.cv_folds = tuning_config.get('cv_folds', 5)
        self.scoring = tuning_config.get('scoring', 'f1_weighted')
        self.n_jobs = tuning_config.get('n_jobs', -1)
        
    def tune_model(self, model, param_grid: Dict[str, Any], 
                   X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[Any, Dict[str, Any], float]:
        """
        Optimiza hiperparámetros del modelo.
        
        Args:
            model: Modelo base a optimizar
            param_grid: Grilla de parámetros a buscar
            X_train: Datos de entrenamiento
            y_train: Target de entrenamiento
            
        Returns:
            Tupla con (mejor_modelo, mejores_params, mejor_score)
        """
        logger.info(f"Iniciando hyperparameter tuning con {self.search_method}")
        
        if self.search_method == 'grid':
            search = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                cv=self.cv_folds,
                scoring=self.scoring,
                n_jobs=self.n_jobs,
                verbose=1
            )
        elif self.search_method == 'random':
            n_iter = self.config.get('n_iter', 50)
            search = RandomizedSearchCV(
                estimator=model,
                param_distributions=param_grid,
                n_iter=n_iter,
                cv=self.cv_folds,
                scoring=self.scoring,
                n_jobs=self.n_jobs,
                verbose=1,
                random_state=42
            )
        else:
            raise ValueError(f"Método de búsqueda no soportado: {self.search_method}")
        
        # Realizar búsqueda
        search.fit(X_train, y_train)
        
        logger.info(f"Mejor score: {search.best_score_:.4f}")
        logger.info(f"Mejores parámetros: {search.best_params_}")
        
        return search.best_estimator_, search.best_params_, search.best_score_


class ModelTrainer:
    """
    Entrenador de modelos para MLOps pipeline.
    
    Implementa entrenamiento robusto de múltiples modelos con
    experiment tracking, hyperparameter tuning y model selection.
    """
    
    def __init__(self, training_config_path: str = "configs/model/model_training.yaml"):
        """
        Inicializa el entrenador.
        
        Args:
            training_config_path: Ruta al archivo de configuración de entrenamiento
        """
        self.config = load_config(training_config_path)
        self.models_config = self.config.get('models', {})
        self.training_config = self.config.get('training', {})
        self.tuning_config = self.config.get('hyperparameter_tuning', {})
        
        # Configurar MLflow
        self.mlflow_config = get_mlflow_config()
        self._setup_mlflow()
        
        # Inicializar tuner si está habilitado
        if self.tuning_config.get('enabled', False):
            self.tuner = HyperparameterTuner(self.tuning_config)
        else:
            self.tuner = None
    
    def _setup_mlflow(self):
        """Configura MLflow para experiment tracking"""
        # Acceder a configuración anidada
        tracking_uri = self.mlflow_config.get('mlflow', {}).get('tracking', {}).get('tracking_uri', 'sqlite:///mlflow.db')
        mlflow.set_tracking_uri(tracking_uri)
        
        # Obtener nombre del experimento de la configuración de MLflow
        experiment_name = self.mlflow_config.get('mlflow', {}).get('experiment', {}).get('name', 'obesity_prediction_training')
        
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(experiment_name)
                logger.info(f"Experimento creado: {experiment_name} (ID: {experiment_id})")
            else:
                experiment_id = experiment.experiment_id
                logger.info(f"Usando experimento existente: {experiment_name} (ID: {experiment_id})")
            
            mlflow.set_experiment(experiment_name)
            
        except Exception as e:
            logger.error(f"Error configurando MLflow: {e}")
            raise ModelTrainingError(f"Error en configuración MLflow: {e}")
    
    def _get_model_instance(self, model_name: str) -> Any:
        """
        Crea instancia del modelo basada en configuración.
        
        Args:
            model_name: Nombre del modelo
            
        Returns:
            Instancia del modelo configurado
        """
        model_config = self.models_config.get(model_name, {})
        params = model_config.get('params', {})
        
        # Mapeo de modelos disponibles
        model_mapping = {
            'random_forest': RandomForestClassifier,
            'gradient_boosting': GradientBoostingClassifier,
            'logistic_regression': LogisticRegression,
            'svm': SVC,
            'knn': KNeighborsClassifier
        }
        
        if model_name not in model_mapping:
            raise ValueError(f"Modelo no soportado: {model_name}")
        
        # Agregar random_state si no está presente
        if 'random_state' not in params and hasattr(model_mapping[model_name](), 'random_state'):
            params['random_state'] = 42
        
        return model_mapping[model_name](**params)
    
    def _calculate_metrics(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """
        Calcula métricas comprensivas del modelo.
        
        Args:
            model: Modelo entrenado
            X_test: Datos de prueba
            y_test: Target de prueba
            
        Returns:
            Diccionario con métricas calculadas
        """
        y_pred = model.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_weighted': f1_score(y_test, y_pred, average='weighted'),
            'f1_macro': f1_score(y_test, y_pred, average='macro'),
            'precision_weighted': precision_score(y_test, y_pred, average='weighted'),
            'precision_macro': precision_score(y_test, y_pred, average='macro'),
            'recall_weighted': recall_score(y_test, y_pred, average='weighted'),
            'recall_macro': recall_score(y_test, y_pred, average='macro')
        }
        
        # AUC para clasificación multiclase si el modelo soporta predict_proba
        try:
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_test)
                auc_score = roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted')
                metrics['auc_weighted'] = auc_score
        except Exception as e:
            logger.warning(f"No se pudo calcular AUC: {e}")
        
        return metrics
    
    def _create_artifacts(self, model: Any, model_name: str, X_test: pd.DataFrame, 
                         y_test: pd.Series, metrics: Dict[str, float], 
                         artifacts_dir: Path) -> Dict[str, str]:
        """
        Crea artefactos del modelo (métricas, plots, reportes).
        
        Args:
            model: Modelo entrenado
            model_name: Nombre del modelo
            X_test: Datos de prueba
            y_test: Target de prueba
            metrics: Métricas calculadas
            artifacts_dir: Directorio para guardar artefactos
            
        Returns:
            Diccionario con rutas de artefactos creados
        """
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        artifacts = {}
        
        # Predicciones
        y_pred = model.predict(X_test)
        
        # 1. Classification Report
        report = classification_report(y_test, y_pred, output_dict=True)
        report_path = artifacts_dir / f"classification_report_{model_name}.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        artifacts['classification_report'] = str(report_path)
        
        # 2. Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=sorted(y_test.unique()),
                   yticklabels=sorted(y_test.unique()))
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        
        cm_path = artifacts_dir / f"confusion_matrix_{model_name}.png"
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()
        artifacts['confusion_matrix'] = str(cm_path)
        
        # 3. Feature Importance (si disponible)
        if hasattr(model, 'feature_importances_'):
            feature_names = X_test.columns if hasattr(X_test, 'columns') else [f'feature_{i}' for i in range(X_test.shape[1])]
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            plt.figure(figsize=(10, 8))
            top_features = importance_df.head(20)  # Top 20 features
            sns.barplot(data=top_features, x='importance', y='feature')
            plt.title(f'Top 20 Feature Importance - {model_name}')
            plt.tight_layout()
            
            importance_path = artifacts_dir / f"feature_importance_{model_name}.png"
            plt.savefig(importance_path, dpi=300, bbox_inches='tight')
            plt.close()
            artifacts['feature_importance'] = str(importance_path)
            
            # Guardar importance como CSV
            importance_csv_path = artifacts_dir / f"feature_importance_{model_name}.csv"
            importance_df.to_csv(importance_csv_path, index=False)
            artifacts['feature_importance_csv'] = str(importance_csv_path)
        
        # 4. Métricas resumidas
        metrics_path = artifacts_dir / f"metrics_{model_name}.json"
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        artifacts['metrics'] = str(metrics_path)
        
        return artifacts
    
    def train_model(self, model_name: str, X_train: pd.DataFrame, y_train: pd.Series,
                   X_test: pd.DataFrame, y_test: pd.Series, 
                   artifacts_dir: Optional[str] = None) -> ModelTrainingResult:
        """
        Entrena un modelo específico.
        
        Args:
            model_name: Nombre del modelo a entrenar
            X_train: Datos de entrenamiento
            y_train: Target de entrenamiento  
            X_test: Datos de prueba
            y_test: Target de prueba
            artifacts_dir: Directorio para guardar artefactos
            
        Returns:
            Resultado del entrenamiento del modelo
        """
        start_time = datetime.now()
        
        with mlflow.start_run(run_name=f"{model_name}_training") as run:
            try:
                logger.info(f"Iniciando entrenamiento de {model_name}")
                
                # Obtener instancia del modelo
                model = self._get_model_instance(model_name)
                
                # Hyperparameter tuning si está habilitado
                best_params = {}
                if self.tuner and model_name in self.tuning_config.get('models', {}):
                    param_grid = self.tuning_config['models'][model_name]
                    model, best_params, tuning_score = self.tuner.tune_model(
                        model, param_grid, X_train, y_train
                    )
                    mlflow.log_params(best_params)
                    mlflow.log_metric("tuning_cv_score", tuning_score)
                    logger.info(f"Hyperparameter tuning completado para {model_name}")
                
                # Entrenar modelo
                model.fit(X_train, y_train)
                training_time = (datetime.now() - start_time).total_seconds()
                
                # Cross-validation
                cv_scores = cross_val_score(
                    model, X_train, y_train, 
                    cv=self.training_config.get('cv_folds', 5),
                    scoring='f1_weighted'
                )
                
                # Calcular métricas en test set
                metrics = self._calculate_metrics(model, X_test, y_test)
                
                # Log métricas en MLflow
                mlflow.log_metrics(metrics)
                mlflow.log_metric("cv_mean", cv_scores.mean())
                mlflow.log_metric("cv_std", cv_scores.std())
                mlflow.log_metric("training_time", training_time)
                
                # Log parámetros del modelo
                model_params = model.get_params()
                mlflow.log_params({f"model_{k}": v for k, v in model_params.items()})
                
                # Crear artefactos
                if artifacts_dir:
                    artifacts_path = Path(artifacts_dir) / model_name
                    artifacts = self._create_artifacts(
                        model, model_name, X_test, y_test, metrics, artifacts_path
                    )
                    
                    # Log artefactos en MLflow
                    for artifact_name, artifact_path in artifacts.items():
                        mlflow.log_artifact(artifact_path)
                
                # Guardar modelo en MLflow
                signature = infer_signature(X_train, model.predict(X_train))
                model_info = mlflow.sklearn.log_model(
                    model, 
                    "model",
                    signature=signature,
                    registered_model_name=f"obesity_prediction_{model_name}"
                )
                
                # Guardar modelo localmente
                model_path = Path(artifacts_dir or "artifacts/models") / f"{model_name}_model.pkl"
                model_path.parent.mkdir(parents=True, exist_ok=True)
                joblib.dump(model, model_path)
                
                logger.info(f"Entrenamiento de {model_name} completado - F1: {metrics['f1_weighted']:.4f}")
                
                return ModelTrainingResult(
                    model_name=model_name,
                    model=model,
                    metrics=metrics,
                    best_params=best_params,
                    cross_val_scores=cv_scores.tolist(),
                    mlflow_run_id=run.info.run_id,
                    model_artifact_path=str(model_path),
                    training_time=training_time
                )
                
            except Exception as e:
                logger.error(f"Error entrenando {model_name}: {e}")
                mlflow.log_param("training_status", "failed")
                mlflow.log_param("error_message", str(e))
                raise ModelTrainingError(f"Error entrenando {model_name}: {e}")
    
    def train_all_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                        X_test: pd.DataFrame, y_test: pd.Series,
                        artifacts_dir: Optional[str] = None) -> TrainingSession:
        """
        Entrena todos los modelos configurados.
        
        Args:
            X_train: Datos de entrenamiento
            y_train: Target de entrenamiento
            X_test: Datos de prueba
            y_test: Target de prueba
            artifacts_dir: Directorio para artefactos
            
        Returns:
            Sesión de entrenamiento con resultados de todos los modelos
        """
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        logger.info(f"Iniciando sesión de entrenamiento: {session_id}")
        
        # Obtener modelos a entrenar
        models_to_train = self.training_config.get('models', list(self.models_config.keys()))
        
        results = []
        for model_name in models_to_train:
            if model_name not in self.models_config:
                logger.warning(f"Configuración no encontrada para modelo: {model_name}")
                continue
            
            try:
                result = self.train_model(
                    model_name, X_train, y_train, X_test, y_test, artifacts_dir
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Falló entrenamiento de {model_name}: {e}")
                continue
        
        # Determinar mejor modelo
        if not results:
            raise ModelTrainingError("No se entrenó ningún modelo exitosamente")
        
        best_result = max(results, key=lambda x: x.metrics['f1_weighted'])
        
        # Crear sesión de entrenamiento
        session = TrainingSession(
            experiment_name=self.training_config.get('experiment_name', 'obesity_prediction_training'),
            session_id=session_id,
            models_results=results,
            best_model_name=best_result.model_name,
            best_model_score=best_result.metrics['f1_weighted'],
            training_data_shape=X_train.shape,
            target_classes=sorted(y_train.unique().tolist()),
            feature_names=X_train.columns.tolist() if hasattr(X_train, 'columns') else [],
            session_config=self.config
        )
        
        # Guardar resumen de la sesión
        if artifacts_dir:
            session_path = Path(artifacts_dir) / f"training_session_{session_id}.json"
            session_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Crear resumen serializable
            session_summary = {
                'session_id': session_id,
                'experiment_name': session.experiment_name,
                'best_model': best_result.model_name,
                'best_score': best_result.metrics['f1_weighted'],
                'models_trained': len(results),
                'training_data_shape': session.training_data_shape,
                'models_results': [r.to_dict() for r in results]
            }
            
            with open(session_path, 'w', encoding='utf-8') as f:
                json.dump(session_summary, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Resumen de sesión guardado: {session_path}")
        
        logger.info(f"Sesión de entrenamiento completada. Mejor modelo: {best_result.model_name} (F1: {best_result.metrics['f1_weighted']:.4f})")
        
        # Integración automática con Model Registry
        try:
            from src.models.registry import register_model_from_training
            
            # Registrar el mejor modelo automáticamente
            if self.config.get('registry', {}).get('auto_register_best_model', True):
                logger.info("Registrando mejor modelo en Model Registry...")
                
                model_info = register_model_from_training(
                    model=best_result.model,
                    model_name=f"obesity_prediction_{best_result.model_name}",
                    run_id=best_result.mlflow_run_id,
                    metrics=best_result.metrics,
                    description=f"Best model from training session {session_id}: {best_result.model_name} with F1={best_result.metrics['f1_weighted']:.4f}"
                )
                
                logger.info(f"Modelo registrado exitosamente: {model_info.name} v{model_info.version}")
                
                # Agregar información del registry a la sesión
                session.best_model_registry_info = {
                    'name': model_info.name,
                    'version': model_info.version,
                    'stage': model_info.stage.value,
                    'registration_time': model_info.creation_timestamp.isoformat()
                }
                
        except ImportError:
            logger.warning("Model Registry no disponible, omitiendo registro automático")
        except Exception as e:
            logger.error(f"Error registrando modelo en registry: {e}")
            # No fallar el entrenamiento por error en registry
        
        return session


@log_pipeline_stage("model_training")
def train_models_pipeline(train_data_path: str, test_data_path: str, 
                         artifacts_dir: str) -> TrainingSession:
    """
    Pipeline completo de entrenamiento de modelos.
    
    Args:
        train_data_path: Ruta a datos de entrenamiento procesados
        test_data_path: Ruta a datos de prueba procesados  
        artifacts_dir: Directorio para artefactos del entrenamiento
        
    Returns:
        Sesión de entrenamiento con resultados
    """
    # Cargar datos
    train_df = pd.read_csv(train_data_path)
    test_df = pd.read_csv(test_data_path)
    
    logger.info(f"Datos de entrenamiento: {train_df.shape}")
    logger.info(f"Datos de prueba: {test_df.shape}")
    
    # Separar features y target
    target_column = 'NObeyesdad'  # Ajustar según configuración
    
    X_train = train_df.drop(columns=[target_column])
    y_train = train_df[target_column]
    X_test = test_df.drop(columns=[target_column])
    y_test = test_df[target_column]
    
    # Inicializar entrenador
    trainer = ModelTrainer()
    
    # Entrenar todos los modelos
    session = trainer.train_all_models(X_train, y_train, X_test, y_test, artifacts_dir)
    
    return session


def train_models_pipeline_xy(X_train_path: str, X_test_path: str,
                            y_train_path: str, y_test_path: str,
                            artifacts_dir: str) -> TrainingSession:
    """
    Pipeline de entrenamiento con archivos X/y separados.
    
    Args:
        X_train_path: Ruta a features de entrenamiento
        X_test_path: Ruta a features de prueba
        y_train_path: Ruta a targets de entrenamiento  
        y_test_path: Ruta a targets de prueba
        artifacts_dir: Directorio para artefactos del entrenamiento
        
    Returns:
        Sesión de entrenamiento con resultados
    """
    # Cargar datos separados
    X_train = pd.read_csv(X_train_path)
    X_test = pd.read_csv(X_test_path)
    y_train = pd.read_csv(y_train_path).iloc[:, 0]  # Primera columna
    y_test = pd.read_csv(y_test_path).iloc[:, 0]    # Primera columna
    
    logger.info(f"Features de entrenamiento: {X_train.shape}")
    logger.info(f"Features de prueba: {X_test.shape}")
    logger.info(f"Target de entrenamiento: {y_train.shape}")
    logger.info(f"Target de prueba: {y_test.shape}")
    
    # Inicializar entrenador
    trainer = ModelTrainer()
    
    # Ejecutar entrenamiento
    session = trainer.train_all_models(X_train, y_train, X_test, y_test, artifacts_dir)
    
    logger.info(f"Entrenamiento completado. Mejor modelo: {session.best_model_name} (Score: {session.best_model_score:.4f})")
    
    return session


def main():
    """
    Función principal de entrenamiento de modelos.
    """
    try:
        # Configurar rutas para archivos separados X/y
        X_train_path = "data/processed/X_train_processed.csv"
        X_test_path = "data/processed/X_test_processed.csv"
        y_train_path = "data/processed/y_train.csv"
        y_test_path = "data/processed/y_test.csv"
        artifacts_dir = "artifacts/training"
        
        # Ejecutar pipeline de entrenamiento con archivos separados
        session = train_models_pipeline_xy(X_train_path, X_test_path, 
                                          y_train_path, y_test_path, artifacts_dir)
        
        # Mostrar resultados
        print(f"Entrenamiento completado exitosamente")
        print(f"  - Modelos entrenados: {len(session.models_results)}")
        print(f"  - Mejor modelo: {session.best_model_name}")
        print(f"  - Mejor score: {session.best_model_score:.4f}")
        print(f"  - Experimento MLflow: {session.experiment_name}")
        
        # Mostrar métricas de todos los modelos
        print("\nResumen de modelos:")
        for result in session.models_results:
            print(f"  {result.model_name}: F1={result.metrics['f1_weighted']:.4f}, "
                  f"Accuracy={result.metrics['accuracy']:.4f}")
        
        # Guardar el mejor modelo para DVC
        import joblib
        from pathlib import Path
        
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        
        best_model_path = models_dir / "best_model.joblib"
        
        # Buscar el mejor modelo directamente
        best_result = None
        for result in session.models_results:
            if result.model_name == session.best_model_name:
                best_result = result
                break
        
        if best_result:
            joblib.dump(best_result.model, best_model_path)
            print(f"Mejor modelo guardado en: {best_model_path}")
        else:
            logger.error("No se pudo encontrar el mejor modelo para guardar")
        
        # Guardar métricas para DVC
        metrics_dir = Path("metrics")
        metrics_dir.mkdir(exist_ok=True)
        
        metrics_path = metrics_dir / "train_metrics.json"
        with open(metrics_path, 'w') as f:
            import json
            json.dump({
                'best_model_name': session.best_model_name,
                'best_model_score': session.best_model_score,
                'models_trained': len(session.models_results),
                'metrics': {result.model_name: result.metrics for result in session.models_results}
            }, f, indent=2)
        print(f"Métricas guardadas en: {metrics_path}")
        
    except Exception as e:
        logger.error(f"Error en entrenamiento de modelos: {e}")
        raise ModelTrainingError(f"Error en pipeline de entrenamiento: {e}")


if __name__ == "__main__":
    main()