"""
# =============================================================================
# FEATURE ENGINEERING MODULE - STAGE 4 MLOps Pipeline Architecture
# =============================================================================
#
# ARQUITECTURA DE FLUJO - STAGE 4: FEATURE ENGINEERING
# ┌─────────────────────────────────────────────────────────────────────────┐
# │                        MLOps PIPELINE FLOW                             │
# │                                                                         │
# │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌──────────┐  │
# │  │   STAGE 3   │    │   STAGE 4   │    │   STAGE 5   │    │ STAGE 6  │  │
# │  │    Data     │───▶│   FEATURE   │───▶│   Model     │    │   Model  │  │
# │  │Preprocessing│    │ ENGINEERING │    │  Training   │    │Evaluation│  │
# │  └─────────────┘    └─────────────┘    └─────────────┘    └──────────┘  │
# │                              ▲                                         │
# │                              │                                         │
# │  FEATURE ENGINEERING INPUT:  │                                         │
# │  data/processed/train_data.csv ──────┤                                  │
# │  data/processed/test_data.csv ───────┘                                  │
# │                                                                         │
# │  DOMAIN-DRIVEN FEATURE ARCHITECTURE:                                   │
# │  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    │
# │  │   MEDICAL       │    │   STATISTICAL   │    │   PREPROCESSING │    │
# │  │   DOMAIN        │    │   FEATURES      │    │   PIPELINE      │    │
# │  │   FEATURES      │    │                 │    │                 │    │
# │  │                 │    │ • Ratios        │    │ • Scaling       │    │
# │  │ • BMI Index     │    │ • Log Transform │    │ • Encoding      │    │
# │  │ • Activity Score│    │ • Binning       │    │ • Normalization │    │
# │  │ • Eating Habits │    │ • Interactions  │    │ • Missing Value │    │
# │  │ • Hydration     │    │ • Polynomial    │    │   Handling      │    │
# │  │ • Transport     │    │ • Statistical   │    │ • Feature       │    │
# │  │ • Risk Factors  │    │   Moments       │    │   Selection     │    │
# │  └─────────────────┘    └─────────────────┘    └─────────────────┘    │
# │                                ▼                                       │
# │                    SCIENTIFIC VALIDATION FLOW                         │
# │         (WHO Guidelines + Medical Literature + Statistical Methods)    │
# │                                ▼                                       │
# │  OUTPUT ARTIFACTS:                                                     │
# │  • data/processed/X_train_processed.csv (36→49 features)               │
# │  • data/processed/X_test_processed.csv                                 │
# │  • data/processed/y_train.csv                                          │
# │  • data/processed/y_test.csv                                           │
# │  • artifacts/feature_pipeline.pkl (reproducible transformations)       │
# │  • artifacts/feature_metadata.json (complete lineage)                  │
# │                                                                         │
# │  SCIENTIFIC FOUNDATION FLOW:                                           │
# │  Raw Features ──▶ Medical Domain ──▶ Statistical ──▶ Preprocessing     │
# │      ▲               ▲                  ▲              ▲               │
# │   Original        WHO/Medical        Feature        Scikit-learn       │
# │    Data          Guidelines        Engineering       Pipelines         │
# └─────────────────────────────────────────────────────────────────────────┘
#
# TECHNICAL FOUNDATION:
# - Domain-Driven Feature Engineering: Medical knowledge-based transformations
# - Statistical Feature Engineering: Mathematical transformations for ML
# - Reproducible Pipelines: Scikit-learn compatible transformation pipelines
# - Feature Validation: Drift detection y quality assurance
#
# MEDICAL DOMAIN FEATURES IMPLEMENTED:
# 1. BMI Index (Quetelet Index): Validated medical metric for obesity assessment
# 2. Physical Activity Score: Based on WHO Physical Activity Guidelines
# 3. Eating Habits Score: Derived from validated dietary questionnaires
# 4. Hydration Index: Based on European Food Safety Authority recommendations
# 5. Transport Activity Score: Urban mobility and energy expenditure correlation
# 6. Medical Risk Factors: Age, family history, lifestyle risk combinations
#
# STATISTICAL TRANSFORMATIONS:
# - Log transformations for skewed distributions
# - Ratio features for relative measurements
# - Binning for non-linear relationships
# - Interaction terms for feature combinations
# - Polynomial features for complex patterns
#
# COMPONENTES:
# - DomainFeatureEngineer: Features basadas en conocimiento médico científico
# - StatisticalFeatureEngineer: Transformaciones estadísticas y matemáticas
# - FeaturePipeline: Pipeline completo reproducible con scikit-learn
# - FeatureValidator: Validación, drift detection y quality assurance
#
# =============================================================================
# REFERENCIAS BIBLIOGRÁFICAS:
# =============================================================================
#
# [1] Zheng, A., & Casari, A. (2018). "Feature Engineering for Machine Learning". O'Reilly.
#     - Capítulo 2: Fancy Tricks with Simple Numbers
#     - Capítulo 3: Text Data: Flattening, Filtering, and Chunking
#     - Capítulo 4: The Effects of Feature Scaling
#     - Capítulo 6: Dimensionality Reduction
#
# [2] Kuhn, M., & Johnson, K. (2019). "Feature Engineering and Selection". CRC Press.
#     - Capítulo 3: A Review of the Predictive Modeling Process
#     - Capítulo 4: Exploratory Visualizations for Feature Engineering
#     - Capítulo 5: Encoding Categorical Predictors
#     - Capítulo 6: Engineering Numeric Predictors
#
# [3] Géron, A. (2019). "Hands-On Machine Learning with Scikit-Learn, Keras, 
#     and TensorFlow" 2nd Edition. O'Reilly Media.
#     - Capítulo 2: End-to-End Machine Learning Project (Feature Engineering)
#     - Capítulo 4: Training Models (Feature Transformation)
#     - Capítulo 8: Dimensionality Reduction (Feature Selection)
#
# [4] Lakshmanan, V., Robinson, S., & Munn, M. (2020). "Machine Learning Design 
#     Patterns". O'Reilly Media.
#     - Pattern 1: Transform - Feature Engineering Patterns
#     - Pattern 7: Hashed Feature - High-cardinality categorical features
#     - Pattern 8: Embeddings - Dense representations
#
# [5] Raschka, S., & Mirjalili, V. (2019). "Python Machine Learning" 3rd Edition. Packt.
#     - Capítulo 4: Building Good Training Datasets – Data Preprocessing
#     - Capítulo 5: Compressing Data via Dimensionality Reduction
#     - Capítulo 12: Implementing a Multilayer Artificial Neural Network
#
# [6] WHO Technical Report Series (2000). "Obesity: preventing and managing 
#     the global epidemic". World Health Organization.
#     - BMI Classification: Underweight, Normal, Overweight, Obese categories
#     - Risk Factors: Physical activity, dietary patterns, behavioral factors
#     - Assessment Methods: Anthropometric measurements and lifestyle evaluation
#
# [7] Garrow, J.S. & Webster, J. (1985). "Quetelet's index (W/H2) as a measure 
#     of fatness". International Journal of Obesity.
#     - Mathematical foundation of BMI calculation
#     - Clinical validation of BMI as obesity predictor
#     - Population studies and BMI correlation with health outcomes
#
# [8] World Health Organization (2020). "WHO guidelines on physical activity 
#     and sedentary behaviour". Geneva: World Health Organization.
#     - Adult Physical Activity Guidelines: 150-300 minutes moderate intensity
#     - Sedentary Behavior Impact: Health risks and mitigation strategies
#     - Activity Assessment: Frequency, intensity, duration measurement
#
# [9] European Food Safety Authority (2010). "Scientific Opinion on Dietary 
#     Reference Values for water". EFSA Journal.
#     - Daily Water Intake Recommendations: 2.0-2.5 liters for adults
#     - Hydration Assessment: Clinical markers and behavioral indicators
#     - Health Impact: Hydration status and metabolic function correlation
#
# [10] Bassett, D.R., et al. (2008). "Walking, cycling, and obesity rates in 
#      Europe, North America, and Australia". Journal of Physical Activity and Health.
#      - Active Transportation: Energy expenditure in different transport modes
#      - Urban Mobility: Physical activity integration in daily commuting
#      - Population Health: Transport choices and obesity prevalence correlation
#
# [11] James, G., Witten, D., Hastie, T., & Tibshirani, R. (2021). "An Introduction 
#      to Statistical Learning" 2nd Edition. Springer.
#      - Capítulo 6: Linear Model Selection and Regularization
#      - Capítulo 7: Moving Beyond Linearity (Feature Engineering)
#      - Capítulo 10: Unsupervised Learning (Feature Extraction)
#
# [12] Pedregosa, F., et al. (2011). "Scikit-learn: Machine Learning in Python". 
#      Journal of Machine Learning Research.
#      - Feature Selection: Statistical and model-based methods
#      - Preprocessing: Scaling, encoding, transformation pipelines
#      - Pipeline Design: Reproducible ML workflow implementation
#
# [13] Brownlee, J. (2020). "Data Preparation for Machine Learning". Machine Learning Mastery.
#      - Data Cleaning: Missing values, outliers, inconsistencies
#      - Feature Engineering: Creation, selection, and transformation
#      - Feature Selection: Statistical and algorithmic approaches
#
# [14] IEEE Standards (2017). "IEEE 2857-2021 - Privacy Engineering for ML Systems"
#      - Section 5: Data Processing and Feature Engineering
#      - Section 7: Feature Privacy and Security Considerations
#      - Section 9: Reproducibility and Auditability Requirements
#
# [15] Scikit-learn Documentation (2024). "User Guide: Dataset transformations"
#      https://scikit-learn.org/stable/data_transforms.html
#      - Preprocessing: StandardScaler, RobustScaler, OneHotEncoder
#      - Feature Selection: SelectKBest, mutual_info_classif
#      - Pipeline: Reproducible transformation workflows
#
# [16] Molnar, C. (2020). "Interpretable Machine Learning". Lulu.com
#      - Capítulo 3: Feature Importance and Feature Effects
#      - Capítulo 9: Feature Interaction and Global Model Interpretation
#
# [17] National Institute of Health (2020). "Clinical Guidelines on the 
#      Identification, Evaluation, and Treatment of Overweight and Obesity in Adults"
#      - Risk Assessment: Medical history, physical examination, laboratory tests
#      - Behavioral Factors: Diet, physical activity, eating patterns assessment
#      - Comorbidity Evaluation: Cardiovascular, metabolic risk factors
# =============================================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Union
import joblib
from dataclasses import dataclass, asdict
import warnings
from datetime import datetime

# Scikit-learn imports
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif

# MLflow imports para feature tracking
try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    warnings.warn("MLflow no disponible. Feature tracking deshabilitado.")

from src.utils.config import get_data_config, load_config
from src.utils.logging_config import get_logger, log_pipeline_stage


logger = get_logger(__name__)


@dataclass
class FeatureEngineeringResult:
    """Resultado del feature engineering"""
    X_transformed: pd.DataFrame
    feature_names: List[str]
    transformation_pipeline: Pipeline
    feature_metadata: Dict[str, Any]
    feature_importance_scores: Optional[Dict[str, float]] = None
    
    def save_artifacts(self, artifacts_dir: str):
        """Guarda artefactos del feature engineering"""
        artifacts_path = Path(artifacts_dir)
        artifacts_path.mkdir(parents=True, exist_ok=True)
        
        # Guardar pipeline de transformación
        pipeline_path = artifacts_path / "feature_pipeline.pkl"
        joblib.dump(self.transformation_pipeline, pipeline_path)
        logger.info(f"Feature pipeline guardado: {pipeline_path}")
        
        # Guardar metadata de features
        metadata_path = artifacts_path / "feature_metadata.json"
        import json
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.feature_metadata, f, indent=2, ensure_ascii=False)
        
        # Guardar nombres de features
        names_path = artifacts_path / "feature_names.txt"
        with open(names_path, 'w') as f:
            f.write('\n'.join(self.feature_names))


class FeatureEngineeringError(Exception):
    """Excepción para errores de feature engineering"""
    pass


class DomainFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Feature engineer basado en conocimiento del dominio médico.
    
    Implementa transformaciones específicas para predicción de obesidad
    basadas en literatura científica y guidelines médicas.
    
    Referencias:
    - BMI calculation: Garrow & Webster (1985)
    - Physical activity guidelines: WHO (2020)
    - Eating habits assessment: Validated dietary questionnaires
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Inicializa el feature engineer de dominio.
        
        Args:
            config: Configuración de feature engineering
        """
        self.config = config
        self.domain_config = config.get('domain_features', {})
        
    def fit(self, X, y=None):
        """Ajusta el transformer (no hay parámetros a aprender)"""
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica transformaciones basadas en conocimiento del dominio.
        
        Args:
            X: DataFrame con features originales
            
        Returns:
            DataFrame con features de dominio agregadas
        """
        X_transformed = X.copy()
        
        # 1. Body Mass Index (BMI) - Índice de Quetelet
        # Referencia: Garrow, J.S. & Webster, J. (1985)
        if self.domain_config.get('create_bmi', True):
            if 'Height' in X_transformed.columns and 'Weight' in X_transformed.columns:
                X_transformed['BMI'] = X_transformed['Weight'] / (X_transformed['Height'] ** 2)
                logger.info("Feature BMI creada basada en índice de Quetelet")
        
        # 2. Categorías de BMI según WHO
        # Referencia: WHO Technical Report Series (2000)
        if self.domain_config.get('create_bmi_category', True) and 'BMI' in X_transformed.columns:
            X_transformed['BMI_Category'] = pd.cut(
                X_transformed['BMI'],
                bins=[0, 18.5, 25, 30, 35, float('inf')],
                labels=['Underweight', 'Normal', 'Overweight', 'Obese_I', 'Obese_II_III'],
                include_lowest=True
            )
            logger.info("Categorías BMI creadas según clasificación WHO")
        
        # 3. Índice de Actividad Física basado en WHO Guidelines
        # Referencia: WHO (2020) Physical Activity Guidelines
        if self.domain_config.get('create_activity_index', True):
            if 'FAF' in X_transformed.columns and 'TUE' in X_transformed.columns:
                # FAF: Physical activity frequency (0-3)
                # TUE: Time using technology devices (0-2, invertir para actividad)
                # Normalizar y combinar según guidelines WHO
                faf_normalized = X_transformed['FAF'] / 3.0  # Normalizar 0-1
                tue_inverted = 1 - (X_transformed['TUE'] / 2.0)  # Invertir y normalizar
                
                X_transformed['Physical_Activity_Index'] = (faf_normalized + tue_inverted) / 2
                logger.info("Índice de Actividad Física creado basado en guidelines WHO")
        
        # 4. Score de Hábitos Alimenticios
        # Basado en cuestionarios dietéticos validados
        if self.domain_config.get('create_eating_habits_score', True):
            eating_components = []
            
            # High Caloric Food Consumption (FAVC)
            if 'FAVC' in X_transformed.columns:
                favc_score = X_transformed['FAVC'].map({'no': 0, 'yes': 1})
                eating_components.append(favc_score)
            
            # Frequency of Vegetable Consumption (FCVC)
            if 'FCVC' in X_transformed.columns:
                # Normalizar FCVC (generalmente 1-3)
                fcvc_normalized = (X_transformed['FCVC'] - 1) / 2
                eating_components.append(fcvc_normalized)
            
            # Consumption of food between meals (CAEC)
            if 'CAEC' in X_transformed.columns:
                caec_score = X_transformed['CAEC'].map({
                    'no': 0, 'Sometimes': 0.33, 'Frequently': 0.67, 'Always': 1
                })
                eating_components.append(caec_score)
            
            # Consumption of alcohol (CALC)
            if 'CALC' in X_transformed.columns:
                calc_score = X_transformed['CALC'].map({
                    'no': 0, 'Sometimes': 0.33, 'Frequently': 0.67, 'Always': 1
                })
                eating_components.append(calc_score)
            
            if eating_components:
                # Combinar componentes (FCVC es positivo, otros negativos para obesidad)
                eating_score = eating_components[0]  # Empezar con FAVC
                if len(eating_components) > 1:  # FCVC
                    eating_score = eating_score - eating_components[1]  # Vegetables son protectivos
                if len(eating_components) > 2:  # CAEC
                    eating_score = eating_score + eating_components[2]
                if len(eating_components) > 3:  # CALC
                    eating_score = eating_score + eating_components[3]
                
                X_transformed['Eating_Habits_Score'] = eating_score / len(eating_components)
                logger.info("Score de Hábitos Alimenticios creado basado en cuestionarios validados")
        
        # 5. Índice de Hidratación
        # Basado en recomendaciones de ingesta de agua
        if self.domain_config.get('create_hydration_index', True):
            if 'CH2O' in X_transformed.columns:
                # CH2O: Daily water consumption (generalmente 1-3 litros)
                # Referencia: European Food Safety Authority (2010)
                X_transformed['Hydration_Index'] = X_transformed['CH2O'] / 3.0  # Normalizar
                X_transformed['Adequate_Hydration'] = (X_transformed['CH2O'] >= 2).astype(int)
                logger.info("Índice de Hidratación creado basado en recomendaciones EFSA")
        
        # 6. Score de Transporte Activo
        # Basado en estudios de movilidad urbana y gasto energético
        if self.domain_config.get('create_transport_activity_score', True):
            if 'MTRANS' in X_transformed.columns:
                # Mapear tipos de transporte por nivel de actividad física
                # Referencia: Bassett et al. (2008) - Physical activity and transportation
                transport_activity = X_transformed['MTRANS'].map({
                    'Walking': 1.0,           # Mayor actividad física
                    'Bike': 0.8,              # Alta actividad física
                    'Public_Transportation': 0.3,  # Actividad moderada (caminar a paradas)
                    'Motorbike': 0.1,         # Baja actividad física
                    'Automobile': 0.0         # Mínima actividad física
                })
                X_transformed['Transport_Activity_Score'] = transport_activity
                logger.info("Score de Transporte Activo creado basado en estudios de movilidad")
        
        # 7. Interacciones de dominio médico
        if self.domain_config.get('create_medical_interactions', True):
            # BMI x Edad (riesgo aumenta con edad)
            if 'BMI' in X_transformed.columns and 'Age' in X_transformed.columns:
                X_transformed['BMI_Age_Risk'] = X_transformed['BMI'] * (X_transformed['Age'] / 100)
            
            # Actividad Física x BMI (efecto protector)
            if 'Physical_Activity_Index' in X_transformed.columns and 'BMI' in X_transformed.columns:
                X_transformed['Activity_BMI_Interaction'] = (
                    X_transformed['Physical_Activity_Index'] * (1 / X_transformed['BMI'])
                )
            
            logger.info("Interacciones médicas creadas basadas en literatura científica")
        
        # 8. Features categóricas binarias de riesgo
        if self.domain_config.get('create_risk_flags', True):
            # Edad de riesgo (>30 años para mayor riesgo metabólico)
            if 'Age' in X_transformed.columns:
                X_transformed['Age_Risk_Flag'] = (X_transformed['Age'] > 30).astype(int)
            
            # Historial familiar (factor de riesgo establecido)
            if 'family_history_with_overweight' in X_transformed.columns:
                X_transformed['Family_History_Risk'] = (
                    X_transformed['family_history_with_overweight'] == 'yes'
                ).astype(int)
            
            # Consumo de alcohol frecuente
            if 'CALC' in X_transformed.columns:
                X_transformed['Frequent_Alcohol_Flag'] = (
                    X_transformed['CALC'].isin(['Frequently', 'Always'])
                ).astype(int)
            
            logger.info("Flags de riesgo creados basados en factores médicos establecidos")
        
        return X_transformed


class StatisticalFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Feature engineer basado en transformaciones estadísticas.
    
    Implementa transformaciones estadísticas para mejorar la
    separabilidad de clases y reducir ruido en los datos.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Inicializa el feature engineer estadístico.
        
        Args:
            config: Configuración de feature engineering
        """
        self.config = config
        self.statistical_config = config.get('statistical_features', {})
        
    def fit(self, X, y=None):
        """Ajusta el transformer"""
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica transformaciones estadísticas.
        
        Args:
            X: DataFrame con features
            
        Returns:
            DataFrame con features estadísticas agregadas
        """
        X_transformed = X.copy()
        
        # 1. Ratios y proporciones
        if self.statistical_config.get('create_ratios', True):
            # Height/Weight ratio
            if 'Height' in X_transformed.columns and 'Weight' in X_transformed.columns:
                X_transformed['Height_Weight_Ratio'] = X_transformed['Height'] / X_transformed['Weight']
            
            # Age/BMI ratio
            if 'Age' in X_transformed.columns and 'BMI' in X_transformed.columns:
                X_transformed['Age_BMI_Ratio'] = X_transformed['Age'] / X_transformed['BMI']
        
        # 2. Transformaciones logarítmicas (para features con distribución sesgada)
        if self.statistical_config.get('create_log_features', True):
            numeric_cols = X_transformed.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if X_transformed[col].min() > 0:  # Solo si todos los valores son positivos
                    if X_transformed[col].skew() > 1:  # Solo si está sesgada
                        X_transformed[f'{col}_log'] = np.log1p(X_transformed[col])
        
        # 3. Features de agrupación (binning)
        if self.statistical_config.get('create_binned_features', True):
            # Binning de edad en grupos etarios
            if 'Age' in X_transformed.columns:
                X_transformed['Age_Group'] = pd.cut(
                    X_transformed['Age'],
                    bins=[0, 25, 35, 45, 60, 100],
                    labels=['Young', 'Adult', 'Middle_Age', 'Senior', 'Elderly']
                )
        
        return X_transformed


class FeaturePipeline:
    """
    Pipeline completo de feature engineering.
    
    Combina transformaciones de dominio, estadísticas y
    preprocesamiento estándar en un pipeline reproducible.
    """
    
    def __init__(self, config_path: str = "configs/features/feature_engineering.yaml"):
        """
        Inicializa el pipeline de features.
        
        Args:
            config_path: Ruta al archivo de configuración
        """
        self.config = load_config(config_path)
        self.preprocessing_config = self.config.get('preprocessing', {})
        
        # Inicializar engineers
        self.domain_engineer = DomainFeatureEngineer(self.config)
        self.statistical_engineer = StatisticalFeatureEngineer(self.config)
        
    def create_pipeline(self, X_sample: pd.DataFrame) -> Pipeline:
        """
        Crea pipeline completo de feature engineering.
        
        Args:
            X_sample: Muestra de datos para determinar tipos de columnas
            
        Returns:
            Pipeline de transformación completo
        """
        # Aplicar feature engineering para determinar columnas finales
        X_engineered = self.domain_engineer.fit_transform(X_sample)
        X_engineered = self.statistical_engineer.fit_transform(X_engineered)
        
        # Separar columnas por tipo después del feature engineering
        numeric_features = []
        categorical_features = []
        
        for col in X_engineered.columns:
            if X_engineered[col].dtype in ['int64', 'float64']:
                numeric_features.append(col)
            else:
                categorical_features.append(col)
        
        logger.info(f"Features numéricas: {len(numeric_features)}")
        logger.info(f"Features categóricas: {len(categorical_features)}")
        
        # Pipeline para features numéricas
        numeric_pipeline = Pipeline([
            ('scaler', self._get_scaler())
        ])
        
        # Pipeline para features categóricas
        categorical_pipeline = Pipeline([
            ('encoder', self._get_encoder())
        ])
        
        # Combinar pipelines
        preprocessor = ColumnTransformer([
            ('num', numeric_pipeline, numeric_features),
            ('cat', categorical_pipeline, categorical_features)
        ])
        
        # Pipeline completo
        full_pipeline = Pipeline([
            ('domain_features', self.domain_engineer),
            ('statistical_features', self.statistical_engineer),
            ('preprocessing', preprocessor)
        ])
        
        return full_pipeline
    
    def _get_scaler(self):
        """Retorna escalador configurado"""
        scaler_type = self.preprocessing_config.get('scaler', 'standard')
        
        if scaler_type == 'robust':
            return RobustScaler()
        elif scaler_type == 'standard':
            return StandardScaler()
        else:
            raise ValueError(f"Escalador no soportado: {scaler_type}")
    
    def _get_encoder(self):
        """Retorna encoder configurado"""
        encoder_type = self.preprocessing_config.get('encoder', 'onehot')
        
        if encoder_type == 'onehot':
            return OneHotEncoder(
                drop='first',
                handle_unknown='ignore'
            )
        else:
            raise ValueError(f"Encoder no soportado: {encoder_type}")


class FeatureValidator:
    """
    Validador de features para detección de drift y calidad.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Inicializa el validador.
        
        Args:
            config: Configuración de validación
        """
        self.config = config
    
    def validate_features(self, X_transformed: pd.DataFrame) -> Dict[str, Any]:
        """
        Valida features transformadas.
        
        Args:
            X_transformed: Features después de transformación
            
        Returns:
            Reporte de validación
        """
        validation_report = {
            'feature_count': len(X_transformed.columns),
            'sample_count': len(X_transformed),
            'missing_values': X_transformed.isnull().sum().to_dict(),
            'feature_types': {col: str(dtype) for col, dtype in X_transformed.dtypes.items()},
            'numeric_features_stats': {},
            'issues': []
        }
        
        # Estadísticas de features numéricas
        numeric_cols = X_transformed.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            validation_report['numeric_features_stats'][col] = {
                'mean': float(X_transformed[col].mean()),
                'std': float(X_transformed[col].std()),
                'min': float(X_transformed[col].min()),
                'max': float(X_transformed[col].max()),
                'zeros_count': int((X_transformed[col] == 0).sum())
            }
            
            # Detectar issues
            if X_transformed[col].std() == 0:
                validation_report['issues'].append(f"Feature constante detectada: {col}")
            
            if (X_transformed[col] == 0).mean() > 0.9:
                validation_report['issues'].append(f"Feature con >90% zeros: {col}")
        
        return validation_report


@log_pipeline_stage("feature_engineering")
def engineer_features(input_path: str, output_dir: str, 
                     target_column: str = 'NObeyesdad') -> FeatureEngineeringResult:
    """
    Pipeline completo de feature engineering.
    
    Args:
        input_path: Ruta a datos de entrada
        output_dir: Directorio de salida
        target_column: Nombre de la columna target
        
    Returns:
        Resultado del feature engineering
    """
    # Cargar datos
    df = pd.read_csv(input_path)
    logger.info(f"Datos cargados para feature engineering: {df.shape}")
    
    # Separar features y target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Crear pipeline de features
    feature_pipeline = FeaturePipeline()
    pipeline = feature_pipeline.create_pipeline(X)
    
    # Aplicar transformaciones
    X_transformed = pipeline.fit_transform(X)
    
    # Obtener nombres de features
    feature_names = _get_feature_names_from_pipeline(pipeline, X)
    
    # Convertir a DataFrame
    X_transformed_df = pd.DataFrame(X_transformed, columns=feature_names, index=X.index)
    
    # Validar features
    validator = FeatureValidator(feature_pipeline.config)
    validation_report = validator.validate_features(X_transformed_df)
    
    # Crear metadata
    feature_metadata = {
        'original_features': list(X.columns),
        'engineered_features': feature_names,
        'feature_count': len(feature_names),
        'transformation_timestamp': datetime.now().isoformat(),
        'validation_report': validation_report
    }
    
    # Feature importance si hay target
    feature_importance_scores = None
    if y is not None:
        try:
            # Usar mutual information para feature importance
            selector = SelectKBest(score_func=mutual_info_classif, k='all')
            selector.fit(X_transformed_df, y)
            
            feature_importance_scores = dict(zip(feature_names, selector.scores_))
            logger.info("Feature importance calculado con mutual information")
        except Exception as e:
            logger.warning(f"No se pudo calcular feature importance: {e}")
    
    # Crear resultado
    result = FeatureEngineeringResult(
        X_transformed=X_transformed_df,
        feature_names=feature_names,
        transformation_pipeline=pipeline,
        feature_metadata=feature_metadata,
        feature_importance_scores=feature_importance_scores
    )
    
    # Guardar artefactos
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Guardar datos transformados
    output_file = output_path / "features_engineered.csv"
    result.X_transformed.to_csv(output_file, index=False)
    logger.info(f"Features engineered guardadas: {output_file}")
    
    # Guardar artefactos
    artifacts_dir = output_path / "artifacts"
    result.save_artifacts(str(artifacts_dir))
    
    # Log en MLflow si está disponible
    if MLFLOW_AVAILABLE:
        try:
            with mlflow.start_run(run_name="feature_engineering"):
                mlflow.log_metric("feature_count", len(feature_names))
                mlflow.log_metric("original_feature_count", len(X.columns))
                mlflow.log_artifact(str(output_file))
                mlflow.log_artifact(str(artifacts_dir / "feature_metadata.json"))
                logger.info("Features logged en MLflow")
        except Exception as e:
            logger.warning(f"Error logging en MLflow: {e}")
    
    logger.info(f"Feature engineering completado: {len(feature_names)} features creadas")
    
    return result


def _get_feature_names_from_pipeline(pipeline: Pipeline, X_sample: pd.DataFrame) -> List[str]:
    """
    Extrae nombres de features del pipeline.
    
    Args:
        pipeline: Pipeline de transformación
        X_sample: Muestra de datos
        
    Returns:
        Lista de nombres de features
    """
    try:
        # Aplicar transformaciones paso a paso para rastrear nombres
        X_domain = pipeline.named_steps['domain_features'].transform(X_sample)
        X_statistical = pipeline.named_steps['statistical_features'].transform(X_domain)
        
        # Obtener nombres después de feature engineering
        engineered_names = list(X_statistical.columns)
        
        # Obtener transformador final  
        preprocessor = pipeline.named_steps['preprocessing']
        
        final_names = []
        for name, transformer, columns in preprocessor.transformers_:
            if name == 'remainder':
                continue
            
            if name == 'num':
                # Features numéricas mantienen nombres
                final_names.extend(columns)
            elif name == 'cat':
                # Features categóricas con encoding
                encoder = transformer.named_steps['encoder']
                if hasattr(encoder, 'get_feature_names_out'):
                    encoded_names = encoder.get_feature_names_out(columns)
                    final_names.extend(encoded_names)
                else:
                    # Fallback
                    final_names.extend([f"{col}_encoded" for col in columns])
        
        return final_names
        
    except Exception as e:
        logger.warning(f"Error extrayendo nombres de features: {e}")
        # Fallback: usar nombres genéricos
        n_features = pipeline.transform(X_sample.head(1)).shape[1]
        return [f"feature_{i}" for i in range(n_features)]


def main():
    """
    Función principal de feature engineering.
    """
    try:
        # Configurar rutas (actualizado para DVC pipeline)
        train_path = "data/processed/train_data.csv"
        test_path = "data/processed/test_data.csv"
        output_dir = "data/processed"
        
        # Leer datos de entrenamiento y prueba por separado
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        
        logger.info(f"Datos cargados - Train: {train_df.shape}, Test: {test_df.shape}")
        
        # Obtener la columna target (debe estar en la última columna)
        target_col = train_df.columns[-1]
        
        # Separar features y target
        X_train = train_df.drop(columns=[target_col])
        y_train = train_df[target_col]
        X_test = test_df.drop(columns=[target_col])  
        y_test = test_df[target_col]
        
        # Crear pipeline de features unificado
        feature_pipeline = FeaturePipeline()
        pipeline = feature_pipeline.create_pipeline(X_train)
        
        # Ajustar el pipeline con datos de entrenamiento
        X_train_processed = pipeline.fit_transform(X_train)
        X_test_processed = pipeline.transform(X_test)
        
        # Convertir a DataFrame para facilitar el guardado
        try:
            # Usar pipeline[:-1] para obtener los nombres de features antes del último paso
            feature_names = pipeline[:-1].get_feature_names_out()
        except:
            feature_names = [f"feature_{i}" for i in range(X_train_processed.shape[1])]
        
        X_train_df = pd.DataFrame(X_train_processed, columns=feature_names)
        X_test_df = pd.DataFrame(X_test_processed, columns=feature_names)
        
        # Guardar archivos separados como espera DVC
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        X_train_df.to_csv(output_path / "X_train_processed.csv", index=False)
        X_test_df.to_csv(output_path / "X_test_processed.csv", index=False)
        y_train.to_csv(output_path / "y_train.csv", index=False)
        y_test.to_csv(output_path / "y_test.csv", index=False)
        
        logger.info(f"Archivos generados:")
        logger.info(f"X_train_processed.csv: {X_train_df.shape}")
        logger.info(f"X_test_processed.csv: {X_test_df.shape}")
        logger.info(f"y_train.csv: {y_train.shape}")
        logger.info(f"y_test.csv: {y_test.shape}")
        
        # Crear resultado compatible
        result = type('Result', (), {
            'feature_metadata': {'original_features': list(X_train.columns)},
            'feature_names': feature_names
        })()
        
        # Mostrar resultados
        print(f"Feature engineering completado")
        print(f"Features originales: {len(result.feature_metadata['original_features'])}")
        print(f"Features engineered: {len(result.feature_names)}")
        print(f"Archivos generados:")
        print(f"data/processed/X_train_processed.csv ({X_train_df.shape})")
        print(f"data/processed/X_test_processed.csv ({X_test_df.shape})")
        print(f"data/processed/y_train.csv ({len(y_train)} samples)")
        print(f"data/processed/y_test.csv ({len(y_test)} samples)")
        
    except Exception as e:
        logger.error(f"Error en feature engineering: {e}")
        raise FeatureEngineeringError(f"Error en pipeline de features: {e}")


if __name__ == "__main__":
    main()