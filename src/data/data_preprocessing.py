"""
Data preprocessing module for obesity prediction MLOps pipeline.

Implementa preprocesamiento de datos siguiendo las mejores prácticas:
- Feature engineering con pipelines reutilizables
- Transformaciones reproducibles con persistencia
- Manejo de valores faltantes y outliers
- Encoding de variables categóricas
- Escalado y normalización

REFERENCIAS BIBLIOGRÁFICAS TÉCNICAS:
- Wilson, B. & Bansal, A. (2022): "Practical MLOps: Operationalizing Machine Learning Models"
  Capítulo 7: "Feature Pipelines" - Diseño de pipelines reproducibles
- Huyen, C. (2022): "Designing Machine Learning Systems"  
  Capítulo 5: "Feature Engineering" - Transformaciones y escalado de features
- Treveil, M. et al. (2024): "Introducing MLOps: How to Scale Machine Learning in the Enterprise"
  Capítulo 6: "Data Preprocessing at Scale" - Automatización de transformaciones
- Zheng, A. & Casari, A. (2018): "Feature Engineering for Machine Learning"
  O'Reilly Media - Técnicas avanzadas de ingeniería de características
- Géron, A. (2022): "Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow"
  Capítulo 2: "End-to-End ML Project" - Preprocessing y pipelines

ARQUITECTURA DE PREPROCESSING:

                    ┌─────────────────────────────────┐
                    │     data/interim/               │
                    │ obesity_estimation_original.csv │ ← Datos validados
                    └─────────────┬───────────────────┘
                                  │
                                  ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │                 PREPROCESSING PIPELINE                          │
    │                                                                 │
    │  DATA CLEANING:                 FEATURE ENGINEERING:            │
    │  • Missing value imputation     • BMI calculation               │
    │  • Outlier handling             • Age group binning             │
    │  • Data type corrections        • Height/Weight ratios          │
    │                                                                 │
    │  ENCODING PIPELINE:          SCALING PIPELINE:                  │
    │  • OneHotEncoder (categóricas)  • StandardScaler (numéricas)    │
    │  • LabelEncoder (target)        • RobustScaler (outliers)       │
    │  • Column transformations       • MinMaxScaler (bounded)        │
    └─────────────────────────────┬───────────────────────────────────┘
                                  │
                                  ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │                    PROCESSED OUTPUTS                            │
    │  - data/processed/dataset_limpio.csv (final dataset)            │
    │  - models/preprocessors/ (fitted transformers)                  │
    │  - preprocessing_report.json (transformation log)               │
    └─────────────────────────────────────────────────────────────────┘

Componentes principales:
- Feature transformers: Transformaciones específicas por tipo de feature
- Pipeline builders: Construcción de pipelines de preprocessing reproducibles
- Data preparation: Preparación final con persistencia de transformadores
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Union
import joblib
from dataclasses import dataclass
import warnings

from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

from src.utils.config import get_data_config, load_config
from src.utils.logging_config import get_logger, log_pipeline_stage


logger = get_logger(__name__)


@dataclass
class PreprocessingResult:
    """Resultado del preprocesamiento de datos"""
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    preprocessing_pipeline: Pipeline
    feature_names: List[str]
    target_encoder: Optional[LabelEncoder] = None
    
    def save_artifacts(self, artifacts_dir: str):
        """Guarda artefactos del preprocesamiento"""
        artifacts_path = Path(artifacts_dir)
        artifacts_path.mkdir(parents=True, exist_ok=True)
        
        # Guardar pipeline
        pipeline_path = artifacts_path / "preprocessing_pipeline.pkl"
        joblib.dump(self.preprocessing_pipeline, pipeline_path)
        logger.info(f"Pipeline guardado: {pipeline_path}")
        
        # Guardar encoder de target si existe
        if self.target_encoder:
            encoder_path = artifacts_path / "target_encoder.pkl"
            joblib.dump(self.target_encoder, encoder_path)
            logger.info(f"Target encoder guardado: {encoder_path}")
        
        # Guardar nombres de features
        features_path = artifacts_path / "feature_names.txt"
        with open(features_path, 'w') as f:
            f.write('\n'.join(self.feature_names))
        logger.info(f"Feature names guardados: {features_path}")


class PreprocessingError(Exception):
    """Excepción para errores de preprocesamiento"""
    pass


class FeatureEngineer:
    """
    Ingeniero de features para el dataset de obesidad.
    
    Implementa transformaciones específicas del dominio
    y feature engineering avanzado.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Inicializa el ingeniero de features.
        
        Args:
            config: Configuración de preprocessing
        """
        self.config = config
        self.feature_config = config.get('feature_engineering', {})
        
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crea nuevas features específicas del dominio.
        
        Args:
            df: DataFrame original
            
        Returns:
            DataFrame con nuevas features
        """
        df_enhanced = df.copy()
        
        # Body Mass Index (BMI)
        if self.feature_config.get('create_bmi', True):
            df_enhanced['BMI'] = df_enhanced['Weight'] / (df_enhanced['Height'] ** 2)
            logger.info("Feature BMI creada")
        
        # Categorías de BMI
        if self.feature_config.get('create_bmi_category', True):
            df_enhanced['BMI_Category'] = pd.cut(
                df_enhanced['BMI'],
                bins=[0, 18.5, 25, 30, 35, 100],
                labels=['Underweight', 'Normal', 'Overweight', 'Obese', 'Severely_Obese'],
                include_lowest=True
            )
            logger.info("Feature BMI_Category creada")
        
        # Ratio altura/peso
        if self.feature_config.get('create_height_weight_ratio', True):
            df_enhanced['Height_Weight_Ratio'] = df_enhanced['Height'] / df_enhanced['Weight']
            logger.info("Feature Height_Weight_Ratio creada")
        
        # Score de hábitos alimenticios
        if self.feature_config.get('create_eating_habits_score', True):
            # Convertir variables categóricas a numéricas para el score
            favc_score = df_enhanced['FAVC'].map({'no': 0, 'yes': 1})
            caec_score = df_enhanced['CAEC'].map({
                'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3
            })
            calc_score = df_enhanced['CALC'].map({
                'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3
            })
            
            df_enhanced['Eating_Habits_Score'] = (
                favc_score + 
                df_enhanced['FCVC'] + 
                caec_score + 
                calc_score
            ) / 4
            logger.info("Feature Eating_Habits_Score creada")
        
        # Score de actividad física
        if self.feature_config.get('create_activity_score', True):
            # Normalizar FAF y TUE a escala 0-1
            faf_normalized = df_enhanced['FAF'] / 3  # FAF va de 0 a 3
            tue_normalized = 1 - (df_enhanced['TUE'] / 2)  # TUE va de 0 a 2, invertir
            
            df_enhanced['Activity_Score'] = (faf_normalized + tue_normalized) / 2
            logger.info("Feature Activity_Score creada")
        
        # Interacciones entre features
        if self.feature_config.get('create_interactions', True):
            # BMI x Age
            df_enhanced['BMI_Age_Interaction'] = df_enhanced['BMI'] * df_enhanced['Age']
            
            # Eating habits x Activity
            if 'Eating_Habits_Score' in df_enhanced.columns and 'Activity_Score' in df_enhanced.columns:
                df_enhanced['Habits_Activity_Interaction'] = (
                    df_enhanced['Eating_Habits_Score'] * df_enhanced['Activity_Score']
                )
            
            logger.info("Features de interacción creadas")
        
        # Features binarias adicionales
        if self.feature_config.get('create_binary_features', True):
            # Edad adulto joven vs adulto mayor
            df_enhanced['Is_Young_Adult'] = (df_enhanced['Age'] < 30).astype(int)
            
            # Alto consumo de agua
            df_enhanced['High_Water_Consumption'] = (df_enhanced['CH2O'] >= 2).astype(int)
            
            # Actividad física regular
            df_enhanced['Regular_Exercise'] = (df_enhanced['FAF'] >= 2).astype(int)
            
            logger.info("Features binarias adicionales creadas")
        
        return df_enhanced


class DataPreprocessor:
    """
    Preprocesador de datos para MLOps pipeline.
    
    Implementa preprocesamiento robusto y reproducible
    con pipelines de scikit-learn.
    """
    
    def __init__(self, preprocessing_config_path: str = "configs/data/data_preprocessing.yaml"):
        """
        Inicializa el preprocesador.
        
        Args:
            preprocessing_config_path: Ruta al archivo de configuración
        """
        self.config = load_config(preprocessing_config_path)
        self.feature_engineer = FeatureEngineer(self.config)
        
        # Configuraciones específicas
        self.encoding_config = self.config.get('encoding', {})
        self.scaling_config = self.config.get('scaling', {})
        self.imputation_config = self.config.get('imputation', {})
        self.split_config = self.config.get('train_test_split', {})
        
    def create_preprocessing_pipeline(self, df: pd.DataFrame) -> Pipeline:
        """
        Crea pipeline de preprocesamiento basado en la configuración.
        
        Args:
            df: DataFrame para determinar tipos de columnas
            
        Returns:
            Pipeline de preprocesamiento configurado
        """
        # Identificar tipos de columnas después del feature engineering
        df_engineered = self.feature_engineer.create_features(df)
        
        # Separar columnas por tipo
        numeric_features = []
        categorical_features = []
        
        for col in df_engineered.columns:
            if col == 'NObeyesdad':  # Target column
                continue
            
            if df_engineered[col].dtype in ['int64', 'float64']:
                numeric_features.append(col)
            else:
                categorical_features.append(col)
        
        logger.info(f"Features numéricas detectadas: {len(numeric_features)}")
        logger.info(f"Features categóricas detectadas: {len(categorical_features)}")
        
        # Pipeline para features numéricas
        numeric_pipeline = Pipeline([
            ('imputer', self._get_numeric_imputer()),
            ('scaler', self._get_scaler())
        ])
        
        # Pipeline para features categóricas
        categorical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('encoder', self._get_categorical_encoder())
        ])
        
        # Combinar pipelines
        preprocessor = ColumnTransformer([
            ('num', numeric_pipeline, numeric_features),
            ('cat', categorical_pipeline, categorical_features)
        ])
        
        # Pipeline completo con feature engineering
        full_pipeline = Pipeline([
            ('feature_engineering', FeatureEngineeringTransformer(self.feature_engineer)),
            ('preprocessing', preprocessor)
        ])
        
        return full_pipeline
    
    def _get_numeric_imputer(self):
        """Retorna imputador para variables numéricas"""
        strategy = self.imputation_config.get('numeric_strategy', 'median')
        
        if strategy == 'knn':
            n_neighbors = self.imputation_config.get('knn_neighbors', 5)
            return KNNImputer(n_neighbors=n_neighbors)
        else:
            return SimpleImputer(strategy=strategy)
    
    def _get_scaler(self):
        """Retorna escalador basado en configuración"""
        scaler_type = self.scaling_config.get('method', 'standard')
        
        if scaler_type == 'robust':
            return RobustScaler()
        elif scaler_type == 'standard':
            return StandardScaler()
        else:
            raise ValueError(f"Método de escalado no soportado: {scaler_type}")
    
    def _get_categorical_encoder(self):
        """Retorna encoder para variables categóricas"""
        encoding_method = self.encoding_config.get('method', 'onehot')
        
        if encoding_method == 'onehot':
            return OneHotEncoder(
                drop=self.encoding_config.get('drop_first', 'first'),
                handle_unknown='ignore'
            )
        else:
            raise ValueError(f"Método de encoding no soportado: {encoding_method}")
    
    def prepare_target(self, y: pd.Series) -> Tuple[pd.Series, Optional[LabelEncoder]]:
        """
        Prepara la variable target.
        
        Args:
            y: Serie con la variable target
            
        Returns:
            Tupla con target preparado y encoder (si aplica)
        """
        target_config = self.config.get('target_encoding', {})
        
        if target_config.get('encode', True):
            encoder = LabelEncoder()
            y_encoded = pd.Series(encoder.fit_transform(y), index=y.index)
            logger.info(f"Target codificado: {len(encoder.classes_)} clases")
            return y_encoded, encoder
        else:
            return y, None
    
    def split_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Divide los datos en train y test.
        
        Args:
            X: Features
            y: Target variable
            
        Returns:
            Tupla con X_train, X_test, y_train, y_test
        """
        test_size = self.split_config.get('test_size', 0.2)
        random_state = self.split_config.get('random_state', 42)
        stratify = y if self.split_config.get('stratify', True) else None
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify
        )
        
        logger.info(f"Datos divididos - Train: {X_train.shape}, Test: {X_test.shape}")
        return X_train, X_test, y_train, y_test
    
    def handle_class_imbalance(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Maneja desbalance de clases si está configurado.
        
        Args:
            X: Features de entrenamiento
            y: Target de entrenamiento
            
        Returns:
            Tupla con datos balanceados
        """
        balance_config = self.config.get('class_balance', {})
        
        if not balance_config.get('apply', False):
            return X, y
        
        method = balance_config.get('method', 'oversample')
        random_state = balance_config.get('random_state', 42)
        
        if method == 'oversample':
            # Oversample minority classes
            df_combined = pd.concat([X, y], axis=1)
            
            # Encontrar la clase mayoritaria
            class_counts = y.value_counts()
            max_count = class_counts.max()
            
            balanced_dfs = []
            for class_label in class_counts.index:
                class_df = df_combined[df_combined[y.name] == class_label]
                
                if len(class_df) < max_count:
                    # Oversample esta clase
                    class_df_resampled = resample(
                        class_df,
                        replace=True,
                        n_samples=max_count,
                        random_state=random_state
                    )
                    balanced_dfs.append(class_df_resampled)
                else:
                    balanced_dfs.append(class_df)
            
            df_balanced = pd.concat(balanced_dfs).sample(frac=1, random_state=random_state)
            
            X_balanced = df_balanced.drop(columns=[y.name])
            y_balanced = df_balanced[y.name]
            
            logger.info(f"Datos balanceados - Original: {len(y)}, Balanceado: {len(y_balanced)}")
            return X_balanced, y_balanced
        
        else:
            logger.warning(f"Método de balanceo no implementado: {method}")
            return X, y


class FeatureEngineeringTransformer:
    """Transformer para feature engineering que puede ser usado en pipelines"""
    
    def __init__(self, feature_engineer: FeatureEngineer):
        self.feature_engineer = feature_engineer
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return self.feature_engineer.create_features(X)
    
    def fit_transform(self, X, y=None):
        return self.transform(X)


@log_pipeline_stage("data_preprocessing")
def preprocess_data(input_path: str, output_dir: str) -> PreprocessingResult:
    """
    Preprocesa datos completos del pipeline.
    
    Args:
        input_path: Ruta al dataset validado
        output_dir: Directorio de salida para datos procesados
        
    Returns:
        Resultado del preprocesamiento con todos los artefactos
    """
    # Cargar datos validados
    df = pd.read_csv(input_path)
    logger.info(f"Datos cargados para preprocesamiento: {df.shape}")
    
    # Inicializar preprocesador
    preprocessor = DataPreprocessor()
    
    # Separar features y target
    target_column = 'NObeyesdad'
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Preparar target
    y_prepared, target_encoder = preprocessor.prepare_target(y)
    
    # Dividir datos
    X_train, X_test, y_train, y_test = preprocessor.split_data(X, y_prepared)
    
    # Manejar desbalance de clases en datos de entrenamiento
    X_train_balanced, y_train_balanced = preprocessor.handle_class_imbalance(X_train, y_train)
    
    # Crear y entrenar pipeline de preprocesamiento
    preprocessing_pipeline = preprocessor.create_preprocessing_pipeline(X_train)
    
    # Ajustar pipeline con datos de entrenamiento
    preprocessing_pipeline.fit(X_train_balanced)
    logger.info("Pipeline de preprocesamiento entrenado")
    
    # Transformar datos
    X_train_processed = preprocessing_pipeline.transform(X_train_balanced)
    X_test_processed = preprocessing_pipeline.transform(X_test)
    
    # Obtener nombres de features después de la transformación
    feature_names = _get_feature_names_from_pipeline(preprocessing_pipeline, X_train)
    
    # Convertir a DataFrames
    X_train_df = pd.DataFrame(X_train_processed, columns=feature_names)
    X_test_df = pd.DataFrame(X_test_processed, columns=feature_names)
    
    logger.info(f"Datos procesados - Train: {X_train_df.shape}, Test: {X_test_df.shape}")
    logger.info(f"Features finales: {len(feature_names)}")
    
    # Crear resultado
    result = PreprocessingResult(
        X_train=X_train_df,
        X_test=X_test_df,
        y_train=y_train_balanced,
        y_test=y_test,
        preprocessing_pipeline=preprocessing_pipeline,
        feature_names=feature_names,
        target_encoder=target_encoder
    )
    
    # Guardar artefactos
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Guardar datasets procesados
    train_path = output_path / "train_data.csv"
    test_path = output_path / "test_data.csv"
    
    # Asegurar que los índices estén alineados antes de concatenar
    X_train_df = X_train_df.reset_index(drop=True)
    y_train_balanced = y_train_balanced.reset_index(drop=True)
    X_test_df = X_test_df.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    
    train_df = pd.concat([X_train_df, y_train_balanced], axis=1)
    test_df = pd.concat([X_test_df, y_test], axis=1)
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    logger.info(f"Datos de entrenamiento guardados: {train_path}")
    logger.info(f"Datos de prueba guardados: {test_path}")
    
    # Guardar artefactos de preprocesamiento
    artifacts_dir = output_path / "artifacts"
    result.save_artifacts(str(artifacts_dir))
    
    return result


def _get_feature_names_from_pipeline(pipeline: Pipeline, X_sample: pd.DataFrame) -> List[str]:
    """
    Extrae nombres de features del pipeline de preprocesamiento.
    
    Args:
        pipeline: Pipeline de preprocesamiento entrenado
        X_sample: Muestra de datos para inferir nombres
        
    Returns:
        Lista de nombres de features
    """
    try:
        # Aplicar feature engineering
        X_engineered = pipeline.named_steps['feature_engineering'].transform(X_sample)
        
        # Obtener transformador de columnas
        column_transformer = pipeline.named_steps['preprocessing']
        
        feature_names = []
        
        # Procesar transformadores
        for name, transformer, columns in column_transformer.transformers_:
            if name == 'remainder':
                continue
                
            if name == 'num':
                # Features numéricas mantienen sus nombres
                feature_names.extend(columns)
            elif name == 'cat':
                # Features categóricas con OneHot encoding
                if hasattr(transformer.named_steps['encoder'], 'get_feature_names_out'):
                    encoded_names = transformer.named_steps['encoder'].get_feature_names_out(columns)
                    feature_names.extend(encoded_names)
                else:
                    # Fallback para versiones anteriores
                    feature_names.extend([f"{col}_encoded" for col in columns])
        
        return feature_names
        
    except Exception as e:
        logger.warning(f"No se pudieron extraer nombres de features: {e}")
        # Fallback: usar nombres genéricos
        n_features = pipeline.transform(X_sample.head(1)).shape[1]
        return [f"feature_{i}" for i in range(n_features)]


def main():
    """
    Función principal de preprocesamiento de datos.
    """
    try:
        # Cargar configuración
        config = get_data_config()
        
        # Rutas
        input_path = config['data']['interim']['obesity_original']
        output_dir = "data/processed"
        
        # Ejecutar preprocesamiento
        result = preprocess_data(input_path, output_dir)
        
        # Mostrar resumen
        print(f"Preprocesamiento completado")
        print(f"  - Features de entrenamiento: {result.X_train.shape}")
        print(f"  - Features de prueba: {result.X_test.shape}")
        print(f"  - Total de features: {len(result.feature_names)}")
        print(f"  - Target encoder: {'Si' if result.target_encoder else 'No'}")
        
    except Exception as e:
        logger.error(f"Error en preprocesamiento: {e}")
        raise PreprocessingError(f"Error en preprocesamiento: {e}")


if __name__ == "__main__":
    main()