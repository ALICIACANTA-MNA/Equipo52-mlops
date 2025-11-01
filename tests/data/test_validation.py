"""
Tests de validación de datos para el proyecto MLOps.
Tests para verificar calidad, integridad y consistencia de los datos.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Agregar el path de src
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

class TestDataQuality:
    """Test suite para calidad de datos."""
    
    def test_no_missing_values_critical_columns(self, sample_obesity_data):
        """Test que no hay valores faltantes en columnas críticas."""
        critical_columns = ['Gender', 'Age', 'Height', 'Weight', 'NObeyesdad']
        
        for col in critical_columns:
            if col in sample_obesity_data.columns:
                missing_count = sample_obesity_data[col].isnull().sum()
                assert missing_count == 0, f"Column {col} has {missing_count} missing values"
    
    def test_no_duplicate_rows(self, sample_obesity_data):
        """Test que no hay filas duplicadas."""
        duplicate_count = sample_obesity_data.duplicated().sum()
        assert duplicate_count == 0, f"Found {duplicate_count} duplicate rows"
    
    def test_data_types_consistency(self, sample_obesity_data):
        """Test consistencia de tipos de datos."""
        expected_types = {
            'Gender': 'object',
            'Age': ['int64', 'int32', 'float64'],
            'Height': ['float64', 'float32'],
            'Weight': ['float64', 'float32'],
            'NObeyesdad': 'object'
        }
        
        for col, expected_type in expected_types.items():
            if col in sample_obesity_data.columns:
                actual_type = str(sample_obesity_data[col].dtype)
                
                if isinstance(expected_type, list):
                    assert actual_type in expected_type, f"Column {col} has type {actual_type}, expected one of {expected_type}"
                else:
                    assert actual_type == expected_type, f"Column {col} has type {actual_type}, expected {expected_type}"
    
    def test_value_ranges_valid(self, sample_obesity_data):
        """Test que los valores están en rangos válidos."""
        range_checks = {
            'Age': (0, 120),
            'Height': (0.5, 2.5),
            'Weight': (10, 300),
            'FCVC': (1, 3),
            'NCP': (1, 4),
            'CH2O': (1, 3),
            'FAF': (0, 3),
            'TUE': (0, 2)
        }
        
        for col, (min_val, max_val) in range_checks.items():
            if col in sample_obesity_data.columns:
                col_min = sample_obesity_data[col].min()
                col_max = sample_obesity_data[col].max()
                
                assert col_min >= min_val, f"Column {col} min value {col_min} is below {min_val}"
                assert col_max <= max_val, f"Column {col} max value {col_max} is above {max_val}"
    
    def test_categorical_values_valid(self, sample_obesity_data):
        """Test que los valores categóricos son válidos."""
        categorical_checks = {
            'Gender': ['Male', 'Female'],
            'family_history_with_overweight': ['yes', 'no'],
            'FAVC': ['yes', 'no'],
            'CAEC': ['no', 'Sometimes', 'Frequently', 'Always'],
            'SMOKE': ['yes', 'no'],
            'SCC': ['yes', 'no'],
            'CALC': ['no', 'Sometimes', 'Frequently', 'Always'],
            'MTRANS': ['Walking', 'Public_Transportation', 'Automobile', 'Bike', 'Motorbike'],
            'NObeyesdad': [
                'Insufficient_Weight', 'Normal_Weight', 'Overweight_Level_I',
                'Overweight_Level_II', 'Obesity_Type_I', 'Obesity_Type_II', 'Obesity_Type_III'
            ]
        }
        
        for col, valid_values in categorical_checks.items():
            if col in sample_obesity_data.columns:
                unique_values = sample_obesity_data[col].unique()
                invalid_values = set(unique_values) - set(valid_values)
                
                assert len(invalid_values) == 0, f"Column {col} has invalid values: {invalid_values}"
    
    def test_numerical_outliers(self, sample_obesity_data):
        """Test detección de outliers en variables numéricas."""
        numerical_columns = ['Age', 'Height', 'Weight']
        
        for col in numerical_columns:
            if col in sample_obesity_data.columns:
                # Usar IQR para detectar outliers
                Q1 = sample_obesity_data[col].quantile(0.25)
                Q3 = sample_obesity_data[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = sample_obesity_data[
                    (sample_obesity_data[col] < lower_bound) | 
                    (sample_obesity_data[col] > upper_bound)
                ]
                
                outlier_percentage = len(outliers) / len(sample_obesity_data) * 100
                
                # Permitir hasta 5% de outliers (normal en datos reales)
                assert outlier_percentage <= 5, f"Column {col} has {outlier_percentage:.2f}% outliers"

class TestDataIntegrity:
    """Test suite para integridad de datos."""
    
    def test_bmi_consistency(self, sample_obesity_data):
        """Test consistencia entre BMI calculado y categoría de obesidad."""
        if all(col in sample_obesity_data.columns for col in ['Height', 'Weight', 'NObeyesdad']):
            # Calcular BMI
            bmi = sample_obesity_data['Weight'] / (sample_obesity_data['Height'] ** 2)
            
            # Verificar consistencia básica
            normal_weight_mask = sample_obesity_data['NObeyesdad'] == 'Normal_Weight'
            if normal_weight_mask.any():
                normal_weight_bmi = bmi[normal_weight_mask]
                # La mayoría de casos "Normal_Weight" deberían tener BMI entre 18.5 y 25
                normal_range_count = ((normal_weight_bmi >= 18.5) & (normal_weight_bmi < 25)).sum()
                consistency_rate = normal_range_count / len(normal_weight_bmi) * 100
                
                # Al menos 70% de consistencia (datos sintéticos pueden variar)
                assert consistency_rate >= 70, f"BMI-Obesity category consistency is only {consistency_rate:.2f}%"
    
    def test_age_gender_distribution(self, sample_obesity_data):
        """Test distribución razonable de edad por género."""
        if all(col in sample_obesity_data.columns for col in ['Age', 'Gender']):
            gender_age_stats = sample_obesity_data.groupby('Gender')['Age'].describe()
            
            # Verificar que ambos géneros tienen representación razonable
            for gender in ['Male', 'Female']:
                if gender in gender_age_stats.index:
                    gender_count = gender_age_stats.loc[gender, 'count']
                    total_count = len(sample_obesity_data)
                    
                    # Cada género debería tener al menos 20% de representación
                    representation = gender_count / total_count * 100
                    assert representation >= 20, f"Gender {gender} has only {representation:.2f}% representation"
    
    def test_lifestyle_consistency(self, sample_obesity_data):
        """Test consistencia entre variables de estilo de vida."""
        lifestyle_columns = ['FAF', 'TUE', 'CAEC', 'SMOKE']
        available_columns = [col for col in lifestyle_columns if col in sample_obesity_data.columns]
        
        if len(available_columns) >= 2:
            # Verificar que no todos los valores sean idénticos (indicaría datos sintéticos pobres)  
            for col in available_columns:
                unique_count = sample_obesity_data[col].nunique()
                assert unique_count > 1, f"Column {col} has only {unique_count} unique value(s)"
    
    def test_transportation_lifestyle_relationship(self, sample_obesity_data):
        """Test relación lógica entre transporte y actividad física."""
        transport_activity_cols = ['MTRANS', 'FAF']
        
        if all(col in sample_obesity_data.columns for col in transport_activity_cols):
            # Aquellos que caminan podrían tener mayor actividad física
            walking_mask = sample_obesity_data['MTRANS'] == 'Walking'
            if walking_mask.any():
                walking_faf = sample_obesity_data.loc[walking_mask, 'FAF'].mean()
                total_faf = sample_obesity_data['FAF'].mean()
                
                # Verificar que la tendencia sea lógica (no estricta por variabilidad de datos)
                # Solo verificamos que no sea inconsistente extremadamente
                assert walking_faf >= total_faf * 0.8, "Inconsistent transportation-activity relationship"

class TestDataSchema:
    """Test suite para validación de schema de datos."""
    
    def test_required_columns_present(self, sample_obesity_data):
        """Test que todas las columnas requeridas están presentes."""
        required_columns = [
            'Gender', 'Age', 'Height', 'Weight', 'family_history_with_overweight',
            'FAVC', 'FCVC', 'NCP', 'CAEC', 'SMOKE', 'CH2O', 'SCC', 'FAF', 'TUE',
            'CALC', 'MTRANS', 'NObeyesdad'
        ]
        
        missing_columns = set(required_columns) - set(sample_obesity_data.columns)
        assert len(missing_columns) == 0, f"Missing required columns: {missing_columns}"
    
    def test_column_count(self, sample_obesity_data):
        """Test que el número de columnas es el esperado."""
        expected_column_count = 17  # Basado en dataset original
        actual_column_count = len(sample_obesity_data.columns)
        
        # Permitir variación para datasets modificados
        assert abs(actual_column_count - expected_column_count) <= 5, \
            f"Expected ~{expected_column_count} columns, got {actual_column_count}"
    
    def test_row_count_reasonable(self, sample_obesity_data):
        """Test que el número de filas es razonable."""
        row_count = len(sample_obesity_data)
        
        # Para datos de prueba, esperamos al menos 10 filas
        assert row_count >= 10, f"Dataset too small: {row_count} rows"
        
        # Para datos de prueba, no esperamos más de 10000 filas
        assert row_count <= 10000, f"Dataset too large for testing: {row_count} rows"
    
    def test_data_not_empty(self, sample_obesity_data):
        """Test que el dataset no esté vacío."""
        assert not sample_obesity_data.empty, "Dataset is empty"
        assert sample_obesity_data.shape[0] > 0, "Dataset has no rows"
        assert sample_obesity_data.shape[1] > 0, "Dataset has no columns"

class TestDataStatistics:
    """Test suite para estadísticas de datos."""
    
    def test_target_distribution(self, sample_obesity_data):
        """Test distribución de la variable objetivo."""
        if 'NObeyesdad' in sample_obesity_data.columns:
            target_counts = sample_obesity_data['NObeyesdad'].value_counts()
            
            # Verificar que hay al menos 2 clases diferentes
            assert len(target_counts) >= 2, "Target variable has less than 2 classes"
            
            # Verificar que no hay una clase que domine excesivamente (>90%)
            max_class_percentage = target_counts.max() / len(sample_obesity_data) * 100
            assert max_class_percentage <= 90, f"Dominant class represents {max_class_percentage:.2f}% of data"
    
    def test_numerical_distributions(self, sample_obesity_data):
        """Test distribuciones de variables numéricas."""
        numerical_columns = ['Age', 'Height', 'Weight']
        
        for col in numerical_columns:
            if col in sample_obesity_data.columns:
                col_stats = sample_obesity_data[col].describe()
                
                # Verificar que hay variabilidad (std > 0)
                assert col_stats['std'] > 0, f"Column {col} has no variability (std = {col_stats['std']})"
                
                # Verificar que los valores no son todos iguales
                assert col_stats['min'] != col_stats['max'], f"Column {col} has constant values"
    
    def test_correlation_patterns(self, sample_obesity_data):
        """Test patrones de correlación esperados."""
        numerical_columns = ['Age', 'Height', 'Weight']
        available_columns = [col for col in numerical_columns if col in sample_obesity_data.columns]
        
        if len(available_columns) >= 2:
            correlation_matrix = sample_obesity_data[available_columns].corr()
            
            # Verificar que no hay correlación perfecta entre variables (indicaría datos duplicados)
            for i in range(len(available_columns)):
                for j in range(i + 1, len(available_columns)):
                    corr_value = abs(correlation_matrix.iloc[i, j])
                    col1, col2 = available_columns[i], available_columns[j]
                    
                    assert corr_value < 0.99, f"Perfect correlation between {col1} and {col2}: {corr_value}"

class TestFeatureEngineeredData:
    """Test suite para datos con features engineered."""
    
    def test_bmi_calculation(self, sample_processed_features):
        """Test cálculo correcto de BMI si está presente."""
        if 'bmi' in sample_processed_features.columns:
            bmi_values = sample_processed_features['bmi']
            
            # BMI debe estar en rango razonable
            assert bmi_values.min() >= 10, f"BMI too low: {bmi_values.min()}"
            assert bmi_values.max() <= 60, f"BMI too high: {bmi_values.max()}"
            
            # No debe haber valores NaN o infinitos
            assert not bmi_values.isnull().any(), "BMI contains null values"
            assert not np.isinf(bmi_values).any(), "BMI contains infinite values"
    
    def test_normalized_features(self, sample_processed_features):
        """Test features normalizadas."""
        normalized_columns = [col for col in sample_processed_features.columns if 'normalized' in col]
        
        for col in normalized_columns:
            col_values = sample_processed_features[col]
            
            # Features normalizadas deberían tener media cercana a 0
            assert abs(col_values.mean()) < 2, f"Normalized feature {col} has mean {col_values.mean()}"
            
            # No debe haber valores extremos (más de 5 desviaciones estándar)
            assert abs(col_values).max() < 5, f"Normalized feature {col} has extreme values"
    
    def test_encoded_features(self, sample_processed_features):
        """Test features codificadas."""
        encoded_columns = [col for col in sample_processed_features.columns if 'encoded' in col]
        
        for col in encoded_columns:
            col_values = sample_processed_features[col]
            unique_values = col_values.unique()
            
            # Features codificadas binarias deberían tener solo 0 y 1
            if len(unique_values) <= 2:
                assert all(val in [0, 1] for val in unique_values), \
                    f"Binary encoded feature {col} has invalid values: {unique_values}"
    
    def test_lifestyle_scores(self, sample_processed_features):
        """Test scores de estilo de vida."""
        lifestyle_columns = [col for col in sample_processed_features.columns if 'score' in col.lower()]
        
        for col in lifestyle_columns:
            col_values = sample_processed_features[col]
            
            # Scores deberían estar en rango [0, 1] o similar
            assert col_values.min() >= 0, f"Lifestyle score {col} has negative values"
            assert col_values.max() <= 1.5, f"Lifestyle score {col} has values > 1.5"

@pytest.mark.parametrize("test_percentage", [0.1, 0.2, 0.3])
class TestDataSplits:
    """Test suite para validación de splits de datos."""
    
    def test_train_test_split_sizes(self, sample_obesity_data, test_percentage):
        """Test tamaños correctos de splits."""
        from sklearn.model_selection import train_test_split
        
        if 'NObeyesdad' in sample_obesity_data.columns:
            X = sample_obesity_data.drop('NObeyesdad', axis=1)
            y = sample_obesity_data['NObeyesdad']
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_percentage, random_state=42
            )
            
            # Verificar tamaños
            expected_test_size = int(len(sample_obesity_data) * test_percentage)
            actual_test_size = len(X_test)
            
            # Permitir diferencia de ±1 por redondeo
            assert abs(actual_test_size - expected_test_size) <= 1
            
            # Verificar que no hay overlap
            assert len(set(X_train.index) & set(X_test.index)) == 0
    
    def test_stratified_split_distribution(self, sample_obesity_data, test_percentage):
        """Test distribución en split estratificado.""" 
        from sklearn.model_selection import train_test_split
        
        if 'NObeyesdad' in sample_obesity_data.columns:
            X = sample_obesity_data.drop('NObeyesdad', axis=1)
            y = sample_obesity_data['NObeyesdad']
            
            # Solo hacer split estratificado si hay suficientes muestras por clase
            if y.value_counts().min() >= 2:
                try:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_percentage, stratify=y, random_state=42
                    )
                    
                    # Verificar que las distribuciones son similares
                    train_dist = y_train.value_counts(normalize=True).sort_index()
                    test_dist = y_test.value_counts(normalize=True).sort_index()
                    
                    # Permitir hasta 10% de diferencia en proporciones
                    for class_label in train_dist.index:
                        if class_label in test_dist.index:
                            diff = abs(train_dist[class_label] - test_dist[class_label])
                            assert diff <= 0.1, f"Class {class_label} distribution differs by {diff:.3f}"
                
                except ValueError:
                    # Split estratificado no es posible con muy pocas muestras
                    pass