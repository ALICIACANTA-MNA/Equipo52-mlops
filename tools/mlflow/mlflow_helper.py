"""
MLflow Configuration Helper - Equipo 52 MLOps
=============================================

FLUJO DE DATOS MLflow:
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Pipeline      │───▶│   MLflow DB     │───▶│   MLflow UI     │
│   (train.py)    │    │   (sqlite+fs)   │    │ (localhost:5000)│
└─────────────────┘    └─────────────────┘    └─────────────────┘

Script de utilidad para configurar y gestionar MLflow con storage híbrido.
Funciona tanto en modo local como con AWS S3 cuando esté disponible.
"""

import os
import sys
import yaml
import subprocess
from pathlib import Path
from typing import Dict, Any


def load_mlflow_config() -> Dict[str, Any]:
    """Carga la configuración de MLflow."""
    config_path = "configs/mlflow/mlflow_config.yaml"
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Error cargando configuración: {e}")
        return {}


def setup_environment_variables():
    """Configura variables de entorno para MLflow."""
    config = load_mlflow_config()
    
    # Configurar AWS si está disponible
    aws_config = config.get('aws', {}).get('credentials', {})
    if aws_config.get('access_key_id'):
        os.environ['AWS_ACCESS_KEY_ID'] = aws_config['access_key_id']
        os.environ['AWS_SECRET_ACCESS_KEY'] = aws_config['secret_access_key']
        os.environ['AWS_DEFAULT_REGION'] = aws_config.get('region', 'us-east-1')
        print("Variables AWS configuradas")
    
    # Configurar MLflow
    mlflow_config = config.get('mlflow', {})
    tracking_uri = mlflow_config.get('tracking', {}).get('tracking_uri', 'sqlite:///mlflow.db')
    
    os.environ['MLFLOW_TRACKING_URI'] = tracking_uri
    print(f"MLflow Tracking URI: {tracking_uri}")
    
    return config


def check_mlflow_status():
    """Verifica el estado actual de MLflow."""
    print("Estado actual de MLflow:")
    print(f"   - Base de datos: {'Existe' if os.path.exists('mlflow.db') else 'No existe'}")
    print(f"   - Directorio mlruns: {'Existe' if os.path.exists('mlruns') else 'No existe'}")
    
    # Contar experimentos y runs
    if os.path.exists('mlflow.db'):
        try:
            import sqlite3
            conn = sqlite3.connect('mlflow.db')
            cursor = conn.cursor()
            
            # Contar experimentos
            cursor.execute("SELECT COUNT(*) FROM experiments")
            exp_count = cursor.fetchone()[0]
            
            # Contar runs
            cursor.execute("SELECT COUNT(*) FROM runs")
            run_count = cursor.fetchone()[0]
            
            print(f"   - Experimentos: {exp_count}")
            print(f"   - Runs totales: {run_count}")
            
            conn.close()
        except Exception as e:
            print(f"   - Error leyendo DB: {e}")


def start_mlflow_ui():
    """Inicia la interfaz web de MLflow."""
    print("Iniciando MLflow UI...")
    
    # Configurar variables de entorno
    setup_environment_variables()
    
    # Comando para iniciar MLflow UI
    cmd = [
        sys.executable, "-m", "mlflow", "ui",
        "--backend-store-uri", "sqlite:///mlflow.db",
        "--host", "127.0.0.1",
        "--port", "5000"
    ]
    
    print(f"Comando: {' '.join(cmd)}")
    print("URL: http://127.0.0.1:5000")
    print("Presiona Ctrl+C para detener")
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("MLflow UI detenido")


def create_example_run():
    """Crea un run de ejemplo para testing."""
    print("Creando run de ejemplo...")
    
    try:
        import mlflow
        import mlflow.sklearn
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.datasets import make_classification
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, f1_score
        
        # Configurar MLflow
        setup_environment_variables()
        mlflow.set_experiment("testing_hybrid_storage")
        
        # Crear datos sintéticos
        X, y = make_classification(n_samples=1000, n_features=20, n_classes=3, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Entrenar modelo
        with mlflow.start_run() as run:
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # Predicciones
            y_pred = model.predict(X_test)
            
            # Métricas
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            # Log de parámetros y métricas
            mlflow.log_param("n_estimators", 100)
            mlflow.log_param("random_state", 42)
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("f1_score", f1)
            
            # Log del modelo
            mlflow.sklearn.log_model(model, "model")
            
            print(f"Run creado: {run.info.run_id}")
            print(f"   - Accuracy: {accuracy:.4f}")
            print(f"   - F1-Score: {f1:.4f}")
        
    except ImportError:
        print("scikit-learn no disponible para crear ejemplo")
    except Exception as e:
        print(f"Error creando run: {e}")


def show_menu():
    """Muestra el menú principal."""
    print("\\n" + "="*60)
    print("MLflow Configuration Helper - Equipo 52 MLOps")
    print("="*60)
    
    check_mlflow_status()
    
    print("\\n Opciones disponibles:")
    print("1. Iniciar MLflow UI")
    print("2. Crear run de ejemplo")
    print("3. Verificar estado")
    print("4. Configurar variables de entorno")
    print("5. Mostrar configuración actual")
    print("6. Abrir documentación MLflow")
    print("7. Ejecutar pipeline completo")
    print("8. Salir")
    
    return input("\\nSelecciona una opción (1-8): ")


def show_current_config():
    """Muestra la configuración actual."""
    config = load_mlflow_config()
    print("Configuración Actual:")
    print("-" * 40)
    
    # MLflow config
    mlflow_config = config.get('mlflow', {})
    print(f"Tracking URI: {mlflow_config.get('tracking', {}).get('tracking_uri', 'N/A')}")
    print(f"Experimento: {mlflow_config.get('experiment', {}).get('name', 'N/A')}")
    print(f"Modelo: {mlflow_config.get('registry', {}).get('registered_model_name', 'N/A')}")
    
    # AWS config
    aws_config = config.get('aws', {})
    if aws_config.get('credentials', {}).get('access_key_id'):
        print(f"AWS Region: {aws_config.get('credentials', {}).get('region', 'N/A')}")
        print(f"S3 Bucket: {aws_config.get('s3', {}).get('bucket_name', 'N/A')}")
    else:
        print("AWS: No configurado")


def run_full_pipeline():
    """Ejecuta el pipeline completo de MLOps."""
    print("Ejecutando pipeline completo...")
    
    stages = [
        "python -m src.data.data_ingestion",
        "python -m src.data.data_validation", 
        "python -m src.data.data_preprocessing",
        "python -m src.models.train",
        "python -m src.models.evaluate",
        "python -m src.models.model_registration"
    ]
    
    for i, stage in enumerate(stages, 1):
        print(f"Stage {i}/{len(stages)}: {stage}")
        try:
            result = subprocess.run(stage.split(), capture_output=True, text=True)
            if result.returncode == 0:
                print(f"Stage {i} completado")
            else:
                print(f"Stage {i} falló: {result.stderr}")
                break
        except Exception as e:
            print(f"Error en stage {i}: {e}")
            break
    
    print("Pipeline completado!")


def main():
    """Función principal."""
    while True:
        choice = show_menu()
        
        if choice == "1":
            start_mlflow_ui()
        elif choice == "2":
            create_example_run()
        elif choice == "3":
            check_mlflow_status()
        elif choice == "4":
            setup_environment_variables()
        elif choice == "5":
            show_current_config()
        elif choice == "6":
            print("Documentación MLflow:")
            print("   - Oficial: https://mlflow.org/docs/latest/")
            print("   - Tracking: https://mlflow.org/docs/latest/tracking.html")
            print("   - Model Registry: https://mlflow.org/docs/latest/model-registry.html")
        elif choice == "7":
            run_full_pipeline()
        elif choice == "8":
            print("¡Hasta luego!")
            break
        else:
            print("Opción inválida")
        
        input("\\nPresiona Enter para continuar...")


if __name__ == "__main__":
    main()