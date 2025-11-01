"""
MLflow Local Configuration - Solo Local (Sin AWS)
================================================

FLUJO SIMPLIFICADO - SOLO LOCAL:
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Pipeline      │───▶│   MLflow Local  │───▶│   MLflow UI     │
│   (train.py)    │    │   (sqlite+fs)   │    │ (localhost:5000)│
└─────────────────┘    └─────────────────┘    └─────────────────┘

ALMACENAMIENTO:
- Base de datos: mlflow.db (SQLite)
- Artefactos: ./mlruns/ (sistema de archivos local)
- AWS S3: DESHABILITADO
"""

import os
import sys
import yaml
import subprocess
from pathlib import Path


def configure_local_only():
    """Configura MLflow para funcionar SOLO en modo local."""
    print("Configurando MLflow - MODO SOLO LOCAL")
    print("=" * 50)
    
    # Variables de entorno para modo local
    os.environ['MLFLOW_TRACKING_URI'] = 'sqlite:///mlflow.db'
    os.environ['MLFLOW_DEFAULT_ARTIFACT_ROOT'] = './mlruns'
    
    # Desactivar AWS (por si hay configuración previa)
    for aws_var in ['AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY', 'AWS_DEFAULT_REGION']:
        if aws_var in os.environ:
            del os.environ[aws_var]
            #print(f"Removida variable: {aws_var}")
    
    print("Configuración local establecida:")
    print(f"Tracking URI: {os.environ['MLFLOW_TRACKING_URI']}")
    print(f"Artifacts: {os.environ['MLFLOW_DEFAULT_ARTIFACT_ROOT']}")
    print(f"AWS S3: DESHABILITADO")


def check_local_setup():
    """Verifica que todo esté configurado para modo local."""
    print("\\n🔍 Verificando configuración local...")
    
    status = {
        "mlflow_db": os.path.exists("mlflow.db"),
        "mlruns_dir": os.path.exists("mlruns"),
        "config_file": os.path.exists("configs/mlflow/mlflow_config.yaml")
    }
    
    for item, exists in status.items():
        status_icon = "✅" if exists else "❌"
        print(f"   {status_icon} {item}: {'Existe' if exists else 'No existe'}")
    
    # Verificar que no hay configuración AWS activa
    aws_vars = ['AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY']
    aws_configured = any(var in os.environ for var in aws_vars)
    
    if aws_configured:
        print("   ⚠️  Variables AWS detectadas (se ignorarán en modo local)")
    else:
        print("   ✅ Sin variables AWS (perfecto para modo local)")
    
    return all(status.values())


def start_local_mlflow_ui():
    """Inicia MLflow UI en modo local."""
    print("\\n🚀 Iniciando MLflow UI - MODO LOCAL")
    print("-" * 40)
    
    # Configurar entorno local
    configure_local_only()
    
    # Comando para MLflow UI
    cmd = [
        sys.executable, "-m", "mlflow", "ui",
        "--backend-store-uri", "sqlite:///mlflow.db",
        "--default-artifact-root", "./mlruns",
        "--host", "127.0.0.1",
        "--port", "5000"
    ]
    
    print(f"📝 Comando: {' '.join(cmd)}")
    print("🌐 URL: http://127.0.0.1:5000")
    print("⛔ Presiona Ctrl+C para detener")
    print("\\n" + "="*50)
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\\n👋 MLflow UI detenido")
    except Exception as e:
        print(f"\\n❌ Error iniciando MLflow UI: {e}")


def create_local_test_run():
    """Crea un run de prueba en modo local."""
    print("\\n🧪 Creando run de prueba - MODO LOCAL")
    print("-" * 40)
    
    try:
        import mlflow
        import mlflow.sklearn
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.datasets import make_classification
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, f1_score
        
        # Configurar MLflow local
        configure_local_only()
        mlflow.set_experiment("local_testing")
        
        print("📊 Generando datos sintéticos...")
        X, y = make_classification(n_samples=500, n_features=10, n_classes=2, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        print("🤖 Entrenando modelo...")
        with mlflow.start_run(run_name="local_test_run") as run:
            # Entrenar modelo
            model = RandomForestClassifier(n_estimators=50, random_state=42)
            model.fit(X_train, y_train)
            
            # Predicciones
            y_pred = model.predict(X_test)
            
            # Métricas
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            # Log en MLflow
            mlflow.log_param("n_estimators", 50)
            mlflow.log_param("mode", "local_only")
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("f1_score", f1)
            
            # Guardar modelo localmente
            mlflow.sklearn.log_model(model, "model")
            
            print(f"✅ Run completado: {run.info.run_id}")
            print(f"   📊 Accuracy: {accuracy:.4f}")
            print(f"   📊 F1-Score: {f1:.4f}")
            print(f"   💾 Modelo guardado en: ./mlruns/")
        
    except ImportError as e:
        print(f"❌ Faltan dependencias: {e}")
        print("💡 Instalar con: pip install scikit-learn")
    except Exception as e:
        print(f"❌ Error creando run: {e}")


def show_local_menu():
    """Muestra menú para operaciones locales."""
    print("\\n" + "="*60)
    print("🏠 MLflow Local Configuration - MODO SOLO LOCAL")
    print("="*60)
    
    # Verificar estado
    setup_ok = check_local_setup()
    
    print("\\n🛠️  Opciones disponibles:")
    print("1. 🚀 Iniciar MLflow UI (solo local)")
    print("2. 🧪 Crear run de prueba local")
    print("3. 🔍 Verificar configuración")
    print("4. ⚙️  Configurar variables de entorno")
    print("5. 📊 Mostrar estadísticas locales")
    print("6. 🔄 Ejecutar pipeline completo (local)")
    print("7. ❌ Salir")
    
    if not setup_ok:
        print("\\n⚠️  ADVERTENCIA: Configuración incompleta")
    
    return input("\\n👉 Selecciona una opción (1-7): ")


def show_local_stats():
    """Muestra estadísticas del MLflow local."""
    print("\\n📊 Estadísticas MLflow Local")
    print("-" * 30)
    
    if os.path.exists("mlflow.db"):
        try:
            import sqlite3
            conn = sqlite3.connect("mlflow.db")
            cursor = conn.cursor()
            
            # Contar experimentos
            cursor.execute("SELECT COUNT(*) FROM experiments")
            exp_count = cursor.fetchone()[0]
            
            # Contar runs
            cursor.execute("SELECT COUNT(*) FROM runs")
            run_count = cursor.fetchone()[0]
            
            # Últimos experimentos
            cursor.execute("SELECT name FROM experiments ORDER BY experiment_id DESC LIMIT 3")
            recent_experiments = [row[0] for row in cursor.fetchall()]
            
            print(f"🧪 Experimentos: {exp_count}")
            print(f"🏃 Runs totales: {run_count}")
            print(f"📈 Experimentos recientes:")
            for exp in recent_experiments:
                print(f"   - {exp}")
            
            conn.close()
            
        except Exception as e:
            print(f"❌ Error leyendo estadísticas: {e}")
    else:
        print("❌ No existe mlflow.db")
    
    # Tamaño de artifacts
    if os.path.exists("mlruns"):
        try:
            total_size = sum(f.stat().st_size for f in Path("mlruns").rglob("*") if f.is_file())
            size_mb = total_size / (1024 * 1024)
            print(f"💾 Tamaño artifacts: {size_mb:.2f} MB")
        except:
            print("💾 Tamaño artifacts: No calculable")


def run_local_pipeline():
    """Ejecuta el pipeline completo en modo local."""
    print("\\n🔄 Ejecutando Pipeline - MODO LOCAL")
    print("-" * 40)
    
    # Configurar entorno local
    configure_local_only()
    
    stages = [
        ("data_ingestion", "python -m src.data.data_ingestion"),
        ("data_validation", "python -m src.data.data_validation"), 
        ("data_preprocessing", "python -m src.data.data_preprocessing"),
        ("model_training", "python -m src.models.train"),
        ("model_evaluation", "python -m src.models.evaluate"),
        ("model_registration", "python -m src.models.model_registration")
    ]
    
    for i, (stage_name, command) in enumerate(stages, 1):
        print(f"\\n📍 Stage {i}/{len(stages)}: {stage_name}")
        try:
            result = subprocess.run(command.split(), capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ {stage_name} completado")
            else:
                print(f"❌ {stage_name} falló:")
                print(f"   Error: {result.stderr[:200]}...")
                break
        except Exception as e:
            print(f"❌ Error en {stage_name}: {e}")
            break
    
    print("\\n🎉 Pipeline local completado!")


def main():
    """Función principal."""
    while True:
        choice = show_local_menu()
        
        if choice == "1":
            start_local_mlflow_ui()
        elif choice == "2":
            create_local_test_run()
        elif choice == "3":
            check_local_setup()
        elif choice == "4":
            configure_local_only()
        elif choice == "5":
            show_local_stats()
        elif choice == "6":
            run_local_pipeline()
        elif choice == "7":
            print("👋 ¡Hasta luego!")
            break
        else:
            print("❌ Opción inválida")
        
        if choice != "1":  # No pausar después de MLflow UI
            input("\\nPresiona Enter para continuar...")


if __name__ == "__main__":
    main()