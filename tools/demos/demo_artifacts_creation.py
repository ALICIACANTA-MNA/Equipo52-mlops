#!/usr/bin/env python3
"""
🎯 DEMOSTRACIÓN: Proceso de Creación de Artefactos MLflow
====================================================

Este script demuestra EXACTAMENTE cuándo y cómo se crean los artefactos
en la carpeta mlruns/ durante un experimento MLflow.

Ejecutar: python demo_artifacts_creation.py
"""

import os
import mlflow
import mlflow.sklearn
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import json
import time

def check_directories():
    """Verificar estado de directorios antes y después de cada paso"""
    mlflow_db_exists = os.path.exists("mlflow.db")
    mlruns_exists = os.path.exists("mlruns")
    
    print(f"  📊 mlflow.db existe: {mlflow_db_exists}")
    print(f"  📁 mlruns/ existe: {mlruns_exists}")
    
    if mlruns_exists:
        # Contar experimentos y runs
        experiments = [d for d in os.listdir("mlruns") if os.path.isdir(os.path.join("mlruns", d)) and d.isdigit()]
        print(f"  🧪 Experimentos: {len(experiments)} -> {experiments}")
        
        for exp in experiments:
            exp_path = os.path.join("mlruns", exp)
            runs = [d for d in os.listdir(exp_path) if os.path.isdir(os.path.join(exp_path, d)) and len(d) > 10]
            if runs:
                print(f"    📝 Experimento {exp}: {len(runs)} runs")
                # Ver artefactos del último run
                last_run = runs[-1]
                artifacts_path = os.path.join(exp_path, last_run, "artifacts")
                if os.path.exists(artifacts_path):
                    artifacts = os.listdir(artifacts_path)
                    print(f"      📦 Artefactos: {artifacts}")

def main():
    print("🎯 DEMOSTRACIÓN: Proceso de Creación de Artefactos MLflow")
    print("=" * 60)
    
    # Configurar MLflow
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("artifact_creation_demo")
    
    print("\n🔍 PASO 0: Estado inicial")
    check_directories()
    
    print("\n" + "="*60)
    print("🚀 PASO 1: mlflow.start_run() - Iniciando experimento")
    print("="*60)
    
    with mlflow.start_run(run_name="artifact_demo") as run:
        print(f"✅ Run iniciado: {run.info.run_id[:8]}...")
        
        print("\n🔍 Estado después de start_run():")
        check_directories()
        
        print("\n" + "="*60)
        print("📊 PASO 2: Generando datos y entrenando modelo")
        print("="*60)
        
        # Datos sintéticos
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=1000, n_features=10, n_classes=3, n_informative=8, n_redundant=2, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Entrenar modelo
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        print("✅ Modelo entrenado en memoria")
        
        print("\n" + "="*60)
        print("📝 PASO 3: mlflow.log_param() - Solo metadatos (DB)")
        print("="*60)
        
        mlflow.log_param("n_estimators", 50)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("test_size", 0.3)
        
        print("✅ Parámetros guardados en mlflow.db")
        print("🔍 Estado después de log_param():")
        check_directories()
        
        print("\n" + "="*60)
        print("📈 PASO 4: mlflow.log_metric() - Solo metadatos (DB)")
        print("="*60)
        
        accuracy = accuracy_score(y_test, y_pred)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("n_samples_train", len(X_train))
        mlflow.log_metric("n_samples_test", len(X_test))
        
        print("✅ Métricas guardadas en mlflow.db")
        print("🔍 Estado después de log_metric():")
        check_directories()
        
        print("\n" + "="*60)
        print("🖼️  PASO 5: PRIMER ARTEFACTO - Matriz de confusión")
        print("="*60)
        print("🎯 ¡AQUÍ SE CREA mlruns/ POR PRIMERA VEZ!")
        
        # Crear matriz de confusión
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Guardar y loggear
        plt.savefig("temp_confusion_matrix.png", dpi=150, bbox_inches='tight')
        mlflow.log_artifact("temp_confusion_matrix.png", "plots")
        plt.close()
        
        print("✅ Matriz de confusión guardada como artefacto")
        print("🔍 Estado después del PRIMER artefacto:")
        check_directories()
        
        print("\n" + "="*60)
        print("🤖 PASO 6: SEGUNDO ARTEFACTO - Modelo completo")
        print("="*60)
        
        # Loggear modelo (crea múltiples archivos)
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name="demo_rf_model"
        )
        
        print("✅ Modelo guardado como artefacto")
        print("🔍 Estado después de log_model():")
        check_directories()
        
        print("\n" + "="*60)
        print("📄 PASO 7: TERCER ARTEFACTO - Reporte de clasificación")
        print("="*60)
        
        # Crear reporte de clasificación
        report = classification_report(y_test, y_pred, output_dict=True)
        with open("temp_classification_report.json", "w") as f:
            json.dump(report, f, indent=2)
        
        mlflow.log_artifact("temp_classification_report.json", "reports")
        
        print("✅ Reporte de clasificación guardado")
        print("🔍 Estado después del reporte:")
        check_directories()
        
        print("\n" + "="*60)
        print("📊 PASO 8: CUARTO ARTEFACTO - Feature importance")
        print("="*60)
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': [f'feature_{i}' for i in range(X.shape[1])],
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Guardar CSV
        feature_importance.to_csv("temp_feature_importance.csv", index=False)
        mlflow.log_artifact("temp_feature_importance.csv", "analysis")
        
        # Crear gráfico
        plt.figure(figsize=(10, 6))
        plt.bar(feature_importance['feature'], feature_importance['importance'])
        plt.title('Feature Importance')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("temp_feature_importance.png", dpi=150, bbox_inches='tight')
        mlflow.log_artifact("temp_feature_importance.png", "plots")
        plt.close()
        
        print("✅ Feature importance guardada")
        print("🔍 Estado FINAL:")
        check_directories()
        
        print(f"\n🎯 Run completado: {run.info.run_id}")
    
    # Limpiar archivos temporales
    temp_files = [
        "temp_confusion_matrix.png",
        "temp_classification_report.json", 
        "temp_feature_importance.csv",
        "temp_feature_importance.png"
    ]
    
    for file in temp_files:
        if os.path.exists(file):
            os.remove(file)
    
    print("\n" + "="*60)
    print("✅ DEMOSTRACIÓN COMPLETADA")
    print("="*60)
    print("📝 RESUMEN:")
    print("1️⃣  mlflow.start_run() → Solo crea entrada en mlflow.db")
    print("2️⃣  mlflow.log_param/metric() → Solo metadatos en mlflow.db") 
    print("3️⃣  mlflow.log_artifact() → 🎯 CREA mlruns/ + guarda archivo")
    print("4️⃣  mlflow.sklearn.log_model() → Múltiples archivos en mlruns/")
    print("5️⃣  Más artefactos → Se acumulan en mlruns/")
    print("\n🌐 Ahora puedes ver los resultados en: http://127.0.0.1:5000")
    print("💡 Ejecuta: python mlflow_local.py (opción 1)")

if __name__ == "__main__":
    main()