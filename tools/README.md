# 🛠️ Tools - Herramientas de Desarrollo

Esta carpeta contiene herramientas y utilidades de apoyo para el desarrollo del proyecto MLOps.

## 📁 Estructura

```
tools/
├── mlflow/                     # Herramientas MLflow
│   ├── mlflow_local.py        # Script principal para MLflow local
│   ├── mlflow_helper.py       # Funciones de ayuda MLflow
│   └── mlflow_config_examples.py  # Ejemplos de configuración
├── demos/                     # Demostraciones y ejemplos
│   └── demo_artifacts_creation.py  # Demo del proceso de artefactos
├── run_tests.py               # ✨ Script principal para ejecutar tests (MOVIDO DESDE RAÍZ)
├── run_dvc_pipeline.py        # ✨ Script para ejecutar pipeline DVC (MOVIDO DESDE RAÍZ)  
├── test_docker_api.py         # ✨ Script para probar API Docker (MOVIDO DESDE RAÍZ)
└── README.md                  # Este archivo
```

## 🚀 Herramientas MLflow

### **mlflow_local.py**
- **Propósito**: Configuración y gestión de MLflow en modo local
- **Uso**: `python tools/mlflow/mlflow_local.py`
- **Características**:
  - Configuración automática de entorno local
  - Interface de menú interactiva
  - Limpieza de variables AWS en modo local
  - Verificación de configuración

### **mlflow_helper.py**
- **Propósito**: Funciones de utilidad para MLflow
- **Uso**: Importar desde otros scripts
- **Características**:
  - Funciones de configuración reutilizables
  - Helpers para manejo de experimentos
  - Utilidades de limpieza y setup

### **mlflow_config_examples.py**
- **Propósito**: Ejemplos y comparaciones de configuración
- **Uso**: `python tools/mlflow/mlflow_config_examples.py`
- **Características**:
  - Comparación entre perfiles de configuración
  - Ejemplos de migración
  - Guías interactivas

## 🧪 Demostraciones

### **demo_artifacts_creation.py**
- **Propósito**: Demostración del proceso de creación de artefactos
- **Uso**: `python tools/demos/demo_artifacts_creation.py`
- **Características**:
  - Muestra paso a paso la creación de mlruns/
  - Verifica estado en cada paso
  - Crea ejemplos de todos los tipos de artefactos

## 🔧 Uso General

### Activar entorno antes de usar las herramientas:
```bash
.venv\Scripts\Activate.ps1  # Windows
```

### Ejecutar herramientas MLflow:
```bash
# Configuración local de MLflow
python tools/mlflow/mlflow_local.py

# Ejemplos de configuración
python tools/mlflow/mlflow_config_examples.py

# Demostración de artefactos
python tools/demos/demo_artifacts_creation.py
```

## 📋 Notas

- Todas las herramientas requieren el entorno virtual activado
- Las configuraciones principales están en `configs/mlflow/`
- Los archivos de datos y resultados siguen en las carpetas principales del proyecto
- Estas herramientas son de apoyo y desarrollo, no parte del pipeline principal