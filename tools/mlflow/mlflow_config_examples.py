"""
MLflow Configuration Examples - Equipo 52 MLOps
==============================================

Este archivo muestra ejemplos prácticos de cómo usar diferentes 
configuraciones de MLflow según el entorno y necesidades.

Author: Equipo 52 MLOps
Date: 2024
"""

import yaml
import os
from typing import Dict, Any


def load_config() -> Dict[str, Any]:
    """Carga la configuración MLflow completa."""
    with open("configs/mlflow/mlflow_config.yaml", 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def show_profile_comparison():
    """Muestra comparación detallada entre perfiles."""
    config = load_config()
    profiles = config.get('profiles', {})
    
    print("📊 COMPARACIÓN DE PERFILES MLflow")
    print("=" * 60)
    
    for profile_name, profile_config in profiles.items():
        print(f"\n🏷️  PERFIL: {profile_name.upper()}")
        print("-" * 40)
        
        # Información básica
        tracking_uri = profile_config.get('tracking_uri', 'N/A')
        artifact_location = profile_config.get('artifact_location', 'N/A')
        aws_sync = profile_config.get('aws_sync', False)
        description = profile_config.get('description', 'Sin descripción')
        
        print(f"📝 Descripción: {description}")
        print(f"🗄️  Tracking URI: {tracking_uri}")
        print(f"📁 Artifacts: {artifact_location}")
        print(f"☁️  AWS Sync: {'✅ Habilitado' if aws_sync else '❌ Deshabilitado'}")
        
        # Análisis del tipo de base de datos
        if 'sqlite' in tracking_uri.lower():
            print(f"🔍 Tipo BD: SQLite (Local)")
            print(f"📊 Escalabilidad: Baja (1 usuario)")
            print(f"⚡ Velocidad: Alta (local)")
            print(f"🔧 Configuración: Ninguna")
        elif 'postgresql' in tracking_uri.lower():
            print(f"🔍 Tipo BD: PostgreSQL (Remoto)")
            print(f"📊 Escalabilidad: Alta (múltiples usuarios)")
            print(f"⚡ Velocidad: Media (red)")
            print(f"🔧 Configuración: Servidor requerido")
        elif 'http' in tracking_uri.lower():
            print(f"🔍 Tipo BD: Servidor MLflow")
            print(f"📊 Escalabilidad: Media (equipo)")
            print(f"⚡ Velocidad: Media (red)")
            print(f"🔧 Configuración: Servidor MLflow")
        
        # Casos de uso recomendados
        print(f"🎯 Casos de uso:")
        if profile_name == 'local_only':
            print(f"   - Desarrollo individual")
            print(f"   - Aprendizaje y experimentación")
            print(f"   - Prototipado rápido")
        elif profile_name == 'collaboration':
            print(f"   - Equipos pequeños (2-5 personas)")
            print(f"   - Compartir experimentos ocasionalmente")
            print(f"   - Backup en S3")
        elif profile_name == 'production':
            print(f"   - Sistemas en producción")
            print(f"   - Múltiples usuarios simultáneos")
            print(f"   - Alta disponibilidad requerida")


def show_evolution_path():
    """Muestra la ruta de evolución entre perfiles."""
    print("\n🚀 RUTA DE EVOLUCIÓN DE PERFILES")
    print("=" * 50)
    
    evolution = [
        {
            "stage": "1. DESARROLLO INDIVIDUAL",
            "profile": "local_only",
            "description": "Un desarrollador, experimentación rápida",
            "setup": "Solo SQLite local",
            "team_size": "1 persona",
            "complexity": "Baja"
        },
        {
            "stage": "2. COLABORACIÓN BÁSICA", 
            "profile": "collaboration",
            "description": "Equipo pequeño, backup en nube",
            "setup": "SQLite + S3 sync",
            "team_size": "2-5 personas",
            "complexity": "Media"
        },
        {
            "stage": "3. PRODUCCIÓN EMPRESARIAL",
            "profile": "production", 
            "description": "Sistema crítico, alta disponibilidad",
            "setup": "PostgreSQL + S3 + Load Balancer",
            "team_size": "5+ personas",
            "complexity": "Alta"
        }
    ]
    
    for stage_info in evolution:
        print(f"\n📍 {stage_info['stage']}")
        print(f"   🏷️  Perfil: {stage_info['profile']}")
        print(f"   💡 Descripción: {stage_info['description']}")
        print(f"   ⚙️  Setup: {stage_info['setup']}")
        print(f"   👥 Tamaño equipo: {stage_info['team_size']}")
        print(f"   🎯 Complejidad: {stage_info['complexity']}")


def show_migration_guide():
    """Muestra guía de migración entre perfiles."""
    print("\n🔄 GUÍA DE MIGRACIÓN ENTRE PERFILES")
    print("=" * 50)
    
    migrations = [
        {
            "from": "local_only",
            "to": "collaboration", 
            "steps": [
                "1. Configurar credenciales AWS",
                "2. Crear bucket S3",
                "3. Cambiar active_profile a 'collaboration'",
                "4. Ejecutar primera sincronización",
                "5. Compartir configuración con equipo"
            ],
            "considerations": [
                "⚠️  Requiere permisos AWS",
                "💰 Costos de S3 (mínimos)",
                "🔐 Gestión de credenciales"
            ]
        },
        {
            "from": "collaboration",
            "to": "production",
            "steps": [
                "1. Configurar servidor PostgreSQL",
                "2. Migrar datos desde SQLite",
                "3. Configurar load balancer (opcional)",
                "4. Actualizar tracking_uri",
                "5. Configurar monitoreo"
            ],
            "considerations": [
                "🏗️  Infraestructura dedicada",
                "💰 Costos de servidor",
                "🔧 Administración de BD",
                "📊 Monitoreo y alertas"
            ]
        }
    ]
    
    for migration in migrations:
        print(f"\n🔄 MIGRACIÓN: {migration['from']} → {migration['to']}")
        print("   📋 Pasos:")
        for step in migration['steps']:
            print(f"      {step}")
        print("   ⚠️  Consideraciones:")
        for consideration in migration['considerations']:
            print(f"      {consideration}")


def show_current_active_profile():
    """Muestra el perfil actualmente activo."""
    config = load_config()
    active_profile = config.get('active_profile', 'No definido')
    
    print(f"\n🎯 PERFIL ACTUALMENTE ACTIVO: {active_profile}")
    print("-" * 40)
    
    if active_profile in config.get('profiles', {}):
        profile_config = config['profiles'][active_profile]
        print(f"🗄️  Tracking URI: {profile_config.get('tracking_uri', 'N/A')}")
        print(f"📁 Artifacts: {profile_config.get('artifact_location', 'N/A')}")
        print(f"☁️  AWS: {'✅ Habilitado' if profile_config.get('aws_sync', False) else '❌ Deshabilitado'}")
        
        # Verificar si los recursos existen
        tracking_uri = profile_config.get('tracking_uri', '')
        if 'sqlite' in tracking_uri.lower():
            db_file = tracking_uri.replace('sqlite:///', '')
            exists = os.path.exists(db_file)
            print(f"🗄️  Base de datos: {'✅ Existe' if exists else '❌ No existe'}")
        
        artifact_location = profile_config.get('artifact_location', '')
        if artifact_location and not artifact_location.startswith('s3://'):
            exists = os.path.exists(artifact_location)
            print(f"📁 Directorio artifacts: {'✅ Existe' if exists else '❌ No existe'}")
    else:
        print("❌ Perfil no encontrado en configuración")


def main():
    """Función principal con menú interactivo."""
    while True:
        print("\n" + "="*60)
        print("📚 MLflow Configuration Examples - Equipo 52")
        print("="*60)
        
        print("\n🔧 Opciones disponibles:")
        print("1. 📊 Comparar todos los perfiles")
        print("2. 🚀 Ver ruta de evolución")
        print("3. 🔄 Guía de migración")
        print("4. 🎯 Ver perfil activo")
        print("5. 📝 Explicar configuración actual")
        print("6. ❌ Salir")
        
        choice = input("\n👉 Selecciona una opción (1-6): ")
        
        if choice == "1":
            show_profile_comparison()
        elif choice == "2":
            show_evolution_path()
        elif choice == "3":
            show_migration_guide()
        elif choice == "4":
            show_current_active_profile()
        elif choice == "5":
            config = load_config()
            print(f"\n📋 CONFIGURACIÓN ACTUAL:")
            print(f"   🧪 Experimento: {config.get('mlflow', {}).get('experiment', {}).get('name', 'N/A')}")
            print(f"   🏷️  Perfil activo: {config.get('active_profile', 'N/A')}")
            print(f"   ☁️  AWS configurado: {'✅ Sí' if config.get('aws') else '❌ No'}")
        elif choice == "6":
            print("👋 ¡Hasta luego!")
            break
        else:
            print("❌ Opción inválida")
        
        input("\nPresiona Enter para continuar...")


if __name__ == "__main__":
    main()