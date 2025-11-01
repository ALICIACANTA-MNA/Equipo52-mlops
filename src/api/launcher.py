#!/usr/bin/env python3
"""
Unified API Launcher - Obesity Prediction MLOps Pipeline

Script unificado que combina las funcionalidades de deployment y ejecución
de la API. Proporciona una interfaz consistente para todos los comandos
relacionados con la API.

Comandos principales:
- dev: Ejecutar API en modo desarrollo (run.py)
- deploy: Orquestar deployment con Docker (deploy.py) 
- status: Ver estado de servicios
- logs: Ver logs de servicios
- stop: Detener servicios

Ejemplos de uso:
    python scripts/api/launcher.py dev                    # Desarrollo simple
    python scripts/api/launcher.py dev --port 8080        # Puerto personalizado
    python scripts/api/launcher.py deploy start --env dev # Deployment desarrollo
    python scripts/api/launcher.py deploy start --env prod # Deployment producción
    python scripts/api/launcher.py status                 # Estado servicios
    python scripts/api/launcher.py logs --service api     # Logs específicos
    python scripts/api/launcher.py stop                   # Detener todo

Desde raíz del proyecto:
    python -m src.api.launcher dev
    python -m src.api.launcher deploy start --env production
"""

import sys
import argparse
from pathlib import Path

# Add scripts directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

def main():
    """Punto de entrada principal del launcher unificado."""
    
    parser = argparse.ArgumentParser(
        description="Unified API Launcher for Obesity Prediction API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  %(prog)s dev                           # Desarrollo simple
  %(prog)s dev --port 8080               # Puerto personalizado  
  %(prog)s deploy start --env dev        # Deployment desarrollo
  %(prog)s deploy start --env prod       # Deployment producción
  %(prog)s status                        # Estado de servicios
  %(prog)s logs --service api            # Ver logs
  %(prog)s stop                          # Detener servicios

Para más información sobre comandos específicos:
  %(prog)s dev --help
  %(prog)s deploy --help
        """)
    
    subparsers = parser.add_subparsers(
        dest='command',
        help='Comandos disponibles',
        metavar='COMMAND'
    )
    
    # === DEV COMMAND (from run.py) ===
    dev_parser = subparsers.add_parser(
        'dev',
        help='Ejecutar API en modo desarrollo',
        description='Lanza la API usando uvicorn para desarrollo local'
    )
    dev_parser.add_argument(
        '--port', '-p',
        type=int,
        default=8000,
        help='Puerto para la API (default: 8000)'
    )
    dev_parser.add_argument(
        '--host',
        default='127.0.0.1',
        help='Host para la API (default: 127.0.0.1)'
    )
    dev_parser.add_argument(
        '--workers',
        type=int,
        default=1,
        help='Número de workers (default: 1 para desarrollo)'
    )
    dev_parser.add_argument(
        '--reload',
        action='store_true',
        default=True,
        help='Auto-reload on file changes (default: True)'
    )
    dev_parser.add_argument(
        '--env',
        choices=['development', 'testing'],
        default='development',
        help='Entorno de desarrollo (default: development)'
    )
    
    # === DEPLOY COMMAND (from deploy.py) ===  
    deploy_parser = subparsers.add_parser(
        'deploy',
        help='Gestionar deployment con Docker',
        description='Orquesta el deployment usando Docker Compose'
    )
    deploy_subparsers = deploy_parser.add_subparsers(
        dest='deploy_action',
        help='Acciones de deployment'
    )
    
    # deploy start
    start_parser = deploy_subparsers.add_parser(
        'start',
        help='Iniciar servicios'
    )
    start_parser.add_argument(
        '--env',
        choices=['development', 'production', 'testing'],
        default='development',
        help='Entorno de deployment (default: development)'
    )
    start_parser.add_argument(
        '--build',
        action='store_true',
        help='Forzar rebuild de imágenes'
    )
    start_parser.add_argument(
        '--detach', '-d',
        action='store_true',
        default=True,
        help='Ejecutar en background (default: True)'
    )
    
    # deploy stop
    deploy_subparsers.add_parser(
        'stop',
        help='Detener servicios'
    )
    
    # deploy status  
    deploy_subparsers.add_parser(
        'status',
        help='Ver estado de servicios'
    )
    
    # deploy logs
    logs_parser = deploy_subparsers.add_parser(
        'logs',
        help='Ver logs de servicios'
    )
    logs_parser.add_argument(
        '--service',
        help='Servicio específico (api, db, redis, etc.)'
    )
    logs_parser.add_argument(
        '--follow', '-f',
        action='store_true',
        help='Seguir logs en tiempo real'
    )
    logs_parser.add_argument(
        '--tail',
        type=int,
        default=100,
        help='Número de líneas a mostrar (default: 100)'
    )
    
    # deploy scale
    scale_parser = deploy_subparsers.add_parser(
        'scale',
        help='Escalar servicios'
    )
    scale_parser.add_argument(
        '--replicas',
        type=int,
        required=True,
        help='Número de réplicas'
    )
    scale_parser.add_argument(
        '--service',
        default='api',
        help='Servicio a escalar (default: api)'
    )
    
    # === QUICK COMMANDS ===
    subparsers.add_parser(
        'status',
        help='Ver estado rápido de servicios (alias para deploy status)'
    )
    
    logs_quick_parser = subparsers.add_parser(
        'logs',
        help='Ver logs rápidamente (alias para deploy logs)'
    )
    logs_quick_parser.add_argument(
        '--service',
        help='Servicio específico'
    )
    logs_quick_parser.add_argument(
        '--follow', '-f',
        action='store_true',
        help='Seguir logs'
    )
    
    subparsers.add_parser(
        'stop',
        help='Detener servicios rápidamente (alias para deploy stop)'
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
        
    # Route to appropriate handler
    try:
        if args.command == 'dev':
            return handle_dev_command(args)
        elif args.command == 'deploy':
            return handle_deploy_command(args)
        elif args.command in ['status', 'logs', 'stop']:
            return handle_quick_command(args)
        else:
            print(f"Comando desconocido: {args.command}")
            return 1
            
    except Exception as e:
        print(f"Error ejecutando comando: {e}")
        return 1

def handle_dev_command(args):
    """Maneja comandos de desarrollo (uvicorn integrado)."""
    import subprocess
    import os
    
    try:
        # Configurar variables de entorno
        os.environ["API_ENV"] = args.env
        
        print("🚀 Starting Obesity Prediction API")
        print("📊 Environment:", args.env)
        print(f"🌐 Server: {args.host}:{args.port}")
        
        # Construir comando uvicorn
        cmd = [
            sys.executable, "-m", "uvicorn",
            "src.api.serve:app",
            "--host", args.host,
            "--port", str(args.port),
        ]
        
        if args.reload or args.env == "development":
            cmd.append("--reload")
            
        cmd.extend(["--log-level", "info"])
        
        print("📋 Command:", " ".join(cmd))
        print("=" * 50)
        
        # Verificar dependencias
        try:
            import uvicorn
            print("✅ FastAPI dependencies available")
        except ImportError:
            print("❌ Missing dependencies: No module named 'uvicorn'")
            print("Install with: pip install fastapi uvicorn")
            return 1
        
        # Verificar modelo existe (desde la raíz del proyecto)
        from pathlib import Path
        project_root = Path(__file__).parent.parent.parent
        model_path = project_root / "models/best_model.joblib"
        if model_path.exists():
            print("✅ Model file found")
        else:
            print("⚠️  Model file not found - API will attempt to load from MLflow")
        
        print("=" * 50)
        
        # Ejecutar comando
        result = subprocess.run(cmd, check=False)
        return result.returncode
        
    except Exception as e:
        print(f"Error ejecutando comando dev: {e}")
        return 1

def handle_deploy_command(args):
    """Maneja comandos de deployment (deploy.py)."""
    try:
        from deploy import main as deploy_main
        
        if not args.deploy_action:
            print("Error: Se requiere una acción para deploy")
            print("Uso: launcher.py deploy {start|stop|status|logs|scale} [opciones]")
            return 1
            
        # Convert args to format expected by deploy.py
        deploy_args = [args.deploy_action]
        
        if args.deploy_action == 'start':
            deploy_args.extend(['--env', args.env])
            if args.build:
                deploy_args.append('--build')
            if args.detach:
                deploy_args.append('--detach')
                
        elif args.deploy_action == 'logs':
            if args.service:
                deploy_args.extend(['--service', args.service])
            if args.follow:
                deploy_args.append('--follow')
            deploy_args.extend(['--tail', str(args.tail)])
            
        elif args.deploy_action == 'scale':
            deploy_args.extend(['--replicas', str(args.replicas)])
            deploy_args.extend(['--service', args.service])
        
        # Override sys.argv for deploy.py
        original_argv = sys.argv.copy()
        sys.argv = ['deploy.py'] + deploy_args
        
        try:
            result = deploy_main()
            return result if result is not None else 0
        finally:
            sys.argv = original_argv
            
    except ImportError as e:
        print(f"Error importando deploy.py: {e}")
        print("Asegúrate de que deploy.py esté en scripts/api/")
        return 1

def handle_quick_command(args):
    """Maneja comandos rápidos (aliases para deploy)."""
    try:
        from deploy import main as deploy_main
        
        # Convert quick commands to deploy equivalents
        if args.command == 'status':
            deploy_args = ['status']
        elif args.command == 'stop':
            deploy_args = ['stop']
        elif args.command == 'logs':
            deploy_args = ['logs']
            if hasattr(args, 'service') and args.service:
                deploy_args.extend(['--service', args.service])
            if hasattr(args, 'follow') and args.follow:
                deploy_args.append('--follow')
        
        # Override sys.argv for deploy.py
        original_argv = sys.argv.copy()
        sys.argv = ['deploy.py'] + deploy_args
        
        try:
            result = deploy_main()
            return result if result is not None else 0
        finally:
            sys.argv = original_argv
            
    except ImportError as e:
        print(f"Error importando deploy.py: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main())