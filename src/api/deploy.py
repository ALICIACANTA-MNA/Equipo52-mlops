#!/usr/bin/env python3
"""
API Deployment Orchestrator for Obesity Prediction MLOps Pipeline.

Script integral para gestionar el despliegue de la API usando la configuración
Docker existente en configs/docker/. Soporta múltiples entornos y proporciona
comandos unificados para el ciclo de vida completo de la API.

Uso:
    python scripts/api/deploy.py start --env development  # Desarrollo
    python scripts/api/deploy.py start --env production   # Producción  
    python scripts/api/deploy.py stop                     # Detener servicios
    python scripts/api/deploy.py status                   # Estado actual
    python scripts/api/deploy.py logs --service api       # Ver logs
    python scripts/api/deploy.py scale --replicas 4       # Escalar API
    
    # O usar el launcher unificado:
    python scripts/api/launcher.py deploy start --env development
    python api_launcher.py deploy start --env development  # Wrapper compatibilidad

Referencias:
- Docker Configuration: configs/docker/
- API Configuration: configs/api/
- Production Guide: docs/DEPLOYMENT.md
"""

import os
import sys
import argparse
import subprocess
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

try:
    import yaml
    import requests
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.live import Live
    from rich.layout import Layout
except ImportError as e:
    print(f"Error: Dependencias faltantes: {e}")
    print("Instalar con: pip install pyyaml requests rich")
    sys.exit(1)

console = Console()


class APIDeploymentManager:
    """Gestor de despliegue para la API de obesidad"""
    
    def __init__(self):
        """Inicializa el gestor de despliegue"""
        self.project_root = Path(__file__).parent.parent.parent
        self.docker_config_dir = self.project_root / "configs" / "docker"
        self.api_config_dir = self.project_root / "configs" / "api"
        
        # Verificar estructura de directorios
        if not self.docker_config_dir.exists():
            raise FileNotFoundError(f"Docker config directory not found: {self.docker_config_dir}")
        if not self.api_config_dir.exists():
            raise FileNotFoundError(f"API config directory not found: {self.api_config_dir}")
    
    def load_environment_config(self, env: str) -> Dict[str, Any]:
        """Carga configuración de entorno"""
        env_file = self.docker_config_dir / f".env.{env}" if env != "dev" else self.docker_config_dir / ".env"
        
        if not env_file.exists():
            raise FileNotFoundError(f"Environment file not found: {env_file}")
        
        config = {}
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    config[key] = value
        
        return config
    
    def check_prerequisites(self) -> bool:
        """Verifica prerequisites para despliegue"""
        console.print("🔍 Checking deployment prerequisites...", style="bold blue")
        
        checks = []
        
        # Docker
        try:
            result = subprocess.run(['docker', '--version'], capture_output=True, text=True)
            if result.returncode == 0:
                checks.append(("Docker", "✅ Available", "green"))
            else:
                checks.append(("Docker", "❌ Not available", "red"))
        except FileNotFoundError:
            checks.append(("Docker", "❌ Not installed", "red"))
        
        # Docker Compose
        try:
            result = subprocess.run(['docker-compose', '--version'], capture_output=True, text=True)
            if result.returncode == 0:
                checks.append(("Docker Compose", "✅ Available", "green"))
            else:
                checks.append(("Docker Compose", "❌ Not available", "red"))
        except FileNotFoundError:
            checks.append(("Docker Compose", "❌ Not installed", "red"))
        
        # Model files
        model_path = self.project_root / "models" / "best_model.joblib"
        if model_path.exists():
            checks.append(("Model File", "✅ Found", "green"))
        else:
            checks.append(("Model File", "⚠️  Not found (will use MLflow)", "yellow"))
        
        # Configuration files
        docker_compose_file = self.docker_config_dir / "docker-compose.yml"
        if docker_compose_file.exists():
            checks.append(("Docker Compose Config", "✅ Found", "green"))
        else:
            checks.append(("Docker Compose Config", "❌ Missing", "red"))
        
        # Display results
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Component")
        table.add_column("Status")
        
        all_good = True
        for component, status, color in checks:
            table.add_row(component, status)
            if "❌" in status:
                all_good = False
        
        console.print(table)
        
        return all_good
    
    def start_services(self, env: str = "dev", services: Optional[List[str]] = None) -> bool:
        """Inicia servicios de la API"""
        console.print(f"🚀 Starting API services (environment: {env})", style="bold green")
        
        # Preparar comando docker-compose
        os.chdir(self.docker_config_dir)
        
        if env == "dev":
            compose_file = "docker-compose.dev.yml"
            env_file = ".env.dev"
        else:
            compose_file = "docker-compose.yml"
            env_file = f".env.{env}"
        
        cmd = ["docker-compose", "-f", compose_file, "--env-file", env_file]
        
        if services:
            cmd.extend(["up", "-d"] + services)
        else:
            cmd.extend(["up", "-d"])
        
        try:
            console.print(f"📋 Command: {' '.join(cmd)}")
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Starting services...", total=None)
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    console.print("✅ Services started successfully", style="green")
                    
                    # Wait for health checks
                    self._wait_for_health_checks(env)
                    
                    return True
                else:
                    console.print(f"❌ Failed to start services: {result.stderr}", style="red")
                    return False
        
        except Exception as e:
            console.print(f"❌ Error starting services: {e}", style="red")
            return False
    
    def stop_services(self) -> bool:
        """Detiene servicios de la API"""
        console.print("🛑 Stopping API services...", style="bold red")
        
        os.chdir(self.docker_config_dir)
        
        try:
            # Try both compose files
            for compose_file in ["docker-compose.yml", "docker-compose.dev.yml"]:
                if Path(compose_file).exists():
                    cmd = ["docker-compose", "-f", compose_file, "down"]
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    
                    if result.returncode == 0:
                        console.print(f"✅ Services stopped using {compose_file}", style="green")
                    else:
                        console.print(f"⚠️  Warning stopping {compose_file}: {result.stderr}", style="yellow")
            
            return True
        
        except Exception as e:
            console.print(f"❌ Error stopping services: {e}", style="red")
            return False
    
    def get_service_status(self) -> Dict[str, Any]:
        """Obtiene estado de los servicios"""
        os.chdir(self.docker_config_dir)
        
        status = {
            "containers": [],
            "api_health": None,
            "mlflow_health": None
        }
        
        try:
            # Get container status
            result = subprocess.run(
                ["docker-compose", "ps", "--format", "json"],
                capture_output=True, text=True
            )
            
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    if line.strip():
                        try:
                            container_info = json.loads(line)
                            status["containers"].append(container_info)
                        except json.JSONDecodeError:
                            pass
            
            # Check API health
            try:
                response = requests.get("http://localhost:8000/health", timeout=5)
                if response.status_code == 200:
                    status["api_health"] = response.json()
                else:
                    status["api_health"] = {"status": "unhealthy", "code": response.status_code}
            except requests.RequestException:
                status["api_health"] = {"status": "unreachable"}
            
            # Check MLflow health
            try:
                response = requests.get("http://localhost:5000", timeout=5)
                if response.status_code == 200:
                    status["mlflow_health"] = {"status": "healthy"}
                else:
                    status["mlflow_health"] = {"status": "unhealthy", "code": response.status_code}
            except requests.RequestException:
                status["mlflow_health"] = {"status": "unreachable"}
        
        except Exception as e:
            console.print(f"⚠️  Error getting status: {e}", style="yellow")
        
        return status
    
    def show_status(self):
        """Muestra estado detallado de los servicios"""
        console.print("📊 Service Status", style="bold blue")
        
        status = self.get_service_status()
        
        # Container status
        if status["containers"]:
            console.print("\n🐳 Container Status:", style="bold")
            
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("Service")
            table.add_column("Status")
            table.add_column("Ports")
            table.add_column("Health")
            
            for container in status["containers"]:
                name = container.get("Service", "Unknown")
                state = container.get("State", "Unknown")
                ports = container.get("Publishers", [])
                port_str = ", ".join([f"{p.get('PublishedPort', '')}:{p.get('TargetPort', '')}" for p in ports]) if ports else "None"
                
                # Health status
                health = "Unknown"
                if state == "running":
                    health = "✅ Running"
                elif state == "exited":
                    health = "❌ Stopped"
                
                table.add_row(name, state, port_str, health)
            
            console.print(table)
        else:
            console.print("No containers found", style="yellow")
        
        # API Health
        console.print("\n🔗 Service Health:", style="bold")
        
        health_table = Table(show_header=True, header_style="bold cyan")
        health_table.add_column("Service")
        health_table.add_column("Status")
        health_table.add_column("URL")
        
        # API
        api_status = status["api_health"]
        if api_status:
            if api_status.get("status") == "healthy":
                health_table.add_row("API", "✅ Healthy", "http://localhost:8000")
            else:
                health_table.add_row("API", "❌ Unhealthy", "http://localhost:8000")
        else:
            health_table.add_row("API", "❓ Unknown", "http://localhost:8000")
        
        # MLflow
        mlflow_status = status["mlflow_health"]
        if mlflow_status:
            if mlflow_status.get("status") == "healthy":
                health_table.add_row("MLflow", "✅ Healthy", "http://localhost:5000")
            else:
                health_table.add_row("MLflow", "❌ Unhealthy", "http://localhost:5000")
        else:
            health_table.add_row("MLflow", "❓ Unknown", "http://localhost:5000")
        
        console.print(health_table)
        
        # Quick links
        console.print("\n🔗 Quick Links:", style="bold green")
        console.print("• API Documentation: http://localhost:8000/docs")
        console.print("• API Health Check: http://localhost:8000/health")
        console.print("• MLflow Tracking: http://localhost:5000")
        console.print("• Prometheus Metrics: http://localhost:8000/metrics")
    
    def show_logs(self, service: Optional[str] = None, follow: bool = False):
        """Muestra logs de servicios"""
        os.chdir(self.docker_config_dir)
        
        cmd = ["docker-compose", "logs"]
        if follow:
            cmd.append("-f")
        if service:
            cmd.append(service)
        
        try:
            console.print(f"📋 Showing logs{' (following)' if follow else ''}...", style="bold blue")
            subprocess.run(cmd)
        except KeyboardInterrupt:
            console.print("\n📋 Log viewing stopped", style="yellow")
        except Exception as e:
            console.print(f"❌ Error showing logs: {e}", style="red")
    
    def scale_service(self, service: str, replicas: int):
        """Escala un servicio"""
        os.chdir(self.docker_config_dir)
        
        cmd = ["docker-compose", "up", "-d", "--scale", f"{service}={replicas}"]
        
        try:
            console.print(f"📈 Scaling {service} to {replicas} replicas...", style="bold blue")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                console.print(f"✅ {service} scaled to {replicas} replicas", style="green")
            else:
                console.print(f"❌ Failed to scale {service}: {result.stderr}", style="red")
        
        except Exception as e:
            console.print(f"❌ Error scaling service: {e}", style="red")
    
    def _wait_for_health_checks(self, env: str, timeout: int = 60):
        """Espera a que los servicios estén saludables"""
        console.print("⏱️  Waiting for services to be healthy...", style="bold yellow")
        
        start_time = time.time()
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Health checking...", total=None)
            
            while time.time() - start_time < timeout:
                try:
                    # Check API
                    api_response = requests.get("http://localhost:8000/health", timeout=5)
                    if api_response.status_code == 200:
                        console.print("✅ API is healthy", style="green")
                        return
                except requests.RequestException:
                    pass
                
                time.sleep(2)
        
        console.print("⚠️  Health check timeout - services may still be starting", style="yellow")


def main():
    """Función principal"""
    parser = argparse.ArgumentParser(description="API Deployment Manager")
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Start command
    start_parser = subparsers.add_parser("start", help="Start API services")
    start_parser.add_argument("--env", choices=["dev", "prod"], default="dev", help="Environment")
    start_parser.add_argument("--services", nargs="*", help="Specific services to start")
    
    # Stop command
    subparsers.add_parser("stop", help="Stop API services")
    
    # Status command
    subparsers.add_parser("status", help="Show service status")
    
    # Logs command
    logs_parser = subparsers.add_parser("logs", help="Show service logs")
    logs_parser.add_argument("--service", help="Specific service")
    logs_parser.add_argument("--follow", "-f", action="store_true", help="Follow logs")
    
    # Scale command
    scale_parser = subparsers.add_parser("scale", help="Scale services")
    scale_parser.add_argument("--service", default="api", help="Service to scale")
    scale_parser.add_argument("--replicas", type=int, required=True, help="Number of replicas")
    
    # Check command
    subparsers.add_parser("check", help="Check prerequisites")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        manager = APIDeploymentManager()
        
        if args.command == "start":
            if not manager.check_prerequisites():
                console.print("❌ Prerequisites not met. Please fix the issues above.", style="red")
                return
            
            manager.start_services(args.env, args.services)
        
        elif args.command == "stop":
            manager.stop_services()
        
        elif args.command == "status":
            manager.show_status()
        
        elif args.command == "logs":
            manager.show_logs(args.service, args.follow)
        
        elif args.command == "scale":
            manager.scale_service(args.service, args.replicas)
        
        elif args.command == "check":
            if manager.check_prerequisites():
                console.print("✅ All prerequisites met!", style="green")
            else:
                console.print("❌ Some prerequisites are missing.", style="red")
    
    except FileNotFoundError as e:
        console.print(f"❌ Configuration error: {e}", style="red")
        console.print("Make sure you're running from the project root directory.", style="yellow")
    except Exception as e:
        console.print(f"❌ Unexpected error: {e}", style="red")


if __name__ == "__main__":
    main()