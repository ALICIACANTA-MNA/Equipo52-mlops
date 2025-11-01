#!/usr/bin/env python3
"""
DVC Pipeline Execution and Management Script

Script completo para ejecutar, gestionar y monitorear el pipeline DVC
con todas las mejores prácticas de MLOps integradas.

Funcionalidades:
- Ejecución completa o por etapas del pipeline
- Validación de dependencias y configuración
- Monitoreo del progreso y métricas
- Gestión de experimentos y artefactos
- Limpieza automática y recuperación
- Reporting detallado de resultados

Uso:
    python run_dvc_pipeline.py [opciones]
    
Ejemplos:
    python run_dvc_pipeline.py --run-all                    # Ejecutar todo el pipeline
    python run_dvc_pipeline.py --stage model_training       # Ejecutar solo entrenamiento
    python run_dvc_pipeline.py --validate                   # Validar configuración
    python run_dvc_pipeline.py --status                     # Ver estado del pipeline
    python run_dvc_pipeline.py --clean                      # Limpiar artefactos
"""

import os
import sys
import subprocess
import json
import time
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import argparse
from contextlib import contextmanager
import yaml
import logging

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/dvc_pipeline.log', mode='a'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class DVCPipelineManager:
    """Gestor completo del pipeline DVC con todas las funcionalidades MLOps"""
    
    def __init__(self, root_path: str = "."):
        self.root_path = Path(root_path)
        self.dvc_file = self.root_path / "dvc.yaml"
        self.params_file = self.root_path / "params.yaml"
        self.logs_dir = self.root_path / "logs"
        self.metrics_dir = self.root_path / "metrics"
        
        # Crear directorios necesarios
        self.logs_dir.mkdir(exist_ok=True)
        self.metrics_dir.mkdir(exist_ok=True)
        
        # Pipeline stages en orden de ejecución
        self.pipeline_stages = [
            "data_ingestion",
            "data_validation", 
            "data_preprocessing",
            "feature_engineering",
            "model_training",
            "model_evaluation",
            "model_comparison"
        ]
        
        logger.info(f"DVC Pipeline Manager inicializado en: {self.root_path}")
    
    def validate_environment(self) -> bool:
        """Validar que el entorno está correctamente configurado"""
        logger.info("Validando entorno del pipeline...")
        
        checks = []
        
        # 1. Verificar archivos DVC esenciales
        if not self.dvc_file.exists():
            logger.error(f"ERROR: Archivo dvc.yaml no encontrado: {self.dvc_file}")
            checks.append(False)
        else:
            logger.info(f"OK: dvc.yaml encontrado: {self.dvc_file}")
            checks.append(True)
            
        if not self.params_file.exists():
            logger.error(f"ERROR: Archivo params.yaml no encontrado: {self.params_file}")
            checks.append(False)
        else:
            logger.info(f"OK: params.yaml encontrado: {self.params_file}")
            checks.append(True)
        
        # 2. Verificar comandos DVC
        try:
            result = subprocess.run(["dvc", "version"], capture_output=True, text=True)
            if result.returncode == 0:
                logger.info(f"OK: DVC disponible: {result.stdout.strip()}")
                checks.append(True)
            else:
                logger.error("ERROR: DVC no está disponible")
                checks.append(False)
        except FileNotFoundError:
            logger.error("ERROR: DVC no está instalado")
            checks.append(False)
        
        # 3. Verificar Git
        try:
            result = subprocess.run(["git", "status"], capture_output=True, text=True)
            if result.returncode == 0:
                logger.info("OK: Repositorio Git válido")
                checks.append(True)
            else:
                logger.warning("WARNING: Problemas con repositorio Git")
                checks.append(True)  # No crítico
        except FileNotFoundError:
            logger.warning("WARNING: Git no disponible")
            checks.append(True)  # No crítico
        
        # 4. Verificar estructura de directorios
        required_dirs = ["src", "data", "models", "configs"]
        for dir_name in required_dirs:
            dir_path = self.root_path / dir_name
            if dir_path.exists():
                logger.info(f"OK: Directorio encontrado: {dir_name}")
                checks.append(True)
            else:
                logger.error(f"ERROR: Directorio faltante: {dir_name}")
                checks.append(False)
        
        # 5. Verificar archivos Python clave
        key_files = [
            "src/data/data_ingestion.py",
            "src/models/train.py",
            "src/features/feature_engineering.py"
        ]
        for file_path in key_files:
            full_path = self.root_path / file_path
            if full_path.exists():
                logger.info(f"OK: Archivo encontrado: {file_path}")
                checks.append(True)
            else:
                logger.error(f"ERROR: Archivo faltante: {file_path}")
                checks.append(False)
        
        success = all(checks)
        if success:
            logger.info("SUCCESS: Validación del entorno completada exitosamente")
        else:
            logger.error("FAILED: Validación del entorno falló")
            
        return success
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Obtener estado actual del pipeline"""
        logger.info("Obteniendo estado del pipeline...")
        
        status = {
            "timestamp": datetime.now().isoformat(),
            "stages": {},
            "overall_status": "unknown"
        }
        
        try:
            # Ejecutar dvc status
            result = subprocess.run(
                ["dvc", "status"], 
                capture_output=True, 
                text=True,
                cwd=self.root_path
            )
            
            if result.returncode == 0:
                if not result.stdout.strip():
                    status["overall_status"] = "up_to_date"
                    logger.info("OK: Pipeline está actualizado")
                else:
                    status["overall_status"] = "changes_detected"
                    logger.info("INFO: Cambios detectados en el pipeline")
                    
                # Parsear salida para obtener detalles por stage
                for line in result.stdout.split('\n'):
                    if line.strip():
                        for stage in self.pipeline_stages:
                            if stage in line:
                                status["stages"][stage] = "modified"
            else:
                status["overall_status"] = "error"
                logger.error(f"Error al obtener estado: {result.stderr}")
                
        except Exception as e:
            logger.error(f"Error ejecutando dvc status: {e}")
            status["overall_status"] = "error"
        
        return status
    
    def run_stage(self, stage_name: str, force: bool = False) -> bool:
        """Ejecutar una etapa específica del pipeline"""
        logger.info(f"RUNNING: Ejecutando etapa: {stage_name}")
        
        if stage_name not in self.pipeline_stages:
            logger.error(f"ERROR: Etapa no válida: {stage_name}")
            return False
        
        try:
            cmd = ["dvc", "repro", stage_name]
            if force:
                cmd.append("--force")
            
            start_time = time.time()
            
            result = subprocess.run(
                cmd,
                cwd=self.root_path,
                capture_output=True,
                text=True
            )
            
            execution_time = time.time() - start_time
            
            if result.returncode == 0:
                logger.info(f"Etapa {stage_name} completada en {execution_time:.2f}s")
                logger.info(f"Salida: {result.stdout}")
                return True
            else:
                logger.error(f"Error en etapa {stage_name}")
                logger.error(f"Error: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Excepción ejecutando {stage_name}: {e}")
            return False
    
    def run_full_pipeline(self, force: bool = False) -> bool:
        """Ejecutar el pipeline completo"""
        logger.info("TARGET: Iniciando ejecución completa del pipeline")
        
        start_time = time.time()
        failed_stages = []
        
        for i, stage in enumerate(self.pipeline_stages, 1):
            logger.info(f"Ejecutando etapa {i}/{len(self.pipeline_stages)}: {stage}")
            
            success = self.run_stage(stage, force=force)
            
            if not success:
                failed_stages.append(stage)
                logger.error(f"Pipeline falló en etapa: {stage}")
                break
            
            logger.info(f"Etapa {stage} completada")
        
        total_time = time.time() - start_time
        
        if not failed_stages:
            logger.info(f"Pipeline completado exitosamente en {total_time:.2f}s")
            self._generate_pipeline_report()
            return True
        else:
            logger.error(f"Pipeline falló. Etapas fallidas: {failed_stages}")
            return False
    
    def clean_pipeline(self, keep_data: bool = True) -> bool:
        """Limpiar artefactos del pipeline"""
        logger.info("🧹 Limpiando artefactos del pipeline...")
        
        try:
            # Limpiar cache DVC
            result = subprocess.run(
                ["dvc", "cache", "dir"],
                capture_output=True,
                text=True,
                cwd=self.root_path
            )
            
            if result.returncode == 0:
                cache_dir = result.stdout.strip()
                logger.info(f"Cache DVC ubicado en: {cache_dir}")
            
            # Limpiar outputs generados
            dirs_to_clean = ["models", "metrics"]
            if not keep_data:
                dirs_to_clean.extend(["data/processed", "data/interim"])
            
            for dir_name in dirs_to_clean:
                dir_path = self.root_path / dir_name
                if dir_path.exists():
                    for item in dir_path.iterdir():
                        if item.is_file():
                            item.unlink()
                            logger.info(f"DELETED: Eliminado: {item}")
                        elif item.is_dir():
                            shutil.rmtree(item)
                            logger.info(f"DELETED: Eliminado directorio: {item}")
            
            logger.info("Limpieza completada")
            return True
            
        except Exception as e:
            logger.error(f"Error durante limpieza: {e}")
            return False
    
    def _generate_pipeline_report(self) -> None:
        """Generar reporte completo del pipeline"""
        logger.info("Generando reporte del pipeline...")
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "pipeline_status": "completed",
            "stages_executed": self.pipeline_stages,
            "metrics": {},
            "artifacts": {}
        }
        
        # Recopilar métricas
        metrics_files = [
            "metrics/train_metrics.json",
            "metrics/evaluation_metrics.json", 
            "metrics/comparison_metrics.json"
        ]
        
        for metrics_file in metrics_files:
            file_path = self.root_path / metrics_file
            if file_path.exists():
                try:
                    with open(file_path, 'r') as f:
                        metrics_data = json.load(f)
                        report["metrics"][metrics_file] = metrics_data
                except Exception as e:
                    logger.warning(f"No se pudo leer {metrics_file}: {e}")
        
        # Listar artefactos generados
        artifact_dirs = ["models", "data/processed", "metrics"]
        for dir_name in artifact_dirs:
            dir_path = self.root_path / dir_name
            if dir_path.exists():
                artifacts = [str(f.relative_to(self.root_path)) 
                           for f in dir_path.rglob("*") if f.is_file()]
                report["artifacts"][dir_name] = artifacts
        
        # Guardar reporte
        report_file = self.root_path / "reports" / f"pipeline_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report_file.parent.mkdir(exist_ok=True)
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Reporte guardado en: {report_file}")
    
    def watch_pipeline(self, stage: Optional[str] = None) -> None:
        """Monitorear ejecución del pipeline en tiempo real"""
        logger.info("Iniciando monitoreo del pipeline...")
        
        try:
            cmd = ["dvc", "repro"]
            if stage:
                cmd.append(stage)
            cmd.extend(["--force", "--verbose"])
            
            process = subprocess.Popen(
                cmd,
                cwd=self.root_path,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            for line in process.stdout:
                print(f"[DVC] {line.rstrip()}")
                logger.info(f"DVC: {line.rstrip()}")
            
            process.wait()
            
            if process.returncode == 0:
                logger.info("Pipeline completado exitosamente")
            else:
                logger.error(f"Pipeline falló con código: {process.returncode}")
                
        except KeyboardInterrupt:
            logger.info("Monitoreo interrumpido por usuario")
        except Exception as e:
            logger.error(f"Error durante monitoreo: {e}")


def create_parser() -> argparse.ArgumentParser:
    """Crear parser de argumentos de línea de comandos"""
    parser = argparse.ArgumentParser(
        description="DVC Pipeline Management Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  %(prog)s --run-all                    # Ejecutar todo el pipeline
  %(prog)s --stage model_training       # Ejecutar solo entrenamiento
  %(prog)s --validate                   # Validar configuración
  %(prog)s --status                     # Ver estado del pipeline
  %(prog)s --clean                      # Limpiar artefactos
  %(prog)s --watch                      # Monitorear ejecución
        """
    )
    
    # Argumentos principales
    parser.add_argument("--run-all", action="store_true",
                       help="Ejecutar todo el pipeline")
    parser.add_argument("--stage", type=str,
                       help="Ejecutar una etapa específica")
    parser.add_argument("--validate", action="store_true",
                       help="Validar configuración del entorno")
    parser.add_argument("--status", action="store_true",
                       help="Mostrar estado del pipeline")
    parser.add_argument("--clean", action="store_true",
                       help="Limpiar artefactos del pipeline")
    parser.add_argument("--watch", action="store_true",
                       help="Monitorear ejecución en tiempo real")
    
    # Argumentos modificadores
    parser.add_argument("--force", action="store_true",
                       help="Forzar re-ejecución de etapas")
    parser.add_argument("--keep-data", action="store_true", default=True,
                       help="Mantener datos al limpiar (default: True)")
    parser.add_argument("--root-path", type=str, default=".",
                       help="Ruta raíz del proyecto (default: .)")
    
    return parser


def main():
    """Función principal"""
    parser = create_parser()
    args = parser.parse_args()
    
    # Crear gestor del pipeline
    manager = DVCPipelineManager(args.root_path)
    
    print("DVC Pipeline Manager")
    print("=" * 50)
    
    # Validación
    if args.validate or not any([args.run_all, args.stage, args.status, 
                                args.clean, args.watch]):
        if not manager.validate_environment():
            print("Validación falló. Revisa los errores arriba.")
            sys.exit(1)
        print("Validación completada exitosamente")
        if args.validate:
            return
    
    # Estado del pipeline
    if args.status:
        status = manager.get_pipeline_status()
        print(f"\nEstado del Pipeline:")
        print(f"Estado general: {status['overall_status']}")
        print(f"Timestamp: {status['timestamp']}")
        if status['stages']:
            print("Etapas modificadas:")
            for stage, state in status['stages'].items():
                print(f"  - {stage}: {state}")
        return
    
    # Limpieza
    if args.clean:
        success = manager.clean_pipeline(keep_data=args.keep_data)
        if success:
            print("Limpieza completada")
        else:
            print("Error durante limpieza")
            sys.exit(1)
        return
    
    # Monitoreo
    if args.watch:
        manager.watch_pipeline(args.stage)
        return
    
    # Ejecución
    if args.run_all:
        success = manager.run_full_pipeline(force=args.force)
        if success:
            print("Pipeline completado exitosamente")
        else:
            print("Pipeline falló")
            sys.exit(1)
    elif args.stage:
        success = manager.run_stage(args.stage, force=args.force)
        if success:
            print(f"Etapa {args.stage} completada")
        else:
            print(f"Etapa {args.stage} falló")
            sys.exit(1)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()