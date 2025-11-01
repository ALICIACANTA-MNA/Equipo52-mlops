"""
MLflow S3 Sync Manager - Equipo 52 MLOps
========================================

FLUJO DE DATOS MLflow HÍBRIDO:
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Pipeline      │───▶│   MLflow Local  │───▶│    AWS S3       │
│   (train.py)    │    │   (sqlite+fs)   │    │   (backup)      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                ▲                       │
                                │      ┌─────────────────┐
                                └──────│   MLflow UI     │
                                       │ (localhost:5000)│
                                       └─────────────────┘

Este módulo gestiona la sincronización automática de artefactos MLflow 
entre el almacenamiento local y AWS S3, permitiendo:

1. DESARROLLO LOCAL: Rápido acceso a experimentos y modelos
2. COLABORACIÓN: Sincronización automática con S3 para compartir
3. BACKUP: Respaldo automático de todos los artefactos importantes
4. PRODUCCIÓN: Despliegue de modelos desde S3

Author: Equipo 52 MLOps
Date: 2024
"""

import os
import json
import yaml
import boto3
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime
from botocore.exceptions import ClientError, NoCredentialsError

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MLflowS3SyncManager:
    """
    Gestor de sincronización MLflow Local <-> AWS S3.
    
    Funcionalidades:
    - Sync automático de artefactos después de cada run
    - Backup completo de experimentos a S3
    - Restauración de experimentos desde S3
    - Gestión de modelos en S3 Model Registry
    """
    
    def __init__(self, config_path: str = "configs/mlflow/mlflow_config.yaml"):
        """
        Inicializa el manager con configuración desde YAML.
        
        Args:
            config_path: Ruta al archivo de configuración MLflow
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.s3_client = None
        self.bucket_name = self.config.get('aws', {}).get('s3', {}).get('bucket_name', 'equipo52-mlops-artifacts')
        
        # Inicializar cliente S3
        self._init_s3_client()
        
    def _load_config(self) -> Dict[str, Any]:
        """Carga configuración desde archivo YAML."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error cargando configuración: {e}")
            return {}
    
    def _init_s3_client(self):
        """Inicializa cliente AWS S3."""
        try:
            aws_config = self.config.get('aws', {})
            credentials = aws_config.get('credentials', {})
            
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=credentials.get('access_key_id'),
                aws_secret_access_key=credentials.get('secret_access_key'),
                region_name=credentials.get('region', 'us-east-1')
            )
            
            # Verificar conexión
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            logger.info(f"Conexión S3 exitosa - Bucket: {self.bucket_name}")
            
        except NoCredentialsError:
            logger.error("Credenciales AWS no encontradas")
            self.s3_client = None
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                logger.warning(f"Bucket {self.bucket_name} no existe - creando...")
                self._create_bucket()
            else:
                logger.error(f"Error conectando a S3: {e}")
                self.s3_client = None
    
    def _create_bucket(self):
        """Crea el bucket S3 si no existe."""
        try:
            region = self.config.get('aws', {}).get('credentials', {}).get('region', 'us-east-1')
            
            if region == 'us-east-1':
                # us-east-1 no necesita LocationConstraint
                self.s3_client.create_bucket(Bucket=self.bucket_name)
            else:
                self.s3_client.create_bucket(
                    Bucket=self.bucket_name,
                    CreateBucketConfiguration={'LocationConstraint': region}
                )
            
            # Habilitar versionado
            self.s3_client.put_bucket_versioning(
                Bucket=self.bucket_name,
                VersioningConfiguration={'Status': 'Enabled'}
            )
            
            logger.info(f"Bucket {self.bucket_name} creado exitosamente")
            
        except ClientError as e:
            logger.error(f"Error creando bucket: {e}")
    
    def sync_artifacts_to_s3(self, local_path: str, s3_prefix: str = "experiments") -> bool:
        """
        Sincroniza artefactos locales a S3.
        
        Args:
            local_path: Ruta local de artefactos (ej: ./mlruns)
            s3_prefix: Prefijo en S3 (ej: experiments)
            
        Returns:
            True si la sincronización fue exitosa
        """
        if not self.s3_client:
            logger.warning("Cliente S3 no disponible - saltando sync")
            return False
            
        try:
            local_path = Path(local_path)
            if not local_path.exists():
                logger.warning(f"Ruta local no existe: {local_path}")
                return False
            
            sync_count = 0
            for file_path in local_path.rglob('*'):
                if file_path.is_file() and not self._should_exclude_file(file_path):
                    # Construir key S3
                    relative_path = file_path.relative_to(local_path)
                    s3_key = f"{s3_prefix}/{relative_path}".replace('\\\\', '/')
                    
                    # Subir archivo
                    self.s3_client.upload_file(
                        str(file_path),
                        self.bucket_name,
                        s3_key
                    )
                    sync_count += 1
            
            logger.info(f"Sincronizados {sync_count} archivos a S3")
            return True
            
        except Exception as e:
            logger.error(f"Error en sync a S3: {e}")
            return False
    
    def _should_exclude_file(self, file_path: Path) -> bool:
        """Verifica si un archivo debe excluirse del sync."""
        exclude_patterns = self.config.get('mlflow', {}).get('artifacts', {}).get('sync_config', {}).get('exclude_patterns', [])
        
        for pattern in exclude_patterns:
            if pattern in str(file_path):
                return True
        return False
    
    def download_from_s3(self, s3_prefix: str, local_path: str) -> bool:
        """
        Descarga artefactos desde S3 a local.
        
        Args:
            s3_prefix: Prefijo en S3
            local_path: Ruta local destino
            
        Returns:
            True si la descarga fue exitosa
        """
        if not self.s3_client:
            logger.warning("Cliente S3 no disponible")
            return False
        
        try:
            local_path = Path(local_path)
            local_path.mkdir(parents=True, exist_ok=True)
            
            # Listar objetos en S3
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=s3_prefix
            )
            
            download_count = 0
            for obj in response.get('Contents', []):
                s3_key = obj['Key']
                # Construir ruta local
                relative_path = s3_key.replace(s3_prefix + '/', '')
                local_file = local_path / relative_path
                
                # Crear directorios padre
                local_file.parent.mkdir(parents=True, exist_ok=True)
                
                # Descargar archivo
                self.s3_client.download_file(
                    self.bucket_name,
                    s3_key,
                    str(local_file)
                )
                download_count += 1
            
            logger.info(f"Descargados {download_count} archivos desde S3")
            return True
            
        except Exception as e:
            logger.error(f"Error descargando desde S3: {e}")
            return False
    
    def create_backup_snapshot(self) -> str:
        """
        Crea un snapshot completo de MLflow local en S3.
        
        Returns:
            ID del snapshot creado
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        snapshot_id = f"snapshot_{timestamp}"
        
        try:
            # Crear metadata del snapshot
            snapshot_metadata = {
                "snapshot_id": snapshot_id,
                "timestamp": timestamp,
                "source": "local_mlflow",
                "bucket": self.bucket_name,
                "created_by": "MLflowS3SyncManager"
            }
            
            # Subir metadata
            metadata_key = f"snapshots/{snapshot_id}/metadata.json"
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=metadata_key,
                Body=json.dumps(snapshot_metadata, indent=2),
                ContentType='application/json'
            )
            
            # Sincronizar todos los artefactos
            self.sync_artifacts_to_s3("./mlruns", f"snapshots/{snapshot_id}/mlruns")
            
            # Backup de la base de datos SQLite
            if os.path.exists("mlflow.db"):
                self.s3_client.upload_file(
                    "mlflow.db",
                    self.bucket_name,
                    f"snapshots/{snapshot_id}/mlflow.db"
                )
            
            logger.info(f"Snapshot creado: {snapshot_id}")
            return snapshot_id
            
        except Exception as e:
            logger.error(f"Error creando snapshot: {e}")
            return ""
    
    def list_snapshots(self) -> List[Dict[str, Any]]:
        """Lista todos los snapshots disponibles en S3."""
        if not self.s3_client:
            return []
        
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix="snapshots/",
                Delimiter='/'
            )
            
            snapshots = []
            for prefix in response.get('CommonPrefixes', []):
                snapshot_id = prefix['Prefix'].split('/')[-2]
                
                # Intentar obtener metadata
                try:
                    metadata_response = self.s3_client.get_object(
                        Bucket=self.bucket_name,
                        Key=f"snapshots/{snapshot_id}/metadata.json"
                    )
                    metadata = json.loads(metadata_response['Body'].read())
                    snapshots.append(metadata)
                except:
                    # Si no hay metadata, crear entrada básica
                    snapshots.append({
                        "snapshot_id": snapshot_id,
                        "timestamp": "unknown",
                        "source": "unknown"
                    })
            
            return sorted(snapshots, key=lambda x: x.get('timestamp', ''), reverse=True)
            
        except Exception as e:
            logger.error(f"Error listando snapshots: {e}")
            return []
    
    def restore_from_snapshot(self, snapshot_id: str) -> bool:
        """
        Restaura MLflow local desde un snapshot en S3.
        
        Args:
            snapshot_id: ID del snapshot a restaurar
            
        Returns:
            True si la restauración fue exitosa
        """
        try:
            logger.info(f"Restaurando desde snapshot: {snapshot_id}")
            
            # Crear backup del estado actual
            current_backup = self.create_backup_snapshot()
            logger.info(f"Backup actual creado: {current_backup}")
            
            # Limpiar mlruns local
            if os.path.exists("./mlruns"):
                shutil.rmtree("./mlruns")
            
            # Restaurar mlruns
            self.download_from_s3(f"snapshots/{snapshot_id}/mlruns", "./mlruns")
            
            # Restaurar base de datos
            try:
                self.s3_client.download_file(
                    self.bucket_name,
                    f"snapshots/{snapshot_id}/mlflow.db",
                    "mlflow.db"
                )
            except ClientError:
                logger.warning("No se encontró mlflow.db en el snapshot")
            
            logger.info(f"Restauración completada desde: {snapshot_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error restaurando snapshot: {e}")
            return False
    
    def get_sync_status(self) -> Dict[str, Any]:
        """Obtiene el estado de sincronización actual."""
        status = {
            "s3_connected": self.s3_client is not None,
            "bucket_name": self.bucket_name,
            "local_mlruns_exists": os.path.exists("./mlruns"),
            "local_db_exists": os.path.exists("mlflow.db"),
            "last_sync": "never",
            "total_snapshots": len(self.list_snapshots())
        }
        
        if self.s3_client:
            try:
                # Verificar última modificación en S3
                response = self.s3_client.list_objects_v2(
                    Bucket=self.bucket_name,
                    Prefix="experiments/",
                    MaxKeys=1
                )
                if response.get('Contents'):
                    status["s3_last_modified"] = response['Contents'][0]['LastModified'].isoformat()
            except:
                pass
        
        return status


def main():
    """Función principal para testing del sync manager."""
    print("MLflow S3 Sync Manager - Equipo 52")
    print("=" * 50)
    
    # Inicializar manager
    sync_manager = MLflowS3SyncManager()
    
    # Mostrar estado
    status = sync_manager.get_sync_status()
    print(f"Estado de Sincronización:")
    for key, value in status.items():
        print(f"   {key}: {value}")
    
    # Opciones interactivas
    print("Opciones disponibles:")
    print("1. Sincronizar a S3")
    print("2. Crear snapshot")
    print("3. Listar snapshots")
    print("4. Restaurar desde snapshot")
    print("5. Salir")
    
    choice = input("\\nSelecciona una opción (1-5): ")
    
    if choice == "1":
        print("Sincronizando a S3...")
        success = sync_manager.sync_artifacts_to_s3("./mlruns")
        print("Sincronización completada" if success else "❌ Error en sincronización")
        
    elif choice == "2":
        print("Creando snapshot...")
        snapshot_id = sync_manager.create_backup_snapshot()
        print(f"Snapshot creado: {snapshot_id}" if snapshot_id else "Error creando snapshot")
        
    elif choice == "3":
        print("Listando snapshots...")
        snapshots = sync_manager.list_snapshots()
        for i, snapshot in enumerate(snapshots, 1):
            print(f"   {i}. {snapshot['snapshot_id']} - {snapshot.get('timestamp', 'N/A')}")
        
    elif choice == "4":
        snapshots = sync_manager.list_snapshots()
        if snapshots:
            print("Snapshots disponibles:")
            for i, snapshot in enumerate(snapshots, 1):
                print(f"   {i}. {snapshot['snapshot_id']} - {snapshot.get('timestamp', 'N/A')}")
            
            try:
                selection = int(input("Selecciona snapshot (número): ")) - 1
                if 0 <= selection < len(snapshots):
                    snapshot_id = snapshots[selection]['snapshot_id']
                    print(f"Restaurando desde: {snapshot_id}")
                    success = sync_manager.restore_from_snapshot(snapshot_id)
                    print("Restauración completada" if success else "Error en restauración")
                else:
                    print("Selección inválida")
            except ValueError:
                print("Entrada inválida")
        else:
            print("No hay snapshots disponibles")
    
    print("¡Hasta luego!")


if __name__ == "__main__":
    main()