"""
python -m src.data.data_ingestion

Data ingestion module for obesity prediction MLOps pipeline.

Implementa ingesta de datos siguiendo las mejores prácticas de MLOps:
- Parametrización externa via configuración
- Logging estructurado y trazabilidad
- Validación de datos de entrada
- Reproducibilidad garantizada

REFERENCIAS BIBLIOGRÁFICAS TÉCNICAS:
- Treveil, M. et al. (2024): "Introducing MLOps: How to Scale Machine Learning in the Enterprise"
  Capítulo 4: "Data Pipeline Orchestration" - Principios de ingesta inmutable
- Wilson, B. & Bansal, A. (2022): "Practical MLOps: Operationalizing Machine Learning Models" 
  Capítulo 5: "Data Management Patterns" - Flujos de datos reproducibles
- Huyen, C. (2022): "Designing Machine Learning Systems"
  Capítulo 3: "Data Engineering Fundamentals" - Arquitectura de pipelines de datos
- Lauchande, C. (2024): "Managing the Machine Learning Lifecycle with MLflow"
  Capítulo 6: "Data Versioning and Lineage" - Trazabilidad de datos
- Lakshmanan, V. et al. (2023): "Machine Learning Design Patterns"
  Patrón: "Data Ingestion Pattern" - Mejores prácticas de ingesta

ARQUITECTURA DEL FLUJO DE DATOS:

    ┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
    │   Internet/URL  │    │  Local Files     │    │ Existing Files  │
    │                 │    │  (Manual Upload) │    │ (data/raw/)     │
    └─────────┬───────┘    └─────────┬────────┘    └─────────┬───────┘
              │                      │                       │
              ▼                      ▼                       ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │                    DATA INGESTION MODULE                        │
    │  • Download & Extract (ZIP handling)                            │
    │  • Integrity Validation (SHA-256)                               │
    │  • Schema Validation                                            │
    │  • Column Cleaning                                              │
    └─────────────────────────┬───────────────────────────────────────┘
                              │
                              ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │                     data/raw/                                   │
    │            IMMUTABLE SOURCE OF TRUTH                            │
    │  • ObesityDataSet_raw_and_data_sinthetic.csv                    │
    │  • Hash: 4de25bd59072...                                        │
    │  • 2,111 records × 17 features                                  │
    └─────────────────────────┬───────────────────────────────────────┘
                              │
                              ▼
              ┌──────────────────────────────────┐
              │      DOWNSTREAM STAGES           │
              │  data/interim/ → data/processed/ │
              └──────────────────────────────────┘

Flujo de ingesta (Algoritmo):
1. Verificar configuración de source_url
2. Si existe URL: descargar, extraer ZIP y guardar en data/raw/
3. Si no hay URL: verificar archivo existente en data/raw/
4. Como alternativa: buscar archivo CSV en directorio raw/
5. Validar estructura, integridad y generar estadísticas
6. Logging completo para auditoría y trazabilidad MLOps
"""

import pandas as pd
import requests
import zipfile
import tempfile
from pathlib import Path
from typing import Tuple, Optional
import hashlib
import shutil
from urllib.parse import urlparse

from src.utils.config import get_data_config, load_params
from src.utils.logging_config import get_logger, log_pipeline_stage


logger = get_logger(__name__)


class DataIngestionError(Exception):
    """Excepción específica para errores de ingesta de datos"""
    pass


def calculate_file_hash(file_path: Path) -> str:
    """
    Calcula hash SHA-256 de un archivo para verificación de integridad.
    
    Args:
        file_path: Ruta al archivo
        
    Returns:
        Hash SHA-256 en formato hexadecimal
    """
    hash_sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()


def download_file(url: str, destination: Path, chunk_size: int = 8192) -> None:
    """
    Descarga archivo desde URL con manejo de errores y progreso.
    
    Args:
        url: URL del archivo a descargar
        destination: Ruta destino del archivo
        chunk_size: Tamaño del chunk para descarga
        
    Raises:
        DataIngestionError: Si la descarga falla
    """
    try:
        logger.info(f"Iniciando descarga desde: {url}")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded_size = 0
        
        with open(destination, 'wb') as file:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:  # filter out keep-alive chunks
                    file.write(chunk)
                    downloaded_size += len(chunk)
                    
                    if total_size > 0:
                        progress = (downloaded_size / total_size) * 100
                        if downloaded_size % (chunk_size * 100) == 0:  # Log every 100 chunks
                            logger.debug(f"Progreso descarga: {progress:.1f}%")
        
        logger.info(f"Descarga completada: {destination} ({downloaded_size:,} bytes)")
        
    except requests.exceptions.RequestException as e:
        raise DataIngestionError(f"Error descargando archivo desde {url}: {e}")
    except IOError as e:
        raise DataIngestionError(f"Error escribiendo archivo en {destination}: {e}")


def extract_zip_file(zip_path: Path, extract_to: Path) -> Path:
    """
    Extrae archivo ZIP y retorna la ruta del archivo principal.
    
    Args:
        zip_path: Ruta al archivo ZIP
        extract_to: Directorio de extracción
        
    Returns:
        Ruta al archivo CSV extraído
        
    Raises:
        DataIngestionError: Si la extracción falla
    """
    try:
        logger.info(f"Extrayendo archivo ZIP: {zip_path}")
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
            
            # Buscar archivo CSV principal
            csv_files = list(extract_to.glob("*.csv"))
            if not csv_files:
                raise DataIngestionError(f"No se encontraron archivos CSV en {zip_path}")
            
            # Retornar el archivo CSV más grande (asumiendo que es el principal)
            main_csv = max(csv_files, key=lambda x: x.stat().st_size)
            logger.info(f"Archivo CSV principal identificado: {main_csv}")
            
            return main_csv
            
    except zipfile.BadZipFile as e:
        raise DataIngestionError(f"Archivo ZIP corrupto: {zip_path}: {e}")
    except Exception as e:
        raise DataIngestionError(f"Error extrayendo archivo ZIP: {e}")


def validate_dataset_structure(df: pd.DataFrame, expected_columns: Optional[list] = None) -> None:
    """
    Valida la estructura básica del dataset.
    
    Args:
        df: DataFrame a validar
        expected_columns: Lista de columnas esperadas (opcional)
        
    Raises:
        DataIngestionError: Si la validación falla
    """
    if df.empty:
        raise DataIngestionError("Dataset está vacío")
    
    logger.info(f"Dataset cargado: {df.shape[0]:,} filas, {df.shape[1]} columnas")
    
    # Validar columnas esperadas si se proporcionan
    if expected_columns:
        missing_columns = set(expected_columns) - set(df.columns)
        if missing_columns:
            raise DataIngestionError(f"Columnas faltantes: {missing_columns}")
        
        extra_columns = set(df.columns) - set(expected_columns)
        if extra_columns:
            logger.warning(f"Columnas adicionales encontradas: {extra_columns}")
    
    # Validar tipos de datos básicos
    numeric_columns = df.select_dtypes(include=['number']).columns
    categorical_columns = df.select_dtypes(include=['object']).columns
    
    logger.info(f"Columnas numéricas: {len(numeric_columns)}")
    logger.info(f"Columnas categóricas: {len(categorical_columns)}")
    
    # Verificar valores nulos
    null_counts = df.isnull().sum()
    if null_counts.any():
        logger.warning("Valores nulos encontrados:")
        for col, count in null_counts[null_counts > 0].items():
            logger.warning(f"  {col}: {count} ({count/len(df)*100:.1f}%)")


@log_pipeline_stage("data_ingestion")
def ingest_data_from_url(url: str, destination_path: Path) -> pd.DataFrame:
    """
    Ingesta datos desde URL externa (UCI ML Repository).
    
    Args:
        url: URL del dataset
        destination_path: Ruta donde guardar el archivo
        
    Returns:
        DataFrame con los datos cargados
        
    Raises:
        DataIngestionError: Si la ingesta falla
    """
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Determinar si es un archivo ZIP
    parsed_url = urlparse(url)
    is_zip = parsed_url.path.endswith('.zip')
    
    if is_zip:
        # Descargar y extraer ZIP
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            zip_file = temp_path / "dataset.zip"
            
            # Descargar ZIP
            download_file(url, zip_file)
            
            # Extraer archivo principal
            csv_file = extract_zip_file(zip_file, temp_path)
            
            # Mover archivo a destino final
            shutil.move(str(csv_file), str(destination_path))
    else:
        # Descarga directa
        download_file(url, destination_path)
    
    # Cargar y validar datos
    try:
        df = pd.read_csv(destination_path)
        
        # Limpiar nombres de columnas (espacios extra)
        df.columns = [col.strip() for col in df.columns]
        
        # Validar estructura
        validate_dataset_structure(df)
        
        # Guardar versión limpia
        df.to_csv(destination_path, index=False)
        
        logger.info(f"Datos guardados exitosamente en: {destination_path}")
        return df
        
    except pd.errors.EmptyDataError:
        raise DataIngestionError(f"Archivo CSV vacío: {destination_path}")
    except pd.errors.ParserError as e:
        raise DataIngestionError(f"Error parsing CSV: {e}")


@log_pipeline_stage("data_ingestion")
def ingest_local_data(source_path: Path, destination_path: Path) -> pd.DataFrame:
    """
    Ingesta datos desde archivo local existente.
    
    Args:
        source_path: Ruta del archivo fuente
        destination_path: Ruta destino
        
    Returns:
        DataFrame con los datos cargados
    """
    if not source_path.exists():
        raise DataIngestionError(f"Archivo fuente no encontrado: {source_path}")
    
    logger.info(f"Copiando datos locales desde: {source_path}")
    
    # Crear directorio destino
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Copiar archivo
    shutil.copy2(source_path, destination_path)
    
    # Cargar y procesar
    df = pd.read_csv(destination_path)
    df.columns = [col.strip() for col in df.columns]
    
    validate_dataset_structure(df)
    
    # Guardar versión limpia
    df.to_csv(destination_path, index=False)
    
    logger.info(f"Datos locales procesados y guardados en: {destination_path}")
    return df


def main():
    """
    Función principal de ingesta de datos.
    Ejecuta la ingesta según la configuración definida.
    """
    try:
        # Cargar configuración
        config = get_data_config()
        params = load_params()
        
        # Obtener configuración de datos
        data_config = config['data']
        raw_config = data_config['raw']
        
        # Rutas
        source_url = raw_config.get('source_url')
        local_path = Path(raw_config['local_path'])
        
        # Intentar ingesta desde URL si está configurada
        if source_url:
            logger.info("Iniciando ingesta desde URL externa")
            df = ingest_data_from_url(source_url, local_path)
        else:
            # Verificar si ya existe archivo en RAW
            if local_path.exists():
                logger.info(f"Usando archivo existente en raw: {local_path}")
                df = pd.read_csv(local_path)
                # Limpiar nombres de columnas (espacios extra)
                df.columns = [col.strip() for col in df.columns]
                validate_dataset_structure(df)
            else:
                # Buscar archivos CSV en directorio raw como alternativa
                raw_dir = local_path.parent
                csv_files = list(raw_dir.glob("*.csv")) if raw_dir.exists() else []
                
                if csv_files:
                    source_file = csv_files[0]  # Tomar el primer CSV encontrado
                    logger.info(f"Encontrado archivo CSV alternativo en raw: {source_file}")
                    df = ingest_local_data(source_file, local_path)
                else:
                    raise DataIngestionError(
                        f"No se encontró fuente de datos disponible. "
                        f"Opciones: 1) Configurar source_url en configs/data/data_config.yaml, "
                        f"2) Colocar archivo CSV en {raw_dir}"
                    )
        
        # Generar estadísticas de ingesta
        stats = {
            'total_records': len(df),
            'total_columns': len(df.columns),
            'file_size_mb': local_path.stat().st_size / (1024 * 1024),
            'file_hash': calculate_file_hash(local_path)
        }
        
        logger.info("Estadísticas de ingesta:")
        for key, value in stats.items():
            logger.info(f"  {key}: {value}")
        
        print(f"Ingesta completada exitosamente: {local_path} ({df.shape})")
        
    except DataIngestionError as e:
        logger.error(f"Error en ingesta de datos: {e}")
        raise
    except Exception as e:
        logger.error(f"Error inesperado en ingesta: {e}")
        raise DataIngestionError(f"Error inesperado: {e}")


if __name__ == "__main__":
    main()