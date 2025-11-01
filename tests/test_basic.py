"""
Test simple para verificar configuración básica.
"""

import pytest
import sys
from pathlib import Path

# Agregar src al path
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

def test_python_version():
    """Test que verifica la versión de Python."""
    assert sys.version_info >= (3, 8), "Python 3.8+ required"

def test_basic_imports():
    """Test que verifica imports básicos."""
    import pandas
    import numpy
    import pytest
    
    assert pandas.__version__ is not None
    assert numpy.__version__ is not None
    assert pytest.__version__ is not None

def test_project_structure():
    """Test que verifica estructura del proyecto."""
    project_root = Path(__file__).parent.parent
    
    # Verificar directorios principales
    assert (project_root / "src").exists()
    assert (project_root / "tests").exists()
    assert (project_root / "configs").exists()
    
def test_sample_data_exists():
    """Test que verifica que existen datos de muestra."""
    project_root = Path(__file__).parent.parent
    sample_file = project_root / "tests" / "fixtures" / "data" / "sample_raw.csv"
    
    assert sample_file.exists(), "Sample data file should exist"

@pytest.mark.smoke
def test_basic_functionality():
    """Test de funcionalidad básica - marcado como smoke test."""
    import pandas as pd
    import numpy as np
    
    # Test básico de pandas
    df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    assert len(df) == 3
    assert list(df.columns) == ['a', 'b']
    
    # Test básico de numpy
    arr = np.array([1, 2, 3])
    assert arr.sum() == 6