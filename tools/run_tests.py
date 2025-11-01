#!/usr/bin/env python3
"""
Script principal para ejecutar todos los tests del proyecto MLOps.
Incluye configuración personalizada y reporting avanzado.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
import json
import time
from datetime import datetime

# Agregar src al path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

def setup_environment():
    """Configurar variables de entorno para testing."""
    os.environ["ENVIRONMENT"] = "test"
    os.environ["PYTHONPATH"] = str(project_root / "src")
    os.environ["LOG_LEVEL"] = "WARNING"
    
    # MLflow para testing
    mlflow_test_dir = project_root / "tests" / "temp" / "mlflow"
    mlflow_test_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MLFLOW_TRACKING_URI"] = f"file://{mlflow_test_dir}"
    
    print("Environment configured for testing")

def create_test_directories():
    """Crear directorios necesarios para testing."""
    directories = [
        "tests/reports",
        "tests/reports/coverage", 
        "tests/temp",
        "tests/fixtures/data"
    ]
    
    for dir_path in directories:
        full_path = project_root / dir_path
        full_path.mkdir(parents=True, exist_ok=True)
    
    print("Test directories created")

def run_pytest(test_type="all", verbose=True, coverage=True, parallel=False):
    """Ejecutar pytest con configuración específica."""
    
    # Comando base de pytest
    cmd = ["python", "-m", "pytest"]
    
    # Configurar argumentos según tipo de test
    if test_type == "unit":
        cmd.extend(["tests/unit", "-m", "unit"])
    elif test_type == "integration":
        cmd.extend(["tests/integration", "-m", "integration"])
    elif test_type == "api":
        cmd.extend(["tests/api", "-m", "api"])
    elif test_type == "data":
        cmd.extend(["tests/data", "-m", "data"])
    elif test_type == "smoke":
        cmd.extend(["-m", "smoke"])
    elif test_type == "fast":
        cmd.extend(["-m", "not slow"])
    else:
        cmd.append("tests/")
    
    # Configuración de verbosidad
    if verbose:
        cmd.append("-v")
    else:
        cmd.append("-q")
    
    # Configuración de coverage
    if coverage:
        cmd.extend([
            "--cov=src",
            "--cov-report=term-missing",
            "--cov-report=html:tests/reports/coverage",
            "--cov-branch",
            "--cov-fail-under=70"
        ])
    
    # Ejecutión en paralelo
    if parallel:
        cmd.extend(["-n", "auto"])
    
    # Configuración adicional
    cmd.extend([
        "--tb=short",
        "--strict-markers",
        "--junitxml=tests/reports/junit.xml",
        "--html=tests/reports/report.html",
        "--self-contained-html"
    ])
    
    print(f"TEST: Running tests: {' '.join(cmd)}")
    
    start_time = time.time()
    result = subprocess.run(cmd, cwd=project_root, capture_output=False)
    end_time = time.time()
    
    execution_time = end_time - start_time
    print(f"TIMER:  Test execution time: {execution_time:.2f} seconds")
    
    return result.returncode

def run_linting():
    """Ejecutar linting y verificaciones de código."""
    print("🧹 Running code quality checks...")
    
    commands = [
        # Flake8 (si está disponible)
        ["python", "-m", "flake8", "src/", "--max-line-length=100", "--ignore=E203,W503"],
        
        # Black check (si está disponible)
        ["python", "-m", "black", "--check", "src/"],
        
        # isort check (si está disponible)
        ["python", "-m", "isort", "--check-only", "src/"]
    ]
    
    results = []
    for cmd in commands:
        try:
            result = subprocess.run(cmd, cwd=project_root, capture_output=True, text=True)
            results.append((cmd[2], result.returncode == 0))
            
            if result.returncode != 0:
                print(f"{cmd[2]} failed:")
                print(result.stdout)
                print(result.stderr)
            else:
                print(f"{cmd[2]} passed")
                
        except FileNotFoundError:
            print(f"WARNING:  {cmd[2]} not available, skipping...")
            results.append((cmd[2], None))
    
    return results

def run_security_checks():
    """Ejecutar verificaciones de seguridad."""
    print("LOCK: Running security checks...")
    
    security_commands = [
        # Safety check (si está disponible)
        ["python", "-m", "safety", "check"],
        
        # Bandit check (si está disponible)  
        ["python", "-m", "bandit", "-r", "src/", "-f", "json"]
    ]
    
    for cmd in security_commands:
        try:
            result = subprocess.run(cmd, cwd=project_root, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"{cmd[2]} - No security issues found")
            else:
                print(f"WARNING:  {cmd[2]} found potential issues:")
                print(result.stdout[:500])  # Mostrar primeros 500 chars
                
        except FileNotFoundError:
            print(f"WARNING:  {cmd[2]} not available, skipping...")

def generate_test_report():
    """Generar reporte de testing."""
    report_data = {
        "timestamp": datetime.now().isoformat(),
        "project": "MLOps Obesity Prediction",
        "test_environment": os.environ.get("ENVIRONMENT", "unknown"),
        "python_version": sys.version,
        "reports_generated": []
    }
    
    # Verificar qué reportes se generaron
    reports_dir = project_root / "tests" / "reports"
    
    if (reports_dir / "junit.xml").exists():
        report_data["reports_generated"].append("junit.xml")
    
    if (reports_dir / "coverage").exists():
        report_data["reports_generated"].append("coverage/")
    
    if (reports_dir / "report.html").exists():
        report_data["reports_generated"].append("report.html")
    
    # Guardar reporte metadata
    with open(reports_dir / "test_summary.json", "w") as f:
        json.dump(report_data, f, indent=2)
    
    print(f"Test report generated: {reports_dir / 'test_summary.json'}")
    
    # Mostrar ubicaciones de reportes
    print("\nFOLDER: Generated Reports:")
    for report in report_data["reports_generated"]:
        print(f"   • {reports_dir / report}")

def main():
    """Función principal."""
    parser = argparse.ArgumentParser(description="Run MLOps project tests")
    
    parser.add_argument(
        "--type", 
        choices=["all", "unit", "integration", "api", "data", "smoke", "fast"],
        default="all",
        help="Type of tests to run"
    )
    
    parser.add_argument(
        "--no-coverage",
        action="store_true",
        help="Skip coverage reporting"
    )
    
    parser.add_argument(
        "--parallel",
        action="store_true", 
        help="Run tests in parallel"
    )
    
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick tests only (excludes slow tests)"
    )
    
    parser.add_argument(
        "--lint",
        action="store_true",
        help="Run linting checks"  
    )
    
    parser.add_argument(
        "--security",
        action="store_true",
        help="Run security checks"
    )
    
    parser.add_argument(
        "--no-setup",
        action="store_true",
        help="Skip environment setup"
    )
    
    args = parser.parse_args()
    
    print("MLOps Testing Suite")
    print("=" * 50)
    
    # Setup
    if not args.no_setup:
        setup_environment()
        create_test_directories()
    
    # Determinar tipo de test
    test_type = "fast" if args.quick else args.type
    
    # Ejecutar linting si se solicita
    if args.lint:
        lint_results = run_linting()
        print()
    
    # Ejecutar security checks si se solicita
    if args.security:
        run_security_checks()
        print()
    
    # Ejecutar tests
    print(f"TEST: Running {test_type} tests...")
    exit_code = run_pytest(
        test_type=test_type,
        coverage=not args.no_coverage,
        parallel=args.parallel
    )
    
    # Generar reportes
    generate_test_report()
    
    # Mostrar resultado final
    print("\n" + "=" * 50)
    if exit_code == 0:
        print("All tests passed!")
    else:
        print("Some tests failed!")
        
        # Mostrar ayuda para debugging
        print("\nINFO: Debugging tips:")
        print("   • Check test output above for specific failures")
        print("   • Run with --type=unit to test individual components")
        print("   • Use --no-coverage for faster execution")
        print("   • Check generated reports in tests/reports/")
    
    print(f"\nTest reports available at: {project_root / 'tests' / 'reports'}")
    
    return exit_code

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)