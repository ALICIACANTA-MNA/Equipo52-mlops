#!/usr/bin/env python3
"""
Cliente de prueba para la API de predicción de obesidad.

Script interactivo para probar la funcionalidad de la API con datos reales
y sintéticos. Incluye ejemplos de uso y casos de prueba comprehensivos.

Uso:
    python api_client.py --test-single      # Prueba predicción individual
    python api_client.py --test-batch       # Prueba predicción en lote
    python api_client.py --test-health      # Prueba health checks
    python api_client.py --stress-test      # Prueba de carga
    python api_client.py --interactive      # Modo interactivo

Referencias:
- API Documentation: http://localhost:8000/docs
- Testing Best Practices: docs/API_TESTING.md
"""

import json
import time
import argparse
import asyncio
from typing import Dict, List, Any, Optional
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import requests
    import pandas as pd
    from rich.console import Console
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.panel import Panel
    from rich.syntax import Syntax
except ImportError as e:
    print(f"Error: Dependencias faltantes: {e}")
    print("Instalar con: pip install requests pandas rich")
    exit(1)

console = Console()


class ObesityAPIClient:
    """Cliente para interactuar con la API de predicción de obesidad"""
    
    def __init__(self, base_url: str = "http://localhost:8000", api_key: Optional[str] = None):
        """
        Inicializa el cliente.
        
        Args:
            base_url: URL base de la API
            api_key: API key si es requerida
        """
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        
        if api_key:
            self.session.headers.update({"X-API-Key": api_key})
            
        self.session.headers.update({
            "Content-Type": "application/json",
            "User-Agent": "ObesityAPIClient/1.0"
        })
    
    def health_check(self) -> Dict[str, Any]:
        """Verifica el estado de salud de la API"""
        try:
            response = self.session.get(f"{self.base_url}/health")
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return {"status": "error", "error": str(e)}
    
    def detailed_health_check(self) -> Dict[str, Any]:
        """Obtiene información detallada de salud"""
        try:
            response = self.session.get(f"{self.base_url}/health/detailed")
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return {"status": "error", "error": str(e)}
    
    def predict_single(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Realiza predicción para un paciente individual.
        
        Args:
            patient_data: Datos del paciente
            
        Returns:
            Resultado de la predicción
        """
        try:
            response = self.session.post(
                f"{self.base_url}/predict",
                json=patient_data
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return {"error": str(e), "status_code": getattr(e.response, 'status_code', None)}
    
    def predict_batch(self, patients_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Realiza predicciones en lote.
        
        Args:
            patients_data: Lista de datos de pacientes
            
        Returns:
            Resultados de las predicciones
        """
        try:
            response = self.session.post(
                f"{self.base_url}/predict/batch",
                json={"patients": patients_data}
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return {"error": str(e), "status_code": getattr(e.response, 'status_code', None)}
    
    def get_model_info(self) -> Dict[str, Any]:
        """Obtiene información del modelo actual"""
        try:
            response = self.session.get(f"{self.base_url}/model/info")
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return {"error": str(e)}
    
    def get_metrics(self) -> str:
        """Obtiene métricas de Prometheus"""
        try:
            response = self.session.get(f"{self.base_url}/metrics")
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            return f"Error: {e}"


def get_sample_patient_data() -> Dict[str, Any]:
    """Genera datos de ejemplo para pruebas"""
    return {
        "Gender": "Female",
        "Age": 28.0,
        "Height": 1.65,
        "Weight": 70.0,
        "family_history_with_overweight": "yes",
        "FAVC": "no",  # Frequent consumption of high caloric food
        "FCVC": 2.0,   # Frequency of consumption of vegetables
        "NCP": 3.0,    # Number of main meals
        "CAEC": "Sometimes",  # Consumption of food between meals
        "SMOKE": "no",
        "CH2O": 2.0,   # Consumption of water daily
        "SCC": "no",   # Calories consumption monitoring
        "FAF": 1.0,    # Physical activity frequency
        "TUE": 1.0,    # Time using technology devices
        "CALC": "no",  # Consumption of alcohol
        "MTRANS": "Public_Transportation"  # Transportation used
    }


def get_test_cases() -> List[Dict[str, Any]]:
    """Genera casos de prueba diversos"""
    return [
        # Caso 1: Peso normal, estilo de vida saludable
        {
            "Gender": "Male",
            "Age": 25.0,
            "Height": 1.75,
            "Weight": 70.0,
            "family_history_with_overweight": "no",
            "FAVC": "no",
            "FCVC": 3.0,
            "NCP": 3.0,
            "CAEC": "no",
            "SMOKE": "no",
            "CH2O": 3.0,
            "SCC": "yes",
            "FAF": 2.0,
            "TUE": 0.0,
            "CALC": "no",
            "MTRANS": "Bike"
        },
        # Caso 2: Sobrepeso, factores de riesgo
        {
            "Gender": "Female",
            "Age": 45.0,
            "Height": 1.60,
            "Weight": 85.0,
            "family_history_with_overweight": "yes",
            "FAVC": "yes",
            "FCVC": 1.0,
            "NCP": 4.0,
            "CAEC": "Frequently",
            "SMOKE": "no",
            "CH2O": 1.0,
            "SCC": "no",
            "FAF": 0.0,
            "TUE": 2.0,
            "CALC": "Sometimes",
            "MTRANS": "Automobile"
        },
        # Caso 3: Obesidad severa
        {
            "Gender": "Male",
            "Age": 50.0,
            "Height": 1.70,
            "Weight": 120.0,
            "family_history_with_overweight": "yes",
            "FAVC": "yes",
            "FCVC": 0.0,
            "NCP": 4.0,
            "CAEC": "Always",
            "SMOKE": "yes",
            "CH2O": 1.0,
            "SCC": "no",
            "FAF": 0.0,
            "TUE": 2.0,
            "CALC": "Frequently",
            "MTRANS": "Automobile"
        }
    ]


def test_health_checks(client: ObesityAPIClient):
    """Prueba los health checks"""
    console.print(Panel.fit("🏥 Testing Health Checks", style="bold blue"))
    
    # Health check básico
    with console.status("[bold green]Checking basic health..."):
        health = client.health_check()
    
    if health.get("status") == "healthy":
        console.print("✅ Basic health check: PASSED", style="green")
    else:
        console.print(f"❌ Basic health check: FAILED - {health}", style="red")
        return False
    
    # Health check detallado
    with console.status("[bold green]Checking detailed health..."):
        detailed_health = client.detailed_health_check()
    
    if detailed_health.get("status") == "healthy":
        console.print("✅ Detailed health check: PASSED", style="green")
        
        # Mostrar detalles
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Component")
        table.add_column("Status")
        table.add_column("Details")
        
        for component, info in detailed_health.get("checks", {}).items():
            status = "✅ OK" if info.get("status") == "healthy" else "❌ FAIL"
            details = info.get("message", "")
            table.add_row(component, status, details)
        
        console.print(table)
    else:
        console.print(f"❌ Detailed health check: FAILED - {detailed_health}", style="red")
    
    return True


def test_single_prediction(client: ObesityAPIClient):
    """Prueba predicción individual"""
    console.print(Panel.fit("🔮 Testing Single Prediction", style="bold blue"))
    
    patient_data = get_sample_patient_data()
    
    # Mostrar datos de entrada
    console.print("📋 Patient Data:", style="bold")
    syntax = Syntax(json.dumps(patient_data, indent=2), "json", theme="monokai")
    console.print(syntax)
    
    # Realizar predicción
    with console.status("[bold green]Making prediction..."):
        result = client.predict_single(patient_data)
    
    if "error" not in result:
        console.print("✅ Single prediction: PASSED", style="green")
        
        # Mostrar resultado
        console.print("\n🎯 Prediction Result:", style="bold")
        
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Field")
        table.add_column("Value")
        
        table.add_row("Prediction", result.get("prediction", "N/A"))
        table.add_row("Confidence", f"{result.get('confidence', 0):.2%}")
        table.add_row("Model Version", result.get("model_version", "N/A"))
        table.add_row("Processing Time", f"{result.get('processing_time_ms', 0):.2f} ms")
        
        console.print(table)
        
        # Mostrar probabilidades por clase si están disponibles
        if "probabilities" in result:
            console.print("\n📊 Class Probabilities:", style="bold")
            prob_table = Table(show_header=True, header_style="bold cyan")
            prob_table.add_column("Class")
            prob_table.add_column("Probability")
            prob_table.add_column("Bar")
            
            for class_name, prob in result["probabilities"].items():
                bar = "█" * int(prob * 20)  # Barra visual
                prob_table.add_row(class_name, f"{prob:.2%}", bar)
            
            console.print(prob_table)
    else:
        console.print(f"❌ Single prediction: FAILED - {result}", style="red")
        return False
    
    return True


def test_batch_prediction(client: ObesityAPIClient):
    """Prueba predicción en lote"""
    console.print(Panel.fit("📦 Testing Batch Prediction", style="bold blue"))
    
    test_cases = get_test_cases()
    
    console.print(f"📋 Testing with {len(test_cases)} patients")
    
    # Realizar predicción en lote
    with console.status("[bold green]Making batch predictions..."):
        result = client.predict_batch(test_cases)
    
    if "error" not in result:
        console.print("✅ Batch prediction: PASSED", style="green")
        
        # Mostrar resultados
        console.print("\n🎯 Batch Results:", style="bold")
        
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Patient")
        table.add_column("Age")
        table.add_column("BMI")
        table.add_column("Prediction")
        table.add_column("Confidence")
        
        predictions = result.get("predictions", [])
        for i, (case, pred) in enumerate(zip(test_cases, predictions)):
            bmi = case["Weight"] / (case["Height"] ** 2)
            table.add_row(
                f"Patient {i+1}",
                f"{case['Age']:.0f}",
                f"{bmi:.1f}",
                pred.get("prediction", "N/A"),
                f"{pred.get('confidence', 0):.1%}"
            )
        
        console.print(table)
        
        # Estadísticas de procesamiento
        processing_time = result.get("total_processing_time_ms", 0)
        console.print(f"\n⏱️  Total processing time: {processing_time:.2f} ms")
        console.print(f"📈 Average per prediction: {processing_time/len(test_cases):.2f} ms")
    else:
        console.print(f"❌ Batch prediction: FAILED - {result}", style="red")
        return False
    
    return True


def test_model_info(client: ObesityAPIClient):
    """Prueba información del modelo"""
    console.print(Panel.fit("ℹ️  Testing Model Info", style="bold blue"))
    
    with console.status("[bold green]Getting model info..."):
        info = client.get_model_info()
    
    if "error" not in info:
        console.print("✅ Model info: PASSED", style="green")
        
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Property")
        table.add_column("Value")
        
        for key, value in info.items():
            if isinstance(value, dict):
                value = json.dumps(value, indent=2)
            table.add_row(key, str(value))
        
        console.print(table)
    else:
        console.print(f"❌ Model info: FAILED - {info}", style="red")
        return False
    
    return True


def stress_test(client: ObesityAPIClient, num_requests: int = 50):
    """Prueba de carga básica"""
    console.print(Panel.fit(f"🚀 Stress Test ({num_requests} requests)", style="bold red"))
    
    patient_data = get_sample_patient_data()
    
    def make_single_request():
        start_time = time.time()
        result = client.predict_single(patient_data)
        end_time = time.time()
        
        return {
            "success": "error" not in result,
            "response_time": (end_time - start_time) * 1000,  # ms
            "result": result
        }
    
    results = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Making requests...", total=num_requests)
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_single_request) for _ in range(num_requests)]
            
            for future in as_completed(futures):
                results.append(future.result())
                progress.advance(task)
    
    # Analizar resultados
    successful = sum(1 for r in results if r["success"])
    failed = num_requests - successful
    response_times = [r["response_time"] for r in results if r["success"]]
    
    # Mostrar estadísticas
    console.print("\n📊 Stress Test Results:", style="bold")
    
    stats_table = Table(show_header=True, header_style="bold green")
    stats_table.add_column("Metric")
    stats_table.add_column("Value")
    
    stats_table.add_row("Total Requests", str(num_requests))
    stats_table.add_row("Successful", f"{successful} ({successful/num_requests:.1%})")
    stats_table.add_row("Failed", f"{failed} ({failed/num_requests:.1%})")
    
    if response_times:
        stats_table.add_row("Avg Response Time", f"{sum(response_times)/len(response_times):.1f} ms")
        stats_table.add_row("Min Response Time", f"{min(response_times):.1f} ms")
        stats_table.add_row("Max Response Time", f"{max(response_times):.1f} ms")
        stats_table.add_row("95th Percentile", f"{sorted(response_times)[int(len(response_times)*0.95)]:.1f} ms")
    
    console.print(stats_table)
    
    # Mostrar errores si los hay
    if failed > 0:
        console.print(f"\n❌ {failed} requests failed", style="red")
        error_samples = [r["result"] for r in results if not r["success"]][:3]
        for i, error in enumerate(error_samples):
            console.print(f"Error {i+1}: {error}")


def interactive_mode(client: ObesityAPIClient):
    """Modo interactivo para pruebas manuales"""
    console.print(Panel.fit("🎮 Interactive Mode", style="bold cyan"))
    console.print("Enter patient data interactively. Press Ctrl+C to exit.")
    
    try:
        while True:
            console.print("\n" + "="*50)
            console.print("Enter patient information:")
            
            # Recolectar datos interactivamente
            patient_data = {}
            
            # Datos básicos
            patient_data["Gender"] = console.input("Gender [Male/Female]: ").strip()
            patient_data["Age"] = float(console.input("Age: "))
            patient_data["Height"] = float(console.input("Height (m): "))
            patient_data["Weight"] = float(console.input("Weight (kg): "))
            
            # Usar valores por defecto para el resto
            defaults = get_sample_patient_data()
            for key, value in defaults.items():
                if key not in patient_data:
                    patient_data[key] = value
            
            # Hacer predicción
            with console.status("[bold green]Making prediction..."):
                result = client.predict_single(patient_data)
            
            # Mostrar resultado
            if "error" not in result:
                console.print(f"\n🎯 Prediction: {result.get('prediction')}", style="bold green")
                console.print(f"🎯 Confidence: {result.get('confidence', 0):.1%}", style="bold green")
                
                bmi = patient_data["Weight"] / (patient_data["Height"] ** 2)
                console.print(f"📊 BMI: {bmi:.1f}", style="blue")
            else:
                console.print(f"❌ Error: {result}", style="red")
    
    except KeyboardInterrupt:
        console.print("\n👋 Goodbye!")


def main():
    """Función principal"""
    parser = argparse.ArgumentParser(description="Obesity API Client")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--api-key", help="API key if required")
    parser.add_argument("--test-health", action="store_true", help="Test health checks")
    parser.add_argument("--test-single", action="store_true", help="Test single prediction")
    parser.add_argument("--test-batch", action="store_true", help="Test batch prediction")
    parser.add_argument("--test-model-info", action="store_true", help="Test model info")
    parser.add_argument("--stress-test", type=int, metavar="N", help="Run stress test with N requests")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    
    args = parser.parse_args()
    
    # Crear cliente
    client = ObesityAPIClient(args.url, args.api_key)
    
    console.print(Panel.fit(f"🧪 Obesity API Client\n🌐 Connecting to: {args.url}", style="bold magenta"))
    
    # Ejecutar pruebas según argumentos
    if args.all or args.test_health:
        test_health_checks(client)
    
    if args.all or args.test_single:
        test_single_prediction(client)
    
    if args.all or args.test_batch:
        test_batch_prediction(client)
    
    if args.all or args.test_model_info:
        test_model_info(client)
    
    if args.stress_test:
        stress_test(client, args.stress_test)
    
    if args.interactive:
        interactive_mode(client)
    
    # Si no se especificó nada, mostrar ayuda
    if not any([args.test_health, args.test_single, args.test_batch, 
               args.test_model_info, args.stress_test, args.interactive, args.all]):
        console.print("ℹ️  No tests specified. Use --help for options or --all to run all tests.")


if __name__ == "__main__":
    main()