#!/usr/bin/env python3
"""
Script de prueba para verificar la configuración Docker simplificada.
"""

import requests
import json
import time

def test_api():
    """Prueba básica de la API"""
    base_url = "http://localhost:8080"
    
    print("Iniciando pruebas de la API...")
    
    # Test 1: Health check
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        print(f"Health check: {response.status_code} - {response.json()}")
    except Exception as e:
        print(f" Health check falló: {e}")
        return False
    
    # Test 2: Root endpoint
    try:
        response = requests.get(f"{base_url}/", timeout=5)
        print(f"Root endpoint: {response.status_code} - {response.json()}")
    except Exception as e:
        print(f"Root endpoint falló: {e}")
    
    # Test 3: Model info
    try:
        response = requests.get(f"{base_url}/model/info", timeout=5)
        print(f"Model info: {response.status_code} - {response.json()}")
    except Exception as e:
        print(f"Model info falló: {e}")
    
    # Test 4: Prediction endpoint
    sample_data = {
        "Age": 25,
        "Gender": "Male",
        "Height": 1.75,
        "Weight": 70,
        "family_history_with_overweight": "no",
        "FAVC": "no",
        "FCVC": 2.0,
        "NCP": 3.0,
        "CAEC": "Sometimes",
        "SMOKE": "no",
        "CH2O": 2.0,
        "SCC": "no",
        "FAF": 1.0,
        "TUE": 1.0,
        "CALC": "no",
        "MTRANS": "Public_Transportation"
    }
    
    try:
        response = requests.post(
            f"{base_url}/predict", 
            json=sample_data, 
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        print(f"✅ Prediction: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Predicción: {result.get('prediction')}")
            print(f"BMI: {result.get('bmi')}")
            print(f"Tiempo: {result.get('processing_time_ms')}ms")
        else:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"Prediction falló: {e}")
    
    print("\n Pruebas completadas")
    return True

if __name__ == "__main__":
    print("Script de prueba para configuración Docker simplificada")
    print("Instrucciones:")
    print("1. docker-compose up -d")
    print("2. python test_docker_api.py")
    print()
    
    # Esperar un poco para que la API esté lista
    print("Esperando que la API esté lista...")
    time.sleep(2)
    
    test_api()