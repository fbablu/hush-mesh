#!/usr/bin/env python3
"""
Comprehensive ML Integration Verification Script
This script tests if the ML model is properly connected to the frontend
"""

import requests
import json
import time
from datetime import datetime

# Test endpoints
ML_SERVER_URL = "http://localhost:9000"
PROXY_SERVER_URL = "http://localhost:8082"

def print_header(text):
    """Print a formatted header"""
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60)

def test_ml_server_health():
    """Test if ML server is running"""
    print_header("1. Testing ML Server (localhost:9000)")
    try:
        response = requests.get(f"{ML_SERVER_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ ML Server is ONLINE")
            print(f"   - Status: {data.get('status')}")
            print(f"   - PyTorch Available: {data.get('torch_available')}")
            print(f"   - NumPy Available: {data.get('numpy_available')}")
            return True
        else:
            print(f"❌ ML Server returned status code: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ ML Server is NOT responding: {e}")
        return False

def test_ml_prediction():
    """Test ML prediction endpoint directly"""
    print_header("2. Testing ML Prediction (Direct to localhost:9000)")
    
    # Sample sensor data
    sample_data = {
        "sensor_data": {
            "convoy_data": {
                "speed_knots": 12.0,
                "vessel_count": 4,
                "position": {"lat": 25.7617, "lon": -80.1918}
            },
            "environmental_conditions": {
                "sea_state": 2,
                "visibility_km": 15.0,
                "wind_speed_knots": 10.0,
                "wave_height_m": 2.0
            },
            "data_quality": {
                "sensor_reliability": 0.9
            },
            "drone_array": [{
                "position": {"altitude_m": 180.0},
                "sensor_suite": {
                    "electro_optical": {
                        "visible_spectrum": {"objects_detected": 1},
                        "infrared": {"thermal_signatures": 1}
                    },
                    "radar": {"contacts": 1},
                    "acoustic": {
                        "hydrophone_data": {"ambient_noise_db": 95.0}
                    },
                    "electronic_warfare": {
                        "rf_spectrum": {"signals_detected": 3}
                    }
                }
            }],
            "threat_detections": [],
            "ground_truth": {"threat_present": False}
        }
    }
    
    try:
        response = requests.post(
            f"{ML_SERVER_URL}/predict",
            json=sample_data,
            timeout=5
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ ML Prediction SUCCESSFUL")
            print(f"   - Threat Detected: {result.get('threat_detected')}")
            print(f"   - Threat Type: {result.get('threat_type')}")
            print(f"   - Confidence: {result.get('confidence', 0):.2%}")
            print(f"   - Top 3 Probabilities:")
            
            all_probs = result.get('all_probabilities', {})
            sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)[:3]
            for threat_class, prob in sorted_probs:
                print(f"      - {threat_class}: {prob:.2%}")
            return True
        else:
            print(f"❌ ML Prediction failed with status: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ ML Prediction request failed: {e}")
        return False

def test_proxy_server():
    """Test if proxy server is running and forwarding"""
    print_header("3. Testing Proxy Server (localhost:8082)")
    
    try:
        # Test proxy health
        response = requests.get(f"{PROXY_SERVER_URL}/", timeout=5)
        if response.status_code == 200:
            print(f"✅ Proxy Server is ONLINE")
            
            # Test if proxy forwards to ML server
            print(f"\n   Testing ML forwarding through proxy...")
            
            sample_data = {
                "sensor_data": {
                    "convoy_data": {"speed_knots": 12.0, "vessel_count": 4},
                    "environmental_conditions": {
                        "sea_state": 2, "visibility_km": 15.0,
                        "wind_speed_knots": 10.0, "wave_height_m": 2.0
                    },
                    "data_quality": {"sensor_reliability": 0.9},
                    "drone_array": [{
                        "position": {"altitude_m": 180.0},
                        "sensor_suite": {
                            "electro_optical": {
                                "visible_spectrum": {"objects_detected": 1},
                                "infrared": {"thermal_signatures": 1}
                            },
                            "radar": {"contacts": 1},
                            "acoustic": {"hydrophone_data": {"ambient_noise_db": 95.0}},
                            "electronic_warfare": {"rf_spectrum": {"signals_detected": 3}}
                        }
                    }],
                    "threat_detections": [],
                    "ground_truth": {"threat_present": False}
                }
            }
            
            response = requests.post(
                f"{PROXY_SERVER_URL}/api/predict",
                json=sample_data,
                timeout=5
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Proxy forwarding WORKS")
                print(f"   - Prediction received: {result.get('threat_type')}")
                return True
            else:
                print(f"❌ Proxy forwarding failed: {response.status_code}")
                return False
        else:
            print(f"❌ Proxy Server returned status: {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Proxy Server is NOT responding: {e}")
        return False

def check_frontend_access():
    """Check if frontend is accessible"""
    print_header("4. Testing Frontend Access")
    
    frontend_urls = [
        "https://d3pka9yj6j75yn.cloudfront.net/ports/8082/enhanced_multi_route.html",
        "https://d3pka9yj6j75yn.cloudfront.net/ports/8082/test_ml.html"
    ]
    
    for url in frontend_urls:
        filename = url.split('/')[-1]
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                print(f"✅ {filename} is accessible")
            else:
                print(f"⚠️  {filename} returned status: {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"❌ {filename} is NOT accessible: {e}")

def generate_summary(results):
    """Generate a summary of test results"""
    print_header("VERIFICATION SUMMARY")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("✅ ALL TESTS PASSED - ML Integration is WORKING!")
        print("\nWhat this means:")
        print("- ML server is running and responding")
        print("- ML model can make predictions")
        print("- Proxy server is forwarding requests correctly")
        print("- Frontend should be able to connect to ML model")
        
        print("\n📋 To verify in the browser:")
        print("1. Open: https://d3pka9yj6j75yn.cloudfront.net/ports/8082/enhanced_multi_route.html")
        print("2. Click '▶️ Start Demo'")
        print("3. Open browser console (F12)")
        print("4. Look for '🤖 ML Request #' messages")
        print("5. Check '🔍 ML Detections' panel for real-time predictions")
        
    else:
        print("⚠️  SOME TESTS FAILED")
        print("\nFailed components:")
        for test_name, passed in results.items():
            if not passed:
                print(f"  ❌ {test_name}")
        
        print("\n🔧 Troubleshooting:")
        if not results.get("ML Server"):
            print("- Start ML server: python3 ml_api_server.py")
        if not results.get("Proxy Server"):
            print("- Start proxy: python3 proxy_server.py")
    
    return all_passed

def main():
    """Run all verification tests"""
    print("\n" + "🔍 ML INTEGRATION VERIFICATION TOOL")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = {
        "ML Server": test_ml_server_health(),
        "ML Prediction": test_ml_prediction(),
        "Proxy Server": test_proxy_server()
    }
    
    time.sleep(0.5)
    check_frontend_access()
    
    time.sleep(0.5)
    all_passed = generate_summary(results)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    exit(main())