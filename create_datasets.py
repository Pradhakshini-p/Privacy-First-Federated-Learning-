#!/usr/bin/env python3
"""
Create Comprehensive Datasets for Federated Learning Analysis
Includes healthcare, financial, and IoT datasets for demonstration
"""

import pandas as pd
import numpy as np
import os
from sklearn.datasets import make_classification, make_regression
from sklearn.preprocessing import StandardScaler
import json
from datetime import datetime

def create_healthcare_dataset():
    """Create a synthetic healthcare dataset for disease prediction"""
    print("🏥 Creating Healthcare Dataset...")
    
    # Generate synthetic patient data
    n_samples = 10000
    np.random.seed(42)
    
    # Patient demographics
    age = np.random.normal(50, 15, n_samples).astype(int)
    age = np.clip(age, 18, 90)
    
    # Medical measurements
    blood_pressure = np.random.normal(120, 20, n_samples)
    cholesterol = np.random.normal(200, 40, n_samples)
    glucose = np.random.normal(100, 25, n_samples)
    bmi = np.random.normal(27, 5, n_samples)
    
    # Lifestyle factors
    smoking = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    exercise_hours = np.random.exponential(2, n_samples)
    alcohol_consumption = np.random.choice([0, 1, 2], n_samples, p=[0.5, 0.3, 0.2])
    
    # Medical history
    family_history = np.random.choice([0, 1], n_samples, p=[0.6, 0.4])
    previous_conditions = np.random.choice([0, 1, 2], n_samples, p=[0.7, 0.2, 0.1])
    
    # Create target variable (disease risk)
    risk_score = (
        (age > 60) * 0.3 +
        (blood_pressure > 140) * 0.25 +
        (cholesterol > 240) * 0.2 +
        (glucose > 125) * 0.15 +
        (bmi > 30) * 0.1 +
        smoking * 0.2 +
        (exercise_hours < 1) * 0.15 +
        (alcohol_consumption > 1) * 0.1 +
        family_history * 0.25 +
        (previous_conditions > 0) * 0.3
    )
    
    # Add noise and create binary target
    risk_score += np.random.normal(0, 0.1, n_samples)
    disease_risk = (risk_score > 0.5).astype(int)
    
    # Create DataFrame
    healthcare_data = pd.DataFrame({
        'patient_id': range(1, n_samples + 1),
        'age': age,
        'blood_pressure': blood_pressure,
        'cholesterol': cholesterol,
        'glucose': glucose,
        'bmi': bmi,
        'smoking': smoking,
        'exercise_hours': exercise_hours,
        'alcohol_consumption': alcohol_consumption,
        'family_history': family_history,
        'previous_conditions': previous_conditions,
        'disease_risk': disease_risk
    })
    
    # Save dataset
    healthcare_data.to_csv('data/healthcare_disease_prediction.csv', index=False)
    print(f"✅ Healthcare dataset saved: {healthcare_data.shape}")
    
    return healthcare_data

def create_financial_dataset():
    """Create a synthetic financial dataset for fraud detection"""
    print("💰 Creating Financial Dataset...")
    
    n_samples = 15000
    np.random.seed(123)
    
    # Transaction features
    transaction_amount = np.random.lognormal(3, 1.5, n_samples)
    transaction_amount = np.clip(transaction_amount, 1, 10000)
    
    # Time-based features
    hour_probabilities = np.array([0.03] * 6 + [0.06] * 6 + [0.08] * 6 + [0.05] * 6)
    hour_probabilities = hour_probabilities / hour_probabilities.sum()
    hour_of_day = np.random.choice(range(24), n_samples, p=hour_probabilities)
    day_of_week = np.random.choice(range(7), n_samples)
    
    # Customer behavior
    customer_age = np.random.normal(35, 12, n_samples).astype(int)
    customer_age = np.clip(customer_age, 18, 80)
    
    account_balance = np.random.lognormal(8, 1, n_samples)
    account_balance = np.clip(account_balance, 100, 100000)
    
    # Transaction patterns
    transactions_last_day = np.random.poisson(5, n_samples)
    transactions_last_week = np.random.poisson(25, n_samples)
    avg_transaction_amount = account_balance / np.random.uniform(10, 100, n_samples)
    
    # Geographic features
    distance_from_home = np.random.exponential(50, n_samples)
    distance_from_home = np.clip(distance_from_home, 0, 1000)
    
    # Device and channel features
    is_mobile = np.random.choice([0, 1], n_samples, p=[0.4, 0.6])
    is_online = np.random.choice([0, 1], n_samples, p=[0.3, 0.7])
    
    # Calculate fraud probability (target)
    fraud_risk = (
        (transaction_amount > 5000) * 0.3 +
        (hour_of_day < 6) * 0.2 +
        (hour_of_day > 22) * 0.15 +
        (distance_from_home > 200) * 0.25 +
        (transactions_last_day > 20) * 0.2 +
        (transaction_amount > avg_transaction_amount * 3) * 0.35 +
        (is_mobile & (transaction_amount > 1000)) * 0.15
    )
    
    # Add noise and create binary target
    fraud_risk += np.random.normal(0, 0.1, n_samples)
    is_fraud = (fraud_risk > 0.3).astype(int)
    
    # Create DataFrame
    financial_data = pd.DataFrame({
        'transaction_id': range(1, n_samples + 1),
        'transaction_amount': transaction_amount,
        'hour_of_day': hour_of_day,
        'day_of_week': day_of_week,
        'customer_age': customer_age,
        'account_balance': account_balance,
        'transactions_last_day': transactions_last_day,
        'transactions_last_week': transactions_last_week,
        'avg_transaction_amount': avg_transaction_amount,
        'distance_from_home': distance_from_home,
        'is_mobile': is_mobile,
        'is_online': is_online,
        'is_fraud': is_fraud
    })
    
    # Save dataset
    financial_data.to_csv('data/financial_fraud_detection.csv', index=False)
    print(f"✅ Financial dataset saved: {financial_data.shape}")
    
    return financial_data

def create_iot_dataset():
    """Create a synthetic IoT dataset for anomaly detection"""
    print("🌐 Creating IoT Dataset...")
    
    n_samples = 20000
    np.random.seed(456)
    
    # Sensor readings
    temperature = np.random.normal(25, 5, n_samples)
    humidity = np.random.normal(60, 15, n_samples)
    pressure = np.random.normal(1013, 10, n_samples)
    
    # Network metrics
    cpu_usage = np.random.beta(2, 5, n_samples) * 100
    memory_usage = np.random.beta(3, 4, n_samples) * 100
    network_latency = np.random.lognormal(2, 0.5, n_samples)
    packet_loss = np.random.beta(1, 20, n_samples)
    
    # Device characteristics
    device_type = np.random.choice(['sensor', 'actuator', 'gateway', 'controller'], n_samples, p=[0.4, 0.3, 0.2, 0.1])
    device_age = np.random.exponential(365, n_samples).astype(int)  # days
    battery_level = np.random.beta(3, 2, n_samples) * 100
    
    # Environmental factors
    vibration = np.random.exponential(0.5, n_samples)
    power_consumption = np.random.lognormal(1, 0.3, n_samples)
    
    # Calculate anomaly probability (target)
    anomaly_risk = (
        (np.abs(temperature - 25) > 15) * 0.2 +
        (np.abs(humidity - 60) > 30) * 0.15 +
        (cpu_usage > 80) * 0.3 +
        (memory_usage > 85) * 0.25 +
        (network_latency > 50) * 0.2 +
        (packet_loss > 0.05) * 0.35 +
        (battery_level < 20) * 0.15 +
        (vibration > 2) * 0.25 +
        (device_age > 1000) * 0.1
    )
    
    # Add noise and create binary target
    anomaly_risk += np.random.normal(0, 0.1, n_samples)
    is_anomaly = (anomaly_risk > 0.4).astype(int)
    
    # Create DataFrame
    iot_data = pd.DataFrame({
        'device_id': range(1, n_samples + 1),
        'temperature': temperature,
        'humidity': humidity,
        'pressure': pressure,
        'cpu_usage': cpu_usage,
        'memory_usage': memory_usage,
        'network_latency': network_latency,
        'packet_loss': packet_loss,
        'device_type': device_type,
        'device_age_days': device_age,
        'battery_level': battery_level,
        'vibration': vibration,
        'power_consumption': power_consumption,
        'is_anomaly': is_anomaly
    })
    
    # Save dataset
    iot_data.to_csv('data/iot_anomaly_detection.csv', index=False)
    print(f"✅ IoT dataset saved: {iot_data.shape}")
    
    return iot_data

def create_classification_dataset():
    """Create a general classification dataset"""
    print("🎯 Creating Classification Dataset...")
    
    X, y = make_classification(
        n_samples=8000,
        n_features=20,
        n_informative=15,
        n_redundant=3,
        n_clusters_per_class=2,
        random_state=42
    )
    
    # Create feature names
    feature_names = [f'feature_{i+1}' for i in range(X.shape[1])]
    
    # Create DataFrame
    classification_data = pd.DataFrame(X, columns=feature_names)
    classification_data['target'] = y
    
    # Save dataset
    classification_data.to_csv('data/general_classification.csv', index=False)
    print(f"✅ Classification dataset saved: {classification_data.shape}")
    
    return classification_data

def create_dataset_metadata():
    """Create metadata for all datasets"""
    print("📋 Creating Dataset Metadata...")
    
    datasets_info = {
        "healthcare_disease_prediction": {
            "name": "Healthcare Disease Prediction",
            "description": "Predict disease risk based on patient demographics and medical measurements",
            "target_column": "disease_risk",
            "task_type": "binary_classification",
            "n_samples": 10000,
            "n_features": 11,
            "use_case": "Hospitals can collaboratively train disease prediction models without sharing patient records",
            "privacy_level": "HIGH",
            "columns": {
                "age": "Patient age (18-90)",
                "blood_pressure": "Systolic blood pressure",
                "cholesterol": "Total cholesterol level",
                "glucose": "Blood glucose level",
                "bmi": "Body Mass Index",
                "smoking": "Smoking status (0=No, 1=Yes)",
                "exercise_hours": "Weekly exercise hours",
                "alcohol_consumption": "Alcohol consumption level (0=None, 1=Low, 2=High)",
                "family_history": "Family disease history (0=No, 1=Yes)",
                "previous_conditions": "Previous medical conditions (0=None, 1=Minor, 2=Major)",
                "disease_risk": "Disease risk prediction (0=Low, 1=High)"
            }
        },
        "financial_fraud_detection": {
            "name": "Financial Fraud Detection",
            "description": "Detect fraudulent transactions based on transaction patterns and customer behavior",
            "target_column": "is_fraud",
            "task_type": "binary_classification",
            "n_samples": 15000,
            "n_features": 11,
            "use_case": "Banks can collaboratively train fraud detection models without sharing customer transaction data",
            "privacy_level": "CRITICAL",
            "columns": {
                "transaction_amount": "Transaction amount in local currency",
                "hour_of_day": "Hour of transaction (0-23)",
                "day_of_week": "Day of week (0-6)",
                "customer_age": "Customer age",
                "account_balance": "Account balance",
                "transactions_last_day": "Number of transactions in last 24 hours",
                "transactions_last_week": "Number of transactions in last 7 days",
                "avg_transaction_amount": "Customer's average transaction amount",
                "distance_from_home": "Distance from usual transaction location",
                "is_mobile": "Mobile transaction (0=No, 1=Yes)",
                "is_online": "Online transaction (0=No, 1=Yes)",
                "is_fraud": "Fraud prediction (0=Legitimate, 1=Fraud)"
            }
        },
        "iot_anomaly_detection": {
            "name": "IoT Anomaly Detection",
            "description": "Detect anomalies in IoT sensor data and device behavior",
            "target_column": "is_anomaly",
            "task_type": "binary_classification",
            "n_samples": 20000,
            "n_features": 12,
            "use_case": "Smart cities can collaboratively train anomaly detection models for IoT infrastructure",
            "privacy_level": "MEDIUM",
            "columns": {
                "temperature": "Temperature reading (°C)",
                "humidity": "Humidity percentage",
                "pressure": "Atmospheric pressure (hPa)",
                "cpu_usage": "CPU usage percentage",
                "memory_usage": "Memory usage percentage",
                "network_latency": "Network latency (ms)",
                "packet_loss": "Packet loss rate",
                "device_type": "Type of IoT device",
                "device_age_days": "Device age in days",
                "battery_level": "Battery level percentage",
                "vibration": "Vibration level",
                "power_consumption": "Power consumption (W)",
                "is_anomaly": "Anomaly prediction (0=Normal, 1=Anomaly)"
            }
        },
        "general_classification": {
            "name": "General Classification",
            "description": "General purpose classification dataset for testing federated learning algorithms",
            "target_column": "target",
            "task_type": "binary_classification",
            "n_samples": 8000,
            "n_features": 20,
            "use_case": "Research and development of federated learning algorithms",
            "privacy_level": "LOW",
            "columns": {
                "feature_1-20": "Synthetic features for classification testing",
                "target": "Binary classification target (0 or 1)"
            }
        }
    }
    
    # Save metadata
    with open('data/dataset_metadata.json', 'w') as f:
        json.dump(datasets_info, f, indent=2)
    
    print("✅ Dataset metadata saved")
    return datasets_info

def update_config_for_datasets():
    """Update config.py to include new datasets"""
    print("Updating configuration for new datasets...")
    
    config_content = '''"""
Configuration file for federated learning data
"""

# CHOOSE YOUR DATA HERE
DATA_CONFIG = {
    # Healthcare Dataset
    "data_path": "data/healthcare_disease_prediction.csv",
    "target_column": "disease_risk",   # Predict disease risk
    
    # Financial Dataset (uncomment to use)
    # "data_path": "data/financial_fraud_detection.csv",
    # "target_column": "is_fraud",   # Predict fraud
    
    # IoT Dataset (uncomment to use)
    # "data_path": "data/iot_anomaly_detection.csv",
    # "target_column": "is_anomaly",   # Predict anomalies
    
    # General Classification (uncomment to use)
    # "data_path": "data/general_classification.csv",
    # "target_column": "target",   # General classification
    
    # Model settings (auto-adjust based on your data)
    "input_dim": None,  # Auto-detected from your data
    "output_dim": None, # Auto-detected from your data
    
    # Federated learning settings
    "n_banks": 5,
    "n_rounds": 10,
    "min_clients": 3,
}

# BANK CONFIGURATIONS
BANK_CONFIGS = {
    "bank_1": {"privacy": True, "noise_multiplier": 1.0},
    "bank_2": {"privacy": True, "noise_multiplier": 1.5}, 
    "bank_3": {"privacy": False, "noise_multiplier": 1.0},
    "bank_4": {"privacy": True, "noise_multiplier": 2.0},
    "bank_5": {"privacy": False, "noise_multiplier": 1.0},
}

def get_data_info():
    """Print current data configuration"""
    print("Current Data Configuration:")
    print(f"   Data file: {DATA_CONFIG['data_path']}")
    print(f"   Target: {DATA_CONFIG['target_column']}")
    print(f"   Number of banks: {DATA_CONFIG['n_banks']}")
    
    # Check if file exists
    import os
    if os.path.exists(DATA_CONFIG['data_path']):
        import pandas as pd
        df = pd.read_csv(DATA_CONFIG['data_path'])
        print(f"   File exists: {df.shape}")
        print(f"   Features: {list(df.columns)}")
    else:
        print(f"   File not found: {DATA_CONFIG['data_path']}")

if __name__ == "__main__":
    get_data_info()
'''
    
    with open('config.py', 'w') as f:
        f.write(config_content)
    
    print("Configuration updated")

def main():
    """Main function to create all datasets"""
    print("🚀 Creating Comprehensive Datasets for Federated Learning Analysis")
    print("=" * 70)
    
    # Create data directory
    os.makedirs('data', exist_ok=True)
    
    # Create all datasets
    healthcare_data = create_healthcare_dataset()
    financial_data = create_financial_dataset()
    iot_data = create_iot_dataset()
    classification_data = create_classification_dataset()
    
    # Create metadata
    metadata = create_dataset_metadata()
    
    # Update configuration
    update_config_for_datasets()
    
    print("\n" + "=" * 70)
    print("🎉 All datasets created successfully!")
    print("\n📊 Available Datasets:")
    for dataset_name, info in metadata.items():
        print(f"   📁 {info['name']}: {info['n_samples']} samples, {info['n_features']} features")
        print(f"      🎯 Use case: {info['use_case']}")
        print(f"      🔒 Privacy level: {info['privacy_level']}")
    
    print(f"\n⚙️ Configuration updated: config.py")
    print(f"📋 Metadata saved: data/dataset_metadata.json")
    print(f"\n🚀 Ready for federated learning analysis!")

if __name__ == "__main__":
    main()
