"""
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
