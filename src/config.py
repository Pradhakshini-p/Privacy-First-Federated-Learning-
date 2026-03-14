"""
Centralized Configuration for Privacy-First Federated Learning Pipeline
Ensures all components use consistent paths and hyperparameters
"""

import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
LOG_DIR = BASE_DIR / "logs"
SRC_DIR = BASE_DIR / "src"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

# Federated Learning Hyperparameters
NUM_CLIENTS = 3
ROUNDS = 5
MIN_CLIENTS = 2
LEARNING_RATE = 0.01
BATCH_SIZE = 32
LOCAL_EPOCHS = 5

# Privacy Parameters
EPSILON = 1.0
DELTA = 1e-5
MAX_GRAD_NORM = 1.0
NOISE_MULTIPLIER = 1.0

# Model Architecture
INPUT_DIM = None  # Auto-detected from data
OUTPUT_DIM = None  # Auto-detected from data
HIDDEN_LAYERS = [64, 32]

# Server Configuration
SERVER_HOST = "localhost"
SERVER_PORT = 8080
HEARTBEAT_INTERVAL = 30  # seconds
TIMEOUT = 120  # seconds

# Logging Configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
METRICS_LOG_FILE = LOG_DIR / "training_metrics.csv"
PRIVACY_LOG_FILE = LOG_DIR / "privacy_metrics.csv"
CLIENT_LOG_FILE = LOG_DIR / "client_metrics.csv"

# Dashboard Configuration
DASHBOARD_REFRESH_INTERVAL = 5  # seconds
MAX_LOG_ENTRIES = 1000

# Data Configuration
DATA_CONFIG = {
    # Healthcare Dataset
    "data_path": str(DATA_DIR / "healthcare_disease_prediction.csv"),
    "target_column": "disease_risk",
    
    # Alternative datasets (commented out)
    # "data_path": str(DATA_DIR / "financial_fraud_detection.csv"),
    # "target_column": "is_fraud",
    
    # "data_path": str(DATA_DIR / "iot_anomaly_detection.csv"),
    # "target_column": "is_anomaly",
}

# Client Configurations
CLIENT_CONFIGS = {
    "client_1": {
        "privacy_enabled": True,
        "noise_multiplier": 1.0,
        "max_grad_norm": 1.0,
        "local_epochs": 5
    },
    "client_2": {
        "privacy_enabled": True,
        "noise_multiplier": 1.5,
        "max_grad_norm": 1.0,
        "local_epochs": 5
    },
    "client_3": {
        "privacy_enabled": False,
        "noise_multiplier": 1.0,
        "max_grad_norm": 1.0,
        "local_epochs": 5
    },
}

# Security Configuration
SECURITY_CONFIG = {
    "enable_encryption": True,
    "enable_authentication": True,
    "enable_attack_detection": True,
    "anomaly_threshold": 0.1,
    "malicious_client_threshold": 0.2
}

# Docker Configuration
DOCKER_CONFIG = {
    "server_image": "fl-server:latest",
    "client_image": "fl-client:latest",
    "dashboard_image": "fl-dashboard:latest",
    "network_name": "fl-network",
    "server_startup_delay": 5,  # seconds
    "client_startup_delay": 2   # seconds
}

def get_data_info():
    """Print current data configuration and verify file exists"""
    print("=== Federated Learning Configuration ===")
    print(f"Data directory: {DATA_DIR}")
    print(f"Log directory: {LOG_DIR}")
    print(f"Source directory: {SRC_DIR}")
    print(f"\nData Configuration:")
    print(f"   Data file: {DATA_CONFIG['data_path']}")
    print(f"   Target column: {DATA_CONFIG['target_column']}")
    print(f"\nFederated Learning Settings:")
    print(f"   Number of clients: {NUM_CLIENTS}")
    print(f"   Training rounds: {ROUNDS}")
    print(f"   Learning rate: {LEARNING_RATE}")
    print(f"   Privacy epsilon: {EPSILON}")
    print(f"   Privacy delta: {DELTA}")
    
    # Check if data file exists
    if os.path.exists(DATA_CONFIG['data_path']):
        import pandas as pd
        try:
            df = pd.read_csv(DATA_CONFIG['data_path'])
            print(f"\n✓ Data file exists: {df.shape}")
            print(f"  Features: {list(df.columns)}")
            if DATA_CONFIG['target_column'] in df.columns:
                print(f"  Target column found: {DATA_CONFIG['target_column']}")
                print(f"  Target distribution: {df[DATA_CONFIG['target_column']].value_counts().to_dict()}")
            else:
                print(f"  ⚠️ Target column '{DATA_CONFIG['target_column']}' not found in data")
        except Exception as e:
            print(f"  ⚠️ Error reading data file: {e}")
    else:
        print(f"\n⚠️ Data file not found: {DATA_CONFIG['data_path']}")
        print("  Please ensure the data file exists in the data/ directory")

def validate_config():
    """Validate configuration settings"""
    errors = []
    
    # Check directories
    if not DATA_DIR.exists():
        errors.append(f"Data directory does not exist: {DATA_DIR}")
    if not LOG_DIR.exists():
        errors.append(f"Log directory does not exist: {LOG_DIR}")
    
    # Check data file
    if not os.path.exists(DATA_CONFIG['data_path']):
        errors.append(f"Data file does not exist: {DATA_CONFIG['data_path']}")
    
    # Check hyperparameters
    if NUM_CLIENTS < 1:
        errors.append("NUM_CLIENTS must be >= 1")
    if ROUNDS < 1:
        errors.append("ROUNDS must be >= 1")
    if LEARNING_RATE <= 0:
        errors.append("LEARNING_RATE must be > 0")
    if EPSILON <= 0:
        errors.append("EPSILON must be > 0")
    if DELTA <= 0 or DELTA >= 1:
        errors.append("DELTA must be in (0, 1)")
    
    if errors:
        print("Configuration validation errors:")
        for error in errors:
            print(f"  - {error}")
        return False
    else:
        print("✓ Configuration validation passed")
        return True

if __name__ == "__main__":
    get_data_info()
    validate_config()
