import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os

def load_credit_card_data(data_path=None, test_size=0.2, random_state=42):
    """
    Simple data loader for federated learning
    """
    if data_path and os.path.exists(data_path):
        # Load from specified path
        df = pd.read_csv(data_path)
    else:
        # Create synthetic data for testing
        n_samples = 1000
        np.random.seed(random_state)
        df = pd.DataFrame({
            'feature_1': np.random.normal(0, 1, n_samples),
            'feature_2': np.random.normal(0, 1, n_samples),
            'feature_3': np.random.normal(0, 1, n_samples),
            'feature_4': np.random.normal(0, 1, n_samples),
            'target': np.random.randint(0, 2, n_samples)
        })
    
    # Split features and target
    X = df.drop('target', axis=1).values
    y = df['target'].values
    
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test

def create_data_silos(X_train, y_train, n_silos=5, random_state=42):
    """
    Split training data into silos for federated learning
    """
    n_samples = len(X_train)
    indices = np.random.permutation(n_samples)
    
    # Calculate samples per silo
    samples_per_silo = n_samples // n_silos
    remaining = n_samples % n_silos
    
    silos = []
    start_idx = 0
    
    for i in range(n_silos):
        # Distribute remaining samples
        n_samples_this_silo = samples_per_silo + (1 if i < remaining else 0)
        
        end_idx = start_idx + n_samples_this_silo
        silo_indices = indices[start_idx:end_idx]
        
        silo_X = X_train[silo_indices]
        silo_y = y_train[silo_indices]
        
        silos.append((silo_X, silo_y))
        start_idx = end_idx
    
    return silos
