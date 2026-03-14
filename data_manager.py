#!/usr/bin/env python3
"""
Data Management Module for Federated Learning
Handles loading, preprocessing, and distributing datasets for federated learning
"""

import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_regression, load_iris, load_wine, load_breast_cancer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
import json
from typing import Dict, List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FederatedDataManager:
    """Manages datasets for federated learning clients"""
    
    def __init__(self):
        self.available_datasets = {}
        self.loaded_data = {}
        self.dataset_info = {}
        self._initialize_builtin_datasets()
    
    def _initialize_builtin_datasets(self):
        """Initialize built-in dataset options"""
        self.available_datasets = {
            "synthetic_classification": {
                "name": "Synthetic Classification",
                "description": "Generated classification dataset with configurable parameters",
                "type": "classification",
                "n_samples": 10000,
                "n_features": 20,
                "n_classes": 2
            },
            "iris": {
                "name": "Iris Dataset",
                "description": "Classic iris flower classification dataset",
                "type": "classification",
                "n_samples": 150,
                "n_features": 4,
                "n_classes": 3
            },
            "wine": {
                "name": "Wine Dataset",
                "description": "Wine classification dataset",
                "type": "classification", 
                "n_samples": 178,
                "n_features": 13,
                "n_classes": 3
            },
            "breast_cancer": {
                "name": "Breast Cancer Dataset",
                "description": "Breast cancer classification dataset",
                "type": "classification",
                "n_samples": 569,
                "n_features": 30,
                "n_classes": 2
            },
            "synthetic_regression": {
                "name": "Synthetic Regression",
                "description": "Generated regression dataset with configurable parameters",
                "type": "regression",
                "n_samples": 10000,
                "n_features": 15
            }
        }
    
    def load_builtin_dataset(self, dataset_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load a built-in dataset"""
        if dataset_name not in self.available_datasets:
            raise ValueError(f"Dataset {dataset_name} not available")
        
        dataset_info = self.available_datasets[dataset_name]
        
        if dataset_name == "synthetic_classification":
            X, y = make_classification(
                n_samples=dataset_info["n_samples"],
                n_features=dataset_info["n_features"],
                n_informative=max(dataset_info["n_features"] // 2, 5),
                n_redundant=dataset_info["n_features"] // 4,
                n_classes=dataset_info["n_classes"],
                random_state=42
            )
        elif dataset_name == "synthetic_regression":
            X, y = make_regression(
                n_samples=dataset_info["n_samples"],
                n_features=dataset_info["n_features"],
                n_informative=max(dataset_info["n_features"] // 2, 5),
                noise=0.1,
                random_state=42
            )
        elif dataset_name == "iris":
            data = load_iris()
            X, y = data.data, data.target
        elif dataset_name == "wine":
            data = load_wine()
            X, y = data.data, data.target
        elif dataset_name == "breast_cancer":
            data = load_breast_cancer()
            X, y = data.data, data.target
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        # Scale features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        # Store dataset info
        self.dataset_info[dataset_name] = {
            "n_samples": X.shape[0],
            "n_features": X.shape[1],
            "n_classes": len(np.unique(y)) if dataset_info["type"] == "classification" else None,
            "type": dataset_info["type"]
        }
        
        self.loaded_data[dataset_name] = (X, y)
        
        logger.info(f"Loaded dataset {dataset_name}: {X.shape[0]} samples, {X.shape[1]} features")
        return X, y
    
    def load_csv_dataset(self, file_path: str, target_column: str, 
                        categorical_columns: List[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Load dataset from CSV file"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"CSV file not found: {file_path}")
        
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            raise ValueError(f"Error reading CSV file: {e}")
        
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in CSV")
        
        # Handle categorical columns
        if categorical_columns:
            le = LabelEncoder()
            for col in categorical_columns:
                if col in df.columns:
                    df[col] = le.fit_transform(df[col].astype(str))
        
        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Convert to numpy arrays
        X = X.values
        y = y.values if hasattr(y, 'values') else y
        
        # Scale features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        # Store dataset info
        dataset_name = f"csv_{os.path.basename(file_path)}"
        self.dataset_info[dataset_name] = {
            "n_samples": X.shape[0],
            "n_features": X.shape[1],
            "n_classes": len(np.unique(y)) if len(np.unique(y)) < 20 else None,
            "type": "classification" if len(np.unique(y)) < 20 else "regression",
            "source": file_path,
            "target_column": target_column
        }
        
        self.loaded_data[dataset_name] = (X, y)
        
        logger.info(f"Loaded CSV dataset: {X.shape[0]} samples, {X.shape[1]} features")
        return X, y
    
    def distribute_data_to_clients(self, dataset_name: str, n_clients: int, 
                                 distribution: str = "iid") -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Distribute dataset to multiple clients"""
        if dataset_name not in self.loaded_data:
            raise ValueError(f"Dataset {dataset_name} not loaded")
        
        X, y = self.loaded_data[dataset_name]
        client_data = {}
        
        if distribution == "iid":
            # IID distribution: random split
            indices = np.random.permutation(len(X))
            client_sizes = np.array_split(indices, n_clients)
            
            for i, client_indices in enumerate(client_sizes):
                client_id = f"client_{i+1:03d}"
                client_X = X[client_indices]
                client_y = y[client_indices]
                client_data[client_id] = (client_X, client_y)
        
        elif distribution == "non_iid":
            # Non-IID distribution: skewed class distribution
            classes = np.unique(y)
            n_classes = len(classes)
            
            # Assign each client preferentially to certain classes
            for i in range(n_clients):
                client_id = f"client_{i+1:03d}"
                preferred_classes = [classes[i % n_classes], classes[(i+1) % n_classes]]
                
                # Select samples from preferred classes
                mask = np.isin(y, preferred_classes)
                client_X = X[mask]
                client_y = y[mask]
                
                # Add some random samples from other classes
                other_mask = ~mask
                if np.sum(other_mask) > 0:
                    other_indices = np.random.choice(np.where(other_mask)[0], 
                                                 min(len(client_X) // 2, np.sum(other_mask)), 
                                                 replace=False)
                    client_X = np.vstack([client_X, X[other_indices]])
                    client_y = np.hstack([client_y, y[other_indices]])
                
                client_data[client_id] = (client_X, client_y)
        
        elif distribution == "quantity_skew":
            # Quantity skew: different amounts of data per client
            total_samples = len(X)
            # Create skewed distribution (some clients have much more data)
            client_sizes = np.random.lognormal(mean=np.log(total_samples/n_clients), 
                                            sigma=1.0, size=n_clients)
            client_sizes = (client_sizes / client_sizes.sum() * total_samples).astype(int)
            
            # Adjust to match total
            client_sizes[-1] += total_samples - client_sizes.sum()
            
            # Shuffle and split
            indices = np.random.permutation(len(X))
            start_idx = 0
            for i, size in enumerate(client_sizes):
                client_id = f"client_{i+1:03d}"
                end_idx = start_idx + size
                client_X = X[start_idx:end_idx]
                client_y = y[start_idx:end_idx]
                client_data[client_id] = (client_X, client_y)
                start_idx = end_idx
        
        logger.info(f"Distributed dataset {dataset_name} to {n_clients} clients using {distribution} distribution")
        return client_data
    
    def get_dataset_info(self, dataset_name: str) -> Dict:
        """Get information about a dataset"""
        if dataset_name not in self.dataset_info:
            return {}
        return self.dataset_info[dataset_name]
    
    def list_available_datasets(self) -> Dict:
        """List all available datasets"""
        return self.available_datasets
    
    def list_loaded_datasets(self) -> List[str]:
        """List all loaded datasets"""
        return list(self.loaded_data.keys())
    
    def create_sample_csv_datasets(self):
        """Create sample CSV datasets for testing"""
        os.makedirs("sample_data", exist_ok=True)
        
        # Sample 1: Student Performance
        np.random.seed(42)
        n_students = 1000
        
        student_data = {
            "study_hours": np.random.uniform(1, 10, n_students),
            "previous_score": np.random.uniform(40, 100, n_students),
            "attendance_rate": np.random.uniform(60, 100, n_students),
            "assignments_completed": np.random.randint(0, 20, n_students),
            "extracurricular": np.random.choice([0, 1], n_students, p=[0.3, 0.7]),
            "parental_education": np.random.choice([1, 2, 3, 4], n_students),
            "sleep_hours": np.random.uniform(4, 10, n_students),
            "stress_level": np.random.randint(1, 10, n_students),
            "final_grade": np.random.choice([0, 1], n_students, p=[0.3, 0.7])  # 0=Fail, 1=Pass
        }
        
        df_students = pd.DataFrame(student_data)
        df_students.to_csv("sample_data/student_performance.csv", index=False)
        
        # Sample 2: Credit Card Fraud (simplified)
        n_transactions = 5000
        
        fraud_data = {
            "transaction_amount": np.random.lognormal(3, 1.5, n_transactions),
            "merchant_category": np.random.choice([1, 2, 3, 4, 5], n_transactions),
            "time_of_day": np.random.randint(0, 24, n_transactions),
            "day_of_week": np.random.randint(0, 7, n_transactions),
            "customer_age": np.random.randint(18, 80, n_transactions),
            "transaction_count_24h": np.random.randint(0, 50, n_transactions),
            "avg_transaction_amount": np.random.lognormal(2.5, 1.0, n_transactions),
            "is_foreign": np.random.choice([0, 1], n_transactions, p=[0.9, 0.1]),
            "is_online": np.random.choice([0, 1], n_transactions, p=[0.4, 0.6]),
            "is_fraud": np.random.choice([0, 1], n_transactions, p=[0.95, 0.05])
        }
        
        df_fraud = pd.DataFrame(fraud_data)
        df_fraud.to_csv("sample_data/credit_card_fraud.csv", index=False)
        
        # Sample 3: Customer Churn
        n_customers = 2000
        
        churn_data = {
            "tenure_months": np.random.randint(1, 72, n_customers),
            "monthly_charges": np.random.uniform(20, 200, n_customers),
            "total_charges": np.random.uniform(100, 10000, n_customers),
            "contract_type": np.random.choice([0, 1, 2], n_customers),  # Month-to-month, 1yr, 2yr
            "payment_method": np.random.choice([0, 1, 2, 3], n_customers),
            "internet_service": np.random.choice([0, 1, 2], n_customers),  # No, DSL, Fiber
            "tech_support": np.random.choice([0, 1], n_customers, p=[0.7, 0.3]),
            "streaming_tv": np.random.choice([0, 1], n_customers, p=[0.5, 0.5]),
            "num_complaints": np.random.randint(0, 10, n_customers),
            "satisfaction_score": np.random.randint(1, 5, n_customers),
            "churned": np.random.choice([0, 1], n_customers, p=[0.7, 0.3])
        }
        
        df_churn = pd.DataFrame(churn_data)
        df_churn.to_csv("sample_data/customer_churn.csv", index=False)
        
        logger.info("Sample CSV datasets created in 'sample_data' directory")

def main():
    """Test the data manager"""
    data_manager = FederatedDataManager()
    
    # Create sample datasets
    data_manager.create_sample_csv_datasets()
    
    # Test loading built-in dataset
    print("Loading built-in dataset...")
    X, y = data_manager.load_builtin_dataset("iris")
    print(f"Loaded iris dataset: {X.shape}")
    
    # Test loading CSV dataset
    print("\nLoading CSV dataset...")
    X, y = data_manager.load_csv_dataset("sample_data/student_performance.csv", "final_grade")
    print(f"Loaded student dataset: {X.shape}")
    
    # Test data distribution
    print("\nDistributing data to clients...")
    client_data = data_manager.distribute_data_to_clients("iris", 5, "non_iid")
    for client_id, (client_X, client_y) in client_data.items():
        print(f"{client_id}: {client_X.shape[0]} samples, Class distribution: {np.bincount(client_y)}")
    
    print("\nDataset information:")
    for dataset_name in data_manager.list_loaded_datasets():
        info = data_manager.get_dataset_info(dataset_name)
        print(f"{dataset_name}: {info}")

if __name__ == "__main__":
    main()
