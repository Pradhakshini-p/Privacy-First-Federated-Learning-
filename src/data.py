import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FraudDetectionDataLoader:
    """
    Data loader for Credit Card Fraud Detection dataset.
    Handles data loading, preprocessing, and splitting into silos.
    """
    
    def __init__(self, data_path=None, random_state=42):
        self.data_path = data_path
        self.random_state = random_state
        self.scaler = RobustScaler()  # Robust to outliers
        self.feature_columns = None
        self.target_column = 'Class'
        
    def load_credit_card_data(self):
        """
        Load data for federated learning.
        Change this to load YOUR data.
        """
        # OPTION 1: Use your CSV file
        if self.data_path and os.path.exists(self.data_path):
            logger.info(f"Loading data from {self.data_path}")
            df = pd.read_csv(self.data_path)
        # OPTION 2: Use built-in datasets (uncomment to use)
        # elif self.data_path == "iris":
        #     from sklearn.datasets import load_iris
        #     iris = load_iris()
        #     df = pd.DataFrame(iris.data, columns=iris.feature_names)
        #     df['Class'] = iris.target
        # OPTION 3: Create synthetic data (default)
        else:
            logger.info("Creating synthetic fraud detection data...")
            df = self._create_synthetic_data()
        
        logger.info(f"Dataset shape: {df.shape}")
        logger.info(f"Fraud cases: {df[self.target_column].sum()} ({df[self.target_column].mean():.4%})")
        
        return df
    
    def _create_synthetic_data(self, n_samples=284807):
        """
        Create synthetic credit card fraud data.
        Mimics the structure of the real Credit Card Fraud Detection dataset.
        """
        np.random.seed(self.random_state)
        
        # Generate features (V1-V28 are PCA components, Time and Amount are original)
        n_features = 30
        feature_names = [f'V{i}' for i in range(1, 29)] + ['Time', 'Amount']
        
        # Generate normal transactions (99.83% of data)
        n_normal = int(n_samples * 0.9983)
        n_fraud = n_samples - n_normal
        
        # Normal transactions
        normal_data = np.random.multivariate_normal(
            mean=np.zeros(n_features),
            cov=np.eye(n_features),
            size=n_normal
        )
        
        # Fraudulent transactions (different distribution)
        fraud_data = np.random.multivariate_normal(
            mean=np.ones(n_features) * 2,
            cov=np.eye(n_features) * 1.5,
            size=n_fraud
        )
        
        # Combine data
        X = np.vstack([normal_data, fraud_data])
        y = np.hstack([np.zeros(n_normal), np.ones(n_fraud)])
        
        # Create DataFrame
        feature_names = [f'V{i}' for i in range(1, 29)] + ['Time', 'Amount']
        df = pd.DataFrame(X, columns=feature_names)
        df['Class'] = y.astype(int)
        
        # Shuffle the dataset
        df = df.sample(frac=1, random_state=self.random_state).reset_index(drop=True)
        
        return df
    
    def preprocess_data(self, df):
        """
        Preprocess the data: scaling and feature selection.
        """
        # Separate features and target
        if self.target_column in df.columns:
            X = df.drop(columns=[self.target_column])
            y = df[self.target_column]
        else:
            raise ValueError(f"Target column '{self.target_column}' not found in dataset")
        
        # Store feature columns
        self.feature_columns = X.columns.tolist()
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=self.feature_columns)
        
        logger.info(f"Preprocessed data shape: {X_scaled.shape}")
        logger.info(f"Feature columns: {len(self.feature_columns)}")
        
        return X_scaled, y
    
    def create_data_silos(self, X, y, n_silos=5, test_size=0.2):
        """
        Split data into silos to simulate different banks/institutions.
        Each silo only sees its own portion of the data.
        
        Args:
            X: Features
            y: Target
            n_silos: Number of silos to create
            test_size: Fraction of data to hold out for global testing
        
        Returns:
            Dictionary with silo data and global test set
        """
        # First, split off a global test set
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        logger.info(f"Global test set size: {len(X_test)} samples")
        logger.info(f"Training data for silos: {len(X_train)} samples")
        
        # Split training data into silos
        silos = {}
        samples_per_silo = len(X_train) // n_silos
        
        for i in range(n_silos):
            start_idx = i * samples_per_silo
            end_idx = start_idx + samples_per_silo if i < n_silos - 1 else len(X_train)
            
            silo_X = X_train.iloc[start_idx:end_idx]
            silo_y = y_train.iloc[start_idx:end_idx]
            
            # Further split each silo into train/validation
            silo_X_train, silo_X_val, silo_y_train, silo_y_val = train_test_split(
                silo_X, silo_y, test_size=0.2, random_state=self.random_state, stratify=silo_y
            )
            
            silos[f'silo_{i+1}'] = {
                'X_train': silo_X_train,
                'X_val': silo_X_val,
                'y_train': silo_y_train,
                'y_val': silo_y_val,
                'n_samples': len(silo_X_train),
                'fraud_rate': silo_y_train.mean()
            }
            
            logger.info(f"Silo {i+1}: {len(silo_X_train)} samples, "
                       f"fraud rate: {silo_y_train.mean():.4%}")
        
        # Add global test set
        silos['global_test'] = {
            'X_test': X_test,
            'y_test': y_test,
            'n_samples': len(X_test),
            'fraud_rate': y_test.mean()
        }
        
        return silos
    
    def create_dataloaders(self, silo_data, batch_size=32):
        """
        Create PyTorch DataLoaders for a silo.
        """
        def create_loader(X, y, shuffle=True):
            # Convert to tensors
            X_tensor = torch.FloatTensor(X.values if hasattr(X, 'values') else X)
            y_tensor = torch.LongTensor(y.values if hasattr(y, 'values') else y)
            
            # Create dataset and dataloader
            dataset = TensorDataset(X_tensor, y_tensor)
            return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        
        loaders = {}
        
        if 'X_train' in silo_data:
            loaders['train'] = create_loader(silo_data['X_train'], silo_data['y_train'], shuffle=True)
        
        if 'X_val' in silo_data:
            loaders['val'] = create_loader(silo_data['X_val'], silo_data['y_val'], shuffle=False)
        
        if 'X_test' in silo_data:
            loaders['test'] = create_loader(silo_data['X_test'], silo_data['y_test'], shuffle=False)
        
        return loaders

def test_data_pipeline():
    """Test the complete data pipeline."""
    logger.info("🧪 Testing data pipeline...")
    
    # Initialize data loader
    data_loader = FraudDetectionDataLoader()
    
    # Load data
    df = data_loader.load_credit_card_data()
    
    # Preprocess
    X, y = data_loader.preprocess_data(df)
    
    # Create silos
    silos = data_loader.create_data_silos(X, y, n_silos=3)
    
    # Create dataloaders for first silo
    silo_1_loaders = data_loader.create_dataloaders(silos['silo_1'])
    
    # Test dataloader
    for batch_idx, (data, target) in enumerate(silo_1_loaders['train']):
        logger.info(f"Batch {batch_idx}: Data shape {data.shape}, Target shape {target.shape}")
        if batch_idx >= 2:  # Only show first few batches
            break
    
    logger.info("✅ Data pipeline test completed successfully!")

if __name__ == "__main__":
    test_data_pipeline()
