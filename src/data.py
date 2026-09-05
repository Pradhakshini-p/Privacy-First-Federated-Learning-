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

class DiabetesDataLoader:
    """
    Data loader for Diabetes Prediction dataset.
    Handles data loading, preprocessing, and splitting into hospital silos.
    """

    def __init__(self, data_path=None, random_state=42):
        self.data_path = data_path or "data/diabetes.csv"
        self.random_state = random_state
        self.scaler = StandardScaler()  # Standard scaling for healthcare data
        self.feature_columns = None
        self.target_column = 'Outcome'
        
    def load_diabetes_data(self):
        """
        Load diabetes dataset for federated learning.
        """
        if os.path.exists(self.data_path):
            logger.info(f"Loading diabetes data from {self.data_path}")
            df = pd.read_csv(self.data_path)
        else:
            logger.error(f"Data file not found: {self.data_path}")
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

        # Clean column names and handle missing values (Pima dataset uses 0 for missing)
        df.columns = df.columns.str.strip()
        df = df.replace('', np.nan)

        zero_invalid_cols = [
            'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI'
        ]
        for col in zero_invalid_cols:
            if col in df.columns:
                df[col] = df[col].replace(0, np.nan)

        # Impute missing values with column medians (standard for Pima dataset)
        for col in df.columns:
            if col != self.target_column and df[col].isna().any():
                df[col] = df[col].fillna(df[col].median())

        df = df.dropna()

        logger.info(f"Dataset shape: {df.shape}")
        logger.info(f"Diabetes cases: {df[self.target_column].sum()} ({df[self.target_column].mean():.4%})")

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
    
    def create_data_silos(self, X, y, n_silos=3, test_size=0.2):
        """
        Split data into silos to simulate different hospitals.
        Each silo only sees its own portion of the data (non-IID distribution).

        Args:
            X: Features
            y: Target
            n_silos: Number of hospital silos to create
            test_size: Fraction of data to hold out for global testing

        Returns:
            Dictionary with silo data and global test set
        """
        # First, split off a global test set
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )

        logger.info(f"Global test set size: {len(X_test)} samples")
        logger.info(f"Training data for hospitals: {len(X_train)} samples")

        # Split training data into hospital silos (non-IID)
        silos = {}
        samples_per_silo = len(X_train) // n_silos

        hospital_names = ['Hospital A', 'Hospital B', 'Hospital C']

        for i in range(n_silos):
            start_idx = i * samples_per_silo
            end_idx = start_idx + samples_per_silo if i < n_silos - 1 else len(X_train)

            silo_X = X_train.iloc[start_idx:end_idx]
            silo_y = y_train.iloc[start_idx:end_idx]

            # Further split each silo into train/validation
            silo_X_train, silo_X_val, silo_y_train, silo_y_val = train_test_split(
                silo_X, silo_y, test_size=0.2, random_state=self.random_state, stratify=silo_y
            )

            silos[f'hospital_{i+1}'] = {
                'X_train': silo_X_train,
                'X_val': silo_X_val,
                'y_train': silo_y_train,
                'y_val': silo_y_val,
                'n_samples': len(silo_X_train),
                'diabetes_rate': silo_y_train.mean(),
                'hospital_name': hospital_names[i]
            }

            logger.info(f"{hospital_names[i]}: {len(silo_X_train)} samples, "
                       f"diabetes rate: {silo_y_train.mean():.4%}")

        # Add global test set
        silos['global_test'] = {
            'X_test': X_test,
            'y_test': y_test,
            'n_samples': len(X_test),
            'diabetes_rate': y_test.mean()
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
    logger.info("🧪 Testing diabetes data pipeline...")

    # Initialize data loader
    data_loader = DiabetesDataLoader()

    # Load data
    df = data_loader.load_diabetes_data()

    # Preprocess
    X, y = data_loader.preprocess_data(df)

    # Create silos
    silos = data_loader.create_data_silos(X, y, n_silos=3)

    # Create dataloaders for first hospital
    hospital_1_loaders = data_loader.create_dataloaders(silos['hospital_1'])

    # Test dataloader
    for batch_idx, (data, target) in enumerate(hospital_1_loaders['train']):
        logger.info(f"Batch {batch_idx}: Data shape {data.shape}, Target shape {target.shape}")
        if batch_idx >= 2:  # Only show first few batches
            break

    logger.info("✅ Diabetes data pipeline test completed successfully!")

if __name__ == "__main__":
    test_data_pipeline()
