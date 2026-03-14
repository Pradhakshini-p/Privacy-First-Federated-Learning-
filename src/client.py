import flwr as fl
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
import time
import argparse

# Import our custom modules
from model import create_model
from data import FraudDetectionDataLoader

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FraudDetectionClient:
    def __init__(self, client_id: int, silo_id: str = None, model_type="mlp"):
        self.client_id = client_id
        self.silo_id = silo_id or f"silo_{client_id}"
        self.model_type = model_type
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize model and data
        self.model = create_model(model_type=model_type, input_dim=30)
        self.model.to(self.device)
        
        # Load silo data
        self._load_silo_data()
        
        logger.info(f"🤖 Client {self.client_id} initialized")
        logger.info(f"📊 Silo: {self.silo_id}")
        logger.info(f"🧠 Model: {model_type.upper()}")
        logger.info(f"💾 Training samples: {len(self.train_loader.dataset)}")
        logger.info(f"🎯 Validation samples: {len(self.val_loader.dataset)}")
        logger.info(f"🔢 Fraud rate in training: {self.fraud_rate:.4%}")
        
    def _load_silo_data(self):
        """Load and prepare the data silo for this client."""
        # Initialize data loader
        data_loader = FraudDetectionDataLoader()
        
        # Load and preprocess data
        df = data_loader.load_credit_card_data()
        X, y = data_loader.preprocess_data(df)
        
        # Create silos
        silos = data_loader.create_data_silos(X, y, n_silos=3)
        
        # Get this client's silo
        if self.silo_id not in silos:
            raise ValueError(f"Silo {self.silo_id} not found. Available silos: {list(silos.keys())}")
        
        silo_data = silos[self.silo_id]
        self.fraud_rate = silo_data['fraud_rate']
        
        # Create data loaders
        loaders = data_loader.create_dataloaders(silo_data, batch_size=32)
        self.train_loader = loaders['train']
        self.val_loader = loaders['val']
    
    def get_parameters(self, config: Dict[str, fl.common.Scalar]) -> List[np.ndarray]:
        """Get model parameters as numpy arrays."""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """Set model parameters from numpy arrays."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)
    
    def fit(self, parameters: List[np.ndarray], config: Dict[str, fl.common.Scalar]) -> Tuple[List[np.ndarray], int, Dict[str, fl.common.Scalar]]:
        """Train the model on local data."""
        # Get training configuration
        local_epochs = config.get("local_epochs", 1)
        
        logger.info(f"🏋️ Client {self.client_id}: Starting training for {local_epochs} epochs...")
        
        # Set model parameters
        self.set_parameters(parameters)
        
        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
        # Training loop
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for epoch in range(local_epochs):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0
            
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(data)
                loss = criterion(outputs, target)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                
                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                epoch_total += target.size(0)
                epoch_correct += (predicted == target).sum().item()
                
                if batch_idx % 10 == 0:
                    logger.info(f"📊 Client {self.client_id}: Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")
            
            epoch_accuracy = epoch_correct / epoch_total
            logger.info(f"✅ Client {self.client_id}: Epoch {epoch+1} completed - Loss: {epoch_loss/len(self.train_loader):.4f}, Accuracy: {epoch_accuracy:.4f}")
            
            total_loss += epoch_loss
            correct += epoch_correct
            total += epoch_total
        
        # Evaluate on validation set
        val_loss, val_accuracy = self._evaluate_model(criterion)
        
        # Return updated parameters and metrics
        updated_params = self.get_parameters(config)
        
        metrics = {
            "train_loss": total_loss / (local_epochs * len(self.train_loader)),
            "train_accuracy": correct / total,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "num_samples": len(self.train_loader.dataset),
            "client_id": self.client_id
        }
        
        logger.info(f"🎯 Client {self.client_id}: Training completed - Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
        
        return updated_params, len(self.train_loader.dataset), metrics
    
    def _evaluate_model(self, criterion):
        """Evaluate model on validation set."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                outputs = self.model(data)
                loss = criterion(outputs, target)
                total_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def evaluate(self, parameters: List[np.ndarray], config: Dict[str, fl.common.Scalar]) -> Tuple[float, int, Dict[str, fl.common.Scalar]]:
        """Evaluate the model on local validation data."""
        logger.info(f"🔍 Client {self.client_id}: Starting evaluation...")
        
        # Set model parameters
        self.set_parameters(parameters)
        
        # Evaluate
        criterion = nn.CrossEntropyLoss()
        val_loss, val_accuracy = self._evaluate_model(criterion)
        
        metrics = {
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "num_samples": len(self.val_loader.dataset),
            "client_id": self.client_id
        }
        
        logger.info(f"📈 Client {self.client_id}: Evaluation completed - Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}")
        
        return val_loss, len(self.val_loader.dataset), metrics

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, client_id: int, silo_id: str = None, model_type="mlp"):
        self.fraud_client = FraudDetectionClient(client_id, silo_id, model_type)
    
    def get_parameters(self, config: Dict[str, fl.common.Scalar]) -> List[np.ndarray]:
        return self.fraud_client.get_parameters(config)
    
    def fit(self, parameters: List[np.ndarray], config: Dict[str, fl.common.Scalar]) -> Tuple[List[np.ndarray], int, Dict[str, fl.common.Scalar]]:
        return self.fraud_client.fit(parameters, config)
    
    def evaluate(self, parameters: List[np.ndarray], config: Dict[str, fl.common.Scalar]) -> Tuple[float, int, Dict[str, fl.common.Scalar]]:
        return self.fraud_client.evaluate(parameters, config)

def start_client(client_id: int, silo_id: str = None, model_type="mlp", server_address="localhost:8080"):
    """Start a Flower client."""
    logger.info(f"🚀 Starting client {client_id}...")
    
    # Create client
    client = FlowerClient(client_id, silo_id, model_type)
    
    # Start client
    fl.client.start_client(
        server_address=server_address,
        client=client,
    )

def main():
    parser = argparse.ArgumentParser(description="Start Flower Fraud Detection Client")
    parser.add_argument("client_id", type=int, help="Client ID (integer)")
    parser.add_argument("--silo", type=str, help="Silo ID (e.g., silo_1, silo_2)")
    parser.add_argument("--model", type=str, default="mlp", choices=["mlp", "cnn"], 
                       help="Model type to use")
    parser.add_argument("--server", type=str, default="localhost:8080", 
                       help="Server address")
    
    args = parser.parse_args()
    
    # Use silo_1, silo_2, silo_3 as defaults if not specified
    silo_id = args.silo or f"silo_{args.client_id}"
    
    start_client(args.client_id, silo_id, args.model, args.server)

if __name__ == "__main__":
    main()
