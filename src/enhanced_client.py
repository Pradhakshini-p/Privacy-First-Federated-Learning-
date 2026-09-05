"""
Enhanced Federated Learning Client with Privacy Support
Integrates with the backend for comprehensive federated learning
"""

import asyncio
import json
import logging
import time
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
import socket

# Import federated learning components
try:
    import flwr as fl
    from flwr.common import Parameters, Scalar, FitRes, EvaluateRes, ndarrays_to_parameters, parameters_to_ndarrays
except ImportError:
    print("Flower not installed. Please install: pip install flwr")
    exit(1)

# Import privacy components
try:
    from opacus import PrivacyEngine
except ImportError:
    print("Opacus not installed. Please install: pip install opacus")
    exit(1)

# Local imports
from config import *
from privacy_engine import FederatedPrivacyEngine
from model import create_model
from data import FraudDetectionDataLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    handlers=[
        logging.FileHandler(LOG_DIR / f"client_{socket.gethostname()}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ClientState:
    """State of the federated learning client"""
    client_id: str
    silo_id: str
    model_type: str
    input_dim: int
    privacy_enabled: bool
    current_round: int
    total_samples: int
    training_time: float
    last_update: str

class EnhancedFederatedClient:
    """Enhanced federated learning client with privacy and monitoring"""
    
    def __init__(self, client_id: str, silo_id: Optional[str] = None, config: Optional[Dict] = None):
        """Initialize the enhanced federated client"""
        self.client_id = client_id
        self.silo_id = silo_id or f"silo_{client_id}"
        self.config = config or {}
        
        # Model configuration
        self.model_type = self.config.get("model_type", "mlp")
        self.input_dim = self.config.get("input_dim", 30)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Training configuration
        self.learning_rate = self.config.get("learning_rate", LEARNING_RATE)
        self.batch_size = self.config.get("batch_size", BATCH_SIZE)
        self.local_epochs = self.config.get("local_epochs", LOCAL_EPOCHS)
        self.server_address = self.config.get("server_address", f"{SERVER_HOST}:{SERVER_PORT}")
        
        # Privacy configuration
        self.privacy_enabled = self.config.get("privacy_enabled", True)
        self.privacy_engine: Optional[FederatedPrivacyEngine] = None
        
        # Initialize model and data
        self.model = create_model(model_type=self.model_type, input_dim=self.input_dim)
        self.model.to(self.device)
        
        # Data loaders
        self.train_loader = None
        self.val_loader = None
        self.fraud_rate = 0.0
        
        # Client state
        self.state = ClientState(
            client_id=self.client_id,
            silo_id=self.silo_id,
            model_type=self.model_type,
            input_dim=self.input_dim,
            privacy_enabled=self.privacy_enabled,
            current_round=0,
            total_samples=0,
            training_time=0.0,
            last_update=datetime.now().isoformat()
        )
        
        # Metrics
        self.training_metrics: List[Dict] = []
        self.privacy_metrics: List[Dict] = []
        
        # Setup data and privacy
        self._setup_data()
        self._setup_privacy()
        
        logger.info(f"Enhanced client {self.client_id} initialized")
        logger.info(f"Model: {self.model_type}, Privacy: {self.privacy_enabled}")
        logger.info(f"Training samples: {len(self.train_loader.dataset) if self.train_loader else 0}")
        logger.info(f"Fraud rate: {self.fraud_rate:.4%}")
    
    def _setup_data(self):
        """Setup data loaders for this client"""
        try:
            data_loader = FraudDetectionDataLoader()
            df = data_loader.load_credit_card_data()
            X, y = data_loader.preprocess_data(df)
            silos = data_loader.create_data_silos(X, y, n_silos=3)
            
            if self.silo_id not in silos:
                raise ValueError(f"Silo {self.silo_id} not found")
            
            silo_data = silos[self.silo_id]
            self.fraud_rate = silo_data['fraud_rate']
            
            # Create data loaders
            loaders = data_loader.create_dataloaders(silo_data, batch_size=self.batch_size)
            self.train_loader = loaders['train']
            self.val_loader = loaders['val']
            
            self.state.total_samples = len(self.train_loader.dataset)
            
            logger.info(f"Data loaded for {self.silo_id}: {len(self.train_loader.dataset)} training samples")
            
        except Exception as e:
            logger.error(f"Error setting up data: {e}")
            raise
    
    def _setup_privacy(self):
        """Setup privacy engine if enabled"""
        if not self.privacy_enabled:
            logger.info(f"Privacy disabled for client {self.client_id}")
            return
        
        try:
            # Create privacy engine
            privacy_config = CLIENT_CONFIGS.get(self.client_id, {})
            
            self.privacy_engine = FederatedPrivacyEngine(
                model=self.model,
                target_epsilon=privacy_config.get("epsilon", EPSILON),
                target_delta=privacy_config.get("delta", DELTA),
                max_grad_norm=privacy_config.get("max_grad_norm", MAX_GRAD_NORM),
                noise_multiplier=privacy_config.get("noise_multiplier", NOISE_MULTIPLIER),
                client_id=self.client_id
            )
            
            logger.info(f"Privacy engine setup complete for {self.client_id}")
            
        except Exception as e:
            logger.error(f"Error setting up privacy: {e}")
            self.privacy_enabled = False
    
    def get_parameters(self, config: Dict[str, Scalar]) -> Parameters:
        """Get model parameters"""
        try:
            ndarrays = [val.cpu().numpy() for _, val in self.model.state_dict().items()]
            return ndarrays_to_parameters(ndarrays)
        except Exception as e:
            logger.error(f"Error getting parameters: {e}")
            return ndarrays_to_parameters([])
    
    def set_parameters(self, parameters: Parameters) -> bool:
        """Set model parameters"""
        try:
            ndarrays = parameters_to_ndarrays(parameters)
            params_dict = zip(self.model.state_dict().keys(), ndarrays)
            state_dict = {k: torch.tensor(v) for k, v in params_dict}
            self.model.load_state_dict(state_dict, strict=True)
            return True
        except Exception as e:
            logger.error(f"Error setting parameters: {e}")
            return False
    
    def fit(self, parameters: Parameters, config: Dict[str, Scalar]) -> Tuple[Parameters, int, Dict[str, Scalar]]:
        """Train the model on local data"""
        start_time = time.time()
        
        try:
            # Get training configuration
            local_epochs = config.get("local_epochs", self.local_epochs)
            round_num = config.get("round_num", 0)
            
            logger.info(f"Client {self.client_id}: Starting training for round {round_num}")
            
            # Set model parameters
            if not self.set_parameters(parameters):
                raise Exception("Failed to set parameters")
            
            # Setup optimizer and criterion
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
            
            # Attach privacy engine if enabled
            if self.privacy_enabled and self.privacy_engine:
                self.privacy_engine.attach_to_optimizer(optimizer)
            
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
                
                epoch_accuracy = epoch_correct / epoch_total if epoch_total > 0 else 0.0
                
                if epoch % 2 == 0:  # Log every 2 epochs
                    logger.info(f"Client {self.client_id}: Epoch {epoch+1}/{local_epochs}, "
                              f"Loss: {epoch_loss/len(self.train_loader):.4f}, "
                              f"Accuracy: {epoch_accuracy:.4f}")
                
                total_loss += epoch_loss
                correct += epoch_correct
                total += epoch_total
            
            # Evaluate on validation set
            val_loss, val_accuracy = self._evaluate_model(criterion)
            
            # Track privacy if enabled
            privacy_spent = 0.0
            privacy_budget_used = 0.0
            if self.privacy_enabled and self.privacy_engine:
                self.privacy_engine.track_privacy_spent(round_num, len(self.train_loader.dataset))
                privacy_spent = self.privacy_engine.current_epsilon
                privacy_budget_used = self.privacy_engine.privacy_budget_used
            
            # Calculate training time
            training_time = time.time() - start_time
            
            # Update state
            self.state.current_round = round_num
            self.state.training_time += training_time
            self.state.last_update = datetime.now().isoformat()
            
            # Prepare metrics
            metrics = {
                "train_loss": total_loss / (local_epochs * len(self.train_loader)),
                "train_accuracy": correct / total if total > 0 else 0.0,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "num_samples": len(self.train_loader.dataset),
                "client_id": self.client_id,
                "round_num": round_num,
                "training_time": training_time,
                "privacy_spent": privacy_spent,
                "privacy_budget_used": privacy_budget_used,
                "privacy_enabled": self.privacy_enabled,
                "silo_id": self.silo_id,
                "fraud_rate": self.fraud_rate
            }
            
            # Log metrics
            self._log_metrics(metrics)
            
            logger.info(f"Client {self.client_id}: Round {round_num} completed - "
                       f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}, "
                       f"Privacy Budget: {privacy_budget_used:.2%}")
            
            # Return updated parameters and metrics
            updated_params = self.get_parameters({})
            return updated_params, len(self.train_loader.dataset), metrics
            
        except Exception as e:
            logger.error(f"Error in training: {e}")
            # Return empty parameters and error metrics
            error_metrics = {
                "error": str(e),
                "client_id": self.client_id,
                "num_samples": 0,
                "training_time": time.time() - start_time
            }
            return parameters, 0, error_metrics
    
    def evaluate(self, parameters: Parameters, config: Dict[str, Scalar]) -> Tuple[float, int, Dict[str, Scalar]]:
        """Evaluate the model on local validation data"""
        try:
            round_num = config.get("round_num", 0)
            
            logger.info(f"Client {self.client_id}: Starting evaluation for round {round_num}")
            
            # Set model parameters
            if not self.set_parameters(parameters):
                raise Exception("Failed to set parameters")
            
            # Evaluate
            criterion = nn.CrossEntropyLoss()
            val_loss, val_accuracy = self._evaluate_model(criterion)
            
            metrics = {
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "num_samples": len(self.val_loader.dataset),
                "client_id": self.client_id,
                "round_num": round_num,
                "privacy_enabled": self.privacy_enabled
            }
            
            logger.info(f"Client {self.client_id}: Evaluation completed - "
                       f"Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}")
            
            return val_loss, len(self.val_loader.dataset), metrics
            
        except Exception as e:
            logger.error(f"Error in evaluation: {e}")
            return 0.0, 0, {"error": str(e), "client_id": self.client_id}
    
    def _evaluate_model(self, criterion: nn.Module) -> Tuple[float, float]:
        """Evaluate model on validation set"""
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
        
        avg_loss = total_loss / len(self.val_loader) if len(self.val_loader) > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0
        
        return avg_loss, accuracy
    
    def _log_metrics(self, metrics: Dict[str, Scalar]):
        """Log training metrics"""
        try:
            metrics["timestamp"] = datetime.now().isoformat()
            self.training_metrics.append(metrics)
            
            # Save to file
            with open(LOG_DIR / f"client_{self.client_id}_metrics.json", "w") as f:
                json.dump(self.training_metrics, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error logging metrics: {e}")
    
    def get_client_state(self) -> Dict[str, Any]:
        """Get current client state"""
        state_dict = {
            "state": self.state.__dict__,
            "privacy_enabled": self.privacy_enabled,
            "privacy_status": self.privacy_engine.get_privacy_status() if self.privacy_engine else None,
            "recent_metrics": self.training_metrics[-5:] if self.training_metrics else []
        }
        return state_dict
    
    def update_privacy_config(self, config: Dict[str, Any]) -> bool:
        """Update privacy configuration"""
        try:
            if not self.privacy_enabled or not self.privacy_engine:
                return False
            
            if "noise_multiplier" in config:
                self.privacy_engine.update_noise_multiplier(config["noise_multiplier"])
            
            logger.info(f"Privacy config updated for {self.client_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating privacy config: {e}")
            return False

class FlowerClientWrapper(fl.client.NumPyClient):
    """Wrapper for EnhancedFederatedClient to work with Flower"""
    
    def __init__(self, enhanced_client: EnhancedFederatedClient):
        self.client = enhanced_client
    
    def get_parameters(self, config: Dict[str, Scalar]) -> List[np.ndarray]:
        """Get model parameters as numpy arrays"""
        params = self.client.get_parameters(config)
        return parameters_to_ndarrays(params)
    
    def set_parameters(self, parameters: List[np.ndarray]) -> bool:
        """Set model parameters from numpy arrays"""
        params = ndarrays_to_parameters(parameters)
        return self.client.set_parameters(params)
    
    def fit(self, parameters: List[np.ndarray], config: Dict[str, Scalar]) -> Tuple[List[np.ndarray], int, Dict[str, Scalar]]:
        """Train the model"""
        params = ndarrays_to_parameters(parameters)
        updated_params, num_samples, metrics = self.client.fit(params, config)
        return parameters_to_ndarrays(updated_params), num_samples, metrics
    
    def evaluate(self, parameters: List[np.ndarray], config: Dict[str, Scalar]) -> Tuple[float, int, Dict[str, Scalar]]:
        """Evaluate the model"""
        params = ndarrays_to_parameters(parameters)
        loss, num_samples, metrics = self.client.evaluate(params, config)
        return loss, num_samples, metrics

def start_enhanced_client(client_id: str, silo_id: Optional[str] = None, config: Optional[Dict] = None):
    """Start an enhanced federated learning client"""
    logger.info(f"Starting enhanced client {client_id}")
    
    # Create enhanced client
    enhanced_client = EnhancedFederatedClient(client_id, silo_id, config)
    
    # Create Flower wrapper
    flower_client = FlowerClientWrapper(enhanced_client)
    
    # Start client
    server_address = config.get("server_address", f"{SERVER_HOST}:{SERVER_PORT}") if config else f"{SERVER_HOST}:{SERVER_PORT}"
    
    try:
        fl.client.start_client(
            server_address=server_address,
            client=flower_client,
        )
    except Exception as e:
        logger.error(f"Error starting client: {e}")
        raise

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Start Enhanced Federated Learning Client")
    parser.add_argument("client_id", type=str, help="Client ID")
    parser.add_argument("--silo", type=str, help="Silo ID")
    parser.add_argument("--server", type=str, default=f"{SERVER_HOST}:{SERVER_PORT}", help="Server address")
    parser.add_argument("--privacy", action="store_true", help="Enable privacy")
    parser.add_argument("--no-privacy", action="store_true", help="Disable privacy")
    
    args = parser.parse_args()
    
    # Determine privacy setting
    privacy_enabled = True
    if args.no_privacy:
        privacy_enabled = False
    elif args.privacy:
        privacy_enabled = True
    
    # Create config
    config = {
        "server_address": args.server,
        "privacy_enabled": privacy_enabled
    }
    
    # Start client
    start_enhanced_client(args.client_id, args.silo, config)
