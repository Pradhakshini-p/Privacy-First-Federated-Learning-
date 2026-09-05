"""
Enhanced Backend Federated Learning System
Integrates privacy management, secure aggregation, and real-time monitoring
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
from dataclasses import dataclass, asdict
import concurrent.futures
from collections import defaultdict

# Import federated learning components
try:
    import flwr as fl
    from flwr.common import Parameters, Scalar, FitRes, EvaluateRes
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
from privacy_engine import FederatedPrivacyEngine, get_privacy_manager
from model import create_model
from data import FraudDetectionDataLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    handlers=[
        logging.FileHandler(LOG_DIR / "backend.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ClientMetrics:
    """Metrics for a single client"""
    client_id: str
    round_num: int
    train_loss: float
    train_accuracy: float
    val_loss: float
    val_accuracy: float
    num_samples: int
    training_time: float
    privacy_spent: float
    privacy_budget_used: float
    timestamp: str

@dataclass
class GlobalMetrics:
    """Global training metrics"""
    round_num: int
    global_accuracy: float
    global_loss: float
    num_clients: int
    total_samples: int
    privacy_budget_used: float
    training_time: float
    timestamp: str

class FederatedLearningBackend:
    """Enhanced backend for federated learning with privacy and monitoring"""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the federated learning backend"""
        self.config = config or {}
        self.model_type = self.config.get("model_type", "mlp")
        self.input_dim = self.config.get("input_dim", 30)
        self.num_clients = self.config.get("num_clients", NUM_CLIENTS)
        self.rounds = self.config.get("rounds", ROUNDS)
        
        # Initialize model
        self.global_model = create_model(model_type=self.model_type, input_dim=self.input_dim)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.global_model.to(self.device)
        
        # Privacy management
        self.privacy_manager = get_privacy_manager()
        
        # Metrics tracking
        self.client_metrics: List[ClientMetrics] = []
        self.global_metrics: List[GlobalMetrics] = []
        self.active_clients: Dict[str, bool] = {}
        self.client_performance: Dict[str, List[float]] = defaultdict(list)
        
        # Training state
        self.current_round = 0
        self.training_active = False
        self.training_lock = threading.Lock()
        
        # Server configuration
        self.server_address = f"{SERVER_HOST}:{SERVER_PORT}"
        
        # Data loader
        self.data_loader = None
        
        logger.info(f"Federated Learning Backend initialized")
        logger.info(f"Model: {self.model_type}, Input dim: {self.input_dim}")
        logger.info(f"Clients: {self.num_clients}, Rounds: {self.rounds}")
    
    def setup_data(self):
        """Setup data for federated learning"""
        try:
            self.data_loader = FraudDetectionDataLoader()
            df = self.data_loader.load_credit_card_data()
            X, y = self.data_loader.preprocess_data(df)
            silos = self.data_loader.create_data_silos(X, y, n_silos=self.num_clients)
            
            logger.info(f"Data setup complete with {len(silos)} silos")
            return silos
            
        except Exception as e:
            logger.error(f"Error setting up data: {e}")
            raise
    
    def create_privacy_engine(self, client_id: str, **privacy_config) -> FederatedPrivacyEngine:
        """Create privacy engine for a client"""
        model = create_model(model_type=self.model_type, input_dim=self.input_dim)
        
        privacy_params = {
            "target_epsilon": privacy_config.get("epsilon", EPSILON),
            "target_delta": privacy_config.get("delta", DELTA),
            "max_grad_norm": privacy_config.get("max_grad_norm", MAX_GRAD_NORM),
            "noise_multiplier": privacy_config.get("noise_multiplier", NOISE_MULTIPLIER),
            "client_id": client_id
        }
        
        privacy_engine = FederatedPrivacyEngine(model, **privacy_params)
        self.privacy_manager.add_client(client_id, privacy_engine)
        
        return privacy_engine
    
    def get_client_config(self, client_id: str) -> Dict[str, Any]:
        """Get configuration for a specific client"""
        base_config = {
            "client_id": client_id,
            "model_type": self.model_type,
            "input_dim": self.input_dim,
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "local_epochs": LOCAL_EPOCHS,
            "server_address": self.server_address
        }
        
        # Add client-specific privacy config
        if client_id in CLIENT_CONFIGS:
            base_config.update(CLIENT_CONFIGS[client_id])
        
        return base_config
    
    def start_training(self, num_rounds: Optional[int] = None) -> bool:
        """Start federated training"""
        with self.training_lock:
            if self.training_active:
                logger.warning("Training already in progress")
                return False
            
            self.training_active = True
            self.rounds = num_rounds or self.rounds
            self.current_round = 0
            
            logger.info(f"Starting federated training for {self.rounds} rounds")
            
            # Start training in background thread
            training_thread = threading.Thread(target=self._run_training)
            training_thread.daemon = True
            training_thread.start()
            
            return True
    
    def _run_training(self):
        """Run the federated training process"""
        try:
            # Setup data
            silos = self.setup_data()
            
            # Create Flower strategy
            strategy = self._create_strategy()
            
            # Start Flower server
            fl.server.start_server(
                server_address=self.server_address,
                config=fl.server.ServerConfig(num_rounds=self.rounds),
                strategy=strategy,
            )
            
        except Exception as e:
            logger.error(f"Error in training: {e}")
        finally:
            self.training_active = False
            logger.info("Training completed")
    
    def _create_strategy(self) -> fl.server.strategy.Strategy:
        """Create Flower strategy with custom callbacks"""
        
        def on_fit_config_fn(round_num: int) -> Dict[str, Scalar]:
            """Configure client training"""
            return {
                "local_epochs": LOCAL_EPOCHS,
                "learning_rate": LEARNING_RATE,
                "round_num": round_num
            }
        
        def on_evaluate_config_fn(round_num: int) -> Dict[str, Scalar]:
            """Configure client evaluation"""
            return {"round_num": round_num}
        
        def fit_metrics_aggregation_fn(metrics: List[Tuple[Dict[str, Scalar], int]]) -> Dict[str, Scalar]:
            """Aggregate client training metrics"""
            aggregated = {}
            
            if not metrics:
                return aggregated
            
            total_samples = sum(num_samples for _, num_samples in metrics)
            
            # Aggregate each metric
            for metric_name in metrics[0][0].keys():
                if isinstance(metrics[0][0][metric_name], (int, float)):
                    weighted_sum = sum(
                        client_metrics[metric_name] * num_samples 
                        for client_metrics, num_samples in metrics
                    )
                    aggregated[metric_name] = weighted_sum / total_samples
            
            # Log client metrics
            self._log_client_metrics(metrics, self.current_round)
            
            return aggregated
        
        def evaluate_metrics_aggregation_fn(metrics: List[Tuple[Dict[str, Scalar], int]]) -> Dict[str, Scalar]:
            """Aggregate client evaluation metrics"""
            return fit_metrics_aggregation_fn(metrics)
        
        def evaluate_fn(server_round: int, parameters: Parameters, config: Dict[str, Scalar]) -> Optional[Tuple[float, Dict[str, Scalar]]]:
            """Global model evaluation"""
            try:
                # Set global model parameters
                params_dict = zip(self.global_model.state_dict().keys(), parameters)
                state_dict = {k: torch.tensor(v) for k, v in params_dict}
                self.global_model.load_state_dict(state_dict, strict=True)
                
                # Evaluate on global test set
                accuracy, loss = self._evaluate_global_model()
                
                # Log global metrics
                self._log_global_metrics(server_round, accuracy, loss)
                
                return loss, {"accuracy": accuracy}
                
            except Exception as e:
                logger.error(f"Error in global evaluation: {e}")
                return 0.0, {"accuracy": 0.0}
        
        strategy = fl.server.strategy.FedAvg(
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=MIN_CLIENTS,
            min_evaluate_clients=MIN_CLIENTS,
            min_available_clients=self.num_clients,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
            evaluate_fn=evaluate_fn,
        )
        
        return strategy
    
    def _evaluate_global_model(self) -> Tuple[float, float]:
        """Evaluate global model on test data"""
        if self.data_loader is None:
            return 0.0, 0.0
        
        try:
            # Load test data
            df = self.data_loader.load_credit_card_data()
            X, y = self.data_loader.preprocess_data(df)
            silos = self.data_loader.create_data_silos(X, y, n_silos=self.num_clients)
            
            if 'global_test' not in silos:
                return 0.0, 0.0
            
            test_data = silos['global_test']
            test_loaders = self.data_loader.create_dataloaders(test_data, batch_size=BATCH_SIZE)
            
            # Evaluate
            self.global_model.eval()
            criterion = nn.CrossEntropyLoss()
            
            total_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for data, target in test_loaders['test']:
                    data, target = data.to(self.device), target.to(self.device)
                    outputs = self.global_model(data)
                    loss = criterion(outputs, target)
                    
                    total_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += target.size(0)
                    correct += (predicted == target).sum().item()
            
            accuracy = correct / total if total > 0 else 0.0
            avg_loss = total_loss / len(test_loaders['test']) if len(test_loaders['test']) > 0 else 0.0
            
            return accuracy, avg_loss
            
        except Exception as e:
            logger.error(f"Error evaluating global model: {e}")
            return 0.0, 0.0
    
    def _log_client_metrics(self, metrics: List[Tuple[Dict[str, Scalar], int]], round_num: int):
        """Log client training metrics"""
        for client_metrics, num_samples in metrics:
            try:
                metric = ClientMetrics(
                    client_id=client_metrics.get("client_id", "unknown"),
                    round_num=round_num,
                    train_loss=client_metrics.get("train_loss", 0.0),
                    train_accuracy=client_metrics.get("train_accuracy", 0.0),
                    val_loss=client_metrics.get("val_loss", 0.0),
                    val_accuracy=client_metrics.get("val_accuracy", 0.0),
                    num_samples=num_samples,
                    training_time=client_metrics.get("training_time", 0.0),
                    privacy_spent=client_metrics.get("privacy_spent", 0.0),
                    privacy_budget_used=client_metrics.get("privacy_budget_used", 0.0),
                    timestamp=datetime.now().isoformat()
                )
                
                self.client_metrics.append(metric)
                
                # Track client performance
                if metric.client_id not in self.client_performance:
                    self.client_performance[metric.client_id] = []
                self.client_performance[metric.client_id].append(metric.val_accuracy)
                
                # Save to file
                self._save_metrics()
                
            except Exception as e:
                logger.error(f"Error logging client metrics: {e}")
    
    def _log_global_metrics(self, round_num: int, accuracy: float, loss: float):
        """Log global training metrics"""
        try:
            metric = GlobalMetrics(
                round_num=round_num,
                global_accuracy=accuracy,
                global_loss=loss,
                num_clients=len(self.active_clients),
                total_samples=sum(
                    m.num_samples for m in self.client_metrics 
                    if m.round_num == round_num
                ),
                privacy_budget_used=self.privacy_manager.get_global_privacy_status()["global_budget_used"],
                training_time=0.0,  # Would need to track this
                timestamp=datetime.now().isoformat()
            )
            
            self.global_metrics.append(metric)
            self.current_round = round_num
            
            # Save to file
            self._save_metrics()
            
            logger.info(f"Round {round_num}: Global Accuracy = {accuracy:.4f}, Loss = {loss:.4f}")
            
        except Exception as e:
            logger.error(f"Error logging global metrics: {e}")
    
    def _save_metrics(self):
        """Save metrics to files"""
        try:
            # Save client metrics
            client_data = [asdict(metric) for metric in self.client_metrics]
            with open(LOG_DIR / "client_metrics.json", "w") as f:
                json.dump(client_data, f, indent=2)
            
            # Save global metrics
            global_data = [asdict(metric) for metric in self.global_metrics]
            with open(LOG_DIR / "global_metrics.json", "w") as f:
                json.dump(global_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving metrics: {e}")
    
    def get_training_status(self) -> Dict[str, Any]:
        """Get current training status"""
        privacy_status = self.privacy_manager.get_global_privacy_status()
        
        return {
            "training_active": self.training_active,
            "current_round": self.current_round,
            "total_rounds": self.rounds,
            "num_clients": self.num_clients,
            "active_clients": len([c for c in self.active_clients.values() if c]),
            "global_accuracy": self.global_metrics[-1].global_accuracy if self.global_metrics else 0.0,
            "global_loss": self.global_metrics[-1].global_loss if self.global_metrics else 0.0,
            "privacy_status": privacy_status,
            "timestamp": datetime.now().isoformat()
        }
    
    def get_client_metrics(self, client_id: Optional[str] = None) -> List[Dict]:
        """Get client metrics"""
        if client_id:
            return [asdict(m) for m in self.client_metrics if m.client_id == client_id]
        return [asdict(m) for m in self.client_metrics]
    
    def get_global_metrics(self) -> List[Dict]:
        """Get global metrics"""
        return [asdict(m) for m in self.global_metrics]
    
    def get_privacy_status(self) -> Dict:
        """Get privacy status"""
        return self.privacy_manager.get_global_privacy_status()
    
    def update_privacy_config(self, client_id: str, config: Dict[str, Any]) -> bool:
        """Update privacy configuration for a client"""
        try:
            if client_id in self.privacy_manager.client_privacy_engines:
                engine = self.privacy_manager.client_privacy_engines[client_id]
                
                if "noise_multiplier" in config:
                    engine.update_noise_multiplier(config["noise_multiplier"])
                
                # Note: Updating epsilon/delta would require recreating the privacy engine
                logger.info(f"Updated privacy config for {client_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error updating privacy config: {e}")
            return False

# Global backend instance
backend_instance = None

def get_backend() -> FederatedLearningBackend:
    """Get the global backend instance"""
    global backend_instance
    if backend_instance is None:
        backend_instance = FederatedLearningBackend()
    return backend_instance

def create_backend(config: Optional[Dict] = None) -> FederatedLearningBackend:
    """Create a new backend instance"""
    return FederatedLearningBackend(config)

if __name__ == "__main__":
    # Example usage
    backend = get_backend()
    
    print("Federated Learning Backend")
    print("=" * 50)
    
    # Start training
    if backend.start_training():
        print("Training started...")
        
        # Monitor training
        while backend.training_active:
            status = backend.get_training_status()
            print(f"Round {status['current_round']}: Accuracy = {status['global_accuracy']:.4f}")
            time.sleep(5)
    
    print("Training completed!")
