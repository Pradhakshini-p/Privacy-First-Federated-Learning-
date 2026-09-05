#!/usr/bin/env python3
"""
Federated Learning Client for Diabetes Prediction
Flower client implementation with differential privacy
"""

import flwr as fl
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
import argparse

# Import our custom modules
from src.model import create_model
from src.data import DiabetesDataLoader
from src.privacy import PrivacyManager, PrivacyConfig, ManualPrivacyTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DiabetesPredictionClient:
    """Federated Learning Client for Diabetes Prediction"""

    def __init__(self, client_id: int, hospital_id: str = None, model_type="mlp", use_privacy=True):
        self.client_id = client_id
        self.hospital_id = hospital_id or f"hospital_{client_id}"
        self.model_type = model_type
        self.use_privacy = use_privacy
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize model and data
        self.model = create_model(model_type=model_type, input_dim=8)
        self.model.to(self.device)

        # Load hospital data
        self._load_hospital_data()

        # Privacy setup (manual DP-SGD for reliable federated training)
        self.privacy_manager = None
        self.manual_privacy = None

        logger.info(f"🤖 Client {self.client_id} initialized")
        logger.info(f"🏥 Hospital: {self.hospital_id}")
        logger.info(f"🧠 Model: {model_type.upper()}")
        logger.info(f"💾 Training samples: {len(self.train_loader.dataset)}")
        logger.info(f"🎯 Validation samples: {len(self.val_loader.dataset)}")
        logger.info(f"🔢 Diabetes rate in training: {self.diabetes_rate:.4%}")
        logger.info(f"🔒 Privacy: {'Enabled' if self.use_privacy else 'Disabled'}")

    def _load_hospital_data(self):
        """Load and prepare the hospital data for this client."""
        # Initialize data loader
        data_loader = DiabetesDataLoader()

        # Load and preprocess data
        df = data_loader.load_diabetes_data()
        X, y = data_loader.preprocess_data(df)

        # Create hospital silos
        silos = data_loader.create_data_silos(X, y, n_silos=3)

        # Get this client's hospital silo
        if self.hospital_id not in silos:
            raise ValueError(f"Hospital {self.hospital_id} not found. Available: {list(silos.keys())}")

        hospital_data = silos[self.hospital_id]
        self.diabetes_rate = hospital_data['diabetes_rate']
        self.hospital_name = hospital_data.get('hospital_name', f'Hospital {self.client_id}')

        # Create data loaders
        loaders = data_loader.create_dataloaders(hospital_data, batch_size=32)
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
        local_epochs = config.get("local_epochs", 5)

        logger.info(f"🏋️ {self.hospital_name}: Starting training for {local_epochs} epochs...")

        self.set_parameters(parameters)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        train_loader = self.train_loader
        manual_privacy = None

        if self.use_privacy:
            privacy_config = PrivacyConfig(
                epsilon=3.0,
                delta=1e-5,
                max_grad_norm=1.0,
                noise_multiplier=1.0
            )
            manual_privacy = ManualPrivacyTrainer(privacy_config)
            logger.info("🔒 Privacy: Manual DP-SGD (clipping + noise)")

        effective_epochs = local_epochs

        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for epoch in range(effective_epochs):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0

            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(data)
                loss = criterion(outputs, target)
                loss.backward()
                if manual_privacy:
                    manual_privacy.apply(self.model)
                optimizer.step()

                epoch_loss += loss.item()

                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                epoch_total += target.size(0)
                epoch_correct += (predicted == target).sum().item()

                if batch_idx % 10 == 0:
                    logger.info(f"📊 {self.hospital_name}: Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")

            epoch_accuracy = epoch_correct / epoch_total
            logger.info(f"✅ {self.hospital_name}: Epoch {epoch+1} completed - Loss: {epoch_loss/len(train_loader):.4f}, Accuracy: {epoch_accuracy:.4f}")

            total_loss += epoch_loss
            correct += epoch_correct
            total += epoch_total

        # Evaluate on validation set
        val_loss, val_accuracy = self._evaluate_model(criterion)

        # Get privacy spent if enabled
        privacy_spent = 0.0
        if manual_privacy:
            epsilon, _ = manual_privacy.get_privacy_spent()
            privacy_spent = epsilon
            logger.info(f"🔒 {self.hospital_name}: Privacy spent - ε={epsilon:.4f}")

        # Return updated parameters and metrics
        updated_params = self.get_parameters(config)

        metrics = {
            "train_loss": total_loss / max(effective_epochs * len(train_loader), 1),
            "train_accuracy": correct / total,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "privacy_spent": privacy_spent
        }

        logger.info(f"🎯 {self.hospital_name}: Training completed - Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

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
        logger.info(f"🔍 {self.hospital_name}: Starting evaluation...")

        if self.use_privacy:
            pass  # model already loaded with global weights

        self.set_parameters(parameters)

        # Evaluate
        criterion = nn.CrossEntropyLoss()
        val_loss, val_accuracy = self._evaluate_model(criterion)

        metrics = {
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
        }

        logger.info(f"📈 {self.hospital_name}: Evaluation completed - Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}")

        return val_loss, len(self.val_loader.dataset), metrics


class FlowerClient(fl.client.NumPyClient):
    """Flower client wrapper"""

    def __init__(self, client_id: int, hospital_id: str = None, model_type="mlp", use_privacy=True):
        self.diabetes_client = DiabetesPredictionClient(client_id, hospital_id, model_type, use_privacy)

    def get_parameters(self, config: Dict[str, fl.common.Scalar]) -> List[np.ndarray]:
        return self.diabetes_client.get_parameters(config)

    def fit(self, parameters: List[np.ndarray], config: Dict[str, fl.common.Scalar]) -> Tuple[List[np.ndarray], int, Dict[str, fl.common.Scalar]]:
        return self.diabetes_client.fit(parameters, config)

    def evaluate(self, parameters: List[np.ndarray], config: Dict[str, fl.common.Scalar]) -> Tuple[float, int, Dict[str, fl.common.Scalar]]:
        return self.diabetes_client.evaluate(parameters, config)


def start_client(client_id: int, hospital_id: str = None, model_type="mlp", use_privacy=True, server_address="localhost:8080"):
    """Start a Flower client."""
    logger.info(f"🚀 Starting client {client_id}...")

    # Create client
    client = FlowerClient(client_id, hospital_id, model_type, use_privacy)

    # Start client (Flower 1.8+ prefers Client over NumPyClient)
    fl.client.start_client(
        server_address=server_address,
        client=client.to_client(),
    )


def main():
    parser = argparse.ArgumentParser(description="Start Flower Diabetes Prediction Client")
    parser.add_argument("client_id", type=int, help="Client ID (integer)")
    parser.add_argument("--hospital", type=str, help="Hospital ID (e.g., hospital_1, hospital_2)")
    parser.add_argument("--model", type=str, default="mlp", choices=["mlp", "cnn"],
                       help="Model type to use")
    parser.add_argument("--no-privacy", action="store_true", help="Disable differential privacy")
    parser.add_argument("--server", type=str, default="localhost:8080",
                       help="Server address")

    args = parser.parse_args()

    # Use hospital_1, hospital_2, hospital_3 as defaults if not specified
    hospital_id = args.hospital or f"hospital_{args.client_id}"

    start_client(args.client_id, hospital_id, args.model, not args.no_privacy, args.server)


if __name__ == "__main__":
    main()
