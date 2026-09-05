import logging
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import create_model
from utils import add_dp_noise, create_secure_mask, create_dataloader, model_to_weights, weights_to_model


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class FederatedClient:
    """Simulated federated learning client."""

    def __init__(
        self,
        client_id: int,
        train_x: np.ndarray,
        train_y: np.ndarray,
        val_x: np.ndarray,
        val_y: np.ndarray,
        model_type: str = "mlp",
        dp_epsilon: float = 2.0,
        mask_seed: int = 100,
    ):
        self.client_id = client_id
        reshape_for_cnn = model_type.lower() == "cnn"
        self.train_loader = create_dataloader(train_x, train_y, batch_size=32, shuffle=True, reshape_for_cnn=reshape_for_cnn)
        self.val_loader = create_dataloader(val_x, val_y, batch_size=64, shuffle=False, reshape_for_cnn=reshape_for_cnn)
        self.model = create_model(model_type=model_type)
        self.dp_epsilon = dp_epsilon
        self.mask_seed = mask_seed
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def set_weights(self, weights: List[np.ndarray]) -> None:
        """Load global model weights into the local model."""
        weights_to_model(self.model, weights)

    def get_weights(self) -> List[np.ndarray]:
        """Return local model parameters as NumPy arrays."""
        return model_to_weights(self.model)

    def train_local_model(
        self,
        global_weights: List[np.ndarray],
        epochs: int = 1,
        learning_rate: float = 0.01,
    ) -> Tuple[List[np.ndarray], List[np.ndarray], int, Dict[str, float]]:
        """Perform local training and return masked updates for secure aggregation."""
        self.set_weights(global_weights)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for epoch in range(epochs):
            for batch_idx, (x_batch, y_batch) in enumerate(self.train_loader):
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(x_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * x_batch.size(0)
                preds = outputs.argmax(dim=1)
                total_correct += (preds == y_batch).sum().item()
                total_samples += x_batch.size(0)

        train_accuracy = total_correct / total_samples if total_samples else 0.0
        train_loss = total_loss / total_samples if total_samples else 0.0

        local_weights = self.get_weights()
        dp_weights = add_dp_noise(local_weights, epsilon=self.dp_epsilon)
        mask = create_secure_mask(dp_weights, seed=self.mask_seed + self.client_id)
        masked_weights = [w + m for w, m in zip(dp_weights, mask)]

        logger.info(
            f"Client {self.client_id} trained locally: samples={total_samples}, "
            f"acc={train_accuracy:.4f}, loss={train_loss:.4f}, dp_epsilon={self.dp_epsilon}"
        )

        return masked_weights, mask, total_samples, {
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
        }

    def evaluate(self, weights: List[np.ndarray]) -> Dict[str, float]:
        """Evaluate a set of weights on the local validation set."""
        self.set_weights(weights)
        self.model.eval()

        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for x_batch, y_batch in self.val_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                outputs = self.model(x_batch)
                loss = criterion(outputs, y_batch)
                total_loss += loss.item() * x_batch.size(0)
                preds = outputs.argmax(dim=1)
                total_correct += (preds == y_batch).sum().item()
                total_samples += x_batch.size(0)

        return {
            "val_loss": total_loss / total_samples if total_samples else 0.0,
            "val_accuracy": total_correct / total_samples if total_samples else 0.0,
        }
