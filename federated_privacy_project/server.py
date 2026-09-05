import logging
import threading
from typing import Dict, List, Optional

import numpy as np
import torch

try:
    from flask import Flask, jsonify, request
except ImportError:  # pragma: no cover
    Flask = None
    jsonify = None
    request = None

from client import FederatedClient
from model import create_model
from utils import (
    aggregate_masked_updates,
    load_digits_dataset,
    create_non_iid_silos,
    create_dataloader,
    model_to_weights,
    weights_to_model,
)


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class FederatedServer:
    """Central server for federated averaging."""

    def __init__(self, model_type: str = "mlp"):
        self.global_model = create_model(model_type=model_type)
        self.history = {"round": [], "accuracy": [], "loss": []}
        self.client_metrics = []
        self.flask_app: Optional[Flask] = None

    def get_global_weights(self) -> List[np.ndarray]:
        """Return the global model weights."""
        return model_to_weights(self.global_model)

    def set_global_weights(self, weights: List[np.ndarray]) -> None:
        """Load weights into the global model."""
        weights_to_model(self.global_model, weights)

    def aggregate_client_updates(
        self,
        masked_updates: List[List[np.ndarray]],
        masks: List[List[np.ndarray]],
        sizes: List[int],
    ) -> List[np.ndarray]:
        """Aggregate updates from clients using secure aggregation simulation."""
        return aggregate_masked_updates(masked_updates, masks, sizes)

    def evaluate_global_model(self, test_loader) -> Dict[str, float]:
        """Evaluate current global model on a test dataset."""
        self.global_model.eval()
        criterion = torch.nn.CrossEntropyLoss()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                outputs = self.global_model(x_batch)
                loss = criterion(outputs, y_batch)
                total_loss += loss.item() * x_batch.size(0)
                preds = outputs.argmax(dim=1)
                total_correct += (preds == y_batch).sum().item()
                total_samples += x_batch.size(0)

        return {
            "loss": total_loss / total_samples if total_samples else 0.0,
            "accuracy": total_correct / total_samples if total_samples else 0.0,
        }

    def run_fedavg(
        self,
        clients: List[FederatedClient],
        test_loader,
        rounds: int = 5,
        local_epochs: int = 1,
    ) -> Dict[str, List[float]]:
        """Run federated learning rounds and track global metrics."""
        global_weights = self.get_global_weights()

        for round_index in range(1, rounds + 1):
            masked_updates = []
            masks = []
            client_sizes = []

            logger.info("\n=== Starting federated round %d ===", round_index)

            for client in clients:
                masked_weights, mask, num_samples, metrics = client.train_local_model(
                    global_weights,
                    epochs=local_epochs,
                )
                masked_updates.append(masked_weights)
                masks.append(mask)
                client_sizes.append(num_samples)
                self.client_metrics.append({
                    "client_id": client.client_id,
                    "round": round_index,
                    **metrics,
                })

            aggregated_weights = self.aggregate_client_updates(masked_updates, masks, client_sizes)
            self.set_global_weights(aggregated_weights)

            evaluation = self.evaluate_global_model(test_loader)
            logger.info(
                "Global model after round %d - accuracy=%.4f, loss=%.4f",
                round_index,
                evaluation["accuracy"],
                evaluation["loss"],
            )

            self.history["round"].append(round_index)
            self.history["accuracy"].append(evaluation["accuracy"])
            self.history["loss"].append(evaluation["loss"])

        return self.history

    def create_api(self, host: str = "127.0.0.1", port: int = 5000) -> Flask:
        """Create a Flask API that exposes server endpoints."""
        app = Flask(__name__)

        @app.route("/status", methods=["GET"])
        def status():
            return jsonify({
                "rounds_completed": self.history["round"],
                "accuracy": self.history["accuracy"],
                "loss": self.history["loss"],
                "message": "Federated server is running",
            })

        @app.route("/weights", methods=["GET"])
        def weights():
            return jsonify({"shape": [w.shape for w in self.get_global_weights()]})

        @app.route("/submit", methods=["POST"])
        def submit():
            payload = request.get_json(force=True)
            return jsonify({"message": "Received payload", "payload_keys": list(payload.keys())})

        self.flask_app = app
        return app

    def run_api(self, host: str = "127.0.0.1", port: int = 5000) -> None:
        """Run the Flask API in a background thread."""
        if Flask is None:
            raise ImportError("Flask is required to start the API. Install it with 'pip install flask'.")

        if self.flask_app is None:
            self.create_api(host=host, port=port)

        thread = threading.Thread(
            target=self.flask_app.run,
            kwargs={"host": host, "port": port, "debug": False, "use_reloader": False},
            daemon=True,
        )
        thread.start()
        logger.info("Flask API running on http://%s:%d", host, port)


def build_clients(n_clients: int = 5, model_type: str = "mlp", dp_epsilon: float = 2.0) -> List[FederatedClient]:
    """Create client objects from non-IID dataset partitions."""
    data = load_digits_dataset()
    silos = create_non_iid_silos(data, n_clients=n_clients)

    clients = []
    for idx, silo in enumerate(silos):
        client = FederatedClient(
            client_id=idx + 1,
            train_x=silo["train_x"],
            train_y=silo["train_y"],
            val_x=silo["val_x"],
            val_y=silo["val_y"],
            model_type=model_type,
            dp_epsilon=dp_epsilon,
            mask_seed=1000,
        )
        clients.append(client)
    return clients
