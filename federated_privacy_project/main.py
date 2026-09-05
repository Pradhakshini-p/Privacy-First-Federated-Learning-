import argparse
import logging
import os
from typing import Dict, List

import torch
from sklearn.model_selection import train_test_split

from server import FederatedServer, build_clients
from utils import (
    load_digits_dataset,
    create_dataloader,
    plot_metrics,
    set_seed,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_test_loader(test_ratio: float = 0.1, batch_size: int = 64, reshape_for_cnn: bool = False):
    """Build a global test loader from the digits dataset."""
    data = load_digits_dataset()
    _, test_df = train_test_split(
        data, test_size=test_ratio, stratify=data["target"], random_state=42
    )
    test_x = test_df.drop(columns=["target"]).values.astype(np.float32)
    test_y = test_df["target"].values.astype(np.int64)
    return create_dataloader(
        test_x,
        test_y,
        batch_size=batch_size,
        shuffle=False,
        reshape_for_cnn=reshape_for_cnn,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the Privacy-Preserving Federated Learning simulation."
    )
    parser.add_argument("--rounds", type=int, default=5, help="Number of federation rounds.")
    parser.add_argument("--clients", type=int, default=5, help="Number of simulated clients.")
    parser.add_argument("--epochs", type=int, default=1, help="Local epochs per client.")
    parser.add_argument("--batch-size", type=int, default=32, help="Local batch size.")
    parser.add_argument("--model", type=str, choices=["mlp", "cnn"], default="mlp", help="Model architecture.")
    parser.add_argument("--dp-epsilon", type=float, default=2.0, help="Differential privacy epsilon value.")
    parser.add_argument("--enable-api", action="store_true", help="Start optional Flask API during simulation.")
    parser.add_argument("--api-port", type=int, default=5000, help="Flask API port.")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(42)

    logger.info("Starting Privacy-Preserving Federated Learning System")
    logger.info("Using model: %s", args.model)
    logger.info("Simulated clients: %s", args.clients)
    logger.info("Federated rounds: %s", args.rounds)

    # Build clients and server.
    clients = build_clients(n_clients=args.clients, model_type=args.model, dp_epsilon=args.dp_epsilon)
    server = FederatedServer(model_type=args.model)

    if args.enable_api:
        server.run_api(port=args.api_port)

    # Build a global test loader by holding out a portion of the dataset.
    test_loader = build_test_loader(
        test_ratio=0.1,
        batch_size=args.batch_size,
        reshape_for_cnn=args.model == "cnn",
    )

    # Run federated learning.
    history = server.run_fedavg(clients, test_loader, rounds=args.rounds, local_epochs=args.epochs)

    # Save metrics plot and summary.
    os.makedirs("plots", exist_ok=True)
    plot_metrics(history, output_path="plots/global_metrics.png")

    logger.info("Federated learning complete. Metrics saved to plots/global_metrics.png")
    logger.info("Summary:")
    for r, acc, loss in zip(history["round"], history["accuracy"], history["loss"]):
        logger.info("Round %d -> accuracy=%.4f loss=%.4f", r, acc, loss)

    logger.info("The system protects privacy by training locally, adding differential privacy noise, and simulating secure aggregation.")


if __name__ == "__main__":
    main()
