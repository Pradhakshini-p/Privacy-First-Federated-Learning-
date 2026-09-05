import random
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_digits_dataset() -> pd.DataFrame:
    """Load the digits dataset and return a Pandas DataFrame."""
    digits = load_digits()
    data = pd.DataFrame(digits.data)
    data["target"] = digits.target
    return data


def create_non_iid_silos(
    data: pd.DataFrame,
    n_clients: int = 5,
    test_size: float = 0.1,
    val_size: float = 0.1,
) -> List[Dict[str, np.ndarray]]:
    """Split data into non-IID client silos and a global test set."""
    client_silos = []
    all_classes = sorted(data["target"].unique())
    np.random.shuffle(all_classes)

    # Reserve a global test set first.
    train_data, test_data = train_test_split(
        data, test_size=test_size, stratify=data["target"], random_state=42
    )

    for cid in range(n_clients):
        # Give each client one or two dominant digit classes.
        classes_for_client = all_classes[cid * 2 : cid * 2 + 2]
        if len(classes_for_client) == 0:
            classes_for_client = [all_classes[-1]]

        client_data = train_data[train_data["target"].isin(classes_for_client)].copy()

        # Add a few samples from other classes to make the distribution slightly mixed.
        extra = train_data[~train_data["target"].isin(classes_for_client)].sample(
            n=min(20, len(train_data) - len(client_data)), random_state=42
        )
        client_data = pd.concat([client_data, extra]).sample(frac=1, random_state=42)

        # Split each client's dataset into training and validation sets.
        train_split, val_split = train_test_split(
            client_data,
            test_size=val_size,
            stratify=client_data["target"],
            random_state=42,
        )

        client_silos.append(
            {
                "train_x": train_split.drop(columns=["target"]).values.astype(np.float32),
                "train_y": train_split["target"].values.astype(np.int64),
                "val_x": val_split.drop(columns=["target"]).values.astype(np.float32),
                "val_y": val_split["target"].values.astype(np.int64),
                "classes": classes_for_client,
            }
        )

    return client_silos


def create_dataloader(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int = 32,
    shuffle: bool = True,
    reshape_for_cnn: bool = False,
) -> DataLoader:
    """Create a PyTorch DataLoader from NumPy arrays."""
    if reshape_for_cnn:
        x = x.reshape(-1, 1, 8, 8).astype(np.float32)
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def model_to_weights(model: torch.nn.Module) -> List[np.ndarray]:
    """Convert a PyTorch model's parameters to a list of NumPy arrays."""
    return [param.detach().cpu().numpy() for param in model.state_dict().values()]


def weights_to_model(model: torch.nn.Module, weights: List[np.ndarray]) -> None:
    """Load NumPy weight arrays into a PyTorch model."""
    state_dict = model.state_dict().copy()
    for key, weight in zip(state_dict.keys(), weights):
        state_dict[key] = torch.tensor(weight)
    model.load_state_dict(state_dict)


def add_dp_noise(weights: List[np.ndarray], epsilon: float = 1.0, sensitivity: float = 1.0) -> List[np.ndarray]:
    """Add Gaussian noise to model weights for differential privacy simulation."""
    noise_scale = sensitivity / epsilon
    noisy_weights = []
    for weight in weights:
        noise = np.random.normal(loc=0.0, scale=noise_scale, size=weight.shape).astype(np.float32)
        noisy_weights.append(weight + noise)
    return noisy_weights


def create_secure_mask(weights: List[np.ndarray], seed: int) -> List[np.ndarray]:
    """Create a reproducible random mask for a weight list."""
    rng = np.random.RandomState(seed)
    return [rng.normal(loc=0.0, scale=0.05, size=w.shape).astype(np.float32) for w in weights]


def aggregate_masked_updates(
    masked_updates: List[List[np.ndarray]],
    masks: List[List[np.ndarray]],
    sizes: List[int],
) -> List[np.ndarray]:
    """Simulate secure aggregation and compute weighted average of client updates."""
    # Recover true updates by subtracting masks from masked updates.
    adjusted_updates = []
    total_size = sum(sizes)

    for client_idx in range(len(masked_updates)):
        recovered = [u - m for u, m in zip(masked_updates[client_idx], masks[client_idx])]
        adjusted_updates.append(recovered)

    weighted_sum = [np.zeros_like(param) for param in adjusted_updates[0]]
    for client_idx, client_update in enumerate(adjusted_updates):
        weight = sizes[client_idx] / float(total_size)
        for idx, param in enumerate(client_update):
            weighted_sum[idx] += param * weight

    return weighted_sum


def plot_metrics(history: Dict[str, List[float]], output_path: str = "metrics.png") -> None:
    """Plot accuracy and loss across federated rounds."""
    rounds = list(range(1, len(history["accuracy"]) + 1))

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(rounds, history["accuracy"], marker="o", color="#1f77b4")
    plt.title("Global Model Accuracy vs Rounds")
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(rounds, history["loss"], marker="o", color="#ff7f0e")
    plt.title("Global Model Loss vs Rounds")
    plt.xlabel("Round")
    plt.ylabel("Loss")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
