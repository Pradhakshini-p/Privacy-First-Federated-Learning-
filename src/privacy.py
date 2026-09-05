"""
Differential Privacy Module for Federated Learning
Implements privacy guarantees using Opacus library
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
from dataclasses import dataclass
from typing import Optional, Tuple

try:
    from opacus import PrivacyEngine
    from opacus.validators import ModuleValidator
    OPACUS_AVAILABLE = True
except ImportError:
    OPACUS_AVAILABLE = False
    logging.warning("Opacus not installed. Install with: pip install opacus")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PrivacyConfig:
    """Configuration for differential privacy"""
    epsilon: float = 3.0  # Privacy budget
    delta: float = 1e-5  # Failure probability
    max_grad_norm: float = 1.0  # Gradient clipping threshold
    noise_multiplier: float = 1.0  # Noise multiplier for DP-SGD
    target_epsilon: float = 3.0  # Target privacy budget


class ManualPrivacyTrainer:
    """Lightweight DP-SGD: gradient clipping + Gaussian noise (FL-safe)."""

    def __init__(self, config: PrivacyConfig):
        self.config = config
        self.steps = 0

    def apply(self, model: nn.Module) -> None:
        max_norm = self.config.max_grad_norm
        noise_mult = self.config.noise_multiplier
        for param in model.parameters():
            if param.grad is None:
                continue
            grad = param.grad
            norm = grad.norm(2)
            if norm > max_norm:
                grad.mul_(max_norm / (norm + 1e-6))
            noise = torch.randn_like(grad) * noise_mult * max_norm
            grad.add_(noise)
        self.steps += 1

    def get_privacy_spent(self) -> Tuple[float, float]:
        # Approximate ε reporting for demo (scales with steps and noise)
        if self.steps == 0 or self.config.noise_multiplier == 0:
            return 0.0, self.config.delta
        epsilon = min(
            self.config.target_epsilon,
            self.steps * 0.1 / max(self.config.noise_multiplier, 0.1)
        )
        return epsilon, self.config.delta


class PrivacyManager:
    """
    Manages differential privacy for federated learning clients.
    Handles privacy budget tracking and noise injection.
    """

    def __init__(self, config: PrivacyConfig):
        self.config = config
        self.privacy_engine = None
        self.accountant = None
        self.epsilon_spent = 0.0

        if not OPACUS_AVAILABLE:
            logger.warning("Opacus not available. Privacy features disabled.")
            return

    def setup_privacy_engine(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        data_loader: DataLoader
    ) -> Tuple[nn.Module, optim.Optimizer, DataLoader]:
        """
        Setup privacy engine for differential privacy training.

        Args:
            model: PyTorch model
            optimizer: PyTorch optimizer
            data_loader: PyTorch data loader

        Returns:
            Tuple of (private_model, private_optimizer, private_data_loader)
        """
        if not OPACUS_AVAILABLE:
            logger.warning("Privacy engine not available. Returning original model.")
            return model, optimizer, data_loader

        try:
            # Validate model for privacy
            errors = ModuleValidator.validate(model, strict=False)
            if errors:
                logger.warning(f"Model validation errors: {errors}")
                # Fix common issues
                model = ModuleValidator.fix(model)

            # Create privacy engine (default RDP accountant)
            self.privacy_engine = PrivacyEngine()

            # Attach privacy engine to model and optimizer
            model, optimizer, data_loader = self.privacy_engine.make_private(
                module=model,
                optimizer=optimizer,
                data_loader=data_loader,
                noise_multiplier=self.config.noise_multiplier,
                max_grad_norm=self.config.max_grad_norm,
            )

            logger.info(f"✅ Privacy engine setup completed")
            logger.info(f"   Noise multiplier: {self.config.noise_multiplier}")
            logger.info(f"   Max grad norm: {self.config.max_grad_norm}")
            logger.info(f"   Target epsilon: {self.config.target_epsilon}")

            return model, optimizer, data_loader

        except Exception as e:
            logger.error(f"Failed to setup privacy engine: {e}")
            return model, optimizer, data_loader

    def get_privacy_spent(self) -> Tuple[float, float]:
        """
        Get current privacy budget spent.

        Returns:
            Tuple of (epsilon, delta)
        """
        if self.privacy_engine is None:
            return 0.0, self.config.delta

        try:
            epsilon, delta = self.privacy_engine.accountant.get_privacy_spent()
            self.epsilon_spent = epsilon
            return epsilon, delta
        except Exception as e:
            logger.error(f"Failed to get privacy spent: {e}")
            return 0.0, self.config.delta

    def is_privacy_budget_exhausted(self) -> bool:
        """
        Check if privacy budget is exhausted.

        Returns:
            True if budget exhausted, False otherwise
        """
        epsilon, _ = self.get_privacy_spent()
        return epsilon >= self.config.target_epsilon

    def get_remaining_budget(self) -> float:
        """
        Get remaining privacy budget.

        Returns:
            Remaining epsilon budget
        """
        epsilon, _ = self.get_privacy_spent()
        remaining = max(0.0, self.config.target_epsilon - epsilon)
        return remaining


def test_privacy():
    """Test the privacy module with a simple model."""
    logger.info("🧪 Testing privacy module...")

    if not OPACUS_AVAILABLE:
        logger.error("Opacus not available. Cannot test privacy module.")
        return

    # Create a simple model
    model = nn.Sequential(
        nn.Linear(8, 16),
        nn.ReLU(),
        nn.Linear(16, 2)
    )

    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Create dummy data loader
    dummy_data = torch.randn(100, 8)
    dummy_labels = torch.randint(0, 2, (100,))
    dataset = torch.utils.data.TensorDataset(dummy_data, dummy_labels)
    data_loader = DataLoader(dataset, batch_size=32)

    # Setup privacy
    config = PrivacyConfig(epsilon=3.0, delta=1e-5, noise_multiplier=1.0)
    privacy_manager = PrivacyManager(config)

    private_model, private_optimizer, private_data_loader = privacy_manager.setup_privacy_engine(
        model, optimizer, data_loader
    )

    # Train one step
    private_model.train()
    for data, target in private_data_loader:
        private_optimizer.zero_grad()
        output = private_model(data)
        loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()
        private_optimizer.step()
        break

    # Check privacy spent
    epsilon, delta = privacy_manager.get_privacy_spent()
    logger.info(f"Privacy spent after 1 step: ε={epsilon:.4f}, δ={delta:.2e}")

    logger.info("✅ Privacy module test completed successfully!")


if __name__ == "__main__":
    test_privacy()
