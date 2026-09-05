#!/usr/bin/env python3
"""
Centralized Training for Diabetes Prediction
Trains model on all data combined (baseline for comparison with federated learning)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
import json
import os
from datetime import datetime
import argparse

# Import our custom modules
from src.model import create_model
from src.data import DiabetesDataLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CentralizedTrainer:
    """Centralized training for diabetes prediction"""

    def __init__(self, model_type="mlp", input_dim=8):
        self.model_type = model_type
        self.input_dim = input_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create directories
        os.makedirs("models", exist_ok=True)
        os.makedirs("results", exist_ok=True)

        # Training history
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []

    def load_data(self):
        """Load and prepare centralized dataset"""
        logger.info("📊 Loading centralized diabetes dataset...")

        data_loader = DiabetesDataLoader()
        df = data_loader.load_diabetes_data()
        X, y = data_loader.preprocess_data(df)

        # Split into train/val/test (80/10/10)
        from sklearn.model_selection import train_test_split
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
        )

        logger.info(f"Train samples: {len(X_train)}")
        logger.info(f"Val samples: {len(X_val)}")
        logger.info(f"Test samples: {len(X_test)}")

        # Create data loaders
        def create_loader(X, y, shuffle=True):
            X_tensor = torch.FloatTensor(X.values if hasattr(X, 'values') else X)
            y_tensor = torch.LongTensor(y.values if hasattr(y, 'values') else y)
            dataset = TensorDataset(X_tensor, y_tensor)
            return DataLoader(dataset, batch_size=32, shuffle=shuffle)

        train_loader = create_loader(X_train, y_train, shuffle=True)
        val_loader = create_loader(X_val, y_val, shuffle=False)
        test_loader = create_loader(X_test, y_test, shuffle=False)

        return train_loader, val_loader, test_loader

    def train(self, num_epochs=20, learning_rate=0.001):
        """Train the model on centralized data"""
        logger.info("=" * 60)
        logger.info("🎯 Starting Centralized Training")
        logger.info("=" * 60)

        # Load data
        train_loader, val_loader, test_loader = self.load_data()

        # Initialize model
        model = create_model(model_type=self.model_type, input_dim=self.input_dim)
        model.to(self.device)

        # Setup training
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        best_val_accuracy = 0.0
        training_start_time = datetime.now()

        # Training loop
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)

                optimizer.zero_grad()
                outputs = model(data)
                loss = criterion(outputs, target)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                train_total += target.size(0)
                train_correct += (predicted == target).sum().item()

            train_accuracy = train_correct / train_total
            avg_train_loss = train_loss / len(train_loader)

            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    outputs = model(data)
                    loss = criterion(outputs, target)
                    val_loss += loss.item()

                    _, predicted = torch.max(outputs.data, 1)
                    val_total += target.size(0)
                    val_correct += (predicted == target).sum().item()

            val_accuracy = val_correct / val_total
            avg_val_loss = val_loss / len(val_loader)

            # Record metrics
            self.train_losses.append(avg_train_loss)
            self.train_accuracies.append(train_accuracy)
            self.val_losses.append(avg_val_loss)
            self.val_accuracies.append(val_accuracy)

            # Save best model
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                torch.save(model.state_dict(), "models/centralized_model.pth")

            logger.info(f"Epoch {epoch+1}/{num_epochs} - "
                       f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f} | "
                       f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

        training_end_time = datetime.now()
        training_time = (training_end_time - training_start_time).total_seconds()

        logger.info("=" * 60)
        logger.info(f"✅ Training completed in {training_time:.2f} seconds")
        logger.info(f"🏆 Best validation accuracy: {best_val_accuracy:.4f}")
        logger.info("=" * 60)

        # Final evaluation on test set
        model.load_state_dict(torch.load("models/centralized_model.pth"))
        test_loss, test_accuracy = self.evaluate(model, test_loader)
        logger.info(f"📊 Final test accuracy: {test_accuracy:.4f}")

        # Save results
        results = {
            "model_type": self.model_type,
            "num_epochs": num_epochs,
            "training_time_seconds": training_time,
            "best_val_accuracy": best_val_accuracy,
            "test_accuracy": test_accuracy,
            "train_losses": self.train_losses,
            "train_accuracies": self.train_accuracies,
            "val_losses": self.val_losses,
            "val_accuracies": self.val_accuracies,
            "training_start": training_start_time.isoformat(),
            "training_end": training_end_time.isoformat()
        }

        with open("results/centralized_results.json", "w") as f:
            json.dump(results, f, indent=2)

        logger.info("✅ Results saved to results/centralized_results.json")

        # Generate plots
        self.plot_training_curves()

        return results

    def evaluate(self, model, test_loader):
        """Evaluate model on test set"""
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                outputs = model(data)
                loss = criterion(outputs, target)
                test_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        accuracy = correct / total
        avg_loss = test_loss / len(test_loader)

        return avg_loss, accuracy

    def plot_training_curves(self):
        """Generate training curves visualization"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Loss plot
        axes[0].plot(self.train_losses, label='Train Loss', marker='o')
        axes[0].plot(self.val_losses, label='Val Loss', marker='s')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True)

        # Accuracy plot
        axes[1].plot(self.train_accuracies, label='Train Accuracy', marker='o')
        axes[1].plot(self.val_accuracies, label='Val Accuracy', marker='s')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Training and Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()
        plt.savefig("results/centralized_training_curves.png", dpi=300, bbox_inches='tight')
        logger.info("✅ Training curves saved to results/centralized_training_curves.png")
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Train centralized model for diabetes prediction")
    parser.add_argument("--model", type=str, default="mlp", choices=["mlp", "cnn"],
                       help="Model type to use")
    parser.add_argument("--epochs", type=int, default=20,
                       help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001,
                       help="Learning rate")

    args = parser.parse_args()

    trainer = CentralizedTrainer(model_type=args.model)
    results = trainer.train(num_epochs=args.epochs, learning_rate=args.lr)

    logger.info("\n" + "=" * 60)
    logger.info("📊 CENTRALIZED TRAINING SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Model: {args.model.upper()}")
    logger.info(f"Training Time: {results['training_time_seconds']:.2f} seconds")
    logger.info(f"Best Val Accuracy: {results['best_val_accuracy']:.4f}")
    logger.info(f"Test Accuracy: {results['test_accuracy']:.4f}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
