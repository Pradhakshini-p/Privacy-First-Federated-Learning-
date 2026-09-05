#!/usr/bin/env python3
"""
Evaluation and Comparison Script
Compares federated learning vs centralized training performance
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import json
import os
import logging
from datetime import datetime

# Import our custom modules
from src.model import create_model
from src.data import DiabetesDataLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluate and compare federated vs centralized models"""

    def __init__(self, model_type="mlp", input_dim=8):
        self.model_type = model_type
        self.input_dim = input_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create directories
        os.makedirs("results", exist_ok=True)

    def load_models(self):
        """Load federated and centralized models"""
        logger.info("📂 Loading models...")

        # Load centralized model
        centralized_model = create_model(model_type=self.model_type, input_dim=self.input_dim)
        centralized_path = "models/centralized_model.pth"
        if os.path.exists(centralized_path):
            centralized_model.load_state_dict(torch.load(centralized_path, map_location=self.device))
            centralized_model.to(self.device)
            logger.info("✅ Centralized model loaded")
        else:
            logger.warning(f"⚠️ Centralized model not found at {centralized_path}")
            centralized_model = None

        # Load federated model
        federated_model = create_model(model_type=self.model_type, input_dim=self.input_dim)
        federated_path = "models/global_model.pth"
        if os.path.exists(federated_path):
            federated_model.load_state_dict(torch.load(federated_path, map_location=self.device))
            federated_model.to(self.device)
            logger.info("✅ Federated model loaded")
        else:
            logger.warning(f"⚠️ Federated model not found at {federated_path}")
            federated_model = None

        return centralized_model, federated_model

    def load_test_data(self):
        """Load test dataset"""
        logger.info("📊 Loading test data...")

        data_loader = DiabetesDataLoader()
        df = data_loader.load_diabetes_data()
        X, y = data_loader.preprocess_data(df)

        # Create silos to get global test set
        silos = data_loader.create_data_silos(X, y, n_silos=3)
        test_data = silos['global_test']

        # Convert to tensors
        X_test = torch.FloatTensor(test_data['X_test'].values if hasattr(test_data['X_test'], 'values') else test_data['X_test'])
        y_test = torch.LongTensor(test_data['y_test'].values if hasattr(test_data['y_test'], 'values') else test_data['y_test'])

        logger.info(f"Test samples: {len(X_test)}")

        return X_test, y_test

    def evaluate_model(self, model, X_test, y_test, model_name):
        """Evaluate a single model"""
        if model is None:
            logger.warning(f"⚠️ {model_name} is None, skipping evaluation")
            return None

        model.eval()
        with torch.no_grad():
            X_test_device = X_test.to(self.device)
            outputs = model(X_test_device)
            probabilities = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(outputs, dim=1)

        # Move to CPU for sklearn metrics
        predictions_np = predictions.cpu().numpy()
        probabilities_np = probabilities.cpu().numpy()
        y_test_np = y_test.numpy()

        # Calculate metrics
        metrics = {
            "accuracy": accuracy_score(y_test_np, predictions_np),
            "precision": precision_score(y_test_np, predictions_np, average='binary'),
            "recall": recall_score(y_test_np, predictions_np, average='binary'),
            "f1_score": f1_score(y_test_np, predictions_np, average='binary'),
            "roc_auc": roc_auc_score(y_test_np, probabilities_np[:, 1])
        }

        # Confusion matrix
        cm = confusion_matrix(y_test_np, predictions_np)

        logger.info(f"📊 {model_name} Results:")
        logger.info(f"   Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"   Precision: {metrics['precision']:.4f}")
        logger.info(f"   Recall: {metrics['recall']:.4f}")
        logger.info(f"   F1-Score: {metrics['f1_score']:.4f}")
        logger.info(f"   ROC-AUC: {metrics['roc_auc']:.4f}")

        return metrics, cm

    def generate_comparison_plots(self, centralized_metrics, federated_metrics, centralized_cm, federated_cm):
        """Generate comparison visualizations"""
        logger.info("📈 Generating comparison plots...")

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        # Metrics comparison bar chart
        metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
        centralized_values = [
            centralized_metrics['accuracy'],
            centralized_metrics['precision'],
            centralized_metrics['recall'],
            centralized_metrics['f1_score'],
            centralized_metrics['roc_auc']
        ]
        federated_values = [
            federated_metrics['accuracy'],
            federated_metrics['precision'],
            federated_metrics['recall'],
            federated_metrics['f1_score'],
            federated_metrics['roc_auc']
        ]

        x = np.arange(len(metrics_names))
        width = 0.35

        axes[0, 0].bar(x - width/2, centralized_values, width, label='Centralized', color='steelblue')
        axes[0, 0].bar(x + width/2, federated_values, width, label='Federated', color='coral')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].set_title('Performance Metrics Comparison')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(metrics_names, rotation=45, ha='right')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_ylim([0, 1])

        # Confusion Matrix - Centralized
        sns.heatmap(centralized_cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 1])
        axes[0, 1].set_title('Confusion Matrix - Centralized')
        axes[0, 1].set_ylabel('True Label')
        axes[0, 1].set_xlabel('Predicted Label')

        # Confusion Matrix - Federated
        sns.heatmap(federated_cm, annot=True, fmt='d', cmap='Reds', ax=axes[1, 0])
        axes[1, 0].set_title('Confusion Matrix - Federated')
        axes[1, 0].set_ylabel('True Label')
        axes[1, 0].set_xlabel('Predicted Label')

        # Performance difference
        differences = np.array(federated_values) - np.array(centralized_values)
        colors = ['green' if d >= 0 else 'red' for d in differences]
        axes[1, 1].barh(metrics_names, differences, color=colors)
        axes[1, 1].set_xlabel('Difference (Federated - Centralized)')
        axes[1, 1].set_title('Performance Difference')
        axes[1, 1].axvline(x=0, color='black', linestyle='--', linewidth=0.5)
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("results/comparison_plots.png", dpi=300, bbox_inches='tight')
        logger.info("✅ Comparison plots saved to results/comparison_plots.png")
        plt.close()

    def load_training_history(self):
        """Load training history from federated and centralized training"""
        history = {
            "federated": None,
            "centralized": None
        }

        # Load federated training history
        if os.path.exists("results/training_results.json"):
            with open("results/training_results.json", "r") as f:
                history["federated"] = json.load(f)
            logger.info("✅ Federated training history loaded")

        # Load centralized training history
        if os.path.exists("results/centralized_results.json"):
            with open("results/centralized_results.json", "r") as f:
                history["centralized"] = json.load(f)
            logger.info("✅ Centralized training history loaded")

        return history

    def plot_training_comparison(self, history):
        """Plot training curves comparison"""
        if history["federated"] is None or history["centralized"] is None:
            logger.warning("⚠️ Missing training history, skipping training comparison plot")
            return

        logger.info("📈 Generating training comparison plot...")

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Accuracy comparison
        if "global_accuracy_history" in history["federated"]:
            fed_acc = history["federated"]["global_accuracy_history"]
            fed_rounds = list(range(1, len(fed_acc) + 1))
            axes[0].plot(fed_rounds, fed_acc, marker='o', label='Federated', color='coral')

        if "val_accuracies" in history["centralized"]:
            cent_acc = history["centralized"]["val_accuracies"]
            cent_epochs = list(range(1, len(cent_acc) + 1))
            axes[0].plot(cent_epochs, cent_acc, marker='s', label='Centralized', color='steelblue')

        axes[0].set_xlabel('Round/Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].set_title('Training Accuracy Comparison')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Loss comparison
        if "global_loss_history" in history["federated"]:
            fed_loss = history["federated"]["global_loss_history"]
            fed_rounds = list(range(1, len(fed_loss) + 1))
            axes[1].plot(fed_rounds, fed_loss, marker='o', label='Federated', color='coral')

        if "val_losses" in history["centralized"]:
            cent_loss = history["centralized"]["val_losses"]
            cent_epochs = list(range(1, len(cent_loss) + 1))
            axes[1].plot(cent_epochs, cent_loss, marker='s', label='Centralized', color='steelblue')

        axes[1].set_xlabel('Round/Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].set_title('Training Loss Comparison')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("results/training_comparison.png", dpi=300, bbox_inches='tight')
        logger.info("✅ Training comparison plot saved to results/training_comparison.png")
        plt.close()

    def generate_report(self, centralized_metrics, federated_metrics):
        """Generate comprehensive comparison report"""
        logger.info("📝 Generating comparison report...")

        report = {
            "evaluation_date": datetime.now().isoformat(),
            "model_type": self.model_type,
            "centralized_metrics": centralized_metrics,
            "federated_metrics": federated_metrics,
            "performance_difference": {
                metric: federated_metrics[metric] - centralized_metrics[metric]
                for metric in centralized_metrics.keys()
            }
        }

        # Save report
        with open("results/comparison_report.json", "w") as f:
            json.dump(report, f, indent=2)

        logger.info("✅ Comparison report saved to results/comparison_report.json")

        # Print summary
        logger.info("\n" + "=" * 60)
        logger.info("📊 COMPARISON SUMMARY")
        logger.info("=" * 60)
        for metric in centralized_metrics.keys():
            diff = federated_metrics[metric] - centralized_metrics[metric]
            symbol = "↑" if diff > 0 else "↓"
            logger.info(f"{metric.capitalize():15s}: Centralized={centralized_metrics[metric]:.4f}, "
                       f"Federated={federated_metrics[metric]:.4f} ({symbol}{abs(diff):.4f})")
        logger.info("=" * 60)

    def run_evaluation(self):
        """Run complete evaluation pipeline"""
        logger.info("=" * 60)
        logger.info("🎯 Starting Model Evaluation")
        logger.info("=" * 60)

        # Load models
        centralized_model, federated_model = self.load_models()

        if centralized_model is None and federated_model is None:
            logger.error("❌ No models found for evaluation")
            logger.info("Please run training first:")
            logger.info("  - Federated: python server.py (and clients)")
            logger.info("  - Centralized: python train_centralized.py")
            return

        # Load test data
        X_test, y_test = self.load_test_data()

        # Evaluate models
        centralized_metrics = None
        federated_metrics = None
        centralized_cm = None
        federated_cm = None

        if centralized_model is not None:
            centralized_metrics, centralized_cm = self.evaluate_model(
                centralized_model, X_test, y_test, "Centralized"
            )

        if federated_model is not None:
            federated_metrics, federated_cm = self.evaluate_model(
                federated_model, X_test, y_test, "Federated"
            )

        # Generate comparison plots if both models are available
        if centralized_metrics is not None and federated_metrics is not None:
            self.generate_comparison_plots(
                centralized_metrics, federated_metrics, centralized_cm, federated_cm
            )
            self.generate_report(centralized_metrics, federated_metrics)

        # Load and plot training history
        history = self.load_training_history()
        self.plot_training_comparison(history)

        logger.info("=" * 60)
        logger.info("✅ Evaluation completed successfully!")
        logger.info("=" * 60)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate and compare federated vs centralized models")
    parser.add_argument("--model", type=str, default="mlp", choices=["mlp", "cnn"],
                       help="Model type to evaluate")

    args = parser.parse_args()

    evaluator = ModelEvaluator(model_type=args.model)
    evaluator.run_evaluation()


if __name__ == "__main__":
    main()
