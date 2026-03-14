#!/usr/bin/env python3
"""
Perfect Federated Learning Platform v2 - With Privacy Engine Integration
Includes real-time epsilon tracking and privacy budget monitoring
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

try:
    from config import (
        NUM_CLIENTS, ROUNDS, LEARNING_RATE, BATCH_SIZE, LOCAL_EPOCHS,
        EPSILON, DELTA, MAX_GRAD_NORM, NOISE_MULTIPLIER,
        DATA_CONFIG, CLIENT_CONFIGS, SERVER_HOST, SERVER_PORT
    )
    from logging_bridge import get_logger, log_training_round, log_client_performance
    from privacy_engine import create_privacy_engine, get_privacy_manager
    from model import create_federated_model
    from data import load_and_preprocess_data
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure all required modules are available in src/")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PrivacyFirstFederatedPlatform:
    """Privacy-first federated learning platform with epsilon tracking"""
    
    def __init__(self, mode: str = "server", client_id: int = 1, 
                 server_host: str = None, server_port: int = None):
        """
        Initialize the federated learning platform
        
        Args:
            mode: 'server' or 'client'
            client_id: Client ID (for client mode)
            server_host: Server host (for client mode)
            server_port: Server port (for client mode)
        """
        self.mode = mode
        self.client_id = client_id
        self.server_host = server_host or SERVER_HOST
        self.server_port = server_port or SERVER_PORT
        
        # Initialize components
        self.global_model = None
        self.client_models = {}
        self.privacy_engines = {}
        self.optimizers = {}
        
        # Data management
        self.client_data = {}
        self.data_loaders = {}
        
        # Training state
        self.current_round = 0
        self.training_history = []
        self.client_performance = {}
        
        # Logging
        self.fl_logger = get_logger()
        self.privacy_manager = get_privacy_manager()
        
        # Initialize based on mode
        if mode == "server":
            self._initialize_server()
        else:
            self._initialize_client()
    
    def _initialize_server(self):
        """Initialize server components"""
        logger.info("Initializing Federated Learning Server")
        
        # Create global model
        self.global_model = create_federated_model()
        logger.info("Global model created")
        
        # Load and prepare data for simulation
        self._prepare_server_data()
        
        logger.info(f"Server initialized on {self.server_host}:{self.server_port}")
        logger.info(f"Expected clients: {NUM_CLIENTS}, Training rounds: {ROUNDS}")
    
    def _initialize_client(self):
        """Initialize client components"""
        logger.info(f"Initializing Client {self.client_id}")
        
        # Create client model
        self.global_model = create_federated_model()
        
        # Load client data
        self._prepare_client_data()
        
        # Initialize privacy engine
        client_config = CLIENT_CONFIGS.get(f"client_{self.client_id}", CLIENT_CONFIGS["client_1"])
        
        if client_config.get("privacy_enabled", True):
            self.privacy_engines[self.client_id] = create_privacy_engine(
                model=self.global_model,
                client_id=f"client_{self.client_id}",
                target_epsilon=EPSILON,
                target_delta=DELTA,
                max_grad_norm=client_config.get("max_grad_norm", MAX_GRAD_NORM),
                noise_multiplier=client_config.get("noise_multiplier", NOISE_MULTIPLIER)
            )
            
            # Create optimizer and attach privacy engine
            optimizer = optim.SGD(
                self.global_model.parameters(),
                lr=LEARNING_RATE,
                momentum=0.9
            )
            
            self.privacy_engines[self.client_id].attach_to_optimizer(optimizer)
            self.optimizers[self.client_id] = optimizer
            
            logger.info(f"Privacy engine enabled for client {self.client_id}")
            logger.info(f"Noise multiplier: {client_config.get('noise_multiplier', NOISE_MULTIPLIER)}")
        else:
            # Standard optimizer without privacy
            self.optimizers[self.client_id] = optim.SGD(
                self.global_model.parameters(),
                lr=LEARNING_RATE,
                momentum=0.9
            )
            logger.info(f"Privacy engine disabled for client {self.client_id}")
        
        logger.info(f"Client {self.client_id} initialized with {len(self.client_data)} samples")
    
    def _prepare_server_data(self):
        """Prepare data for server simulation"""
        try:
            # Load full dataset
            X, y = load_and_preprocess_data(DATA_CONFIG["data_path"], DATA_CONFIG["target_column"])
            
            # Split data among clients
            client_data_size = len(X) // NUM_CLIENTS
            
            for i in range(1, NUM_CLIENTS + 1):
                start_idx = (i - 1) * client_data_size
                end_idx = start_idx + client_data_size if i < NUM_CLIENTS else len(X)
                
                client_X = X[start_idx:end_idx]
                client_y = y[start_idx:end_idx]
                
                # Create data loader
                dataset = torch.utils.data.TensorDataset(
                    torch.FloatTensor(client_X),
                    torch.LongTensor(client_y)
                )
                
                self.client_data[i] = {
                    'X': client_X,
                    'y': client_y,
                    'dataset': dataset,
                    'size': len(client_X)
                }
                
                logger.info(f"Client {i} data prepared: {len(client_X)} samples")
        
        except Exception as e:
            logger.error(f"Error preparing server data: {e}")
            raise
    
    def _prepare_client_data(self):
        """Prepare data for client"""
        try:
            # For simulation, load a subset of data
            X, y = load_and_preprocess_data(DATA_CONFIG["data_path"], DATA_CONFIG["target_column"])
            
            # Simulate client data partition
            client_data_size = len(X) // NUM_CLIENTS
            start_idx = (self.client_id - 1) * client_data_size
            end_idx = start_idx + client_data_size if self.client_id < NUM_CLIENTS else len(X)
            
            client_X = X[start_idx:end_idx]
            client_y = y[start_idx:end_idx]
            
            # Create data loader
            dataset = torch.utils.data.TensorDataset(
                torch.FloatTensor(client_X),
                torch.LongTensor(client_y)
            )
            
            self.client_data[self.client_id] = {
                'X': client_X,
                'y': client_y,
                'dataset': dataset,
                'size': len(client_X)
            }
            
            self.data_loaders[self.client_id] = torch.utils.data.DataLoader(
                dataset,
                batch_size=BATCH_SIZE,
                shuffle=True
            )
            
        except Exception as e:
            logger.error(f"Error preparing client data: {e}")
            raise
    
    def train_client(self, client_id: int, global_weights: Dict) -> Dict:
        """
        Train a single client
        
        Args:
            client_id: Client identifier
            global_weights: Global model weights
            
        Returns:
            Client training results
        """
        try:
            # Load global weights
            self.global_model.load_state_dict(global_weights)
            
            # Get client data
            client_info = self.client_data[client_id]
            data_loader = self.data_loaders.get(client_id)
            
            if data_loader is None:
                data_loader = torch.utils.data.DataLoader(
                    client_info['dataset'],
                    batch_size=BATCH_SIZE,
                    shuffle=True
                )
            
            # Training loop
            self.global_model.train()
            criterion = nn.CrossEntropyLoss()
            optimizer = self.optimizers[client_id]
            
            initial_loss = 0.0
            final_loss = 0.0
            correct_predictions = 0
            total_predictions = 0
            
            start_time = time.time()
            
            for epoch in range(LOCAL_EPOCHS):
                epoch_loss = 0.0
                epoch_correct = 0
                epoch_total = 0
                
                for batch_idx, (data, target) in enumerate(data_loader):
                    optimizer.zero_grad()
                    output = self.global_model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    _, predicted = torch.max(output.data, 1)
                    epoch_correct += (predicted == target).sum().item()
                    epoch_total += target.size(0)
                
                if epoch == 0:
                    initial_loss = epoch_loss / len(data_loader)
                if epoch == LOCAL_EPOCHS - 1:
                    final_loss = epoch_loss / len(data_loader)
                
                correct_predictions += epoch_correct
                total_predictions += epoch_total
            
            training_time = time.time() - start_time
            
            # Calculate metrics
            accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
            
            # Track privacy if enabled
            privacy_status = {}
            if client_id in self.privacy_engines:
                privacy_engine = self.privacy_engines[client_id]
                
                # Track privacy spent for this round
                privacy_available = privacy_engine.track_privacy_spent(
                    round_num=self.current_round,
                    data_size=len(client_info['X'])
                )
                
                privacy_status = privacy_engine.get_privacy_status()
                
                if not privacy_available:
                    logger.warning(f"Client {client_id} privacy budget exhausted!")
            
            # Get model weights
            client_weights = {k: v.cpu().numpy() for k, v in self.global_model.state_dict().items()}
            
            # Log client performance
            log_client_performance(
                round_num=self.current_round,
                client_id=f"client_{client_id}",
                local_accuracy=accuracy,
                local_loss=final_loss,
                data_size=len(client_info['X']),
                training_time=training_time,
                communication_cost=len(str(client_weights)) / 1024,  # KB
                status="active" if privacy_status.get("is_active", True) else "privacy_exhausted"
            )
            
            results = {
                'client_id': client_id,
                'weights': client_weights,
                'accuracy': accuracy,
                'loss': final_loss,
                'data_size': len(client_info['X']),
                'training_time': training_time,
                'privacy_status': privacy_status
            }
            
            logger.info(f"Client {client_id} training completed:")
            logger.info(f"  Accuracy: {accuracy:.4f}")
            logger.info(f"  Loss: {final_loss:.4f}")
            logger.info(f"  Data size: {len(client_info['X'])}")
            logger.info(f"  Training time: {training_time:.2f}s")
            
            if privacy_status:
                logger.info(f"  Privacy budget used: {privacy_status.get('privacy_budget_used', 0):.2%}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error training client {client_id}: {e}")
            raise
    
    def aggregate_models(self, client_results: List[Dict]) -> Dict:
        """
        Aggregate client models using FedAvg
        
        Args:
            client_results: List of client training results
            
        Returns:
            Aggregated global weights
        """
        try:
            if not client_results:
                raise ValueError("No client results to aggregate")
            
            # Calculate total data size
            total_data = sum(result['data_size'] for result in client_results)
            
            # Initialize aggregated weights
            aggregated_weights = {}
            
            # Get weight keys from first client
            first_client_weights = client_results[0]['weights']
            
            for key in first_client_weights.keys():
                # Weighted average of client weights
                weighted_sum = np.zeros_like(first_client_weights[key])
                
                for result in client_results:
                    weight = result['weights'][key]
                    data_weight = result['data_size'] / total_data
                    weighted_sum += weight * data_weight
                
                aggregated_weights[key] = weighted_sum
            
            # Calculate global metrics
            global_accuracy = np.mean([result['accuracy'] for result in client_results])
            global_loss = np.mean([result['loss'] for result in client_results])
            avg_client_accuracy = global_accuracy
            
            # Log training round
            log_training_round(
                round_num=self.current_round,
                global_accuracy=global_accuracy,
                global_loss=global_loss,
                avg_client_accuracy=avg_client_accuracy,
                communication_rounds=len(client_results)
            )
            
            logger.info(f"Round {self.current_round} aggregation completed:")
            logger.info(f"  Global accuracy: {global_accuracy:.4f}")
            logger.info(f"  Global loss: {global_loss:.4f}")
            logger.info(f"  Participating clients: {len(client_results)}")
            
            return aggregated_weights
            
        except Exception as e:
            logger.error(f"Error aggregating models: {e}")
            raise
    
    def run_server_training(self):
        """Run federated learning training on server"""
        logger.info("Starting federated learning training")
        
        # Get initial global weights
        global_weights = {k: v.cpu().numpy() for k, v in self.global_model.state_dict().items()}
        
        for round_num in range(1, ROUNDS + 1):
            self.current_round = round_num
            
            logger.info(f"=== Starting Round {round_num}/{ROUNDS} ===")
            
            # Train clients
            client_results = []
            active_clients = 0
            
            for client_id in range(1, NUM_CLIENTS + 1):
                try:
                    # Create client instance for training
                    client_platform = PrivacyFirstFederatedPlatform(
                        mode="client",
                        client_id=client_id
                    )
                    
                    # Train client with current global weights
                    result = client_platform.train_client(client_id, global_weights)
                    client_results.append(result)
                    
                    if result.get('privacy_status', {}).get('is_active', True):
                        active_clients += 1
                    
                except Exception as e:
                    logger.error(f"Error training client {client_id}: {e}")
                    continue
            
            if not client_results:
                logger.error("No clients successfully trained in this round")
                continue
            
            # Aggregate models
            global_weights = self.aggregate_models(client_results)
            
            # Update global model
            state_dict = {k: torch.FloatTensor(v) for k, v in global_weights.items()}
            self.global_model.load_state_dict(state_dict)
            
            # Log round summary
            logger.info(f"Round {round_num} completed:")
            logger.info(f"  Active clients: {active_clients}/{NUM_CLIENTS}")
            
            # Get global privacy status
            if hasattr(self, 'privacy_manager'):
                global_privacy = self.privacy_manager.get_global_privacy_status()
                logger.info(f"  Global privacy budget used: {global_privacy.get('global_budget_used', 0):.2%}")
            
            # Check if all clients exhausted privacy budget
            if active_clients == 0:
                logger.warning("All clients have exhausted privacy budget. Stopping training.")
                break
        
        logger.info("Federated learning training completed")
        
        # Final evaluation
        self._evaluate_global_model()
    
    def _evaluate_global_model(self):
        """Evaluate the final global model"""
        try:
            logger.info("Evaluating final global model")
            
            self.global_model.eval()
            
            # Evaluate on all client data
            total_correct = 0
            total_samples = 0
            
            with torch.no_grad():
                for client_id in range(1, NUM_CLIENTS + 1):
                    if client_id in self.client_data:
                        client_info = self.client_data[client_id]
                        data_loader = torch.utils.data.DataLoader(
                            client_info['dataset'],
                            batch_size=BATCH_SIZE,
                            shuffle=False
                        )
                        
                        client_correct = 0
                        client_total = 0
                        
                        for data, target in data_loader:
                            output = self.global_model(data)
                            _, predicted = torch.max(output.data, 1)
                            client_correct += (predicted == target).sum().item()
                            client_total += target.size(0)
                        
                        total_correct += client_correct
                        total_samples += client_total
                        
                        accuracy = client_correct / client_total if client_total > 0 else 0
                        logger.info(f"Client {client_id} final accuracy: {accuracy:.4f}")
            
            final_accuracy = total_correct / total_samples if total_samples > 0 else 0
            logger.info(f"Final global accuracy: {final_accuracy:.4f}")
            
        except Exception as e:
            logger.error(f"Error evaluating global model: {e}")
    
    def run_client_training(self):
        """Run client training (for client mode)"""
        logger.info(f"Client {self.client_id} starting training")
        
        # In client mode, this would connect to server
        # For simulation, we'll just train locally
        global_weights = {k: v.cpu().numpy() for k, v in self.global_model.state_dict().items()}
        
        result = self.train_client(self.client_id, global_weights)
        
        logger.info(f"Client {self.client_id} training completed")
        return result

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Privacy-First Federated Learning Platform")
    parser.add_argument("--mode", choices=["server", "client"], default="server",
                       help="Run as server or client")
    parser.add_argument("--client-id", type=int, default=1,
                       help="Client ID (for client mode)")
    parser.add_argument("--server-host", type=str, default=SERVER_HOST,
                       help="Server host (for client mode)")
    parser.add_argument("--server-port", type=int, default=SERVER_PORT,
                       help="Server port (for client mode)")
    parser.add_argument("--rounds", type=int, default=ROUNDS,
                       help="Number of training rounds")
    parser.add_argument("--min-clients", type=int, default=2,
                       help="Minimum clients required")
    
    args = parser.parse_args()
    
    # Update global config from command line
    global ROUNDS
    ROUNDS = args.rounds
    
    try:
        # Create platform
        platform = PrivacyFirstFederatedPlatform(
            mode=args.mode,
            client_id=args.client_id,
            server_host=args.server_host,
            server_port=args.server_port
        )
        
        # Run based on mode
        if args.mode == "server":
            platform.run_server_training()
        else:
            platform.run_client_training()
    
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    main()
