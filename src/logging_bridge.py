"""
Logging Bridge for Real-time Dashboard Communication
Handles CSV-based logging for federated learning metrics
"""

import csv
import os
import time
import threading
from datetime import datetime
from typing import Dict, List, Optional, Any
import pandas as pd
from .config import (
    METRICS_LOG_FILE, PRIVACY_LOG_FILE, CLIENT_LOG_FILE,
    LOG_LEVEL, LOG_FORMAT, MAX_LOG_ENTRIES
)

class FederatedLearningLogger:
    """Centralized logger for federated learning metrics"""
    
    def __init__(self):
        self.training_metrics = []
        self.privacy_metrics = []
        self.client_metrics = []
        self._lock = threading.Lock()
        
        # Initialize CSV files with headers
        self._initialize_csv_files()
        
    def _initialize_csv_files(self):
        """Create CSV files with proper headers"""
        # Training metrics headers
        training_headers = [
            'timestamp', 'round', 'global_accuracy', 'global_loss',
            'avg_client_accuracy', 'communication_rounds', 'convergence_score'
        ]
        
        # Privacy metrics headers  
        privacy_headers = [
            'timestamp', 'round', 'client_id', 'epsilon_spent', 'delta',
            'noise_multiplier', 'max_grad_norm', 'privacy_budget_used'
        ]
        
        # Client metrics headers
        client_headers = [
            'timestamp', 'round', 'client_id', 'local_accuracy', 'local_loss',
            'data_size', 'training_time', 'communication_cost', 'status'
        ]
        
        # Write headers to files
        self._write_csv_header(METRICS_LOG_FILE, training_headers)
        self._write_csv_header(PRIVACY_LOG_FILE, privacy_headers)
        self._write_csv_header(CLIENT_LOG_FILE, client_headers)
    
    def _write_csv_header(self, file_path, headers):
        """Write headers to CSV file"""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        if not os.path.exists(file_path):
            with open(file_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
    
    def log_training_metrics(self, round_num: int, global_accuracy: float, 
                           global_loss: float, avg_client_accuracy: float,
                           communication_rounds: int = 0, convergence_score: float = 0.0):
        """Log global training metrics"""
        with self._lock:
            timestamp = datetime.now().isoformat()
            metrics = {
                'timestamp': timestamp,
                'round': round_num,
                'global_accuracy': global_accuracy,
                'global_loss': global_loss,
                'avg_client_accuracy': avg_client_accuracy,
                'communication_rounds': communication_rounds,
                'convergence_score': convergence_score
            }
            
            # Append to memory
            self.training_metrics.append(metrics)
            
            # Keep only recent entries
            if len(self.training_metrics) > MAX_LOG_ENTRIES:
                self.training_metrics = self.training_metrics[-MAX_LOG_ENTRIES:]
            
            # Write to CSV
            self._append_to_csv(METRICS_LOG_FILE, metrics)
    
    def log_privacy_metrics(self, round_num: int, client_id: str, 
                          epsilon_spent: float, delta: float,
                          noise_multiplier: float, max_grad_norm: float,
                          privacy_budget_used: float):
        """Log privacy metrics for a client"""
        with self._lock:
            timestamp = datetime.now().isoformat()
            metrics = {
                'timestamp': timestamp,
                'round': round_num,
                'client_id': client_id,
                'epsilon_spent': epsilon_spent,
                'delta': delta,
                'noise_multiplier': noise_multiplier,
                'max_grad_norm': max_grad_norm,
                'privacy_budget_used': privacy_budget_used
            }
            
            # Append to memory
            self.privacy_metrics.append(metrics)
            
            # Keep only recent entries
            if len(self.privacy_metrics) > MAX_LOG_ENTRIES:
                self.privacy_metrics = self.privacy_metrics[-MAX_LOG_ENTRIES:]
            
            # Write to CSV
            self._append_to_csv(PRIVACY_LOG_FILE, metrics)
    
    def log_client_metrics(self, round_num: int, client_id: str,
                         local_accuracy: float, local_loss: float,
                         data_size: int, training_time: float,
                         communication_cost: float, status: str = "active"):
        """Log client-specific metrics"""
        with self._lock:
            timestamp = datetime.now().isoformat()
            metrics = {
                'timestamp': timestamp,
                'round': round_num,
                'client_id': client_id,
                'local_accuracy': local_accuracy,
                'local_loss': local_loss,
                'data_size': data_size,
                'training_time': training_time,
                'communication_cost': communication_cost,
                'status': status
            }
            
            # Append to memory
            self.client_metrics.append(metrics)
            
            # Keep only recent entries
            if len(self.client_metrics) > MAX_LOG_ENTRIES:
                self.client_metrics = self.client_metrics[-MAX_LOG_ENTRIES:]
            
            # Write to CSV
            self._append_to_csv(CLIENT_LOG_FILE, metrics)
    
    def _append_to_csv(self, file_path: str, data: Dict[str, Any]):
        """Append data to CSV file"""
        try:
            # Read existing data
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
            else:
                df = pd.DataFrame()
            
            # Append new data
            new_row = pd.DataFrame([data])
            df = pd.concat([df, new_row], ignore_index=True)
            
            # Write back to file
            df.to_csv(file_path, index=False)
            
        except Exception as e:
            print(f"Error writing to CSV {file_path}: {e}")
    
    def get_latest_training_metrics(self, n: int = 10) -> List[Dict]:
        """Get latest n training metrics"""
        with self._lock:
            return self.training_metrics[-n:] if self.training_metrics else []
    
    def get_latest_privacy_metrics(self, n: int = 10) -> List[Dict]:
        """Get latest n privacy metrics"""
        with self._lock:
            return self.privacy_metrics[-n:] if self.privacy_metrics else []
    
    def get_latest_client_metrics(self, n: int = 10) -> List[Dict]:
        """Get latest n client metrics"""
        with self._lock:
            return self.client_metrics[-n:] if self.client_metrics else []
    
    def get_training_data_from_csv(self) -> pd.DataFrame:
        """Read training metrics from CSV file"""
        try:
            if os.path.exists(METRICS_LOG_FILE):
                return pd.read_csv(METRICS_LOG_FILE)
            else:
                return pd.DataFrame()
        except Exception as e:
            print(f"Error reading training metrics CSV: {e}")
            return pd.DataFrame()
    
    def get_privacy_data_from_csv(self) -> pd.DataFrame:
        """Read privacy metrics from CSV file"""
        try:
            if os.path.exists(PRIVACY_LOG_FILE):
                return pd.read_csv(PRIVACY_LOG_FILE)
            else:
                return pd.DataFrame()
        except Exception as e:
            print(f"Error reading privacy metrics CSV: {e}")
            return pd.DataFrame()
    
    def get_client_data_from_csv(self) -> pd.DataFrame:
        """Read client metrics from CSV file"""
        try:
            if os.path.exists(CLIENT_LOG_FILE):
                return pd.read_csv(CLIENT_LOG_FILE)
            else:
                return pd.DataFrame()
        except Exception as e:
            print(f"Error reading client metrics CSV: {e}")
            return pd.DataFrame()
    
    def clear_logs(self):
        """Clear all log files and memory"""
        with self._lock:
            self.training_metrics.clear()
            self.privacy_metrics.clear()
            self.client_metrics.clear()
            
            # Clear CSV files
            for file_path in [METRICS_LOG_FILE, PRIVACY_LOG_FILE, CLIENT_LOG_FILE]:
                if os.path.exists(file_path):
                    os.remove(file_path)
            
            # Reinitialize headers
            self._initialize_csv_files()

# Global logger instance
fl_logger = FederatedLearningLogger()

def get_logger():
    """Get the global federated learning logger instance"""
    return fl_logger

# Convenience functions for direct access
def log_training_round(round_num: int, global_accuracy: float, global_loss: float,
                      avg_client_accuracy: float, **kwargs):
    """Convenience function to log training round"""
    fl_logger.log_training_metrics(round_num, global_accuracy, global_loss, 
                                 avg_client_accuracy, **kwargs)

def log_privacy_spent(round_num: int, client_id: str, epsilon_spent: float,
                     delta: float = 1e-5, **kwargs):
    """Convenience function to log privacy spending"""
    from .config import NOISE_MULTIPLIER, MAX_GRAD_NORM
    fl_logger.log_privacy_metrics(round_num, client_id, epsilon_spent, delta,
                                 NOISE_MULTIPLIER, MAX_GRAD_NORM, **kwargs)

def log_client_performance(round_num: int, client_id: str, local_accuracy: float,
                          local_loss: float, data_size: int, training_time: float,
                          **kwargs):
    """Convenience function to log client performance"""
    fl_logger.log_client_metrics(round_num, client_id, local_accuracy, local_loss,
                                data_size, training_time, **kwargs)
