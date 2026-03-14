#!/usr/bin/env python3
"""
Simple server that works without complex dependencies
"""

import time
import json
import os
from datetime import datetime

def simple_server():
    """Simple federated learning server simulation"""
    
    print("🌸 SIMPLE FEDERATED LEARNING SERVER")
    print("=" * 50)
    
    # Create logs directory
    os.makedirs('logs', exist_ok=True)
    
    # Simulate training rounds
    n_rounds = 5
    n_clients = 4
    
    print(f"🎯 Starting {n_rounds} rounds with {n_clients} clients...")
    print(f"📊 Dataset: Customer Churn Prediction")
    print(f"🏦 Clients: 4 Telecom Companies")
    print(f"🎯 Target: Customer Churn (0=No, 1=Yes)")
    
    # Simulate federated learning
    training_log = []
    
    for round_num in range(1, n_rounds + 1):
        print(f"\n🔄 Round {round_num}/{n_rounds}")
        
        # Simulate client updates
        client_accuracies = []
        client_losses = []
        
        for client_id in range(1, n_clients + 1):
            # Simulate training (getting better over time)
            base_accuracy = 0.6 + (round_num * 0.05)
            accuracy = min(0.95, base_accuracy + (client_id * 0.01))
            loss = max(0.1, 1.0 - (round_num * 0.15))
            
            client_accuracies.append(accuracy)
            client_losses.append(loss)
            
            print(f"   🏦 Client {client_id}: Accuracy={accuracy:.3f}, Loss={loss:.3f}")
        
        # Aggregate (FedAvg)
        global_accuracy = sum(client_accuracies) / len(client_accuracies)
        global_loss = sum(client_losses) / len(client_losses)
        
        # Log results
        log_entry = {
            'round': round_num,
            'accuracy': global_accuracy,
            'loss': global_loss,
            'timestamp': datetime.now().isoformat()
        }
        training_log.append(log_entry)
        
        print(f"   🌍 Global: Accuracy={global_accuracy:.3f}, Loss={global_loss:.3f}")
        
        # Simulate delay
        time.sleep(1)
    
    # Save logs
    with open('logs/training_log.json', 'w') as f:
        json.dump(training_log, f, indent=2)
    
    # Create client metrics
    client_metrics = []
    for client_id in range(1, n_clients + 1):
        for round_num in range(1, n_rounds + 1):
            metrics = {
                'client_id': f'client_{client_id}',
                'round': round_num,
                'val_accuracy': 0.7 + (round_num * 0.03) + (client_id * 0.01),
                'train_loss': 1.0 - (round_num * 0.1),
                'val_loss': 0.9 - (round_num * 0.08),
                'timestamp': datetime.now().isoformat()
            }
            client_metrics.append(metrics)
    
    with open('logs/client_metrics.json', 'w') as f:
        json.dump(client_metrics, f, indent=2)
    
    print(f"\n✅ Training completed!")
    print(f"📊 Final accuracy: {global_accuracy:.3f}")
    print(f"📁 Logs saved to: logs/training_log.json")
    print(f"📁 Client metrics: logs/client_metrics.json")
    
    return training_log

if __name__ == "__main__":
    simple_server()
