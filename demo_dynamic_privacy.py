#!/usr/bin/env python3
"""
Demo Script for Dynamic Privacy Configuration
Shows how dashboard controls affect training in real-time
"""

import time
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

def update_privacy_config(epsilon=1.0, noise_multiplier=1.0, max_grad_norm=1.0):
    """Update the privacy configuration that dashboard reads"""
    config = {
        "privacy": {
            "epsilon": epsilon,
            "noise_multiplier": noise_multiplier,
            "max_grad_norm": max_grad_norm,
            "privacy_budget_limit": 8.0
        },
        "training": {
            "learning_rate": 0.01,
            "batch_size": 32,
            "local_epochs": 5,
            "rounds": 5
        },
        "system": {
            "num_clients": 3,
            "min_clients": 2,
            "auto_stop_on_budget_exhausted": True
        },
        "last_updated": datetime.now().isoformat()
    }
    
    with open('config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Privacy config updated:")
    print(f"   Epsilon: {epsilon}")
    print(f"   Noise Multiplier: {noise_multiplier}")
    print(f"   Max Grad Norm: {max_grad_norm}")

def simulate_training_round(round_num, epsilon, noise_multiplier):
    """Simulate a training round with given privacy parameters"""
    
    # Higher privacy (lower epsilon, higher noise) = lower accuracy
    base_accuracy = 0.85
    
    # Privacy impact on accuracy
    privacy_impact = (epsilon / 2.0) * 0.1 - (noise_multiplier / 5.0) * 0.15
    accuracy = base_accuracy + privacy_impact + np.random.normal(0, 0.02)
    accuracy = max(0.5, min(0.95, accuracy))  # Clamp between 0.5 and 0.95
    
    # Loss inversely related to accuracy
    loss = 1.5 - accuracy + np.random.normal(0, 0.05)
    loss = max(0.1, loss)
    
    # Privacy spending
    epsilon_spent = round_num * epsilon * 0.1 + np.random.normal(0, 0.01)
    
    return {
        'timestamp': datetime.now().isoformat(),
        'round': round_num,
        'global_accuracy': accuracy,
        'global_loss': loss,
        'avg_client_accuracy': accuracy - 0.02,
        'communication_rounds': 3,
        'convergence_score': accuracy * 0.9,
        'epsilon_spent': epsilon_spent,
        'privacy_budget_used': epsilon_spent / 8.0
    }

def demo_privacy_vs_utility():
    """Demonstrate the privacy vs utility tradeoff"""
    
    print("🎯 Dynamic Privacy vs Utility Demo")
    print("=" * 50)
    
    # Create logs directory
    Path('logs').mkdir(exist_ok=True)
    
    # Scenario 1: High Privacy, Low Utility
    print("\n📊 Scenario 1: High Privacy (ε=0.5, σ=2.0)")
    update_privacy_config(epsilon=0.5, noise_multiplier=2.0)
    
    training_data = []
    privacy_data = []
    
    for round_num in range(1, 6):
        metrics = simulate_training_round(round_num, 0.5, 2.0)
        training_data.append({
            'timestamp': metrics['timestamp'],
            'round': metrics['round'],
            'global_accuracy': metrics['global_accuracy'],
            'global_loss': metrics['global_loss'],
            'avg_client_accuracy': metrics['avg_client_accuracy'],
            'communication_rounds': metrics['communication_rounds'],
            'convergence_score': metrics['convergence_score']
        })
        
        for client_id in ['client_1', 'client_2', 'client_3']:
            privacy_data.append({
                'timestamp': metrics['timestamp'],
                'round': metrics['round'],
                'client_id': client_id,
                'epsilon_spent': metrics['epsilon_spent'],
                'delta': 1e-5,
                'noise_multiplier': 2.0,
                'max_grad_norm': 1.0,
                'privacy_budget_used': metrics['privacy_budget_used']
            })
        
        print(f"  Round {round_num}: Accuracy = {metrics['global_accuracy']:.3f}, ε spent = {metrics['epsilon_spent']:.3f}")
        time.sleep(1)  # Simulate training time
    
    # Save data
    pd.DataFrame(training_data).to_csv('logs/training_metrics.csv', index=False)
    pd.DataFrame(privacy_data).to_csv('logs/privacy_metrics.csv', index=False)
    
    print("\n📈 Scenario 2: Low Privacy, High Utility (ε=2.0, σ=0.5)")
    update_privacy_config(epsilon=2.0, noise_multiplier=0.5)
    
    print("  🎛️ Dashboard sliders now show new values!")
    print("  📊 Privacy-Utility tradeoff visualization updated!")
    print("  🔐 Secure aggregation shows different encryption levels!")
    
    time.sleep(3)
    
    # Show how config changes affect training
    print("\n🔄 Real-time Configuration Changes:")
    print("  1. Move ε slider on dashboard → config.json updates")
    print("  2. Training clients read new values next round")
    print("  3. Accuracy responds to privacy changes")
    print("  4. Privacy budget tracking updates in real-time")
    
    print("\n✨ Demo Features Demonstrated:")
    print("  ✅ Dynamic privacy controls")
    print("  ✅ Real-time config updates")
    print("  ✅ Privacy vs utility tradeoff")
    print("  ✅ Budget tracking")
    print("  ✅ Interactive dashboard")
    
    print(f"\n🌐 Dashboard running at: http://localhost:8501")
    print("📖 Try moving the privacy sliders to see real-time changes!")

if __name__ == "__main__":
    demo_privacy_vs_utility()
