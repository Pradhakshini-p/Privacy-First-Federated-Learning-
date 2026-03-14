#!/usr/bin/env python3
"""
Simple client that works without complex dependencies
"""

import time
import random
import json
import os
from datetime import datetime

def simple_client(client_id):
    """Simple federated learning client simulation"""
    
    print(f"🏦 CLIENT {client_id} - TELECOM COMPANY {client_id}")
    print("=" * 50)
    
    # Load customer churn data
    try:
        import pandas as pd
        df = pd.read_csv('customer_churn.csv')
        
        # Simulate data split (each client gets different portion)
        samples_per_client = len(df) // 4
        start_idx = (client_id - 1) * samples_per_client
        end_idx = start_idx + samples_per_client if client_id < 4 else len(df)
        
        client_data = df.iloc[start_idx:end_idx]
        
        print(f"📊 Dataset Info:")
        print(f"   📋 Total samples: {len(client_data)}")
        print(f"   🎯 Churn rate: {client_data['churned'].mean():.2%}")
        print(f"   📏 Features: {len(client_data.columns) - 1}")
        
        # Simulate training rounds
        n_rounds = 5
        
        print(f"\n🔄 Training for {n_rounds} rounds...")
        
        for round_num in range(1, n_rounds + 1):
            # Simulate local training
            accuracy = 0.6 + (round_num * 0.04) + (client_id * 0.01)
            loss = max(0.1, 1.0 - (round_num * 0.12))
            
            print(f"   Round {round_num}: Accuracy={accuracy:.3f}, Loss={loss:.3f}")
            
            # Simulate training time
            time.sleep(0.5)
        
        print(f"\n✅ Client {client_id} training completed!")
        print(f"🎯 Final accuracy: {accuracy:.3f}")
        print(f"📊 Data privacy: ✅ Raw data never left this client")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    import sys
    client_id = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    simple_client(client_id)
