#!/usr/bin/env python3
"""
Simplified Attack Simulation for Federated Learning Security
Demonstrates various attack scenarios without torch dependencies
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import logging
import json
import time
from dataclasses import dataclass
from enum import Enum
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AttackType(Enum):
    """Types of attacks in federated learning"""
    DATA_POISONING = "data_poisoning"
    MODEL_POISONING = "model_poisoning"
    FREE_RIDER = "free_rider"
    BYZANTINE = "byzantine"
    EAVESDROPPING = "eavesdropping"

@dataclass
class AttackConfig:
    """Configuration for attack simulation"""
    attack_type: AttackType
    malicious_clients: List[int]
    attack_intensity: float  # 0.0 to 1.0
    start_round: int
    end_round: int
    description: str

class SimplifiedAttackSimulator:
    """Simplified attack simulator for federated learning systems"""
    
    def __init__(self, n_clients=10, n_rounds=10):
        self.n_clients = n_clients
        self.n_rounds = n_rounds
        self.attack_history = []
        self.defense_history = []
        self.client_reputation = {i: 1.0 for i in range(n_clients)}
        self.round_results = []
        
        logger.info(f"Attack Simulator initialized with {n_clients} clients for {n_rounds} rounds")
    
    def simulate_data_poisoning_attack(self, client_id: int, intensity: float) -> Dict:
        """Simulate data poisoning attack where malicious client sends manipulated gradients"""
        logger.info(f"Simulating data poisoning attack from client {client_id} with intensity {intensity}")
        
        # Simulate malicious model updates
        base_gradients = np.random.randn(10, 5) * 0.1
        
        # Add poisoning based on intensity
        if intensity > 0.7:
            # High intensity: completely reverse gradients
            poisoned_gradients = -base_gradients * 2
            attack_severity = "HIGH"
        elif intensity > 0.4:
            # Medium intensity: add noise and bias
            noise = np.random.randn(10, 5) * intensity
            bias = np.ones((10, 5)) * intensity * 0.5
            poisoned_gradients = base_gradients + noise + bias
            attack_severity = "MEDIUM"
        else:
            # Low intensity: subtle manipulation
            poisoned_gradients = base_gradients + np.random.randn(10, 5) * intensity * 0.1
            attack_severity = "LOW"
        
        return {
            "client_id": client_id,
            "attack_type": AttackType.DATA_POISONING.value,
            "original_gradients": base_gradients.tolist(),
            "poisoned_gradients": poisoned_gradients.tolist(),
            "intensity": intensity,
            "severity": attack_severity,
            "timestamp": time.time()
        }
    
    def simulate_model_poisoning_attack(self, client_id: int, intensity: float) -> Dict:
        """Simulate model poisoning attack with malicious model weights"""
        logger.info(f"Simulating model poisoning attack from client {client_id} with intensity {intensity}")
        
        # Simulate model weights
        base_weights = np.random.randn(5, 3) * 0.5
        
        # Create malicious weights
        if intensity > 0.8:
            # Extreme values to disrupt convergence
            poisoned_weights = np.random.randn(5, 3) * 10
            attack_severity = "CRITICAL"
        elif intensity > 0.5:
            # Scale weights to cause divergence
            poisoned_weights = base_weights * (1 + intensity * 2)
            attack_severity = "HIGH"
        else:
            # Subtle weight manipulation
            poisoned_weights = base_weights * (1 + intensity * 0.5)
            attack_severity = "MEDIUM"
        
        return {
            "client_id": client_id,
            "attack_type": AttackType.MODEL_POISONING.value,
            "original_weights": base_weights.tolist(),
            "poisoned_weights": poisoned_weights.tolist(),
            "intensity": intensity,
            "severity": attack_severity,
            "timestamp": time.time()
        }
    
    def simulate_byzantine_attack(self, client_id: int, intensity: float) -> Dict:
        """Simulate Byzantine attack where clients send arbitrary malicious updates"""
        logger.info(f"Simulating Byzantine attack from client {client_id} with intensity {intensity}")
        
        # Byzantine clients can send completely arbitrary updates
        if intensity > 0.9:
            # Send random garbage
            malicious_update = np.random.randn(100, 50) * 100
            attack_severity = "EXTREME"
        elif intensity > 0.6:
            # Send inverted updates
            malicious_update = np.random.randn(10, 5) * -5
            attack_severity = "HIGH"
        else:
            # Send slightly corrupted updates
            malicious_update = np.random.randn(10, 5) * 2
            attack_severity = "MEDIUM"
        
        return {
            "client_id": client_id,
            "attack_type": AttackType.BYZANTINE.value,
            "malicious_update": malicious_update.tolist(),
            "intensity": intensity,
            "severity": attack_severity,
            "timestamp": time.time()
        }
    
    def detect_anomalies(self, client_updates: Dict) -> List[Dict]:
        """Detect anomalies in client updates using various statistical methods"""
        anomalies = []
        
        for client_id, update in client_updates.items():
            anomaly_score = 0
            reasons = []
            
            # Check for extreme values
            if 'poisoned_gradients' in update:
                gradients = np.array(update['poisoned_gradients'])
                if np.any(np.abs(gradients) > 5):
                    anomaly_score += 0.4
                    reasons.append("EXTREME_GRADIENTS")
            
            # Check for unusual patterns
            if 'poisoned_weights' in update:
                weights = np.array(update['poisoned_weights'])
                weight_std = np.std(weights)
                if weight_std > 2:
                    anomaly_score += 0.3
                    reasons.append("HIGH_WEIGHT_VARIANCE")
            
            # Check client reputation
            if self.client_reputation[client_id] < 0.5:
                anomaly_score += 0.3
                reasons.append("LOW_REPUTATION")
            
            if anomaly_score > 0.5:
                anomalies.append({
                    "client_id": client_id,
                    "anomaly_score": anomaly_score,
                    "reasons": reasons,
                    "detected_at": time.time()
                })
                
                # Update client reputation
                self.client_reputation[client_id] *= 0.8
        
        return anomalies
    
    def apply_defense_mechanisms(self, anomalies: List[Dict]) -> Dict:
        """Apply defense mechanisms against detected attacks"""
        defense_actions = {
            "clients_removed": [],
            "clients_warned": [],
            "aggregation_method": "weighted_average",
            "privacy_budget_enforced": False
        }
        
        for anomaly in anomalies:
            client_id = anomaly["client_id"]
            anomaly_score = anomaly["anomaly_score"]
            
            if anomaly_score > 0.8:
                # Remove malicious client
                defense_actions["clients_removed"].append(client_id)
                logger.warning(f"Client {client_id} removed due to high anomaly score: {anomaly_score}")
            elif anomaly_score > 0.6:
                # Warn client
                defense_actions["clients_warned"].append(client_id)
                logger.warning(f"Client {client_id} warned for suspicious behavior")
        
        # If many attacks detected, use more robust aggregation
        if len(anomalies) > len(self.client_reputation) * 0.3:
            defense_actions["aggregation_method"] = "trimmed_mean"
            logger.info("Switching to trimmed mean aggregation due to multiple attacks")
        
        return defense_actions
    
    def run_attack_simulation(self, attack_configs: List[AttackConfig]) -> Dict:
        """Run comprehensive attack simulation"""
        logger.info(f"Starting attack simulation with {len(attack_configs)} attack configurations")
        
        simulation_results = {
            "attack_configs": [
                {
                    "attack_type": ac.attack_type.value if hasattr(ac.attack_type, 'value') else str(ac.attack_type),
                    "malicious_clients": ac.malicious_clients,
                    "attack_intensity": ac.attack_intensity,
                    "start_round": ac.start_round,
                    "end_round": ac.end_round,
                    "description": ac.description
                } for ac in attack_configs
            ],
            "round_results": [],
            "total_attacks_detected": 0,
            "clients_compromised": set(),
            "defense_effectiveness": 0
        }
        
        for round_num in range(self.n_rounds):
            round_result = {
                "round": round_num,
                "attacks_launched": [],
                "attacks_detected": [],
                "defenses_applied": {},
                "system_health": 1.0
            }
            
            # Check which attacks should be active this round
            active_attacks = []
            for config in attack_configs:
                if config.start_round <= round_num <= config.end_round:
                    active_attacks.append(config)
            
            # Simulate attacks
            client_updates = {}
            for config in active_attacks:
                for client_id in config.malicious_clients:
                    if config.attack_type == AttackType.DATA_POISONING:
                        attack_result = self.simulate_data_poisoning_attack(client_id, config.attack_intensity)
                    elif config.attack_type == AttackType.MODEL_POISONING:
                        attack_result = self.simulate_model_poisoning_attack(client_id, config.attack_intensity)
                    elif config.attack_type == AttackType.BYZANTINE:
                        attack_result = self.simulate_byzantine_attack(client_id, config.attack_intensity)
                    
                    client_updates[client_id] = attack_result
                    round_result["attacks_launched"].append(attack_result)
                    simulation_results["clients_compromised"].add(client_id)
            
            # Detect anomalies
            anomalies = self.detect_anomalies(client_updates)
            round_result["attacks_detected"] = anomalies
            simulation_results["total_attacks_detected"] += len(anomalies)
            
            # Apply defenses
            defenses = self.apply_defense_mechanisms(anomalies)
            round_result["defenses_applied"] = defenses
            
            # Calculate system health
            total_clients = self.n_clients
            healthy_clients = total_clients - len(defenses["clients_removed"])
            round_result["system_health"] = healthy_clients / total_clients
            
            simulation_results["round_results"].append(round_result)
            
            logger.info(f"Round {round_num}: {len(active_attacks)} attacks, {len(anomalies)} detected")
        
        # Calculate defense effectiveness
        total_attacks_launched = sum(len(r["attacks_launched"]) for r in simulation_results["round_results"])
        if total_attacks_launched > 0:
            simulation_results["defense_effectiveness"] = (
                simulation_results["total_attacks_detected"] / total_attacks_launched
            ) * 100
        
        simulation_results["clients_compromised"] = list(simulation_results["clients_compromised"])
        
        return simulation_results
    
    def generate_attack_report(self, results: Dict) -> str:
        """Generate a comprehensive attack simulation report"""
        report = []
        report.append("=" * 80)
        report.append("FEDERATED LEARNING ATTACK SIMULATION REPORT")
        report.append("=" * 80)
        
        # Summary
        report.append("\nSIMULATION SUMMARY:")
        report.append(f"   Total Rounds: {self.n_rounds}")
        report.append(f"   Total Clients: {self.n_clients}")
        report.append(f"   Clients Compromised: {len(results['clients_compromised'])}")
        report.append(f"   Attacks Detected: {results['total_attacks_detected']}")
        report.append(f"   Defense Effectiveness: {results['defense_effectiveness']:.1f}%")
        
        # Attack configurations
        report.append("\nATTACK CONFIGURATIONS:")
        for config in results['attack_configs']:
            attack_type = config.get('attack_type', 'UNKNOWN')
            if isinstance(attack_type, str):
                attack_name = attack_type.upper()
            else:
                attack_name = str(attack_type).split('.')[-1].upper()
            report.append(f"   - {attack_name}: Clients {config.get('malicious_clients', [])}")
            report.append(f"     Intensity: {config.get('intensity', 0)}, Rounds {config.get('start_round', 0)}-{config.get('end_round', 0)}")
            report.append(f"     Description: {config.get('description', 'No description')}")
        
        # Round-by-round analysis
        report.append("\nROUND-BY-ROUND ANALYSIS:")
        for round_result in results['round_results']:
            report.append(f"\n   Round {round_result['round']}:")
            report.append(f"     Attacks Launched: {len(round_result['attacks_launched'])}")
            report.append(f"     Attacks Detected: {len(round_result['attacks_detected'])}")
            report.append(f"     System Health: {round_result['system_health']:.1%}")
            
            if round_result['defenses_applied']['clients_removed']:
                report.append(f"     Clients Removed: {round_result['defenses_applied']['clients_removed']}")
        
        # Client reputation
        report.append("\nCLIENT REPUTATION (Final):")
        for client_id, reputation in sorted(self.client_reputation.items()):
            status = "HEALTHY" if reputation > 0.8 else "SUSPICIOUS" if reputation > 0.5 else "MALICIOUS"
            report.append(f"   Client {client_id}: {status} ({reputation:.2f})")
        
        # Recommendations
        report.append("\nSECURITY RECOMMENDATIONS:")
        if results['defense_effectiveness'] < 70:
            report.append("   WARNING: Consider implementing stronger anomaly detection")
        if len(results['clients_compromised']) > self.n_clients * 0.3:
            report.append("   WARNING: High client compromise rate - review client onboarding process")
        
        report.append("   INFO: Continue monitoring client behavior patterns")
        report.append("   INFO: Implement adaptive reputation systems")
        report.append("   INFO: Consider zero-knowledge proofs for client verification")
        
        report.append("\n" + "=" * 80)
        
        return "\n".join(report)
    
    def save_visualizations(self, results: Dict):
        """Create and save visualization plots"""
        os.makedirs("logs", exist_ok=True)
        
        # Create DataFrame for analysis
        round_data = []
        for round_result in results['round_results']:
            round_data.append({
                "round": round_result['round'],
                "attacks_launched": len(round_result['attacks_launched']),
                "attacks_detected": len(round_result['attacks_detected']),
                "system_health": round_result['system_health']
            })
        
        df = pd.DataFrame(round_data)
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Federated Learning Attack Simulation Results', fontsize=16)
        
        # Plot 1: Attacks over rounds
        axes[0, 0].plot(df['round'], df['attacks_launched'], 'r-', label='Launched', linewidth=2)
        axes[0, 0].plot(df['round'], df['attacks_detected'], 'g-', label='Detected', linewidth=2)
        axes[0, 0].set_title('Attack Detection Over Time')
        axes[0, 0].set_xlabel('Round')
        axes[0, 0].set_ylabel('Number of Attacks')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: System health
        axes[0, 1].plot(df['round'], df['system_health'], 'b-', linewidth=2)
        axes[0, 1].set_title('System Health Over Time')
        axes[0, 1].set_xlabel('Round')
        axes[0, 1].set_ylabel('System Health (%)')
        axes[0, 1].set_ylim(0, 1.1)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Client reputation
        client_ids = list(self.client_reputation.keys())
        reputations = list(self.client_reputation.values())
        colors = ['green' if rep > 0.8 else 'orange' if rep > 0.5 else 'red' for rep in reputations]
        
        axes[1, 0].bar(client_ids, reputations, color=colors, alpha=0.7)
        axes[1, 0].set_title('Final Client Reputation')
        axes[1, 0].set_xlabel('Client ID')
        axes[1, 0].set_ylabel('Reputation Score')
        axes[1, 0].set_ylim(0, 1.1)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Attack types distribution
        attack_types = {}
        for config in results['attack_configs']:
            attack_type = config['attack_type']
            attack_types[attack_type] = attack_types.get(attack_type, 0) + len(config['malicious_clients'])
        
        if attack_types:
            axes[1, 1].pie(attack_types.values(), labels=attack_types.keys(), autopct='%1.1f%%')
            axes[1, 1].set_title('Attack Types Distribution')
        
        plt.tight_layout()
        plt.savefig('logs/attack_simulation_visualization.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Visualization saved: logs/attack_simulation_visualization.png")

def create_sample_attack_scenarios() -> List[AttackConfig]:
    """Create sample attack scenarios for demonstration"""
    scenarios = [
        AttackConfig(
            attack_type=AttackType.DATA_POISONING,
            malicious_clients=[1, 2],
            attack_intensity=0.7,
            start_round=2,
            end_round=5,
            description="Medium intensity data poisoning by clients 1 and 2"
        ),
        AttackConfig(
            attack_type=AttackType.MODEL_POISONING,
            malicious_clients=[3],
            attack_intensity=0.9,
            start_round=4,
            end_round=7,
            description="High intensity model poisoning by client 3"
        ),
        AttackConfig(
            attack_type=AttackType.BYZANTINE,
            malicious_clients=[4, 5],
            attack_intensity=0.8,
            start_round=6,
            end_round=9,
            description="Byzantine attack by clients 4 and 5"
        ),
    ]
    return scenarios

def main():
    """Main function to run attack simulation"""
    logger.info("Starting Federated Learning Attack Simulation")
    
    # Create simulator
    simulator = SimplifiedAttackSimulator(n_clients=10, n_rounds=10)
    
    # Create attack scenarios
    attack_scenarios = create_sample_attack_scenarios()
    
    # Run simulation
    results = simulator.run_attack_simulation(attack_scenarios)
    
    # Generate report
    report = simulator.generate_attack_report(results)
    print(report)
    
    # Save results
    with open("logs/attack_simulation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Create visualizations
    simulator.save_visualizations(results)
    
    logger.info("Attack simulation completed. Results saved to logs/")

if __name__ == "__main__":
    main()
