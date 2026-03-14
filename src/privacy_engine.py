"""
Privacy Engine for Federated Learning with Epsilon Tracking
Implements differential privacy with real-time epsilon monitoring
"""

import numpy as np
import torch
import torch.nn as nn
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

try:
    from config import EPSILON, DELTA, MAX_GRAD_NORM, NOISE_MULTIPLIER
    from logging_bridge import log_privacy_spent
except ImportError:
    # Fallback defaults
    EPSILON = 1.0
    DELTA = 1e-5
    MAX_GRAD_NORM = 1.0
    NOISE_MULTIPLIER = 1.0

logger = logging.getLogger(__name__)

class FederatedPrivacyEngine:
    """Privacy engine for federated learning with epsilon tracking"""
    
    def __init__(self, model: nn.Module, target_epsilon: float = EPSILON, 
                 target_delta: float = DELTA, max_grad_norm: float = MAX_GRAD_NORM,
                 noise_multiplier: float = NOISE_MULTIPLIER, client_id: str = "default"):
        """
        Initialize privacy engine
        
        Args:
            model: PyTorch model to protect
            target_epsilon: Target privacy budget (ε)
            target_delta: Target privacy parameter (δ)
            max_grad_norm: Maximum gradient norm for clipping
            noise_multiplier: Noise multiplier for DP
            client_id: Unique identifier for the client
        """
        self.client_id = client_id
        self.target_epsilon = target_epsilon
        self.target_delta = target_delta
        self.max_grad_norm = max_grad_norm
        self.noise_multiplier = noise_multiplier
        
        # Privacy tracking
        self.current_epsilon = 0.0
        self.epsilon_history = []
        self.round_count = 0
        self.privacy_spent = 0.0
        self.privacy_budget_used = 0.0
        
        # Initialize Opacus privacy engine
        self.privacy_engine = None
        self.model = model
        self.optimizer = None
        
        # Validate and prepare model
        self._prepare_model()
        
        logger.info(f"Privacy engine initialized for {client_id}")
        logger.info(f"Target ε: {target_epsilon}, δ: {target_delta}")
        logger.info(f"Max grad norm: {max_grad_norm}, Noise multiplier: {noise_multiplier}")
    
    def _prepare_model(self):
        """Prepare model for differential privacy"""
        try:
            # Validate model for privacy
            errors = ModuleValidator.validate(self.model, strict=False)
            if errors:
                logger.warning(f"Model validation warnings: {errors}")
            
            # Fix model if needed
            self.model = ModuleValidator.fix(self.model)
            
            logger.info("Model prepared for differential privacy")
            
        except Exception as e:
            logger.error(f"Error preparing model for privacy: {e}")
            raise
    
    def attach_to_optimizer(self, optimizer: torch.optim.Optimizer):
        """Attach privacy engine to optimizer"""
        try:
            # Create privacy engine
            self.privacy_engine = PrivacyEngine()
            
            # Attach to optimizer and model
            self.optimizer = self.privacy_engine.make_private(
                module=self.model,
                optimizer=optimizer,
                noise_multiplier=self.noise_multiplier,
                max_grad_norm=self.max_grad_norm,
            )
            
            logger.info(f"Privacy engine attached to optimizer for {self.client_id}")
            
        except Exception as e:
            logger.error(f"Error attaching privacy engine: {e}")
            raise
    
    def get_epsilon(self, delta: Optional[float] = None) -> float:
        """
        Get current epsilon spent
        
        Args:
            delta: Delta value for privacy calculation
            
        Returns:
            Current epsilon spent
        """
        if self.privacy_engine is None:
            logger.warning("Privacy engine not attached, returning 0")
            return 0.0
        
        try:
            delta = delta or self.target_delta
            self.current_epsilon = self.privacy_engine.get_epsilon(delta=delta)
            return self.current_epsilon
            
        except Exception as e:
            logger.error(f"Error calculating epsilon: {e}")
            return 0.0
    
    def track_privacy_spent(self, round_num: int, data_size: int = None):
        """
        Track privacy spending for current round
        
        Args:
            round_num: Current training round
            data_size: Size of training data
        """
        try:
            # Get current epsilon
            current_epsilon = self.get_epsilon()
            
            # Calculate epsilon spent in this round
            if len(self.epsilon_history) > 0:
                epsilon_spent_this_round = current_epsilon - self.epsilon_history[-1]
            else:
                epsilon_spent_this_round = current_epsilon
            
            # Update tracking
            self.epsilon_history.append(current_epsilon)
            self.privacy_spent = current_epsilon
            self.privacy_budget_used = current_epsilon / self.target_epsilon
            self.round_count = round_num
            
            # Log privacy metrics
            privacy_budget_used_pct = min(100.0, self.privacy_budget_used * 100)
            
            logger.info(f"Round {round_num} Privacy Update for {self.client_id}:")
            logger.info(f"  Current ε: {current_epsilon:.4f}")
            logger.info(f"  ε spent this round: {epsilon_spent_this_round:.4f}")
            logger.info(f"  Privacy budget used: {privacy_budget_used_pct:.2f}%")
            logger.info(f"  Data size: {data_size or 'Unknown'}")
            
            # Log to CSV via logging bridge
            try:
                log_privacy_spent(
                    round_num=round_num,
                    client_id=self.client_id,
                    epsilon_spent=current_epsilon,
                    delta=self.target_delta,
                    noise_multiplier=self.noise_multiplier,
                    max_grad_norm=self.max_grad_norm,
                    privacy_budget_used=self.privacy_budget_used
                )
            except Exception as e:
                logger.warning(f"Could not log privacy metrics: {e}")
            
            # Check privacy budget
            if self.privacy_budget_used >= 1.0:
                logger.warning(f"⚠️ PRIVACY BUDGET EXHAUSTED for {self.client_id}!")
                logger.warning(f"  Current ε: {current_epsilon:.4f}")
                logger.warning(f"  Target ε: {self.target_epsilon:.4f}")
                return False  # Privacy budget exhausted
            
            return True  # Privacy budget available
            
        except Exception as e:
            logger.error(f"Error tracking privacy spent: {e}")
            return False
    
    def is_privacy_budget_available(self) -> bool:
        """Check if privacy budget is still available"""
        return self.privacy_budget_used < 1.0
    
    def get_privacy_status(self) -> Dict:
        """Get comprehensive privacy status"""
        return {
            "client_id": self.client_id,
            "current_epsilon": self.current_epsilon,
            "target_epsilon": self.target_epsilon,
            "privacy_budget_used": self.privacy_budget_used,
            "privacy_budget_remaining": max(0.0, 1.0 - self.privacy_budget_used),
            "round_count": self.round_count,
            "noise_multiplier": self.noise_multiplier,
            "max_grad_norm": self.max_grad_norm,
            "is_active": self.is_privacy_budget_available(),
            "status": "Active" if self.is_privacy_budget_available() else "Exhausted"
        }
    
    def reset_privacy_tracking(self):
        """Reset privacy tracking (for new training session)"""
        self.current_epsilon = 0.0
        self.epsilon_history.clear()
        self.privacy_spent = 0.0
        self.privacy_budget_used = 0.0
        self.round_count = 0
        
        logger.info(f"Privacy tracking reset for {self.client_id}")
    
    def update_noise_multiplier(self, new_noise_multiplier: float):
        """Update noise multiplier"""
        if new_noise_multiplier <= 0:
            raise ValueError("Noise multiplier must be positive")
        
        self.noise_multiplier = new_noise_multiplier
        logger.info(f"Noise multiplier updated to {new_noise_multiplier} for {self.client_id}")
    
    def get_privacy_recommendation(self) -> Dict:
        """Get privacy recommendations based on current usage"""
        recommendations = []
        
        if self.privacy_budget_used > 0.8:
            recommendations.append("Privacy budget nearly exhausted - consider stopping training")
        
        if self.noise_multiplier < 0.5:
            recommendations.append("Low noise multiplier - consider increasing for better privacy")
        
        if self.max_grad_norm > 2.0:
            recommendations.append("High gradient norm - consider reducing for better privacy")
        
        if len(self.epsilon_history) > 10:
            recent_epsilons = self.epsilon_history[-10:]
            epsilon_growth = recent_epsilons[-1] - recent_epsilons[0]
            if epsilon_growth > 0.1:
                recommendations.append("High epsilon growth - reduce learning rate or batch size")
        
        return {
            "recommendations": recommendations,
            "urgency": "High" if self.privacy_budget_used > 0.8 else "Medium" if self.privacy_budget_used > 0.5 else "Low",
            "next_round_recommended": self.is_privacy_budget_available()
        }

class PrivacyBudgetManager:
    """Manages privacy budget across multiple clients"""
    
    def __init__(self):
        self.client_privacy_engines: Dict[str, FederatedPrivacyEngine] = {}
        self.global_epsilon_budget = 0.0
        self.global_epsilon_spent = 0.0
    
    def add_client(self, client_id: str, privacy_engine: FederatedPrivacyEngine):
        """Add a client's privacy engine to management"""
        self.client_privacy_engines[client_id] = privacy_engine
        self.global_epsilon_budget += privacy_engine.target_epsilon
        logger.info(f"Added {client_id} to privacy budget management")
    
    def get_global_privacy_status(self) -> Dict:
        """Get global privacy status across all clients"""
        total_epsilon_spent = sum(engine.current_epsilon for engine in self.client_privacy_engines.values())
        total_budget = sum(engine.target_epsilon for engine in self.client_privacy_engines.values())
        
        active_clients = sum(1 for engine in self.client_privacy_engines.values() 
                           if engine.is_privacy_budget_available())
        
        return {
            "total_clients": len(self.client_privacy_engines),
            "active_clients": active_clients,
            "total_epsilon_spent": total_epsilon_spent,
            "total_epsilon_budget": total_budget,
            "global_budget_used": total_epsilon_spent / total_budget if total_budget > 0 else 0,
            "exhausted_clients": len(self.client_privacy_engines) - active_clients
        }
    
    def get_client_privacy_summary(self) -> pd.DataFrame:
        """Get privacy summary for all clients"""
        try:
            import pandas as pd
            
            summary_data = []
            for client_id, engine in self.client_privacy_engines.items():
                status = engine.get_privacy_status()
                summary_data.append(status)
            
            return pd.DataFrame(summary_data)
            
        except ImportError:
            logger.warning("Pandas not available, returning list of dictionaries")
            return [engine.get_privacy_status() for engine in self.client_privacy_engines.values()]

# Global privacy budget manager
privacy_manager = PrivacyBudgetManager()

def get_privacy_manager():
    """Get the global privacy budget manager"""
    return privacy_manager

def create_privacy_engine(model: nn.Module, client_id: str, **kwargs) -> FederatedPrivacyEngine:
    """
    Create and configure a privacy engine for a client
    
    Args:
        model: PyTorch model to protect
        client_id: Unique client identifier
        **kwargs: Additional privacy parameters
        
    Returns:
        Configured FederatedPrivacyEngine
    """
    engine = FederatedPrivacyEngine(model, client_id=client_id, **kwargs)
    privacy_manager.add_client(client_id, engine)
    return engine
