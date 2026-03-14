"""
Dynamic Configuration Manager for Real-time Privacy Updates
Allows dashboard to update privacy parameters that clients read during training
"""

import json
import threading
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class DynamicConfigManager:
    """Manages dynamic configuration updates for federated learning"""
    
    def __init__(self, config_file: Path = None):
        """
        Initialize dynamic configuration manager
        
        Args:
            config_file: Path to config file (defaults to ../config.json)
        """
        if config_file is None:
            config_file = Path(__file__).parent.parent / "config.json"
        
        self.config_file = config_file
        self.config = {}
        self.last_modified = None
        self._lock = threading.Lock()
        
        # Initialize config file
        self._initialize_config()
        
        # Start config watcher
        self._watch_config()
    
    def _initialize_config(self):
        """Initialize configuration file with defaults"""
        try:
            if self.config_file.exists():
                self.load_config()
            else:
                # Create default config
                self.config = {
                    "privacy": {
                        "epsilon": 1.0,
                        "noise_multiplier": 1.0,
                        "max_grad_norm": 1.0,
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
                self.save_config()
                logger.info("Created default configuration file")
        
        except Exception as e:
            logger.error(f"Error initializing config: {e}")
    
    def _watch_config(self):
        """Watch for config file changes in background thread"""
        def watcher():
            while True:
                try:
                    if self.config_file.exists():
                        current_modified = self.config_file.stat().st_mtime
                        if self.last_modified is None or current_modified > self.last_modified:
                            self.load_config()
                            logger.info("Configuration updated from file")
                    time.sleep(1)  # Check every second
                except Exception as e:
                    logger.error(f"Error watching config: {e}")
                    time.sleep(5)  # Wait longer on error
        
        watcher_thread = threading.Thread(target=watcher, daemon=True)
        watcher_thread.start()
        logger.info("Config watcher started")
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from file"""
        try:
            with self._lock:
                with open(self.config_file, 'r') as f:
                    self.config = json.load(f)
                self.last_modified = self.config_file.stat().st_mtime
                return self.config
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return {}
    
    def save_config(self):
        """Save configuration to file"""
        try:
            with self._lock:
                self.config['last_updated'] = datetime.now().isoformat()
                with open(self.config_file, 'w') as f:
                    json.dump(self.config, f, indent=2)
                self.last_modified = self.config_file.stat().st_mtime
        except Exception as e:
            logger.error(f"Error saving config: {e}")
    
    def get_privacy_config(self) -> Dict[str, Any]:
        """Get privacy configuration"""
        return self.config.get('privacy', {})
    
    def get_training_config(self) -> Dict[str, Any]:
        """Get training configuration"""
        return self.config.get('training', {})
    
    def get_system_config(self) -> Dict[str, Any]:
        """Get system configuration"""
        return self.config.get('system', {})
    
    def update_privacy_config(self, **kwargs):
        """Update privacy configuration"""
        with self._lock:
            if 'privacy' not in self.config:
                self.config['privacy'] = {}
            
            self.config['privacy'].update(kwargs)
            self.save_config()
            logger.info(f"Privacy config updated: {kwargs}")
    
    def update_training_config(self, **kwargs):
        """Update training configuration"""
        with self._lock:
            if 'training' not in self.config:
                self.config['training'] = {}
            
            self.config['training'].update(kwargs)
            self.save_config()
            logger.info(f"Training config updated: {kwargs}")
    
    def get_epsilon(self) -> float:
        """Get current epsilon value"""
        return self.get_privacy_config().get('epsilon', 1.0)
    
    def get_noise_multiplier(self) -> float:
        """Get current noise multiplier"""
        return self.get_privacy_config().get('noise_multiplier', 1.0)
    
    def get_max_grad_norm(self) -> float:
        """Get current max gradient norm"""
        return self.get_privacy_config().get('max_grad_norm', 1.0)
    
    def get_privacy_budget_limit(self) -> float:
        """Get privacy budget limit"""
        return self.get_privacy_config().get('privacy_budget_limit', 8.0)
    
    def is_auto_stop_enabled(self) -> bool:
        """Check if auto-stop on budget exhaustion is enabled"""
        return self.get_system_config().get('auto_stop_on_budget_exhausted', True)
    
    def should_stop_training(self, current_epsilon: float) -> bool:
        """Check if training should stop based on privacy budget"""
        if not self.is_auto_stop_enabled():
            return False
        
        budget_limit = self.get_privacy_budget_limit()
        return current_epsilon >= budget_limit
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get configuration summary for display"""
        return {
            "privacy": self.get_privacy_config(),
            "training": self.get_training_config(),
            "system": self.get_system_config(),
            "last_updated": self.config.get('last_updated'),
            "file_path": str(self.config_file)
        }

# Global config manager instance
_config_manager = None

def get_config_manager() -> DynamicConfigManager:
    """Get the global configuration manager instance"""
    global _config_manager
    if _config_manager is None:
        _config_manager = DynamicConfigManager()
    return _config_manager

def reload_config():
    """Force reload configuration from file"""
    manager = get_config_manager()
    return manager.load_config()

def update_privacy_settings(epsilon: float = None, noise_multiplier: float = None, 
                          max_grad_norm: float = None, budget_limit: float = None):
    """Update privacy settings"""
    manager = get_config_manager()
    updates = {}
    
    if epsilon is not None:
        updates['epsilon'] = epsilon
    if noise_multiplier is not None:
        updates['noise_multiplier'] = noise_multiplier
    if max_grad_norm is not None:
        updates['max_grad_norm'] = max_grad_norm
    if budget_limit is not None:
        updates['privacy_budget_limit'] = budget_limit
    
    if updates:
        manager.update_privacy_config(**updates)
    
    return manager.get_privacy_config()
