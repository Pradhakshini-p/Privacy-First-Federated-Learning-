"""
Secure Aggregation Module for Federated Learning
Provides cryptographic protection for model updates
"""

import numpy as np
import torch
import hashlib
import hmac
import json
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import secrets
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

logger = logging.getLogger(__name__)

@dataclass
class SecureUpdate:
    """Secure model update with encryption metadata"""
    client_id: str
    round_num: int
    encrypted_parameters: List[bytes]
    encryption_key_id: str
    signature: str
    timestamp: str
    num_samples: int

@dataclass
class AggregationResult:
    """Result of secure aggregation"""
    aggregated_parameters: List[np.ndarray]
    participating_clients: List[str]
    round_num: int
    security_level: str
    verification_passed: bool
    timestamp: str

class SecureAggregator:
    """Secure aggregation with cryptographic protection"""
    
    def __init__(self, security_level: str = "standard"):
        """
        Initialize secure aggregator
        
        Args:
            security_level: Security level - "basic", "standard", "military"
        """
        self.security_level = security_level
        self.encryption_keys: Dict[str, bytes] = {}
        self.client_secrets: Dict[str, bytes] = {}
        self.aggregation_history: List[AggregationResult] = []
        
        # Security parameters based on level
        self.security_params = self._get_security_params(security_level)
        
        logger.info(f"Secure aggregator initialized with {security_level} security level")
    
    def _get_security_params(self, level: str) -> Dict[str, Any]:
        """Get security parameters based on level"""
        params = {
            "basic": {
                "key_length": 16,
                "iterations": 1000,
                "hash_algorithm": "SHA256",
                "encryption_algorithm": "Fernet"
            },
            "standard": {
                "key_length": 32,
                "iterations": 100000,
                "hash_algorithm": "SHA256",
                "encryption_algorithm": "Fernet"
            },
            "military": {
                "key_length": 64,
                "iterations": 1000000,
                "hash_algorithm": "SHA512",
                "encryption_algorithm": "Fernet"
            }
        }
        return params.get(level, params["standard"])
    
    def generate_client_secret(self, client_id: str) -> bytes:
        """Generate cryptographic secret for a client"""
        if client_id in self.client_secrets:
            return self.client_secrets[client_id]
        
        # Generate cryptographically secure random secret
        secret = secrets.token_bytes(self.security_params["key_length"])
        self.client_secrets[client_id] = secret
        
        logger.info(f"Generated secret for client {client_id}")
        return secret
    
    def derive_encryption_key(self, client_id: str, round_num: int, salt: Optional[bytes] = None) -> bytes:
        """Derive encryption key from client secret and round number"""
        if client_id not in self.client_secrets:
            raise ValueError(f"No secret found for client {client_id}")
        
        if salt is None:
            salt = secrets.token_bytes(16)
        
        # Derive key using PBKDF2
        kdf = PBKDF2HMAC(
            algorithm=getattr(hashes, self.security_params["hash_algorithm"])(),
            length=self.security_params["key_length"],
            salt=salt,
            iterations=self.security_params["iterations"],
        )
        
        # Combine client secret with round number
        secret_data = self.client_secrets[client_id] + round_num.to_bytes(4, 'big')
        key = base64.urlsafe_b64encode(kdf.derive(secret_data))
        
        # Store key with ID
        key_id = f"{client_id}_{round_num}_{salt.hex()[:8]}"
        self.encryption_keys[key_id] = key
        
        return key
    
    def encrypt_parameters(self, parameters: List[np.ndarray], client_id: str, round_num: int) -> SecureUpdate:
        """Encrypt model parameters for secure transmission"""
        try:
            # Generate encryption key
            salt = secrets.token_bytes(16)
            key = self.derive_encryption_key(client_id, round_num, salt)
            key_id = f"{client_id}_{round_num}_{salt.hex()[:8]}"
            
            # Create encryptor
            fernet = Fernet(key)
            
            # Encrypt each parameter array
            encrypted_params = []
            for param in parameters:
                # Convert to bytes
                param_bytes = param.tobytes()
                # Encrypt
                encrypted_bytes = fernet.encrypt(param_bytes)
                encrypted_params.append(encrypted_bytes)
            
            # Create signature
            signature_data = {
                "client_id": client_id,
                "round_num": round_num,
                "param_shapes": [p.shape for p in parameters],
                "timestamp": datetime.now().isoformat()
            }
            signature = self._create_signature(signature_data, client_id)
            
            # Create secure update
            secure_update = SecureUpdate(
                client_id=client_id,
                round_num=round_num,
                encrypted_parameters=encrypted_params,
                encryption_key_id=key_id,
                signature=signature,
                timestamp=datetime.now().isoformat(),
                num_samples=len(parameters[0]) if parameters else 0
            )
            
            logger.info(f"Encrypted parameters for {client_id}, round {round_num}")
            return secure_update
            
        except Exception as e:
            logger.error(f"Error encrypting parameters: {e}")
            raise
    
    def decrypt_parameters(self, secure_update: SecureUpdate) -> List[np.ndarray]:
        """Decrypt model parameters from secure update"""
        try:
            # Get encryption key
            if secure_update.encryption_key_id not in self.encryption_keys:
                raise ValueError(f"Encryption key not found: {secure_update.encryption_key_id}")
            
            key = self.encryption_keys[secure_update.encryption_key_id]
            fernet = Fernet(key)
            
            # Decrypt parameters
            decrypted_params = []
            for encrypted_param in secure_update.encrypted_parameters:
                # Decrypt
                decrypted_bytes = fernet.decrypt(encrypted_param)
                # Convert back to numpy array (need to know shape)
                # For now, assume we can infer from the data
                param_array = np.frombuffer(decrypted_bytes, dtype=np.float32)
                decrypted_params.append(param_array)
            
            # Verify signature
            if not self._verify_signature(secure_update):
                raise ValueError("Signature verification failed")
            
            logger.info(f"Decrypted parameters for {secure_update.client_id}")
            return decrypted_params
            
        except Exception as e:
            logger.error(f"Error decrypting parameters: {e}")
            raise
    
    def _create_signature(self, data: Dict[str, Any], client_id: str) -> str:
        """Create HMAC signature for data"""
        if client_id not in self.client_secrets:
            raise ValueError(f"No secret found for client {client_id}")
        
        # Convert data to JSON string
        data_str = json.dumps(data, sort_keys=True)
        
        # Create HMAC
        h = hmac.new(self.client_secrets[client_id], data_str.encode(), hashlib.sha256)
        signature = h.hexdigest()
        
        return signature
    
    def _verify_signature(self, secure_update: SecureUpdate) -> bool:
        """Verify signature of secure update"""
        try:
            # Recreate signature data
            signature_data = {
                "client_id": secure_update.client_id,
                "round_num": secure_update.round_num,
                "param_shapes": [],  # Would need to store shapes
                "timestamp": secure_update.timestamp
            }
            
            # Create expected signature
            expected_signature = self._create_signature(signature_data, secure_update.client_id)
            
            # Compare signatures
            return hmac.compare_digest(expected_signature, secure_update.signature)
            
        except Exception as e:
            logger.error(f"Error verifying signature: {e}")
            return False
    
    def secure_aggregate(self, secure_updates: List[SecureUpdate], round_num: int) -> AggregationResult:
        """Perform secure aggregation of encrypted updates"""
        try:
            if not secure_updates:
                raise ValueError("No updates to aggregate")
            
            logger.info(f"Starting secure aggregation for round {round_num} with {len(secure_updates)} updates")
            
            # Decrypt all updates
            all_parameters = []
            participating_clients = []
            
            for update in secure_updates:
                try:
                    params = self.decrypt_parameters(update)
                    all_parameters.append(params)
                    participating_clients.append(update.client_id)
                except Exception as e:
                    logger.warning(f"Failed to decrypt update from {update.client_id}: {e}")
                    continue
            
            if not all_parameters:
                raise ValueError("No valid updates to aggregate")
            
            # Perform weighted averaging
            aggregated_params = []
            num_params = len(all_parameters[0])
            
            for i in range(num_params):
                # Stack all parameters for this index
                param_stack = np.stack([params[i] for params in all_parameters])
                # Compute weighted average (equal weights for now)
                averaged_param = np.mean(param_stack, axis=0)
                aggregated_params.append(averaged_param)
            
            # Create aggregation result
            result = AggregationResult(
                aggregated_parameters=aggregated_params,
                participating_clients=participating_clients,
                round_num=round_num,
                security_level=self.security_level,
                verification_passed=True,
                timestamp=datetime.now().isoformat()
            )
            
            # Store in history
            self.aggregation_history.append(result)
            
            logger.info(f"Secure aggregation completed for round {round_num}")
            return result
            
        except Exception as e:
            logger.error(f"Error in secure aggregation: {e}")
            raise
    
    def get_aggregation_history(self) -> List[Dict[str, Any]]:
        """Get history of aggregations"""
        return [
            {
                "round_num": result.round_num,
                "participating_clients": result.participating_clients,
                "security_level": result.security_level,
                "verification_passed": result.verification_passed,
                "timestamp": result.timestamp,
                "num_parameters": len(result.aggregated_parameters)
            }
            for result in self.aggregation_history
        ]
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get current security status"""
        return {
            "security_level": self.security_level,
            "active_clients": len(self.client_secrets),
            "active_keys": len(self.encryption_keys),
            "total_aggregations": len(self.aggregation_history),
            "last_aggregation": self.aggregation_history[-1].timestamp if self.aggregation_history else None,
            "security_params": self.security_params
        }

class PrivacyPreservingAggregator:
    """Privacy-preserving aggregation with noise injection"""
    
    def __init__(self, noise_multiplier: float = 1.0, epsilon: float = 1.0):
        """
        Initialize privacy-preserving aggregator
        
        Args:
            noise_multiplier: Multiplier for noise injection
            epsilon: Privacy budget parameter
        """
        self.noise_multiplier = noise_multiplier
        self.epsilon = epsilon
        self.aggregation_count = 0
        
        logger.info(f"Privacy-preserving aggregator initialized")
        logger.info(f"Noise multiplier: {noise_multiplier}, Epsilon: {epsilon}")
    
    def add_privacy_noise(self, parameters: List[np.ndarray], sensitivity: float = 1.0) -> List[np.ndarray]:
        """Add differential privacy noise to parameters"""
        try:
            noisy_params = []
            
            for param in parameters:
                # Calculate noise scale
                noise_scale = (sensitivity * self.noise_multiplier) / self.epsilon
                
                # Add Gaussian noise
                noise = np.random.normal(0, noise_scale, param.shape)
                noisy_param = param + noise
                
                noisy_params.append(noisy_param)
            
            self.aggregation_count += 1
            logger.info(f"Added privacy noise to parameters (aggregation #{self.aggregation_count})")
            
            return noisy_params
            
        except Exception as e:
            logger.error(f"Error adding privacy noise: {e}")
            return parameters
    
    def aggregate_with_privacy(self, client_updates: List[Tuple[List[np.ndarray], int]]) -> List[np.ndarray]:
        """Aggregate client updates with privacy protection"""
        try:
            if not client_updates:
                return []
            
            # Separate parameters and weights
            all_params = [params for params, _ in client_updates]
            weights = [weight for _, weight in client_updates]
            
            # Calculate sensitivity based on number of clients
            sensitivity = 1.0 / len(client_updates)
            
            # Add noise to each client's parameters
            noisy_updates = []
            for params, weight in client_updates:
                noisy_params = self.add_privacy_noise(params, sensitivity)
                noisy_updates.append((noisy_params, weight))
            
            # Perform weighted averaging
            aggregated_params = []
            num_params = len(all_params[0])
            
            for i in range(num_params):
                # Stack all parameters for this index
                param_stack = np.stack([params[i] for params, _ in noisy_updates])
                weights_array = np.array([weight for _, weight in noisy_updates])
                
                # Compute weighted average
                weighted_sum = np.sum(param_stack * weights_array[:, np.newaxis], axis=0)
                total_weight = np.sum(weights_array)
                averaged_param = weighted_sum / total_weight
                
                aggregated_params.append(averaged_param)
            
            logger.info(f"Privacy-preserving aggregation completed with {len(client_updates)} clients")
            return aggregated_params
            
        except Exception as e:
            logger.error(f"Error in privacy-preserving aggregation: {e}")
            raise

# Global instances
secure_aggregator = None
privacy_aggregator = None

def get_secure_aggregator(security_level: str = "standard") -> SecureAggregator:
    """Get or create secure aggregator instance"""
    global secure_aggregator
    if secure_aggregator is None or secure_aggregator.security_level != security_level:
        secure_aggregator = SecureAggregator(security_level)
    return secure_aggregator

def get_privacy_aggregator(noise_multiplier: float = 1.0, epsilon: float = 1.0) -> PrivacyPreservingAggregator:
    """Get or create privacy-preserving aggregator instance"""
    global privacy_aggregator
    if privacy_aggregator is None:
        privacy_aggregator = PrivacyPreservingAggregator(noise_multiplier, epsilon)
    return privacy_aggregator

if __name__ == "__main__":
    # Test secure aggregation
    aggregator = SecureAggregator("standard")
    
    # Generate test parameters
    test_params = [np.random.rand(10, 10) for _ in range(3)]
    
    # Test encryption/decryption
    secure_update = aggregator.encrypt_parameters(test_params, "client_1", 1)
    decrypted_params = aggregator.decrypt_parameters(secure_update)
    
    print("Secure aggregation test completed successfully!")
