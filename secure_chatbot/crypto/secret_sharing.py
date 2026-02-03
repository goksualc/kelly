"""
Additive Secret Sharing over Finite Field Zq

For value x: [x] = ([x]₁, [x]₂) where [x]₁ + [x]₂ = x mod q
Each share is uniformly random and reveals 0 bits about the secret.
"""

import numpy as np
from typing import Tuple, List
import secrets


class SecretSharing:
    """Additive secret sharing over finite field Zq"""
    
    def __init__(self, q: int = 2**31 - 1):
        """
        Initialize secret sharing scheme.
        
        Args:
            q: Prime modulus (default: 2^31 - 1, largest 32-bit prime)
        """
        self.q = q
        self.half_q = q // 2
    
    def share(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Secret share a value or array.
        
        Args:
            x: Value(s) to share (can be scalar, 1D, or multi-dimensional array)
            
        Returns:
            Tuple of two shares ([x]_1, [x]_2) such that [x]_1 + [x]_2 = x mod q
        """
        x = np.asarray(x, dtype=np.int64)
        shape = x.shape
        
        # Generate random share
        share1 = np.random.randint(0, self.q, size=shape, dtype=np.int64)
        
        # Compute second share: share2 = x - share1 mod q
        share2 = (x - share1) % self.q
        
        return share1, share2
    
    def reconstruct(self, share1: np.ndarray, share2: np.ndarray) -> np.ndarray:
        """
        Reconstruct secret from shares.
        
        Args:
            share1: First share [x]_1
            share2: Second share [x]_2
            
        Returns:
            Reconstructed value x = [x]_1 + [x]_2 mod q
        """
        share1 = np.asarray(share1, dtype=np.int64)
        share2 = np.asarray(share2, dtype=np.int64)
        
        result = (share1 + share2) % self.q
        return result
    
    def add(self, share1_a: np.ndarray, share1_b: np.ndarray,
            share2_a: np.ndarray, share2_b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Add two secret-shared values (local operation, no communication).
        
        Args:
            share1_a, share1_b: CP1's shares of a and b
            share2_a, share2_b: CP2's shares of a and b
            
        Returns:
            Shares of a + b: ([a+b]_1, [a+b]_2)
        """
        result_share1 = (share1_a + share1_b) % self.q
        result_share2 = (share2_a + share2_b) % self.q
        return result_share1, result_share2
    
    def subtract(self, share1_a: np.ndarray, share1_b: np.ndarray,
                 share2_a: np.ndarray, share2_b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Subtract two secret-shared values (local operation).
        
        Args:
            share1_a, share1_b: CP1's shares of a and b
            share2_a, share2_b: CP2's shares of a and b
            
        Returns:
            Shares of a - b: ([a-b]_1, [a-b]_2)
        """
        result_share1 = (share1_a - share1_b) % self.q
        result_share2 = (share2_a - share2_b) % self.q
        return result_share1, result_share2
    
    def multiply_public(self, share1: np.ndarray, share2: np.ndarray,
                       scalar: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Multiply secret-shared value by public scalar (local operation).
        
        Args:
            share1, share2: Shares of [x]
            scalar: Public scalar value
            
        Returns:
            Shares of scalar * x: ([scalar*x]_1, [scalar*x]_2)
        """
        scalar = int(scalar) % self.q
        result_share1 = (share1 * scalar) % self.q
        result_share2 = (share2 * scalar) % self.q
        return result_share1, result_share2
    
    def is_uniform_random(self, share: np.ndarray, num_samples: int = 1000) -> bool:
        """
        Test if share appears uniformly random (for security validation).
        
        Args:
            share: Share to test
            num_samples: Number of samples to check
            
        Returns:
            True if share appears uniformly distributed
        """
        if share.size == 0:
            return True
        
        # Flatten and sample
        flat = share.flatten()
        samples = np.random.choice(flat, size=min(num_samples, len(flat)), replace=False)
        
        # Check distribution (simplified: check range and variance)
        if samples.min() < 0 or samples.max() >= self.q:
            return False
        
        # Variance should be high for uniform distribution
        variance = np.var(samples)
        expected_variance = (self.q ** 2) / 12  # Variance of uniform [0, q)
        
        # Allow some tolerance
        return abs(variance - expected_variance) < expected_variance * 0.5
