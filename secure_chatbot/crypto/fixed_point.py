"""
Fixed-Point Arithmetic for Secure Computation

Encodes floating-point numbers to fixed-point representation:
Encode(x) = ⌊x · 2^f⌋ mod q

Parameters:
- k = 32 total bits
- f = 16 fractional bits  
- q = 2^31 - 1 (prime modulus)
"""

import numpy as np
from typing import Tuple
import secrets


class FixedPoint:
    """Fixed-point arithmetic for secure computation"""
    
    def __init__(self, k: int = 32, f: int = 16, q: int = 2**31 - 1):
        """
        Initialize fixed-point encoding.
        
        Args:
            k: Total bits (default: 32)
            f: Fractional bits (default: 16)
            q: Prime modulus (default: 2^31 - 1)
        """
        self.k = k
        self.f = f
        self.q = q
        self.scale = 2 ** f
        self.half_scale = self.scale // 2
        
        # Range: [-2^(k-f-1), 2^(k-f-1) - 1]
        self.max_int = 2 ** (k - f - 1) - 1
        self.min_int = -(2 ** (k - f - 1))
    
    def encode(self, x: np.ndarray) -> np.ndarray:
        """
        Encode floating-point to fixed-point integer.
        
        Args:
            x: Floating-point value(s) to encode
            
        Returns:
            Fixed-point encoded integer: ⌊x · 2^f⌋ mod q
        """
        x = np.asarray(x, dtype=np.float64)
        
        # Scale and round
        encoded = np.round(x * self.scale).astype(np.int64)
        
        # Handle overflow: wrap around mod q
        encoded = encoded % self.q
        
        # Handle negative values (convert to positive mod q)
        # If encoded > q/2, treat as negative
        mask = encoded > self.q // 2
        encoded = np.where(mask, encoded - self.q, encoded)
        
        # Clamp to valid range
        encoded = np.clip(encoded, self.min_int, self.max_int)
        
        # Convert back to positive mod q for secret sharing
        encoded = encoded % self.q
        
        return encoded.astype(np.int64)
    
    def decode(self, encoded: np.ndarray) -> np.ndarray:
        """
        Decode fixed-point integer to floating-point.
        
        Args:
            encoded: Fixed-point encoded integer(s)
            
        Returns:
            Floating-point value: encoded / 2^f
        """
        encoded = np.asarray(encoded, dtype=np.int64)
        
        # Handle negative values (if > q/2, subtract q)
        mask = encoded > self.q // 2
        signed = np.where(mask, encoded.astype(np.int64) - self.q, encoded.astype(np.int64))
        
        # Convert to float and scale down
        decoded = signed.astype(np.float64) / self.scale
        
        return decoded
    
    def truncate(self, share1: np.ndarray, share2: np.ndarray,
                 statistical_security: int = 64) -> Tuple[np.ndarray, np.ndarray]:
        """
        Truncate secret-shared value (divide by 2^f) with statistical security.
        
        Uses secure truncation protocol:
        1. Add random r to shares
        2. Reconstruct and truncate
        3. Subtract truncated r
        
        Args:
            share1, share2: Shares of value to truncate
            statistical_security: Statistical security parameter (default: 64)
            
        Returns:
            Shares of truncated value
        """
        share1 = np.asarray(share1, dtype=np.int64)
        share2 = np.asarray(share2, dtype=np.int64)
        shape = share1.shape
        
        # Generate random truncation mask
        # In real protocol, this would be precomputed and shared
        # For now, we simulate by generating random shares
        r_share1 = np.random.randint(0, self.q, size=shape, dtype=np.int64)
        r_share2 = np.random.randint(0, self.q, size=shape, dtype=np.int64)
        
        # Add mask to shares
        masked_share1 = (share1 + r_share1) % self.q
        masked_share2 = (share2 + r_share2) % self.q
        
        # Reconstruct masked value
        masked = (masked_share1 + masked_share2) % self.q
        
        # Handle negative values
        mask = masked > self.q // 2
        masked_signed = np.where(mask, masked.astype(np.int64) - self.q, masked.astype(np.int64))
        
        # Truncate (divide by scale)
        truncated_masked = masked_signed // self.scale
        
        # Convert back to mod q
        truncated_masked = truncated_masked % self.q
        
        # Truncate the mask
        r_signed = np.where(
            r_share1 + r_share2 > self.q // 2,
            (r_share1 + r_share2) % self.q - self.q,
            (r_share1 + r_share2) % self.q
        )
        truncated_r = (r_signed // self.scale) % self.q
        
        # Subtract truncated mask from shares
        # In real protocol, truncated_r would be secret-shared
        # For simplicity, we split it here
        truncated_r_share1 = np.random.randint(0, self.q, size=shape, dtype=np.int64)
        truncated_r_share2 = (truncated_r - truncated_r_share1) % self.q
        
        result_share1 = (truncated_masked - truncated_r_share1) % self.q
        result_share2 = (0 - truncated_r_share2) % self.q  # CP2's share of truncated_masked is 0
        
        return result_share1, result_share2
    
    def add(self, share1_a: np.ndarray, share1_b: np.ndarray,
            share2_a: np.ndarray, share2_b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Add two fixed-point secret-shared values (local operation).
        
        Args:
            share1_a, share1_b: CP1's shares
            share2_a, share2_b: CP2's shares
            
        Returns:
            Shares of a + b
        """
        result_share1 = (share1_a + share1_b) % self.q
        result_share2 = (share2_a + share2_b) % self.q
        return result_share1, result_share2
    
    def multiply(self, share1_a: np.ndarray, share1_b: np.ndarray,
                 share2_a: np.ndarray, share2_b: np.ndarray,
                 beaver_triple: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Multiply two fixed-point secret-shared values using Beaver triple.
        
        Args:
            share1_a, share1_b: CP1's shares of a and b
            share2_a, share2_b: CP2's shares of a and b
            beaver_triple: ([r_a]_1, [r_b]_1, [r_a*r_b]_1, [r_a]_2, [r_b]_2, [r_a*r_b]_2)
            
        Returns:
            Shares of a * b (truncated)
        """
        r_a_share1, r_b_share1, r_ab_share1, r_a_share2, r_b_share2, r_ab_share2 = beaver_triple
        
        # Compute blinded values: e = a - r_a, f = b - r_b
        e_share1 = (share1_a - r_a_share1) % self.q
        e_share2 = (share2_a - r_a_share2) % self.q
        f_share1 = (share1_b - r_b_share1) % self.q
        f_share2 = (share2_b - r_b_share2) % self.q
        
        # Reconstruct e and f (requires communication)
        e = (e_share1 + e_share2) % self.q
        f = (f_share1 + f_share2) % self.q
        
        # Handle negatives
        e_signed = np.where(e > self.q // 2, e.astype(np.int64) - self.q, e.astype(np.int64))
        f_signed = np.where(f > self.q // 2, f.astype(np.int64) - self.q, f.astype(np.int64))
        
        # Compute result: ab = r_a*r_b + e*r_b + f*r_a + e*f
        # Local part: r_a*r_b + e*r_b + f*r_a
        result_share1 = (r_ab_share1 + (e_signed * r_b_share1) % self.q + (f_signed * r_a_share1) % self.q) % self.q
        result_share2 = (r_ab_share2 + (e_signed * r_b_share2) % self.q + (f_signed * r_a_share2) % self.q) % self.q
        
        # Public multiplication: e*f
        ef = (e_signed * f_signed) % self.q
        result_share1 = (result_share1 + ef // 2) % self.q  # Split ef between parties
        result_share2 = (result_share2 + (ef - ef // 2)) % self.q
        
        # Truncate result
        result_share1, result_share2 = self.truncate(result_share1, result_share2)
        
        return result_share1, result_share2
