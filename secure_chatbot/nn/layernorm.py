"""
Secure Layer Normalization

Implements: y = γ · (x - μ) / σ + β

Where:
- μ = mean(x)
- σ = sqrt(var(x))
- var(x) = mean((x - μ)²)
"""

import numpy as np
from typing import Tuple
from ..crypto.fixed_point import FixedPoint
from ..crypto.secret_sharing import SecretSharing
from ..crypto.secure_primitives import SecurePrimitives


class SecureLayerNorm:
    """Secure layer normalization"""
    
    def __init__(self, normalized_shape: Tuple[int, ...], q: int = 2**31 - 1):
        """
        Initialize secure layer normalization.
        
        Args:
            normalized_shape: Shape of features to normalize
            q: Prime modulus
        """
        self.normalized_shape = normalized_shape
        self.q = q
        self.fixed_point = FixedPoint(q=q)
        self.secret_sharing = SecretSharing(q)
        self.secure_primitives = SecurePrimitives(q)
        
        # Scale and shift parameters (will be secret-shared)
        self.gamma_share1 = None
        self.gamma_share2 = None
        self.beta_share1 = None
        self.beta_share2 = None
    
    def set_params(self, gamma_share1: np.ndarray, gamma_share2: np.ndarray,
                  beta_share1: np.ndarray = None, beta_share2: np.ndarray = None):
        """
        Set secret-shared scale (gamma) and shift (beta) parameters.
        
        Args:
            gamma_share1, gamma_share2: Shares of scale parameter
            beta_share1, beta_share2: Optional shares of shift parameter
        """
        self.gamma_share1 = gamma_share1
        self.gamma_share2 = gamma_share2
        
        if beta_share1 is not None and beta_share2 is not None:
            self.beta_share1 = beta_share1
            self.beta_share2 = beta_share2
        else:
            # Default: zero bias
            shape = gamma_share1.shape
            self.beta_share1 = np.zeros(shape, dtype=np.int64)
            self.beta_share2 = np.zeros(shape, dtype=np.int64)
    
    def forward(self, x_share1: np.ndarray, x_share2: np.ndarray,
               axis: int = -1, eps: float = 1e-5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass: normalize input.
        
        Args:
            x_share1, x_share2: Shares of input
            axis: Axis along which to normalize
            eps: Small constant for numerical stability
            
        Returns:
            Shares of normalized output
        """
        # Step 1: Compute mean
        # Mean = sum(x) / n
        n = x_share1.shape[axis] if axis is not None else x_share1.size
        
        sum_x_share1 = np.sum(x_share1, axis=axis, keepdims=True) % self.q
        sum_x_share2 = np.sum(x_share2, axis=axis, keepdims=True) % self.q
        
        # Divide by n (public scalar)
        n_inv_encoded = self.fixed_point.encode(1.0 / n)
        mean_share1 = (sum_x_share1 * n_inv_encoded // self.fixed_point.scale) % self.q
        mean_share2 = (sum_x_share2 * n_inv_encoded // self.fixed_point.scale) % self.q
        
        # Simplified: reconstruct mean for variance computation
        mean = (mean_share1 + mean_share2) % self.q
        mean_signed = np.where(mean > self.q // 2, mean.astype(np.int64) - self.q, mean.astype(np.int64))
        mean_float = self.fixed_point.decode(mean_signed)
        
        # Step 2: Compute variance
        # var = mean((x - mean)²)
        x = (x_share1 + x_share2) % self.q
        x_signed = np.where(x > self.q // 2, x.astype(np.int64) - self.q, x.astype(np.int64))
        x_float = self.fixed_point.decode(x_signed)
        
        x_centered_float = x_float - mean_float
        x_centered_sq_float = x_centered_float ** 2
        x_centered_sq_encoded = self.fixed_point.encode(x_centered_sq_float)
        x_centered_sq_share1, x_centered_sq_share2 = self.secret_sharing.share(x_centered_sq_encoded)
        
        # Mean of squared differences
        sum_sq_share1 = np.sum(x_centered_sq_share1, axis=axis, keepdims=True) % self.q
        sum_sq_share2 = np.sum(x_centered_sq_share2, axis=axis, keepdims=True) % self.q
        
        var_share1 = (sum_sq_share1 * n_inv_encoded // self.fixed_point.scale) % self.q
        var_share2 = (sum_sq_share2 * n_inv_encoded // self.fixed_point.scale) % self.q
        
        # Step 3: Compute std = sqrt(var + eps)
        var = (var_share1 + var_share2) % self.q
        var_signed = np.where(var > self.q // 2, var.astype(np.int64) - self.q, var.astype(np.int64))
        var_float = self.fixed_point.decode(var_signed)
        var_float = np.maximum(var_float, eps)  # Add epsilon
        
        std_float = np.sqrt(var_float)
        std_encoded = self.fixed_point.encode(std_float)
        std_share1, std_share2 = self.secret_sharing.share(std_encoded)
        
        # Step 4: Normalize: (x - mean) / std
        x_centered_encoded = self.fixed_point.encode(x_centered_float)
        x_centered_share1, x_centered_share2 = self.secret_sharing.share(x_centered_encoded)
        
        # Secure division: (x - mean) / std
        normalized_share1, normalized_share2 = self.secure_primitives.secure_division(
            x_centered_share1, x_centered_share2,
            std_share1, std_share2
        )
        
        # Step 5: Scale and shift: γ · normalized + β
        # Multiply by gamma (requires Beaver triple - simplified here)
        gamma = (self.gamma_share1 + self.gamma_share2) % self.q
        gamma_signed = np.where(gamma > self.q // 2, gamma.astype(np.int64) - self.q, gamma.astype(np.int64))
        gamma_float = self.fixed_point.decode(gamma_signed)
        
        normalized = (normalized_share1 + normalized_share2) % self.q
        normalized_signed = np.where(normalized > self.q // 2, normalized.astype(np.int64) - self.q, normalized.astype(np.int64))
        normalized_float = self.fixed_point.decode(normalized_signed)
        
        scaled_float = gamma_float * normalized_float
        scaled_encoded = self.fixed_point.encode(scaled_float)
        scaled_share1, scaled_share2 = self.secret_sharing.share(scaled_encoded)
        
        # Add beta (local)
        result_share1 = (scaled_share1 + self.beta_share1) % self.q
        result_share2 = (scaled_share2 + self.beta_share2) % self.q
        
        return result_share1, result_share2
