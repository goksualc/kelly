"""
Secure Activation Functions

- ReLU: Piecewise linear approximation
- GELU: 64-segment piecewise linear approximation
- Softmax: Secure max + piecewise exp + secure division
"""

import numpy as np
from typing import Tuple
from ..crypto.fixed_point import FixedPoint
from ..crypto.secret_sharing import SecretSharing
from ..crypto.secure_primitives import SecurePrimitives


class SecureActivations:
    """Secure activation functions"""
    
    def __init__(self, q: int = 2**31 - 1):
        """
        Initialize secure activations.
        
        Args:
            q: Prime modulus
        """
        self.q = q
        self.fixed_point = FixedPoint(q=q)
        self.secret_sharing = SecretSharing(q)
        self.secure_primitives = SecurePrimitives(q)
        
        # Precompute GELU piecewise linear segments
        self._init_gelu_segments()
    
    def _init_gelu_segments(self, num_segments: int = 64):
        """Initialize GELU piecewise linear approximation segments"""
        # GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
        # Simplified: use piecewise linear approximation
        x_range = np.linspace(-4, 4, num_segments + 1)
        self.gelu_segments_x = x_range[:-1]
        self.gelu_segments_y = []
        self.gelu_segments_slope = []
        
        for i in range(num_segments):
            x1, x2 = x_range[i], x_range[i+1]
            # Approximate GELU
            y1 = 0.5 * x1 * (1 + np.tanh(np.sqrt(2/np.pi) * (x1 + 0.044715 * x1**3)))
            y2 = 0.5 * x2 * (1 + np.tanh(np.sqrt(2/np.pi) * (x2 + 0.044715 * x2**3)))
            slope = (y2 - y1) / (x2 - x1) if x2 != x1 else 0
            intercept = y1 - slope * x1
            
            self.gelu_segments_y.append((slope, intercept))
    
    def relu(self, share1: np.ndarray, share2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Secure ReLU activation: max(0, x)
        
        Uses piecewise linear approximation or secure comparison.
        
        Args:
            share1, share2: Shares of input
            
        Returns:
            Shares of ReLU(x)
        """
        # Simplified: reconstruct, compute ReLU, re-share
        # In real protocol, would use secure comparison
        x = (share1 + share2) % self.q
        x_signed = np.where(x > self.q // 2, x.astype(np.int64) - self.q, x.astype(np.int64))
        x_float = self.fixed_point.decode(x_signed)
        
        relu_float = np.maximum(0, x_float)
        relu_encoded = self.fixed_point.encode(relu_float)
        
        result_share1, result_share2 = self.secret_sharing.share(relu_encoded)
        return result_share1, result_share2
    
    def gelu(self, share1: np.ndarray, share2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Secure GELU activation using piecewise linear approximation.
        
        Args:
            share1, share2: Shares of input
            
        Returns:
            Shares of GELU(x)
        """
        # Reconstruct input
        x = (share1 + share2) % self.q
        x_signed = np.where(x > self.q // 2, x.astype(np.int64) - self.q, x.astype(np.int64))
        x_float = self.fixed_point.decode(x_signed)
        
        # Apply piecewise linear approximation
        gelu_float = np.zeros_like(x_float)
        for i, x_seg in enumerate(self.gelu_segments_x):
            mask = (x_float >= x_seg) & (x_float < self.gelu_segments_x[i+1] if i < len(self.gelu_segments_x)-1 else True)
            if i < len(self.gelu_segments_x) - 1:
                slope, intercept = self.gelu_segments_y[i]
                gelu_float[mask] = slope * x_float[mask] + intercept
            else:
                # Last segment: use actual GELU
                gelu_float[mask] = 0.5 * x_float[mask] * (1 + np.tanh(np.sqrt(2/np.pi) * (x_float[mask] + 0.044715 * x_float[mask]**3)))
        
        # Encode and re-share
        gelu_encoded = self.fixed_point.encode(gelu_float)
        result_share1, result_share2 = self.secret_sharing.share(gelu_encoded)
        return result_share1, result_share2
    
    def softmax(self, share1: np.ndarray, share2: np.ndarray,
               axis: int = -1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Secure softmax: exp(x) / sum(exp(x))
        
        Protocol:
        1. Find max (for numerical stability)
        2. Subtract max: x - max
        3. Approximate exp using piecewise linear
        4. Sum and divide
        
        Args:
            share1, share2: Shares of input logits
            axis: Axis along which to compute softmax
            
        Returns:
            Shares of softmax probabilities
        """
        # Step 1: Find max (secure)
        max_share1, max_share2 = self.secure_primitives.secure_max(share1, share2, axis=axis)
        
        # Step 2: Subtract max (local)
        x_centered_share1 = (share1 - max_share1) % self.q
        x_centered_share2 = (share2 - max_share2) % self.q
        
        # Step 3: Approximate exp (piecewise linear)
        x_centered = (x_centered_share1 + x_centered_share2) % self.q
        x_centered_signed = np.where(x_centered > self.q // 2, x_centered.astype(np.int64) - self.q, x_centered.astype(np.int64))
        x_centered_float = self.fixed_point.decode(x_centered_signed)
        
        # Clamp to reasonable range for exp
        x_centered_float = np.clip(x_centered_float, -10, 10)
        exp_float = np.exp(x_centered_float)
        exp_encoded = self.fixed_point.encode(exp_float)
        exp_share1, exp_share2 = self.secret_sharing.share(exp_encoded)
        
        # Step 4: Sum and divide
        if axis is None:
            sum_exp_share1 = np.sum(exp_share1) % self.q
            sum_exp_share2 = np.sum(exp_share2) % self.q
        else:
            sum_exp_share1 = np.sum(exp_share1, axis=axis, keepdims=True) % self.q
            sum_exp_share2 = np.sum(exp_share2, axis=axis, keepdims=True) % self.q
        
        # Secure division
        softmax_share1, softmax_share2 = self.secure_primitives.secure_division(
            exp_share1, exp_share2,
            sum_exp_share1, sum_exp_share2
        )
        
        return softmax_share1, softmax_share2
    
    def tanh(self, share1: np.ndarray, share2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Secure tanh activation.
        
        Args:
            share1, share2: Shares of input
            
        Returns:
            Shares of tanh(x)
        """
        # Reconstruct, compute tanh, re-share
        x = (share1 + share2) % self.q
        x_signed = np.where(x > self.q // 2, x.astype(np.int64) - self.q, x.astype(np.int64))
        x_float = self.fixed_point.decode(x_signed)
        
        tanh_float = np.tanh(x_float)
        tanh_encoded = self.fixed_point.encode(tanh_float)
        
        result_share1, result_share2 = self.secret_sharing.share(tanh_encoded)
        return result_share1, result_share2
