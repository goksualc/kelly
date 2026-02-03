"""
Secure Linear Layers

Implements: [y] = [W] @ [x] + [b]
Uses Beaver partition for matrix multiplication.
"""

import numpy as np
from typing import Tuple, Dict
from ..crypto.beaver_triples import BeaverTriples
from ..crypto.fixed_point import FixedPoint


class SecureLinear:
    """Secure linear transformation layer"""
    
    def __init__(self, in_features: int, out_features: int, q: int = 2**31 - 1):
        """
        Initialize secure linear layer.
        
        Args:
            in_features: Input dimension
            out_features: Output dimension
            q: Prime modulus
        """
        self.in_features = in_features
        self.out_features = out_features
        self.q = q
        self.beaver = BeaverTriples(q)
        self.fixed_point = FixedPoint(q=q)
        
        # Weight and bias (will be secret-shared)
        self.weight = None
        self.bias = None
    
    def set_weights(self, weight_share1: np.ndarray, weight_share2: np.ndarray,
                   bias_share1: np.ndarray = None, bias_share2: np.ndarray = None):
        """
        Set secret-shared weights and bias.
        
        Args:
            weight_share1, weight_share2: Shares of weight matrix (out_features, in_features)
            bias_share1, bias_share2: Optional shares of bias vector (out_features,)
        """
        self.weight_share1 = weight_share1
        self.weight_share2 = weight_share2
        self.bias_share1 = bias_share1 if bias_share1 is not None else np.zeros((self.out_features,), dtype=np.int64)
        self.bias_share2 = bias_share2 if bias_share2 is not None else np.zeros((self.out_features,), dtype=np.int64)
    
    def forward(self, x_share1: np.ndarray, x_share2: np.ndarray,
               matmul_triple: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass: [y] = [W] @ [x] + [b]
        
        Args:
            x_share1, x_share2: Shares of input (batch_size, in_features)
            matmul_triple: Beaver partition for matrix multiplication
            
        Returns:
            Shares of output (batch_size, out_features)
        """
        # Reshape if needed
        if x_share1.ndim == 1:
            x_share1 = x_share1.reshape(1, -1)
            x_share2 = x_share2.reshape(1, -1)
        
        batch_size = x_share1.shape[0]
        
        # Transpose weight for multiplication: W @ x (W is out_features x in_features)
        # We need: (batch_size, in_features) @ (in_features, out_features) = (batch_size, out_features)
        # So we compute: x @ W^T
        W_T_share1 = self.weight_share1.T
        W_T_share2 = self.weight_share2.T
        
        # Generate triple for x @ W^T if not provided
        if matmul_triple is None:
            matmul_triple = self.beaver.generate_matmul_triple(
                batch_size, self.in_features, self.out_features
            )
        
        # Secure matrix multiplication: x @ W^T
        y_share1, y_share2 = self.beaver.secure_matmul(
            x_share1, x_share2,
            W_T_share1, W_T_share2,
            matmul_triple
        )
        
        # Add bias (local operation)
        bias_broadcast1 = np.broadcast_to(self.bias_share1, y_share1.shape)
        bias_broadcast2 = np.broadcast_to(self.bias_share2, y_share2.shape)
        
        y_share1 = (y_share1 + bias_broadcast1) % self.q
        y_share2 = (y_share2 + bias_broadcast2) % self.q
        
        return y_share1, y_share2
