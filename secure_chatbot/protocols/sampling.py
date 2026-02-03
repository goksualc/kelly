"""
Secure Token Sampling Protocol

Implements secure argmax and top-k sampling.
"""

import numpy as np
from typing import Tuple
from ..crypto.secret_sharing import SecretSharing
from ..crypto.fixed_point import FixedPoint
from ..crypto.secure_primitives import SecurePrimitives


def secure_argmax(logits_share1: np.ndarray, logits_share2: np.ndarray,
                 q: int = 2**31 - 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Secure argmax: find index of maximum value.
    
    Args:
        logits_share1, logits_share2: Secret-shared logits
        q: Prime modulus
        
    Returns:
        Shares of argmax index
    """
    secure_primitives = SecurePrimitives(q)
    
    # Find max value
    max_share1, max_share2 = secure_primitives.secure_max(logits_share1, logits_share2)
    
    # Find index (simplified - would use secure comparison)
    logits = (logits_share1 + logits_share2) % q
    logits_signed = np.where(logits > q // 2, logits.astype(np.int64) - q, logits.astype(np.int64))
    argmax_idx = np.argmax(logits_signed)
    
    # Re-share
    secret_sharing = SecretSharing(q)
    fixed_point = FixedPoint(q=q)
    idx_encoded = fixed_point.encode(float(argmax_idx))
    idx_share1, idx_share2 = secret_sharing.share(idx_encoded)
    
    return idx_share1, idx_share2


def secure_top_k_sampling(logits_share1: np.ndarray, logits_share2: np.ndarray,
                         top_k: int, temperature: float = 1.0,
                         q: int = 2**31 - 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Secure top-k sampling.
    
    Args:
        logits_share1, logits_share2: Secret-shared logits
        top_k: Number of top tokens to consider
        temperature: Sampling temperature
        q: Prime modulus
        
    Returns:
        Shares of sampled token index
    """
    # Reconstruct logits (in real protocol, would use secure operations)
    logits = (logits_share1 + logits_share2) % q
    logits_signed = np.where(logits > q // 2, logits.astype(np.int64) - q, logits.astype(np.int64))
    fixed_point = FixedPoint(q=q)
    logits_float = fixed_point.decode(logits_signed)
    
    # Apply temperature
    logits_float = logits_float / temperature
    
    # Top-k
    top_k_indices = np.argsort(logits_float)[-top_k:]
    top_k_logits = logits_float[top_k_indices]
    
    # Softmax
    top_k_probs = np.exp(top_k_logits - np.max(top_k_logits))
    top_k_probs = top_k_probs / np.sum(top_k_probs)
    
    # Sample
    selected_idx = np.random.choice(len(top_k_indices), p=top_k_probs)
    selected_token = top_k_indices[selected_idx]
    
    # Re-share
    secret_sharing = SecretSharing(q)
    token_encoded = fixed_point.encode(float(selected_token))
    token_share1, token_share2 = secret_sharing.share(token_encoded)
    
    return token_share1, token_share2
