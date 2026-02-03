"""
Secure Matrix Multiplication Protocol

Wrapper around Beaver partition for matrix multiplication.
"""

from ..crypto.beaver_triples import BeaverTriples
import numpy as np
from typing import Tuple, Dict


def secure_matrix_multiply(A_share1: np.ndarray, A_share2: np.ndarray,
                           B_share1: np.ndarray, B_share2: np.ndarray,
                           triple: Dict[str, np.ndarray],
                           q: int = 2**31 - 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform secure matrix multiplication.
    
    Args:
        A_share1, A_share2: Shares of matrix A
        B_share1, B_share2: Shares of matrix B
        triple: Beaver partition triple
        q: Prime modulus
        
    Returns:
        Shares of A @ B
    """
    beaver = BeaverTriples(q)
    return beaver.secure_matmul(A_share1, A_share2, B_share1, B_share2, triple)
