"""
Beaver Multiplication Triples for Secure Computation

Traditional Beaver triple: For [a]·[b] = [c], use triple ([r_a], [r_b], [r_a·r_b])

FLUID generalization: Beaver partition for polynomials
- Input blinding: x - r (revealed), [r] (secret)
- Offline: CP0 computes f(r₁,...,rₙ) and intermediate terms
- Online: Linear combination (no communication!)
"""

import numpy as np
from typing import Tuple, List, Dict
import secrets


class BeaverTriples:
    """Generate and manage Beaver multiplication triples"""
    
    def __init__(self, q: int = 2**31 - 1):
        """
        Initialize Beaver triple generator.
        
        Args:
            q: Prime modulus
        """
        self.q = q
    
    def generate_triple(self, shape: Tuple[int, ...]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate Beaver multiplication triple for given shape.
        
        Args:
            shape: Shape of values to multiply
            
        Returns:
            Triple shares: ([r_a]_1, [r_b]_1, [r_a*r_b]_1, [r_a]_2, [r_b]_2, [r_a*r_b]_2)
        """
        # Generate random values
        r_a = np.random.randint(0, self.q, size=shape, dtype=np.int64)
        r_b = np.random.randint(0, self.q, size=shape, dtype=np.int64)
        r_ab = (r_a * r_b) % self.q
        
        # Secret share
        r_a_share1 = np.random.randint(0, self.q, size=shape, dtype=np.int64)
        r_a_share2 = (r_a - r_a_share1) % self.q
        
        r_b_share1 = np.random.randint(0, self.q, size=shape, dtype=np.int64)
        r_b_share2 = (r_b - r_b_share1) % self.q
        
        r_ab_share1 = np.random.randint(0, self.q, size=shape, dtype=np.int64)
        r_ab_share2 = (r_ab - r_ab_share1) % self.q
        
        return (r_a_share1, r_b_share1, r_ab_share1,
                r_a_share2, r_b_share2, r_ab_share2)
    
    def generate_matmul_triple(self, d1: int, d2: int, d3: int) -> Dict[str, np.ndarray]:
        """
        Generate Beaver partition for matrix multiplication A @ B where:
        - A: shape (d1, d2)
        - B: shape (d2, d3)
        - Result: shape (d1, d3)
        
        Uses FLUID optimization: single round, O(d1*d2 + d2*d3) communication.
        
        Args:
            d1, d2, d3: Matrix dimensions
            
        Returns:
            Dictionary with shares for CP1 and CP2:
            {
                'A_mask_share1': (d1, d2),
                'A_mask_share2': (d1, d2),
                'B_mask_share1': (d2, d3),
                'B_mask_share2': (d2, d3),
                'AB_mask_share1': (d1, d3),
                'AB_mask_share2': (d1, d3),
            }
        """
        # Generate random masks
        A_mask = np.random.randint(0, self.q, size=(d1, d2), dtype=np.int64)
        B_mask = np.random.randint(0, self.q, size=(d2, d3), dtype=np.int64)
        
        # Compute masked product offline
        AB_mask = (A_mask @ B_mask) % self.q
        
        # Secret share A_mask
        A_mask_share1 = np.random.randint(0, self.q, size=(d1, d2), dtype=np.int64)
        A_mask_share2 = (A_mask - A_mask_share1) % self.q
        
        # Secret share B_mask
        B_mask_share1 = np.random.randint(0, self.q, size=(d2, d3), dtype=np.int64)
        B_mask_share2 = (B_mask - B_mask_share1) % self.q
        
        # Secret share AB_mask
        AB_mask_share1 = np.random.randint(0, self.q, size=(d1, d3), dtype=np.int64)
        AB_mask_share2 = (AB_mask - AB_mask_share1) % self.q
        
        return {
            'A_mask_share1': A_mask_share1,
            'A_mask_share2': A_mask_share2,
            'B_mask_share1': B_mask_share1,
            'B_mask_share2': B_mask_share2,
            'AB_mask_share1': AB_mask_share1,
            'AB_mask_share2': AB_mask_share2,
        }
    
    def secure_matmul(self, A_share1: np.ndarray, A_share2: np.ndarray,
                     B_share1: np.ndarray, B_share2: np.ndarray,
                     triple: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform secure matrix multiplication using Beaver partition.
        
        Protocol:
        1. Compute blinded inputs: E = A - A_mask, F = B - B_mask
        2. Reconstruct E and F (single communication round)
        3. Compute result: AB = AB_mask + E @ B_mask + A_mask @ F + E @ F
        
        Args:
            A_share1, A_share2: Shares of matrix A (d1, d2)
            B_share1, B_share2: Shares of matrix B (d2, d3)
            triple: Beaver partition triple
            
        Returns:
            Shares of A @ B: (result_share1, result_share2) of shape (d1, d3)
        """
        A_mask_share1 = triple['A_mask_share1']
        A_mask_share2 = triple['A_mask_share2']
        B_mask_share1 = triple['B_mask_share1']
        B_mask_share2 = triple['B_mask_share2']
        AB_mask_share1 = triple['AB_mask_share1']
        AB_mask_share2 = triple['AB_mask_share2']
        
        # Step 1: Compute blinded inputs
        E_share1 = (A_share1 - A_mask_share1) % self.q
        E_share2 = (A_share2 - A_mask_share2) % self.q
        F_share1 = (B_share1 - B_mask_share1) % self.q
        F_share2 = (B_share2 - B_mask_share2) % self.q
        
        # Step 2: Reconstruct E and F (requires communication)
        E = (E_share1 + E_share2) % self.q
        F = (F_share1 + F_share2) % self.q
        
        # Handle negative values
        E_signed = np.where(E > self.q // 2, E.astype(np.int64) - self.q, E.astype(np.int64))
        F_signed = np.where(F > self.q // 2, F.astype(np.int64) - self.q, F.astype(np.int64))
        
        # Step 3: Compute result shares
        # Local computation: AB_mask + E @ B_mask + A_mask @ F
        EB_mask = (E_signed @ B_mask_share1) % self.q
        result_share1 = (AB_mask_share1 + EB_mask) % self.q
        
        EB_mask2 = (E_signed @ B_mask_share2) % self.q
        result_share2 = (AB_mask_share2 + EB_mask2) % self.q
        
        A_mask_F = (A_mask_share1 @ F_signed) % self.q
        result_share1 = (result_share1 + A_mask_F) % self.q
        
        A_mask_F2 = (A_mask_share2 @ F_signed) % self.q
        result_share2 = (result_share2 + A_mask_F2) % self.q
        
        # Public computation: E @ F (split between parties)
        EF = (E_signed @ F_signed) % self.q
        result_share1 = (result_share1 + EF // 2) % self.q
        result_share2 = (result_share2 + (EF - EF // 2)) % self.q
        
        return result_share1, result_share2
