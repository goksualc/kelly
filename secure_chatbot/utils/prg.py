"""
Pseudorandom Generator (PRG) for Shared Randomness

Instead of CP0 sending random r to both parties:
- seed_01 = shared_between(CP0, CP1)
- seed_02 = shared_between(CP0, CP2)

CP1 generates: r_1 = PRG(seed_01)
CP2 generates: r_2 = PRG(seed_02)
CP0 computes offline: r = r_1 + r_2

Result: ZERO communication from CP0 in online phase!
"""

import numpy as np
import hashlib
import secrets
from typing import Tuple


class PRG:
    """Pseudorandom generator for shared randomness"""
    
    def __init__(self, seed: bytes):
        """
        Initialize PRG with seed.
        
        Args:
            seed: Seed bytes for PRG
        """
        self.seed = seed
        self.state = hashlib.sha256(seed).digest()
    
    def generate(self, shape: Tuple[int, ...], dtype=np.int64, q: int = 2**31 - 1) -> np.ndarray:
        """
        Generate pseudorandom array.
        
        Args:
            shape: Output shape
            dtype: Output dtype
            q: Modulus for values
            
        Returns:
            Pseudorandom array
        """
        size = np.prod(shape)
        num_bytes = size * np.dtype(dtype).itemsize
        
        # Generate random bytes
        random_bytes = b''
        while len(random_bytes) < num_bytes:
            self.state = hashlib.sha256(self.state).digest()
            random_bytes += self.state
        
        # Convert to array
        random_bytes = random_bytes[:num_bytes]
        # Use frombuffer to convert bytes to numpy array
        arr = np.frombuffer(random_bytes, dtype=dtype)
        arr = arr.reshape(shape)
        
        # Mod q
        arr = arr % q
        
        return arr
    
    @staticmethod
    def generate_shared_seeds() -> Tuple[bytes, bytes]:
        """
        Generate shared seeds for CP0-CP1 and CP0-CP2.
        
        Returns:
            Tuple of (seed_01, seed_02)
        """
        seed_01 = secrets.token_bytes(32)
        seed_02 = secrets.token_bytes(32)
        return seed_01, seed_02
    
    @staticmethod
    def generate_shared_randomness(seed_01: bytes, seed_02: bytes, 
                                   shape: Tuple[int, ...], q: int = 2**31 - 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate shared randomness using PRG.
        
        CP1 generates r_1 = PRG(seed_01)
        CP2 generates r_2 = PRG(seed_02)
        CP0 computes r = r_1 + r_2 (offline)
        
        Args:
            seed_01: Seed shared between CP0 and CP1
            seed_02: Seed shared between CP0 and CP2
            shape: Shape of random values
            q: Modulus
            
        Returns:
            Tuple of (r_1, r_2) where r_1 + r_2 = r mod q
        """
        prg1 = PRG(seed_01)
        prg2 = PRG(seed_02)
        
        r_1 = prg1.generate(shape, q=q)
        r_2 = prg2.generate(shape, q=q)
        
        return r_1, r_2
