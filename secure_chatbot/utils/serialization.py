"""
Efficient Serialization for Secret Shares

Optimizes data transfer for large arrays.
"""

import numpy as np
import pickle
import zlib
from typing import Any


def serialize_shares(share1: np.ndarray, share2: np.ndarray, compress: bool = True) -> bytes:
    """
    Serialize secret shares efficiently.
    
    Args:
        share1, share2: Shares to serialize
        compress: Whether to compress data
        
    Returns:
        Serialized bytes
    """
    data = {
        'share1': share1,
        'share2': share2,
        'shape': share1.shape,
        'dtype': str(share1.dtype)
    }
    
    serialized = pickle.dumps(data)
    
    if compress:
        serialized = zlib.compress(serialized)
    
    return serialized


def deserialize_shares(data: bytes, compressed: bool = True) -> tuple:
    """
    Deserialize secret shares.
    
    Args:
        data: Serialized bytes
        compressed: Whether data is compressed
        
    Returns:
        Tuple of (share1, share2)
    """
    if compressed:
        data = zlib.decompress(data)
    
    obj = pickle.loads(data)
    return obj['share1'], obj['share2']


def serialize_triples(triples: dict, compress: bool = True) -> bytes:
    """
    Serialize Beaver triples.
    
    Args:
        triples: Dictionary of Beaver triples
        compress: Whether to compress
        
    Returns:
        Serialized bytes
    """
    serialized = pickle.dumps(triples)
    
    if compress:
        serialized = zlib.compress(serialized)
    
    return serialized


def deserialize_triples(data: bytes, compressed: bool = True) -> dict:
    """
    Deserialize Beaver triples.
    
    Args:
        data: Serialized bytes
        compressed: Whether data is compressed
        
    Returns:
        Dictionary of Beaver triples
    """
    if compressed:
        data = zlib.decompress(data)
    
    return pickle.loads(data)
