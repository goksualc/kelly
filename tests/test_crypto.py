"""
Tests for cryptographic primitives
"""

import numpy as np
import pytest
from secure_chatbot.crypto.secret_sharing import SecretSharing
from secure_chatbot.crypto.fixed_point import FixedPoint
from secure_chatbot.crypto.beaver_triples import BeaverTriples


def test_secret_sharing():
    """Test secret sharing and reconstruction"""
    ss = SecretSharing()
    
    # Test scalar
    x = 42
    share1, share2 = ss.share(x)
    reconstructed = ss.reconstruct(share1, share2)
    assert reconstructed == x
    
    # Test array
    x = np.array([1, 2, 3, 4, 5])
    share1, share2 = ss.share(x)
    reconstructed = ss.reconstruct(share1, share2)
    assert np.array_equal(x, reconstructed)
    
    # Test matrix
    x = np.random.randint(0, 100, size=(5, 5))
    share1, share2 = ss.share(x)
    reconstructed = ss.reconstruct(share1, share2)
    assert np.allclose(x, reconstructed)


def test_secret_sharing_add():
    """Test addition of secret-shared values"""
    ss = SecretSharing()
    
    a = np.array([10, 20, 30])
    b = np.array([5, 15, 25])
    
    a_share1, a_share2 = ss.share(a)
    b_share1, b_share2 = ss.share(b)
    
    result_share1, result_share2 = ss.add(a_share1, a_share2, b_share1, b_share2)
    result = ss.reconstruct(result_share1, result_share2)
    
    assert np.array_equal(result, (a + b) % ss.q)


def test_fixed_point_encode_decode():
    """Test fixed-point encoding and decoding"""
    fp = FixedPoint()
    
    x = np.array([3.14159, -2.5, 100.0, 0.0])
    encoded = fp.encode(x)
    decoded = fp.decode(encoded)
    
    # Allow small error due to quantization
    assert np.allclose(x, decoded, atol=1e-3)


def test_beaver_triple_generation():
    """Test Beaver triple generation"""
    beaver = BeaverTriples()
    
    triple = beaver.generate_matmul_triple(3, 4, 5)
    
    assert 'A_mask_share1' in triple
    assert 'A_mask_share2' in triple
    assert 'B_mask_share1' in triple
    assert 'B_mask_share2' in triple
    assert 'AB_mask_share1' in triple
    assert 'AB_mask_share2' in triple
    
    # Check shapes
    assert triple['A_mask_share1'].shape == (3, 4)
    assert triple['B_mask_share1'].shape == (4, 5)
    assert triple['AB_mask_share1'].shape == (3, 5)


def test_secure_matrix_multiplication():
    """Test secure matrix multiplication using Beaver triples"""
    beaver = BeaverTriples()
    ss = SecretSharing()
    
    # Generate test matrices
    A = np.random.randint(0, 100, size=(3, 4))
    B = np.random.randint(0, 100, size=(4, 5))
    expected = (A @ B) % beaver.q
    
    # Secret share
    A_share1, A_share2 = ss.share(A)
    B_share1, B_share2 = ss.share(B)
    
    # Generate triple
    triple = beaver.generate_matmul_triple(3, 4, 5)
    
    # Secure multiplication
    C_share1, C_share2 = beaver.secure_matmul(
        A_share1, A_share2,
        B_share1, B_share2,
        triple
    )
    
    # Reconstruct
    C = ss.reconstruct(C_share1, C_share2)
    
    # Check result (mod q)
    assert np.allclose(expected, C % beaver.q)


if __name__ == '__main__':
    test_secret_sharing()
    test_secret_sharing_add()
    test_fixed_point_encode_decode()
    test_beaver_triple_generation()
    test_secure_matrix_multiplication()
    print("All crypto tests passed!")
