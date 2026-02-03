"""
Tests for secure neural network operations
"""

import numpy as np
from secure_chatbot.nn.linear import SecureLinear
from secure_chatbot.nn.activations import SecureActivations
from secure_chatbot.crypto.secret_sharing import SecretSharing
from secure_chatbot.crypto.fixed_point import FixedPoint


def test_secure_linear():
    """Test secure linear layer"""
    in_features = 10
    out_features = 5
    batch_size = 3
    
    linear = SecureLinear(in_features, out_features)
    
    # Generate weights
    fp = FixedPoint()
    W = np.random.randn(out_features, in_features).astype(np.float32)
    b = np.random.randn(out_features).astype(np.float32)
    
    W_encoded = fp.encode(W)
    b_encoded = fp.encode(b)
    
    ss = SecretSharing()
    W_share1, W_share2 = ss.share(W_encoded)
    b_share1, b_share2 = ss.share(b_encoded)
    
    linear.set_weights(W_share1, W_share2, b_share1, b_share2)
    
    # Generate input
    x = np.random.randn(batch_size, in_features).astype(np.float32)
    x_encoded = fp.encode(x)
    x_share1, x_share2 = ss.share(x_encoded)
    
    # Forward pass (simplified - would need Beaver triple)
    # This is a placeholder test
    print(f"Linear layer test: input shape {x.shape}, weight shape {W.shape}")


def test_secure_activations():
    """Test secure activation functions"""
    activations = SecureActivations()
    ss = SecretSharing()
    fp = FixedPoint()
    
    # Test ReLU
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    x_encoded = fp.encode(x)
    x_share1, x_share2 = ss.share(x_encoded)
    
    relu_share1, relu_share2 = activations.relu(x_share1, x_share2)
    relu = fp.decode(ss.reconstruct(relu_share1, relu_share2))
    
    expected = np.maximum(0, x)
    assert np.allclose(relu, expected, atol=0.1)
    
    print("ReLU test passed")


if __name__ == '__main__':
    test_secure_linear()
    test_secure_activations()
    print("All NN tests passed!")
