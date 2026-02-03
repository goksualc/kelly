"""
Example usage of secure chatbot system
"""

import numpy as np
from secure_chatbot.crypto.secret_sharing import SecretSharing
from secure_chatbot.crypto.fixed_point import FixedPoint
from secure_chatbot.crypto.beaver_triples import BeaverTriples
from secure_chatbot.nn.transformer import SecureTransformer


def example_secret_sharing():
    """Example: Secret sharing and reconstruction"""
    print("=" * 60)
    print("Example 1: Secret Sharing")
    print("=" * 60)
    
    ss = SecretSharing()
    
    # Share a value
    x = np.array([42, 100, 255])
    share1, share2 = ss.share(x)
    
    print(f"Original value: {x}")
    print(f"Share 1: {share1}")
    print(f"Share 2: {share2}")
    
    # Reconstruct
    reconstructed = ss.reconstruct(share1, share2)
    print(f"Reconstructed: {reconstructed}")
    print(f"Match: {np.array_equal(x, reconstructed)}\n")


def example_fixed_point():
    """Example: Fixed-point arithmetic"""
    print("=" * 60)
    print("Example 2: Fixed-Point Arithmetic")
    print("=" * 60)
    
    fp = FixedPoint()
    
    # Encode and decode
    x = np.array([3.14159, -2.5, 100.0])
    encoded = fp.encode(x)
    decoded = fp.decode(encoded)
    
    print(f"Original: {x}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")
    print(f"Error: {np.abs(x - decoded)}\n")


def example_beaver_triples():
    """Example: Beaver multiplication triples"""
    print("=" * 60)
    print("Example 3: Beaver Multiplication Triples")
    print("=" * 60)
    
    beaver = BeaverTriples()
    
    # Generate triple for matrix multiplication
    triple = beaver.generate_matmul_triple(3, 4, 5)
    print(f"Generated triple for (3, 4) @ (4, 5) matrix multiplication")
    print(f"Triple keys: {list(triple.keys())}")
    
    # Test secure matrix multiplication
    A = np.random.randint(0, 100, size=(3, 4))
    B = np.random.randint(0, 100, size=(4, 5))
    expected = A @ B
    
    # Secret share
    ss = SecretSharing()
    A_share1, A_share2 = ss.share(A)
    B_share1, B_share2 = ss.share(B)
    
    # Secure multiplication
    C_share1, C_share2 = beaver.secure_matmul(
        A_share1, A_share2,
        B_share1, B_share2,
        triple
    )
    
    # Reconstruct
    C = ss.reconstruct(C_share1, C_share2)
    
    print(f"A shape: {A.shape}")
    print(f"B shape: {B.shape}")
    print(f"Expected A @ B shape: {expected.shape}")
    print(f"Computed C shape: {C.shape}")
    print(f"Match: {np.allclose(expected % beaver.q, C % beaver.q)}\n")


def example_transformer():
    """Example: Secure transformer initialization"""
    print("=" * 60)
    print("Example 4: Secure Transformer")
    print("=" * 60)
    
    config = {
        'num_layers': 2,
        'hidden_dim': 128,
        'num_heads': 4,
        'vocab_size': 1000,
        'max_seq_len': 32,
        'ffn_dim': 512
    }
    
    transformer = SecureTransformer(config)
    print(f"Initialized transformer with config: {config}")
    print(f"Number of layers: {transformer.num_layers}")
    print(f"Hidden dimension: {transformer.hidden_dim}")
    print(f"Number of heads: {transformer.num_heads}\n")


def example_medical_chatbot():
    """Example: Medical chatbot use case"""
    print("=" * 60)
    print("Example 5: Medical Chatbot Use Case")
    print("=" * 60)
    
    print("User query: 'I'm 55, diabetic, HbA1c 8.5. Constant fatigue. Help?'")
    print("\nPrivacy guarantees:")
    print("  ✓ User's medical information is secret-shared")
    print("  ✓ Neither CP1 nor CP2 sees the query")
    print("  ✓ Model weights remain secret-shared")
    print("  ✓ Only final response is revealed")
    print("\nExpected response (example):")
    print("  'Your HbA1c is above target. Fatigue may be related to high")
    print("   blood sugar. Consult your endocrinologist about medication")
    print("   adjustment. Regular exercise and diet review recommended.'\n")


if __name__ == '__main__':
    example_secret_sharing()
    example_fixed_point()
    example_beaver_triples()
    example_transformer()
    example_medical_chatbot()
    
    print("=" * 60)
    print("All examples completed!")
    print("=" * 60)
