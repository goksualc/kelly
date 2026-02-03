# Tutorial: Building a Secure AI Chatbot with MPC

This tutorial provides a step-by-step guide to understanding and using the secure AI chatbot system.

## Table of Contents

1. [Understanding MPC Concepts](#understanding-mpc-concepts)
2. [Setting Up the System](#setting-up-the-system)
3. [Basic Secret Sharing](#basic-secret-sharing)
4. [Secure Computation](#secure-computation)
5. [Building a Secure Transformer](#building-a-secure-transformer)
6. [Running the Complete System](#running-the-complete-system)
7. [Extending the System](#extending-the-system)

## Understanding MPC Concepts

### What is Secure Multiparty Computation?

MPC allows multiple parties to compute a function over their inputs while keeping those inputs private. In our system:

- **User** wants to query an AI model without revealing their question
- **Model Owner** wants to provide AI services without revealing model weights
- **Solution**: Secret share everything and compute on shares

### Secret Sharing

The foundation of our system is **additive secret sharing**:

```python
from secure_chatbot.crypto.secret_sharing import SecretSharing

ss = SecretSharing()

# Share a secret value
secret = 42
share1, share2 = ss.share(secret)

# Each share alone reveals nothing
print(f"Share 1: {share1}")  # Random number
print(f"Share 2: {share2}")  # Random number

# But together, they reconstruct the secret
reconstructed = ss.reconstruct(share1, share2)
print(f"Reconstructed: {reconstructed}")  # 42
```

**Key Property**: Each share is uniformly random and reveals 0 bits about the secret.

### Fixed-Point Arithmetic

Neural networks use floating-point numbers, but MPC works with integers. We use **fixed-point encoding**:

```python
from secure_chatbot.crypto.fixed_point import FixedPoint

fp = FixedPoint()

# Encode float to integer
x = 3.14159
encoded = fp.encode(x)  # Integer representation

# Decode back to float
decoded = fp.decode(encoded)  # ≈ 3.14159
```

## Setting Up the System

### Installation

```bash
# Clone repository
git clone <repository-url>
cd kelly-ai

# Install dependencies
pip install -r requirements.txt
```

### Verify Installation

```bash
# Run examples
python example_usage.py

# Run tests
pytest tests/
```

## Basic Secret Sharing

### Example 1: Sharing a Number

```python
import numpy as np
from secure_chatbot.crypto.secret_sharing import SecretSharing

ss = SecretSharing()

# Share a single number
x = 100
share1, share2 = ss.share(x)
print(f"Original: {x}")
print(f"Share 1: {share1}")
print(f"Share 2: {share2}")
print(f"Reconstructed: {ss.reconstruct(share1, share2)}")
```

### Example 2: Sharing Arrays

```python
# Share an array
arr = np.array([1, 2, 3, 4, 5])
share1, share2 = ss.share(arr)
reconstructed = ss.reconstruct(share1, share2)
assert np.array_equal(arr, reconstructed)
```

### Example 3: Adding Secret-Shared Values

```python
# Addition is local (no communication needed!)
a = np.array([10, 20, 30])
b = np.array([5, 15, 25])

a_share1, a_share2 = ss.share(a)
b_share1, b_share2 = ss.share(b)

# Add shares locally
result_share1, result_share2 = ss.add(a_share1, a_share2, b_share1, b_share2)
result = ss.reconstruct(result_share1, result_share2)

print(f"{a} + {b} = {result}")
```

## Secure Computation

### Beaver Multiplication Triples

Multiplication requires **Beaver triples** (precomputed offline):

```python
from secure_chatbot.crypto.beaver_triples import BeaverTriples

beaver = BeaverTriples()

# Generate triple for matrix multiplication
triple = beaver.generate_matmul_triple(3, 4, 5)  # (3,4) @ (4,5)

# Use for secure multiplication
A = np.random.randint(0, 100, size=(3, 4))
B = np.random.randint(0, 100, size=(4, 5))

A_share1, A_share2 = ss.share(A)
B_share1, B_share2 = ss.share(B)

# Secure matrix multiplication
C_share1, C_share2 = beaver.secure_matmul(
    A_share1, A_share2,
    B_share1, B_share2,
    triple
)

C = ss.reconstruct(C_share1, C_share2)
expected = A @ B
print(f"Match: {np.allclose(expected % beaver.q, C % beaver.q)}")
```

## Building a Secure Transformer

### Step 1: Initialize Transformer

```python
from secure_chatbot.nn.transformer import SecureTransformer

config = {
    'num_layers': 2,
    'hidden_dim': 128,
    'num_heads': 4,
    'vocab_size': 1000,
    'max_seq_len': 32,
    'ffn_dim': 512
}

transformer = SecureTransformer(config)
```

### Step 2: Set Secret-Shared Weights

```python
from secure_chatbot.crypto.fixed_point import FixedPoint

fp = FixedPoint()

# Generate random weights (in practice, would load from model)
token_emb = np.random.randn(config['vocab_size'], config['hidden_dim']).astype(np.float32)
token_emb_encoded = fp.encode(token_emb)

# Secret share
token_emb_share1, token_emb_share2 = ss.share(token_emb_encoded)

# Set embeddings
transformer.set_embeddings(token_emb_share1, token_emb_share2)
```

### Step 3: Forward Pass

```python
# Prepare input
input_ids = np.array([1, 2, 3, 4, 5])  # Token IDs
input_ids_share1, input_ids_share2 = ss.share(input_ids)

# Forward pass (requires Beaver triples)
beaver_triples = [...]  # Precomputed triples
logits_share1, logits_share2 = transformer.forward(
    input_ids_share1, input_ids_share2,
    seq_len=len(input_ids),
    beaver_triples=beaver_triples
)

# Reconstruct logits
logits = ss.reconstruct(logits_share1, logits_share2)
```

## Running the Complete System

### Step 1: Start Preprocessing Server (CP0)

```bash
python main.py --mode cp0
```

This generates Beaver triples offline (can take time, but only needs to run once).

### Step 2: Start Computing Parties

```bash
# Terminal 2
python main.py --mode cp1 --port 8001

# Terminal 3
python main.py --mode cp2 --port 8002
```

### Step 3: Start Gateway

```bash
# Terminal 4
python main.py --mode gateway --port 8000
```

### Step 4: Send Query

```python
from secure_chatbot.parties.gateway import Gateway

gateway = Gateway()
response = gateway.chat("Hello, how are you?", tokenizer)
print(response)
```

## Extending the System

### Adding New Activation Functions

```python
from secure_chatbot.nn.activations import SecureActivations

class SecureActivations:
    def sigmoid(self, share1, share2):
        # Implement secure sigmoid
        # Use piecewise linear approximation
        pass
```

### Adding New Layers

```python
from secure_chatbot.nn.linear import SecureLinear

class SecureConv2d:
    def __init__(self, in_channels, out_channels, kernel_size):
        # Implement secure convolution
        pass
```

### Optimizing Communication

The system uses several optimization techniques:

1. **PRG-Based Randomness**: Reduces communication from CP0
2. **Variable Reuse**: Reuse Beaver partitions across iterations
3. **Batching**: Batch multiple operations in single round

See `DESIGN.md` for details.

## Common Patterns

### Pattern 1: Secret Share → Compute → Reconstruct

```python
# 1. Secret share
share1, share2 = ss.share(input)

# 2. Compute (on shares)
result_share1, result_share2 = secure_operation(share1, share2)

# 3. Reconstruct
result = ss.reconstruct(result_share1, result_share2)
```

### Pattern 2: Local Operations

```python
# Addition, subtraction, public multiplication are local
result_share1, result_share2 = ss.add(a_share1, a_share2, b_share1, b_share2)
```

### Pattern 3: Operations Requiring Communication

```python
# Multiplication, comparison, etc. require communication
# Use Beaver triples for multiplication
result_share1, result_share2 = beaver.secure_matmul(A_share1, A_share2, B_share1, B_share2, triple)
```

## Troubleshooting

### Issue: Shares don't reconstruct correctly

**Solution**: Ensure you're using the same modulus `q` for sharing and reconstruction.

### Issue: Fixed-point encoding loses precision

**Solution**: Adjust fractional bits `f` in `FixedPoint` initialization.

### Issue: Communication errors

**Solution**: Check that all parties are connected and using correct ports.

## Next Steps

1. Read [DESIGN.md](DESIGN.md) for detailed protocol descriptions
2. Read [API.md](API.md) for complete API reference
3. Explore `example_usage.py` for more examples
4. Run tests to verify correctness: `pytest tests/`

## References

- FLUID Paper: Cho, H., Wu, D. J., & Berger, B. (2018). Secure genome-wide association analysis using multiparty computation. Nature Biotechnology, 36(6), 547-551.
- MPC Tutorial: https://www.mpcprotocol.io/
