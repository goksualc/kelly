# Design Document: Secure AI Chatbot with MPC

## Overview

This document describes the design and implementation of a privacy-preserving AI chatbot system using Secure Multiparty Computation (MPC) based on the FLUID protocol.

## Cryptographic Primitives

### Secret Sharing

**Additive Secret Sharing over Zq**

For a value x, we create shares [x] = ([x]₁, [x]₂) such that:
- [x]₁ + [x]₂ = x mod q
- [x]₁ is uniformly random
- Each share reveals 0 bits about the secret

**Properties:**
- Addition/subtraction are local (no communication)
- Multiplication requires Beaver triples
- Information-theoretic security

### Fixed-Point Arithmetic

**Encoding:**
```
Encode(x) = ⌊x · 2^f⌋ mod q
```

**Parameters:**
- k = 32 total bits
- f = 16 fractional bits
- q = 2^31 - 1 (prime modulus)

**Operations:**
- Addition: Local (no communication)
- Multiplication: Requires Beaver triple + truncation
- Truncation: Secure protocol with statistical security κ=64

### Beaver Multiplication Triples

**Traditional Beaver Triple:**
For [a]·[b] = [c], use triple ([r_a], [r_b], [r_a·r_b])

**FLUID Beaver Partition:**
For polynomial f(x₁,...,xₙ):
1. Input blinding: x - r (revealed), [r] (secret)
2. Offline: CP0 computes f(r₁,...,rₙ) and intermediate terms
3. Online: Linear combination (no communication!)

**Matrix Multiplication:**
- Communication: O(d₁·d₂ + d₂·d₃) instead of O(d₁·d₂·d₃)
- Single round protocol
- Variable reuse: partition once, use many times

## System Architecture

### Party Roles

#### CP0 (Preprocessing Server)
- **Purpose**: Generate Beaver triples offline
- **Input Dependency**: None (can precompute days in advance)
- **Online Participation**: None
- **Output**: Shares of triples distributed to CP1 and CP2

#### CP1 & CP2 (Computing Parties)
- **Purpose**: Perform secure computation
- **Hold**: Shares of model weights and user inputs
- **Communication**: Only during Beaver partition reconstruction
- **Security**: Can corrupt one party (not both)

#### Gateway
- **Purpose**: User interface and secret sharing
- **Functions**:
  - Accept user inputs
  - Split inputs into shares
  - Distribute to CP1 and CP2
  - Reconstruct final outputs

## Secure Neural Network Operations

### Secure Linear Layer

```
[y] = [W] @ [x] + [b]
```

**Protocol:**
1. Use Beaver partition for W @ x
2. Addition of bias is local

### Secure Activation Functions

**ReLU:**
- Piecewise linear approximation
- Or secure comparison protocol

**GELU:**
- 64-segment piecewise linear approximation
- Precomputed segments for efficiency

**Softmax:**
1. Secure max finding (for numerical stability)
2. Piecewise exp approximation
3. Secure division (Goldschmidt algorithm)

### Secure Attention

```
Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
```

**Steps:**
1. [scores] = [Q] @ [K]^T (secure matmul)
2. [attn] = softmax([scores]) (secure softmax)
3. [output] = [attn] @ [V] (secure matmul)

### Secure Layer Normalization

```
y = γ · (x - μ) / σ + β
```

**Steps:**
1. [mean] = sum([x]) / n
2. [var] = sum(([x] - [mean])²) / n
3. [std] = sqrt([var])
4. [normalized] = ([x] - [mean]) / [std]
5. [output] = γ · [normalized] + β

## Optimization Techniques

### PRG-Based Shared Randomness

Instead of CP0 sending random r to both parties:
- seed_01 = shared_between(CP0, CP1)
- seed_02 = shared_between(CP0, CP2)

CP1 generates: r_1 = PRG(seed_01)
CP2 generates: r_2 = PRG(seed_02)
CP0 computes offline: r = r_1 + r_2

**Result**: ZERO communication from CP0 in online phase!

### Variable Reuse Pattern

For power iteration: A^T @ A @ x (repeated T times)
- DON'T Beaver partition A each time (wasteful)
- DO partition A once, reuse across all T iterations

**Example**: Embedding matrix used for every token
- Partition once at start
- Reuse for all lookups

### Depth-1 Circuit Batching

When multiple matrix multiplications have depth=1:
- Batch all input blinding into single communication round
- Compute all offline results together
- Reconstruct all outputs in parallel

## Protocol Execution Flow

### Offline Phase (CP0)

```python
def preprocessing_phase(model_config, num_conversations):
    for conv_id in range(num_conversations):
        for layer_idx in range(num_layers):
            # Generate triples for attention
            qk_triple = generate_matmul_triple(seq_len, hidden, seq_len)
            av_triple = generate_matmul_triple(seq_len, seq_len, hidden)
            
            # Generate triples for feedforward
            ff1_triple = generate_matmul_triple(seq_len, hidden, ffn_dim)
            ff2_triple = generate_matmul_triple(seq_len, ffn_dim, hidden)
            
            # Share with CP1, CP2
            distribute_shares(qk_triple, av_triple, ff1_triple, ff2_triple)
```

### Online Phase (CP1, CP2, User)

```python
def inference_phase(user_input):
    # 1. User → Gateway: Send query
    query = user_input
    
    # 2. Gateway: Secret share
    [query]_1, [query]_2 = secret_share(query)
    send([query]_1, to=CP1)
    send([query]_2, to=CP2)
    
    # 3. CP1, CP2: Secure inference
    [response]_1, [response]_2 = secure_transformer_forward([query])
    
    # 4. Reconstruct output
    response = reconstruct([response]_1, [response]_2)
    
    # 5. Gateway → User: Return response
    return response
```

## Security Analysis

### Information-Theoretic Privacy

Each share is uniformly random → reveals 0 bits about secret.

View of CP1/CP2 consists only of:
- Their own shares (uniform random)
- Blinded values x-r where r is uniform random

### Threat Model

- **Semi-honest adversaries** (honest-but-curious)
- Can corrupt CP1 OR CP2 (not both)
- Can corrupt CP0 in offline phase (but no online collusion)
- Active security extension possible via SPDZ MACs

### Leakage Analysis

**What is revealed:**
- Message length (can be padded)
- Timing (can be made constant-time)
- Final output (necessary for functionality)

**What is NEVER revealed:**
- User input: Information-theoretic
- Model weights: Information-theoretic
- Intermediate activations: Information-theoretic
- Attention patterns: Information-theoretic

## Performance Analysis

### Communication Complexity

**Per token:**
- Input sharing: O(vocab_size) or O(hidden_dim)
- Per layer: O(seq_len · hidden_dim)
- Total: O(num_layers · seq_len · hidden_dim)

### Round Complexity

- Per layer: 1 round (Beaver partition reconstruction)
- Total: num_layers rounds

### Time Estimates (GPT-2 Small, 512 tokens)

- Per token generation: ~5 seconds
- 50 token response: ~4 minutes
- Communication per token: ~275 MB

## Implementation Details

### Data Structures

**Secret Shares:**
- Stored as numpy arrays (int64)
- Mod q arithmetic
- Shape preserved

**Beaver Triples:**
- Dictionary structure:
  ```python
  {
      'A_mask_share1': np.ndarray,
      'A_mask_share2': np.ndarray,
      'B_mask_share1': np.ndarray,
      'B_mask_share2': np.ndarray,
      'AB_mask_share1': np.ndarray,
      'AB_mask_share2': np.ndarray,
  }
  ```

### Communication Protocol

- TCP sockets for reliable delivery
- Pickle for serialization
- Compression for large arrays
- Threading for async communication

## Future Extensions

### Multi-Party Extension
- Support n parties (tolerate n-1 collusions)
- n-out-of-n secret sharing scheme

### Active Security
- SPDZ-style MACs for malicious adversaries
- Verification of computation correctness

### Differential Privacy
- Add calibrated noise to outputs
- ε-differential privacy guarantee

### Model Fine-tuning
- Secure gradient computation
- Secure parameter updates
- Privacy-preserving backpropagation

### RAG Integration
- Secure document retrieval
- Private knowledge base querying
