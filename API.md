# API Reference

Complete API documentation for the Secure AI Chatbot system.

## Cryptographic Primitives

### SecretSharing

```python
class SecretSharing:
    def __init__(self, q: int = 2**31 - 1)
    
    def share(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]
    def reconstruct(self, share1: np.ndarray, share2: np.ndarray) -> np.ndarray
    def add(self, share1_a, share1_b, share2_a, share2_b) -> Tuple[np.ndarray, np.ndarray]
    def subtract(self, share1_a, share1_b, share2_a, share2_b) -> Tuple[np.ndarray, np.ndarray]
    def multiply_public(self, share1, share2, scalar: int) -> Tuple[np.ndarray, np.ndarray]
```

### FixedPoint

```python
class FixedPoint:
    def __init__(self, k: int = 32, f: int = 16, q: int = 2**31 - 1)
    
    def encode(self, x: np.ndarray) -> np.ndarray
    def decode(self, encoded: np.ndarray) -> np.ndarray
    def truncate(self, share1, share2, statistical_security: int = 64) -> Tuple[np.ndarray, np.ndarray]
    def multiply(self, share1_a, share1_b, share2_a, share2_b, beaver_triple) -> Tuple[np.ndarray, np.ndarray]
```

### BeaverTriples

```python
class BeaverTriples:
    def __init__(self, q: int = 2**31 - 1)
    
    def generate_triple(self, shape: Tuple[int, ...]) -> Tuple[np.ndarray, ...]
    def generate_matmul_triple(self, d1: int, d2: int, d3: int) -> Dict[str, np.ndarray]
    def secure_matmul(self, A_share1, A_share2, B_share1, B_share2, triple) -> Tuple[np.ndarray, np.ndarray]
```

## Neural Network Operations

### SecureLinear

```python
class SecureLinear:
    def __init__(self, in_features: int, out_features: int, q: int = 2**31 - 1)
    
    def set_weights(self, weight_share1, weight_share2, bias_share1=None, bias_share2=None)
    def forward(self, x_share1, x_share2, matmul_triple) -> Tuple[np.ndarray, np.ndarray]
```

### SecureActivations

```python
class SecureActivations:
    def __init__(self, q: int = 2**31 - 1)
    
    def relu(self, share1, share2) -> Tuple[np.ndarray, np.ndarray]
    def gelu(self, share1, share2) -> Tuple[np.ndarray, np.ndarray]
    def softmax(self, share1, share2, axis: int = -1) -> Tuple[np.ndarray, np.ndarray]
    def tanh(self, share1, share2) -> Tuple[np.ndarray, np.ndarray]
```

### SecureMultiHeadAttention

```python
class SecureMultiHeadAttention:
    def __init__(self, hidden_dim: int, num_heads: int, q: int = 2**31 - 1)
    
    def set_weights(self, q_share1, q_share2, k_share1, k_share2, v_share1, v_share2, 
                   out_share1=None, out_share2=None)
    def forward(self, x_share1, x_share2, seq_len: int, triples: Dict) -> Tuple[np.ndarray, np.ndarray]
```

### SecureLayerNorm

```python
class SecureLayerNorm:
    def __init__(self, normalized_shape: Tuple[int, ...], q: int = 2**31 - 1)
    
    def set_params(self, gamma_share1, gamma_share2, beta_share1=None, beta_share2=None)
    def forward(self, x_share1, x_share2, axis: int = -1, eps: float = 1e-5) -> Tuple[np.ndarray, np.ndarray]
```

### SecureTransformer

```python
class SecureTransformer:
    def __init__(self, config: Dict)
    
    def set_embeddings(self, token_emb_share1, token_emb_share2, 
                      pos_emb_share1=None, pos_emb_share2=None)
    def set_layer_weights(self, layer_idx: int, ...)
    def set_lm_head_weights(self, weight_share1, weight_share2, bias_share1=None, bias_share2=None)
    def embed(self, input_ids_share1, input_ids_share2, seq_len: int) -> Tuple[np.ndarray, np.ndarray]
    def forward(self, input_ids_share1, input_ids_share2, seq_len: int, beaver_triples) -> Tuple[np.ndarray, np.ndarray]
    def sample(self, logits_share1, logits_share2, top_k: int = 50, temperature: float = 1.0) -> Tuple[np.ndarray, np.ndarray]
```

## MPC Parties

### PreprocessingServer

```python
class PreprocessingServer:
    def __init__(self, q: int = 2**31 - 1)
    
    def generate_triples_for_layer(self, layer_idx, seq_len, hidden_dim, num_heads, ffn_dim, vocab_size) -> Dict
    def generate_triples_for_model(self, config: Dict, num_conversations: int = 100) -> List[Dict]
    def distribute_triples(self, triples: Dict, cp1_channel, cp2_channel)
    def generate_shared_seeds(self, num_seeds: int = 1000) -> List[tuple]
```

### ComputingParty

```python
class ComputingParty:
    def __init__(self, party_id: str, q: int = 2**31 - 1)
    
    def load_model_share(self, weights_share: Dict)
    def initialize_model(self, config: Dict)
    def load_beaver_triples(self, triples: Dict)
    def receive_input_shares(self, input_share: np.ndarray) -> np.ndarray
    def secure_forward(self, input_share, seq_len, other_party_id: str) -> Tuple[np.ndarray, Dict]
    def send_output_share(self, output_share, gateway_id: str = 'gateway')
```

### Gateway

```python
class Gateway:
    def __init__(self, q: int = 2**31 - 1)
    
    def secret_share_input(self, user_input: np.ndarray) -> Tuple[np.ndarray, np.ndarray]
    def distribute_input(self, input_share1, input_share2)
    def reconstruct_output(self, output_share1, output_share2, decode: bool = True) -> np.ndarray
    def receive_outputs(self) -> Tuple[np.ndarray, np.ndarray]
    def chat(self, user_message: str, tokenizer, max_tokens: int = 50) -> str
```

## Utilities

### PRG

```python
class PRG:
    def __init__(self, seed: bytes)
    def generate(self, shape: Tuple[int, ...], dtype=np.int64, q: int = 2**31 - 1) -> np.ndarray
    
    @staticmethod
    def generate_shared_seeds() -> Tuple[bytes, bytes]
    @staticmethod
    def generate_shared_randomness(seed_01, seed_02, shape, q) -> Tuple[np.ndarray, np.ndarray]
```

### MPCChannel

```python
class MPCChannel:
    def __init__(self, host: str, port: int, is_server: bool = False)
    
    def connect(self)
    def send(self, message: Any)
    def receive(self, timeout: Optional[float] = None) -> Any
    def close(self)
```

## Configuration

### Model Configuration

```python
config = {
    'num_layers': 12,          # Number of transformer layers
    'hidden_dim': 768,         # Hidden dimension
    'num_heads': 12,            # Number of attention heads
    'vocab_size': 50257,        # Vocabulary size
    'max_seq_len': 512,         # Maximum sequence length
    'ffn_dim': 3072,            # Feedforward network dimension
    'q': 2**31 - 1             # Prime modulus
}
```

## Example Usage

See `example_usage.py` for complete examples.
