"""
Complete Secure Transformer Implementation

Implements full transformer architecture with:
- Embedding layers
- Multi-head self-attention
- Feedforward networks
- Layer normalization
- Residual connections
"""

import numpy as np
from typing import Tuple, Dict, List
from ..crypto.fixed_point import FixedPoint
from ..crypto.secret_sharing import SecretSharing
from .attention import SecureMultiHeadAttention
from .linear import SecureLinear
from .activations import SecureActivations
from .layernorm import SecureLayerNorm


class SecureTransformer:
    """Complete secure transformer model"""
    
    def __init__(self, config: Dict):
        """
        Initialize secure transformer.
        
        Args:
            config: Configuration dictionary with:
                - num_layers: Number of transformer layers
                - hidden_dim: Hidden dimension
                - num_heads: Number of attention heads
                - vocab_size: Vocabulary size
                - max_seq_len: Maximum sequence length
                - ffn_dim: Feedforward network dimension
        """
        self.config = config
        self.num_layers = config['num_layers']
        self.hidden_dim = config['hidden_dim']
        self.num_heads = config['num_heads']
        self.vocab_size = config['vocab_size']
        self.max_seq_len = config.get('max_seq_len', 512)
        self.ffn_dim = config.get('ffn_dim', 4 * self.hidden_dim)
        self.q = config.get('q', 2**31 - 1)
        
        self.fixed_point = FixedPoint(q=self.q)
        self.secret_sharing = SecretSharing(q=self.q)
        self.activations = SecureActivations(q=self.q)
        
        # Embedding layers
        self.token_embedding = None
        self.position_embedding = None
        
        # Transformer layers
        self.layers = []
        for _ in range(self.num_layers):
            layer = {
                'attention': SecureMultiHeadAttention(self.hidden_dim, self.num_heads, self.q),
                'ffn': {
                    'linear1': SecureLinear(self.hidden_dim, self.ffn_dim, self.q),
                    'linear2': SecureLinear(self.ffn_dim, self.hidden_dim, self.q),
                },
                'layernorm1': SecureLayerNorm((self.hidden_dim,), self.q),
                'layernorm2': SecureLayerNorm((self.hidden_dim,), self.q),
            }
            self.layers.append(layer)
        
        # Language model head
        self.lm_head = SecureLinear(self.hidden_dim, self.vocab_size, self.q)
    
    def set_embeddings(self, token_emb_share1: np.ndarray, token_emb_share2: np.ndarray,
                      pos_emb_share1: np.ndarray = None, pos_emb_share2: np.ndarray = None):
        """
        Set secret-shared embedding weights.
        
        Args:
            token_emb_share1, token_emb_share2: Token embedding shares (vocab_size, hidden_dim)
            pos_emb_share1, pos_emb_share2: Optional position embedding shares (max_seq_len, hidden_dim)
        """
        self.token_embedding_share1 = token_emb_share1
        self.token_embedding_share2 = token_emb_share2
        
        if pos_emb_share1 is not None and pos_emb_share2 is not None:
            self.position_embedding_share1 = pos_emb_share1
            self.position_embedding_share2 = pos_emb_share2
        else:
            # Generate position embeddings
            pos_emb = np.random.randn(self.max_seq_len, self.hidden_dim).astype(np.float32)
            pos_emb_encoded = self.fixed_point.encode(pos_emb)
            self.position_embedding_share1, self.position_embedding_share2 = self.secret_sharing.share(pos_emb_encoded)
    
    def set_layer_weights(self, layer_idx: int, 
                         attn_q_share1: np.ndarray, attn_q_share2: np.ndarray,
                         attn_k_share1: np.ndarray, attn_k_share2: np.ndarray,
                         attn_v_share1: np.ndarray, attn_v_share2: np.ndarray,
                         attn_out_share1: np.ndarray, attn_out_share2: np.ndarray,
                         ffn_w1_share1: np.ndarray, ffn_w1_share2: np.ndarray,
                         ffn_b1_share1: np.ndarray, ffn_b1_share2: np.ndarray,
                         ffn_w2_share1: np.ndarray, ffn_w2_share2: np.ndarray,
                         ffn_b2_share1: np.ndarray, ffn_b2_share2: np.ndarray,
                         ln1_gamma_share1: np.ndarray, ln1_gamma_share2: np.ndarray,
                         ln1_beta_share1: np.ndarray, ln1_beta_share2: np.ndarray,
                         ln2_gamma_share1: np.ndarray, ln2_gamma_share2: np.ndarray,
                         ln2_beta_share1: np.ndarray, ln2_beta_share2: np.ndarray):
        """Set weights for a transformer layer"""
        layer = self.layers[layer_idx]
        
        # Attention weights
        layer['attention'].set_weights(
            attn_q_share1, attn_q_share2,
            attn_k_share1, attn_k_share2,
            attn_v_share1, attn_v_share2,
            attn_out_share1, attn_out_share2
        )
        
        # FFN weights
        layer['ffn']['linear1'].set_weights(ffn_w1_share1, ffn_w1_share2, ffn_b1_share1, ffn_b1_share2)
        layer['ffn']['linear2'].set_weights(ffn_w2_share1, ffn_w2_share2, ffn_b2_share1, ffn_b2_share2)
        
        # Layer norm weights
        layer['layernorm1'].set_params(ln1_gamma_share1, ln1_gamma_share2, ln1_beta_share1, ln1_beta_share2)
        layer['layernorm2'].set_params(ln2_gamma_share1, ln2_gamma_share2, ln2_beta_share1, ln2_beta_share2)
    
    def set_lm_head_weights(self, weight_share1: np.ndarray, weight_share2: np.ndarray,
                           bias_share1: np.ndarray = None, bias_share2: np.ndarray = None):
        """Set language model head weights"""
        self.lm_head.set_weights(weight_share1, weight_share2, bias_share1, bias_share2)
    
    def embed(self, input_ids_share1: np.ndarray, input_ids_share2: np.ndarray,
             seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Embed input tokens and positions.
        
        Args:
            input_ids_share1, input_ids_share2: Secret-shared token IDs
            seq_len: Sequence length
            
        Returns:
            Shares of embedded input (seq_len, hidden_dim)
        """
        # Reconstruct token IDs (in real protocol, would use secure table lookup)
        input_ids = (input_ids_share1 + input_ids_share2) % self.q
        input_ids = input_ids.astype(np.int32)
        
        # Lookup token embeddings
        token_emb = self.token_embedding_share1[input_ids] + self.token_embedding_share2[input_ids]
        token_emb = token_emb % self.q
        
        # Add position embeddings
        pos_ids = np.arange(seq_len, dtype=np.int32)
        pos_emb = (self.position_embedding_share1[pos_ids] + self.position_embedding_share2[pos_ids]) % self.q
        
        # Combine (local operation)
        embedded = (token_emb + pos_emb) % self.q
        
        # Re-share
        embedded_share1 = np.random.randint(0, self.q, size=embedded.shape, dtype=np.int64)
        embedded_share2 = (embedded - embedded_share1) % self.q
        
        return embedded_share1, embedded_share2
    
    def forward(self, input_ids_share1: np.ndarray, input_ids_share2: np.ndarray,
               seq_len: int, beaver_triples: List[Dict[str, Dict[str, np.ndarray]]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass through transformer.
        
        Args:
            input_ids_share1, input_ids_share2: Secret-shared input token IDs
            seq_len: Sequence length
            beaver_triples: List of Beaver triple dictionaries for each layer
            
        Returns:
            Shares of output logits (seq_len, vocab_size)
        """
        # Step 1: Embedding
        x_share1, x_share2 = self.embed(input_ids_share1, input_ids_share2, seq_len)
        
        # Step 2: Transformer layers
        for layer_idx, layer in enumerate(self.layers):
            triples = beaver_triples[layer_idx] if layer_idx < len(beaver_triples) else {}
            
            # Self-attention with residual
            attn_out_share1, attn_out_share2 = layer['attention'].forward(
                x_share1, x_share2, seq_len, triples.get('attention', {})
            )
            
            # Residual connection (local)
            x_share1 = (x_share1 + attn_out_share1) % self.q
            x_share2 = (x_share2 + attn_out_share2) % self.q
            
            # Layer norm 1
            x_share1, x_share2 = layer['layernorm1'].forward(x_share1, x_share2)
            
            # Feedforward network
            ffn_triples = triples.get('ffn', {})
            h_share1, h_share2 = layer['ffn']['linear1'].forward(
                x_share1, x_share2, ffn_triples.get('linear1', None)
            )
            
            # Activation (GELU)
            h_share1, h_share2 = self.activations.gelu(h_share1, h_share2)
            
            # Linear 2
            ffn_out_share1, ffn_out_share2 = layer['ffn']['linear2'].forward(
                h_share1, h_share2, ffn_triples.get('linear2', None)
            )
            
            # Residual connection (local)
            x_share1 = (x_share1 + ffn_out_share1) % self.q
            x_share2 = (x_share2 + ffn_out_share2) % self.q
            
            # Layer norm 2
            x_share1, x_share2 = layer['layernorm2'].forward(x_share1, x_share2)
        
        # Step 3: Language model head
        lm_triple = beaver_triples[-1].get('lm_head', {}) if beaver_triples else {}
        logits_share1, logits_share2 = self.lm_head.forward(
            x_share1, x_share2, lm_triple
        )
        
        return logits_share1, logits_share2
    
    def sample(self, logits_share1: np.ndarray, logits_share2: np.ndarray,
              top_k: int = 50, temperature: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample next token from logits (secure argmax or top-k).
        
        Args:
            logits_share1, logits_share2: Shares of logits (vocab_size,)
            top_k: Top-k sampling parameter
            temperature: Sampling temperature
            
        Returns:
            Shares of selected token ID
        """
        # Reconstruct logits (in real protocol, would use secure comparison)
        logits = (logits_share1 + logits_share2) % self.q
        logits_signed = np.where(logits > self.q // 2, logits.astype(np.int64) - self.q, logits.astype(np.int64))
        logits_float = self.fixed_point.decode(logits_signed)
        
        # Apply temperature
        logits_float = logits_float / temperature
        
        # Top-k sampling
        if top_k > 0:
            top_k_indices = np.argsort(logits_float)[-top_k:]
            top_k_logits = logits_float[top_k_indices]
            # Softmax over top-k
            top_k_probs = np.exp(top_k_logits - np.max(top_k_logits))
            top_k_probs = top_k_probs / np.sum(top_k_probs)
            # Sample
            selected_idx = np.random.choice(len(top_k_indices), p=top_k_probs)
            selected_token = top_k_indices[selected_idx]
        else:
            # Greedy: argmax
            selected_token = np.argmax(logits_float)
        
        # Re-share
        token_encoded = self.fixed_point.encode(float(selected_token))
        token_share1, token_share2 = self.secret_sharing.share(token_encoded)
        
        return token_share1, token_share2
