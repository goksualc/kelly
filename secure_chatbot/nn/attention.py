"""
Secure Multi-Head Self-Attention

Implements: Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
"""

import numpy as np
from typing import Tuple, Dict
from ..crypto.beaver_triples import BeaverTriples
from ..crypto.fixed_point import FixedPoint
from ..crypto.secret_sharing import SecretSharing
from ..nn.activations import SecureActivations


class SecureMultiHeadAttention:
    """Secure multi-head self-attention mechanism"""
    
    def __init__(self, hidden_dim: int, num_heads: int, q: int = 2**31 - 1):
        """
        Initialize secure multi-head attention.
        
        Args:
            hidden_dim: Hidden dimension
            num_heads: Number of attention heads
            q: Prime modulus
        """
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.q = q
        self.beaver = BeaverTriples(q)
        self.fixed_point = FixedPoint(q=q)
        self.secret_sharing = SecretSharing(q)
        self.activations = SecureActivations(q)
        
        # Scale factor for attention scores
        self.scale = 1.0 / np.sqrt(self.head_dim)
        self.scale_encoded = self.fixed_point.encode(self.scale)
    
    def set_weights(self, q_share1: np.ndarray, q_share2: np.ndarray,
                   k_share1: np.ndarray, k_share2: np.ndarray,
                   v_share1: np.ndarray, v_share2: np.ndarray,
                   out_share1: np.ndarray = None, out_share2: np.ndarray = None):
        """
        Set secret-shared query, key, value, and output projection weights.
        
        Args:
            q_share1, q_share2: Shares of Q projection (hidden_dim, hidden_dim)
            k_share1, k_share2: Shares of K projection (hidden_dim, hidden_dim)
            v_share1, v_share2: Shares of V projection (hidden_dim, hidden_dim)
            out_share1, out_share2: Optional shares of output projection (hidden_dim, hidden_dim)
        """
        self.q_proj_share1 = q_share1
        self.q_proj_share2 = q_share2
        self.k_proj_share1 = k_share1
        self.k_proj_share2 = k_share2
        self.v_proj_share1 = v_share1
        self.v_proj_share2 = v_share2
        
        if out_share1 is not None and out_share2 is not None:
            self.out_proj_share1 = out_share1
            self.out_proj_share2 = out_share2
        else:
            # Identity projection
            self.out_proj_share1 = np.eye(self.hidden_dim, dtype=np.int64) * self.fixed_point.scale
            self.out_proj_share2 = np.zeros((self.hidden_dim, self.hidden_dim), dtype=np.int64)
    
    def forward(self, x_share1: np.ndarray, x_share2: np.ndarray,
               seq_len: int, triples: Dict[str, Dict[str, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass: multi-head self-attention.
        
        Args:
            x_share1, x_share2: Shares of input (seq_len, hidden_dim)
            seq_len: Sequence length
            triples: Dictionary containing Beaver triples for:
                - 'qk': Q @ K^T multiplication
                - 'av': attention @ V multiplication
                - 'q_proj': input @ Q projection
                - 'k_proj': input @ K projection
                - 'v_proj': input @ V projection
                - 'out_proj': output projection
        
        Returns:
            Shares of attention output (seq_len, hidden_dim)
        """
        # Step 1: Project to Q, K, V
        # Q = x @ W_q, K = x @ W_k, V = x @ W_v
        
        # Q projection
        q_triple = triples.get('q_proj', self.beaver.generate_matmul_triple(seq_len, self.hidden_dim, self.hidden_dim))
        q_share1, q_share2 = self.beaver.secure_matmul(
            x_share1, x_share2,
            self.q_proj_share1.T, self.q_proj_share2.T,
            q_triple
        )
        
        # K projection
        k_triple = triples.get('k_proj', self.beaver.generate_matmul_triple(seq_len, self.hidden_dim, self.hidden_dim))
        k_share1, k_share2 = self.beaver.secure_matmul(
            x_share1, x_share2,
            self.k_proj_share1.T, self.k_proj_share2.T,
            k_triple
        )
        
        # V projection
        v_triple = triples.get('v_proj', self.beaver.generate_matmul_triple(seq_len, self.hidden_dim, self.hidden_dim))
        v_share1, v_share2 = self.beaver.secure_matmul(
            x_share1, x_share2,
            self.v_proj_share1.T, self.v_proj_share2.T,
            v_triple
        )
        
        # Reshape for multi-head: (seq_len, num_heads, head_dim)
        q_share1 = q_share1.reshape(seq_len, self.num_heads, self.head_dim)
        q_share2 = q_share2.reshape(seq_len, self.num_heads, self.head_dim)
        k_share1 = k_share1.reshape(seq_len, self.num_heads, self.head_dim)
        k_share2 = k_share2.reshape(seq_len, self.num_heads, self.head_dim)
        v_share1 = v_share1.reshape(seq_len, self.num_heads, self.head_dim)
        v_share2 = v_share2.reshape(seq_len, self.num_heads, self.head_dim)
        
        # Step 2: Compute attention scores: scores = Q @ K^T / sqrt(d_k)
        # For each head: (seq_len, head_dim) @ (head_dim, seq_len) = (seq_len, seq_len)
        
        # Transpose K: (seq_len, head_dim) -> (head_dim, seq_len)
        k_t_share1 = np.transpose(k_share1, (0, 2, 1))  # (num_heads, head_dim, seq_len)
        k_t_share2 = np.transpose(k_share2, (0, 2, 1))
        
        # Reshape for batch matrix multiplication
        q_flat_share1 = q_share1.reshape(seq_len * self.num_heads, self.head_dim)
        q_flat_share2 = q_share2.reshape(seq_len * self.num_heads, self.head_dim)
        k_t_flat_share1 = k_t_share1.reshape(self.num_heads, self.head_dim, seq_len)
        k_t_flat_share2 = k_t_share2.reshape(self.num_heads, self.head_dim, seq_len)
        
        # Compute Q @ K^T for each head
        scores_share1_list = []
        scores_share2_list = []
        
        for head_idx in range(self.num_heads):
            q_head_share1 = q_flat_share1[head_idx * seq_len:(head_idx + 1) * seq_len]
            q_head_share2 = q_flat_share2[head_idx * seq_len:(head_idx + 1) * seq_len]
            k_t_head_share1 = k_t_flat_share1[head_idx]
            k_t_head_share2 = k_t_flat_share2[head_idx]
            
            qk_triple = triples.get('qk', self.beaver.generate_matmul_triple(seq_len, self.head_dim, seq_len))
            scores_head_share1, scores_head_share2 = self.beaver.secure_matmul(
                q_head_share1, q_head_share2,
                k_t_head_share1, k_t_head_share2,
                qk_triple
            )
            
            scores_share1_list.append(scores_head_share1)
            scores_share2_list.append(scores_head_share2)
        
        scores_share1 = np.stack(scores_share1_list, axis=0)  # (num_heads, seq_len, seq_len)
        scores_share2 = np.stack(scores_share2_list, axis=0)
        
        # Scale scores: divide by sqrt(d_k)
        # Multiply by scale factor (public)
        scores_share1 = (scores_share1 * self.scale_encoded // self.fixed_point.scale) % self.q
        scores_share2 = (scores_share2 * self.scale_encoded // self.fixed_point.scale) % self.q
        
        # Step 3: Apply softmax
        # Reshape for softmax: (num_heads * seq_len, seq_len)
        scores_flat_share1 = scores_share1.reshape(self.num_heads * seq_len, seq_len)
        scores_flat_share2 = scores_share2.reshape(self.num_heads * seq_len, seq_len)
        
        attn_flat_share1, attn_flat_share2 = self.activations.softmax(
            scores_flat_share1, scores_flat_share2, axis=-1
        )
        
        attn_share1 = attn_flat_share1.reshape(self.num_heads, seq_len, seq_len)
        attn_share2 = attn_flat_share2.reshape(self.num_heads, seq_len, seq_len)
        
        # Step 4: Apply attention to values: output = attention @ V
        # For each head: (seq_len, seq_len) @ (seq_len, head_dim) = (seq_len, head_dim)
        output_share1_list = []
        output_share2_list = []
        
        for head_idx in range(self.num_heads):
            attn_head_share1 = attn_share1[head_idx]
            attn_head_share2 = attn_share2[head_idx]
            v_head_share1 = v_share1[:, head_idx, :]
            v_head_share2 = v_share2[:, head_idx, :]
            
            av_triple = triples.get('av', self.beaver.generate_matmul_triple(seq_len, seq_len, self.head_dim))
            output_head_share1, output_head_share2 = self.beaver.secure_matmul(
                attn_head_share1, attn_head_share2,
                v_head_share1, v_head_share2,
                av_triple
            )
            
            output_share1_list.append(output_head_share1)
            output_share2_list.append(output_head_share2)
        
        # Concatenate heads: (seq_len, num_heads, head_dim) -> (seq_len, hidden_dim)
        output_share1 = np.concatenate(output_share1_list, axis=-1)
        output_share2 = np.concatenate(output_share2_list, axis=-1)
        
        # Step 5: Output projection
        out_triple = triples.get('out_proj', self.beaver.generate_matmul_triple(seq_len, self.hidden_dim, self.hidden_dim))
        final_share1, final_share2 = self.beaver.secure_matmul(
            output_share1, output_share2,
            self.out_proj_share1.T, self.out_proj_share2.T,
            out_triple
        )
        
        return final_share1, final_share2
