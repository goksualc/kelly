"""
Preprocessing Server (CP0)

Generates Beaver triples offline (no input dependency).
Can precompute days in advance.
No online participation.
"""

import numpy as np
from typing import Dict, List
from ..crypto.beaver_triples import BeaverTriples
from ..crypto.fixed_point import FixedPoint
from ..utils.prg import PRG


class PreprocessingServer:
    """Preprocessing server for offline triple generation"""
    
    def __init__(self, q: int = 2**31 - 1):
        """
        Initialize preprocessing server.
        
        Args:
            q: Prime modulus
        """
        self.q = q
        self.beaver = BeaverTriples(q)
        self.fixed_point = FixedPoint(q=q)
    
    def generate_triples_for_layer(self, layer_idx: int, seq_len: int, hidden_dim: int,
                                   num_heads: int, ffn_dim: int, vocab_size: int) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Generate all Beaver triples needed for one transformer layer.
        
        Args:
            layer_idx: Layer index
            seq_len: Sequence length
            hidden_dim: Hidden dimension
            num_heads: Number of attention heads
            ffn_dim: Feedforward network dimension
            vocab_size: Vocabulary size
            
        Returns:
            Dictionary of triples for:
                - attention: QK and AV multiplications
                - ffn: Linear layer multiplications
        """
        head_dim = hidden_dim // num_heads
        
        triples = {
            'attention': {},
            'ffn': {}
        }
        
        # Attention triples
        # Q @ K^T: (seq_len, head_dim) @ (head_dim, seq_len) = (seq_len, seq_len)
        for head_idx in range(num_heads):
            qk_triple = self.beaver.generate_matmul_triple(seq_len, head_dim, seq_len)
            triples['attention'][f'qk_head_{head_idx}'] = qk_triple
        
        # Attention @ V: (seq_len, seq_len) @ (seq_len, head_dim) = (seq_len, head_dim)
        for head_idx in range(num_heads):
            av_triple = self.beaver.generate_matmul_triple(seq_len, seq_len, head_dim)
            triples['attention'][f'av_head_{head_idx}'] = av_triple
        
        # Projection triples (if needed)
        # Input @ Q_proj: (seq_len, hidden_dim) @ (hidden_dim, hidden_dim) = (seq_len, hidden_dim)
        triples['attention']['q_proj'] = self.beaver.generate_matmul_triple(seq_len, hidden_dim, hidden_dim)
        triples['attention']['k_proj'] = self.beaver.generate_matmul_triple(seq_len, hidden_dim, hidden_dim)
        triples['attention']['v_proj'] = self.beaver.generate_matmul_triple(seq_len, hidden_dim, hidden_dim)
        triples['attention']['out_proj'] = self.beaver.generate_matmul_triple(seq_len, hidden_dim, hidden_dim)
        
        # FFN triples
        # Linear1: (seq_len, hidden_dim) @ (hidden_dim, ffn_dim) = (seq_len, ffn_dim)
        triples['ffn']['linear1'] = self.beaver.generate_matmul_triple(seq_len, hidden_dim, ffn_dim)
        
        # Linear2: (seq_len, ffn_dim) @ (ffn_dim, hidden_dim) = (seq_len, hidden_dim)
        triples['ffn']['linear2'] = self.beaver.generate_matmul_triple(seq_len, ffn_dim, hidden_dim)
        
        return triples
    
    def generate_triples_for_model(self, config: Dict, num_conversations: int = 100) -> List[Dict[str, Dict[str, np.ndarray]]]:
        """
        Generate all Beaver triples for entire model.
        
        Args:
            config: Model configuration
            num_conversations: Number of conversations to preprocess
            
        Returns:
            List of triple dictionaries (one per conversation)
        """
        num_layers = config['num_layers']
        seq_len = config.get('max_seq_len', 512)
        hidden_dim = config['hidden_dim']
        num_heads = config['num_heads']
        ffn_dim = config.get('ffn_dim', 4 * hidden_dim)
        vocab_size = config['vocab_size']
        
        all_triples = []
        
        for conv_idx in range(num_conversations):
            conversation_triples = []
            
            for layer_idx in range(num_layers):
                layer_triples = self.generate_triples_for_layer(
                    layer_idx, seq_len, hidden_dim, num_heads, ffn_dim, vocab_size
                )
                conversation_triples.append(layer_triples)
            
            # Language model head triples
            lm_head_triple = self.beaver.generate_matmul_triple(seq_len, hidden_dim, vocab_size)
            conversation_triples.append({'lm_head': lm_head_triple})
            
            all_triples.append(conversation_triples)
        
        return all_triples
    
    def distribute_triples(self, triples: Dict, cp1_channel, cp2_channel):
        """
        Distribute triples to computing parties.
        
        Args:
            triples: Triple dictionary
            cp1_channel: Communication channel to CP1
            cp2_channel: Communication channel to CP2
        """
        # Split each triple into shares for CP1 and CP2
        triples_cp1 = {}
        triples_cp2 = {}
        
        for key, triple_dict in triples.items():
            if isinstance(triple_dict, dict):
                triples_cp1[key] = {}
                triples_cp2[key] = {}
                
                for sub_key, triple in triple_dict.items():
                    # Extract CP1 and CP2 shares
                    triples_cp1[key][sub_key] = {
                        'A_mask_share1': triple['A_mask_share1'],
                        'B_mask_share1': triple['B_mask_share1'],
                        'AB_mask_share1': triple['AB_mask_share1'],
                    }
                    triples_cp2[key][sub_key] = {
                        'A_mask_share2': triple['A_mask_share2'],
                        'B_mask_share2': triple['B_mask_share2'],
                        'AB_mask_share2': triple['AB_mask_share2'],
                    }
        
        # Send to parties
        cp1_channel.send(triples_cp1)
        cp2_channel.send(triples_cp2)
    
    def generate_shared_seeds(self, num_seeds: int = 1000) -> List[tuple]:
        """
        Generate shared PRG seeds for CP0-CP1 and CP0-CP2.
        
        Args:
            num_seeds: Number of seed pairs to generate
            
        Returns:
            List of (seed_01, seed_02) tuples
        """
        seeds = []
        for _ in range(num_seeds):
            seed_01, seed_02 = PRG.generate_shared_seeds()
            seeds.append((seed_01, seed_02))
        return seeds
