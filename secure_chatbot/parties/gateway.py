"""
Secret Sharing Gateway

Accepts user inputs, splits into shares, distributes to CP1 and CP2.
Reconstructs final outputs.
"""

import numpy as np
from typing import Dict, Tuple, Optional
from ..crypto.secret_sharing import SecretSharing
from ..crypto.fixed_point import FixedPoint
from ..utils.communication import PartyCommunicator


class Gateway:
    """Gateway for secret sharing and user interface"""
    
    def __init__(self, q: int = 2**31 - 1):
        """
        Initialize gateway.
        
        Args:
            q: Prime modulus
        """
        self.q = q
        self.secret_sharing = SecretSharing(q)
        self.fixed_point = FixedPoint(q=q)
        self.communicator = PartyCommunicator('gateway')
    
    def secret_share_input(self, user_input: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Secret share user input.
        
        Args:
            user_input: User input (token IDs or embeddings)
            
        Returns:
            Tuple of (share1, share2) to send to CP1 and CP2
        """
        # Encode if needed (if input is float)
        if user_input.dtype == np.float32 or user_input.dtype == np.float64:
            user_input_encoded = self.fixed_point.encode(user_input)
        else:
            user_input_encoded = user_input.astype(np.int64) % self.q
        
        # Secret share
        share1, share2 = self.secret_sharing.share(user_input_encoded)
        
        return share1, share2
    
    def distribute_input(self, input_share1: np.ndarray, input_share2: np.ndarray):
        """
        Distribute input shares to computing parties.
        
        Args:
            input_share1: Share for CP1
            input_share2: Share for CP2
        """
        self.communicator.send_to('cp1', {
            'type': 'input_share',
            'share': input_share1
        })
        self.communicator.send_to('cp2', {
            'type': 'input_share',
            'share': input_share2
        })
    
    def reconstruct_output(self, output_share1: np.ndarray, output_share2: np.ndarray,
                          decode: bool = True) -> np.ndarray:
        """
        Reconstruct output from shares.
        
        Args:
            output_share1: CP1's output share
            output_share2: CP2's output share
            decode: Whether to decode from fixed-point
            
        Returns:
            Reconstructed output
        """
        # Reconstruct
        output = self.secret_sharing.reconstruct(output_share1, output_share2)
        
        # Decode if needed
        if decode:
            output = self.fixed_point.decode(output)
        
        return output
    
    def receive_outputs(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Receive output shares from computing parties.
        
        Returns:
            Tuple of (output_share1, output_share2)
        """
        msg1 = self.communicator.receive_from('cp1')
        msg2 = self.communicator.receive_from('cp2')
        
        output_share1 = msg1['share']
        output_share2 = msg2['share']
        
        return output_share1, output_share2
    
    def chat(self, user_message: str, tokenizer, max_tokens: int = 50) -> str:
        """
        Process chat message through secure inference.
        
        Args:
            user_message: User's message
            tokenizer: Tokenizer for encoding text
            max_tokens: Maximum tokens to generate
            
        Returns:
            Model response
        """
        # Tokenize input
        input_ids = tokenizer.encode(user_message)
        input_ids = np.array(input_ids, dtype=np.int32)
        
        # Secret share
        share1, share2 = self.secret_share_input(input_ids)
        
        # Distribute
        self.distribute_input(share1, share2)
        
        # Wait for outputs (in real implementation, would handle async)
        output_share1, output_share2 = self.receive_outputs()
        
        # Reconstruct
        output_ids = self.reconstruct_output(output_share1, output_share2, decode=False)
        output_ids = output_ids.astype(np.int32)
        
        # Decode tokens
        response = tokenizer.decode(output_ids.tolist())
        
        return response
