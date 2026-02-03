"""
Computing Party (CP1 or CP2)

Holds share of model weights and user inputs.
Performs secure computation collaboratively.
Communicates only during Beaver partition reconstruction.
"""

import numpy as np
from typing import Dict, Tuple, Optional
from ..crypto.secret_sharing import SecretSharing
from ..crypto.fixed_point import FixedPoint
from ..nn.transformer import SecureTransformer
from ..utils.communication import PartyCommunicator


class ComputingParty:
    """Computing party for secure MPC"""
    
    def __init__(self, party_id: str, q: int = 2**31 - 1):
        """
        Initialize computing party.
        
        Args:
            party_id: Party identifier ('cp1' or 'cp2')
            q: Prime modulus
        """
        self.party_id = party_id
        self.q = q
        self.secret_sharing = SecretSharing(q)
        self.fixed_point = FixedPoint(q=q)
        self.communicator = PartyCommunicator(party_id)
        
        # Model state
        self.model = None
        self.model_weights_share = None
        self.beaver_triples = None
    
    def load_model_share(self, weights_share: Dict):
        """
        Load secret-shared model weights.
        
        Args:
            weights_share: Dictionary of weight shares for this party
        """
        self.model_weights_share = weights_share
    
    def initialize_model(self, config: Dict):
        """
        Initialize transformer model with secret-shared weights.
        
        Args:
            config: Model configuration
        """
        self.model = SecureTransformer(config)
        
        # Set embeddings
        if 'token_embedding' in self.model_weights_share:
            if self.party_id == 'cp1':
                self.model.set_embeddings(
                    self.model_weights_share['token_embedding']['share1'],
                    None,  # CP2's share not needed here
                    self.model_weights_share.get('position_embedding', {}).get('share1'),
                    None
                )
            else:  # cp2
                self.model.set_embeddings(
                    None,
                    self.model_weights_share['token_embedding']['share2'],
                    None,
                    self.model_weights_share.get('position_embedding', {}).get('share2')
                )
        
        # Set layer weights (simplified - would need proper structure)
        # In practice, would iterate through layers and set weights
    
    def load_beaver_triples(self, triples: Dict):
        """
        Load Beaver triples for computation.
        
        Args:
            triples: Dictionary of Beaver triple shares
        """
        self.beaver_triples = triples
    
    def receive_input_shares(self, input_share: np.ndarray) -> np.ndarray:
        """
        Receive secret-shared input from gateway.
        
        Args:
            input_share: This party's share of the input
            
        Returns:
            Input share (stored for computation)
        """
        return input_share
    
    def secure_forward(self, input_share: np.ndarray, seq_len: int,
                      other_party_id: str) -> Tuple[np.ndarray, Dict]:
        """
        Perform secure forward pass.
        
        Args:
            input_share: This party's share of input
            seq_len: Sequence length
            other_party_id: ID of other computing party
            
        Returns:
            Tuple of (output_share, communication_log)
        """
        if self.model is None:
            raise ValueError("Model not initialized")
        
        if self.beaver_triples is None:
            raise ValueError("Beaver triples not loaded")
        
        # Get other party's input share (requires communication)
        other_input_share = self.communicator.receive_from(other_party_id)
        
        # Perform secure computation
        # In practice, this would involve multiple rounds of communication
        # For now, simplified version
        
        # Reconstruct input (in real protocol, would keep secret-shared)
        input_reconstructed = (input_share + other_input_share) % self.q
        
        # Convert to proper format
        input_ids = input_reconstructed.astype(np.int32)
        input_ids_share1, input_ids_share2 = self.secret_sharing.share(input_ids)
        
        if self.party_id == 'cp1':
            input_ids_share = input_ids_share1
        else:
            input_ids_share = input_ids_share2
        
        # Forward pass (simplified - would use proper secure operations)
        # This is a placeholder - real implementation would use secure operations throughout
        output_share = input_share  # Placeholder
        
        communication_log = {
            'rounds': 1,
            'bytes_sent': 0,
            'bytes_received': 0
        }
        
        return output_share, communication_log
    
    def send_output_share(self, output_share: np.ndarray, gateway_id: str = 'gateway'):
        """
        Send output share to gateway for reconstruction.
        
        Args:
            output_share: This party's share of output
            gateway_id: Gateway party ID
        """
        self.communicator.send_to(gateway_id, {
            'type': 'output_share',
            'share': output_share,
            'party_id': self.party_id
        })
