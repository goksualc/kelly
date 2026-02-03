"""
Main entry point for secure chatbot system
"""

import argparse
import sys
from secure_chatbot.parties.gateway import Gateway
from secure_chatbot.parties.computing_party import ComputingParty
from secure_chatbot.parties.preprocessing_server import PreprocessingServer


def main():
    parser = argparse.ArgumentParser(description='Secure AI Chatbot using MPC')
    parser.add_argument('--mode', choices=['gateway', 'cp1', 'cp2', 'cp0'], required=True,
                       help='Mode to run in')
    parser.add_argument('--host', default='localhost', help='Host address')
    parser.add_argument('--port', type=int, default=8000, help='Port number')
    parser.add_argument('--config', type=str, help='Configuration file path')
    
    args = parser.parse_args()
    
    if args.mode == 'gateway':
        gateway = Gateway()
        print("Gateway started. Waiting for connections...")
        # In real implementation, would start server
        gateway.communicator.connect_all()
        
    elif args.mode == 'cp1':
        cp1 = ComputingParty('cp1')
        print("CP1 started. Waiting for connections...")
        cp1.communicator.connect_all()
        
    elif args.mode == 'cp2':
        cp2 = ComputingParty('cp2')
        print("CP2 started. Waiting for connections...")
        cp2.communicator.connect_all()
        
    elif args.mode == 'cp0':
        cp0 = PreprocessingServer()
        print("Preprocessing server (CP0) started.")
        # Generate triples
        config = {
            'num_layers': 12,
            'hidden_dim': 768,
            'num_heads': 12,
            'vocab_size': 50257,
            'max_seq_len': 512,
            'ffn_dim': 3072
        }
        triples = cp0.generate_triples_for_model(config, num_conversations=10)
        print(f"Generated {len(triples)} sets of triples")
        
    print("Running...")


if __name__ == '__main__':
    main()
